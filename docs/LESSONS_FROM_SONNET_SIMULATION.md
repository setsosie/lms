# Lessons from the Sonnet-team simulation (2026-08-20/21)

**Status: IN PROGRESS.** Generations 1–3 complete, run continues to generation 10.
Sections marked *(interim)* will be revised when the run finishes.

## What this experiment is

A hand-orchestrated LMS run in which the agents are Claude Sonnet subagents
instead of the box's `lms-generalist` (Qwen3.6-27B-FP8), grading through the
project's **real** oracle: `Society._verify_admissible` → `RealLeanVerifier`
→ `default_gate_runner` → `FoundationFile`, against a real Mathlib on the local
box. Only *who writes the Lean* is simulated. Verdicts mean exactly what they
mean in `committee_fix_c`.

Structure, deliberately simplified from the real harness: one long-lived
coordinator agent allocates three tasks per generation; three fresh researcher
agents each get one shot with no repair loop; no planning panel; no peer review.

**Goal: `stacks-ch4-phase1`.** Note this goal sets
`forbidden_imports: ['Mathlib.CategoryTheory']` — agents must bootstrap category
theory from scratch. Every artifact is therefore N0 **by construction**, and
**this goal cannot produce a nonzero CVFN regardless of which model runs it.**
It is a harness shakedown, not a novelty measurement.

## Scoreboard (interim)

| Gen | Sonnet team | Qwen `committee_fix_c` |
|-----|-------------|------------------------|
| 1 | 3/3 verified | 3/3 — one was a placeholder comment |
| 2 | 3/3 verified | 2/9 |
| 3 | 2/3 verified | 3/12 |
| 4 | 1/3 as submitted; 2/3 after one mechanical repair | 1/4 |
| 5 | 3/3 verified — **goal complete, 9/9 tags** | 0/7 |
| 6–10 | *(goal already complete at gen 5)* | 0/6, 0/6, 0/7, 0/7, 1/10 |

**The nine-tag goal was completed in five generations**, versus 2/9 in ten on
the box. Final foundation: **303 lines, 27 declarations, 11 theorems**,
`lake build LMS.Foundation` green. The box's after ten generations: ~65 lines,
4 structures, **zero theorems**.

Verified/submitted: 12/15 as submitted, 13/15 counting one one-line repair.
Box: 10/71, of which 3 were the prompt scaffold.

Both milestones landed with real proofs — `yonedaEmbedding_fullyFaithful`
(reusing the already-proved Yoneda lemmas rather than reproving them) and
`pullback_pasting`. Four of these textbook results scored **N1**; see below.

Foundation after generation 3: **157 lines, 14 declarations, `lake build
LMS.Foundation` green (501 jobs).** Contains `Category` (+2 proved models),
`Functor`, `Functor.id`, `Category.op` (+2 lemmas), `IsProduct`,
`isProduct_unique_up_to_iso`, `NatTrans`, `NatTrans.id`, `homFunctor`.

Qwen's foundation after ten generations: 4 structures, no theorems.

## Harness defects found (fix these regardless of model)

### 1. The verifier is stricter than the foundation it writes into

`Foundation.lean` declares `universe u v w` (line 15) *above* `namespace
LMS.Foundation` (line 24). `foundation.py` strips `universe` lines out of
incoming artifacts, on the stated grounds that the header already has them. But
`RealLeanVerifier._wrap_in_storage_namespace` (`real.py:185`) wraps a candidate
in the **namespace only** — no universe header.

Net effect: an artifact that omits `universe u v` fails verification on code
that would compile fine once merged. This cost generation 3 a complete and
correct `IsPullback` plus a uniqueness-up-to-iso theorem, on
`error: unknown universe level 'v'`.

**Fix:** wrap candidates in the same preamble the foundation uses. One line.
Weaker models omit boilerplate more often, so this taxes Qwen hardest.

### 2. Conversations are never recorded

`TraceStore.add_conversation()` and `add_trace()` exist and are **never called
anywhere in `lms/`**. The only reference is `trace_store.save()` at
`society.py:1586`. So `conversations.json` and `reasoning_traces.json` are
empty (2 bytes) in every run, in every mode.

Committee transcripts *are* captured, but they go to the textbook instead
(`society.py:1202`, `entry_type="transcript"`) — which is why `textbook.json` is
4.2 MB. The data exists; it is filed under the wrong name and mixed in with
pedagogical entries.

**Fix:** call `add_conversation` from the committee path, or drop `TraceStore`
and document that transcripts live in the textbook. The present state is the
worst of both — a file that looks like a feature and is always empty.

### 3. `FoundationFile.add_artifact` dedupes silently

Returns early, with no exception and no return value, on an artifact id it has
seen. A caller cannot distinguish "added" from "already present", so it reports
a successful promotion while accumulating nothing. Latent in production (ids are
unique there) but it cost an hour here.

**Fix:** return `bool`, or raise.

### 4. `lake build` of the merged foundation is still not wired

`foundation.py:131-134` already predicts this failure mode. Running
`lake build LMS.Foundation` after each generation takes seconds (501 jobs) and
is the **only** check that catches cross-artifact incompatibility — which is
exactly what killed `committee_fix_c` generations 5–9.

### 4b. `N1` does not mean novel — the gate can be fooled by a bespoke API

**This is the most consequential finding here and it threatens CVFN directly.**

The Yoneda Lemma, proved over this run's hand-rolled bundled `Category`, was
classified **N1 at confidence 0.90 with `needs_review: false`** by Gate 4 —
the maximum-strength novelty verdict, requiring no D4 sign-off. Measured
verbatim (Mathlib rev `fe3134f0`):

```json
{"level": "N1", "confidence": 0.9, "needs_review": false,
 "evidence": ["not found by: name, loogle, exact_probe, semantic"],
 "stages_available": ["name", "loogle", "exact_probe", "semantic"],
 "stages_unavailable": []}
```

**All four stages ran.** This is not a degraded-coverage artifact — the ladder
had full reach and still missed. The classification was also run *with* the
informal statement "Yoneda lemma: natural transformations Hom(X,-) => F
correspond to elements of F(X)" supplied, so the semantic backend had the words
"Yoneda lemma" and still returned nothing above threshold.

Yoneda is the opposite of novel. It scored N1 because the search ladder looks
for *statement-level* matches in Mathlib, and a from-scratch API matches
nothing there — different names, different types, different structure. So:

> **N1 means "not found in Mathlib", not "mathematically novel."**

Any goal that forbids Mathlib and forces bootstrapping will systematically
produce N1 for results that are centuries old in substance. If CVFN is computed
as *verified ∧ N1*, such a goal inflates it structurally — and nobody
downstream would be able to tell from the number.

Note the direction of the error matters: the gate's design assumes an N0 hit is
evidence of duplication and its absence is evidence of novelty. That inference
is invalid whenever the artifact's vocabulary is disjoint from Mathlib's.

**Suggested fixes, in order:**
1. Treat N1 as INCONCLUSIVE whenever the artifact does not import the Mathlib
   namespace the concept would live in — absence of a match is not evidence
   when the search could not have matched.
2. Score novelty on the *informal* statement as well as the Lean, so "Yoneda
   lemma" is recognisable regardless of the API it is written against.
3. At minimum, never report a CVFN computed on a goal with
   `forbidden_imports` covering the relevant Mathlib area.

### 4c. Measure pass@k, not pass@1

Generation 4 was accidentally a k=2 experiment on Yoneda: two agents,
independent structurings (Mathlib `Equiv` vs. three separate declarations).
Both failed as submitted. Both had the **correct mathematical construction**
and died on trailing mechanical steps — one rewriting under a structure
literal, the other needing `typeCategory.id` unfolded to the identity function.
Adding a single `rfl` verified the second one.

Generation 1 was accidentally k=3 on `Category`: 3/3, near-identical.

A pass@1 harness reports "capable but doesn't land" and "not capable"
identically, as `failed`. That is why `committee_fix_c`'s 10/71 is
uninterpretable: it cannot distinguish "Qwen cannot do category theory" from
"Qwen cannot finish a proof it has already correctly set up" — and those imply
completely different fixes (change the model vs. build the repair loop).

**Build a pass@k harness.** Fix a task, draw k independent samples, report
pass@1, pass@k, and a **failure-class histogram**. The histogram is the payload.
Cheap locally — wall-clock, not API spend. This also feeds the model-selection
pass@k axis already recorded for the BFS-Prover comparison.

## Design lessons (transferable to weaker models)

### 5. Seed the foundational representation; don't let generation 1 vote on it

The single most load-bearing decision in the whole run is how `Category` is
represented, and it is currently made by whichever agent happens to land first.

- Box: `structure Category (C : Type u)` — *parameterized*. This invites
  `[C : Category]`, and Qwen wrote exactly that for five straight generations:
  `invalid binder annotation, type is not a class`.
- Sonnet team: `structure Category where Obj : Type u` — *bundled*. There is no
  separate type argument, so the binder mistake is structurally impossible.

All three Sonnet agents independently chose bundled, but that is luck, not
architecture. **Ship a hand-written, human-reviewed axiom layer as generation
0.** This is probably the highest-leverage change for a weak model.

### 6. Put exact signatures in each assignment, not only in the shared context

The foundation context block *already* warns: "Even when a name below matches a
Mathlib concept, Mathlib's API for it does NOT exist here… class-style
`Category C`, `.Hom`, `𝟙` — is an unknown identifier." Qwen read that and did it
anyway, across five generations.

The coordinator here pasted the literal field list into **every task string**.
Across nine submissions, not one researcher made a shape error. Per-task
specificity beats a global preamble — the same information, positioned
differently, changed the outcome.

### 7. Require a concrete instance with every definition

Unprompted, one researcher exhibited `Type u`-with-functions and a one-object
category as models of its `Category` axioms, and Lean accepted both. That proves
the axiomatization is *satisfiable*, not merely well-formed.

This is the check that would have caught `committee_fix_c`'s `Opposite`, whose
`Hom_op : C → C → Type v` was a free field with no relation whatsoever to
`C.Hom` — vacuous, and verified. `gates/vacuity.py` cannot evaluate a structure;
"exhibit a model and let Lean check it" can.

**Make it a prompt requirement and a gate.**

### 8. Make the planner dependency-aware, including sibling invisibility

The coordinator refused to assign Yoneda in generation 3 because `NatTrans` was
being built that same generation and would be invisible to its sibling. It
assigned Hom-functor scaffolding instead, shortening generation 4's critical
path.

The box's planning panel repeatedly assigned every group to 0019 and invented
tags (`STANDBY`, `STANDBY_B`, `TAG`) that the guard then dropped. This is
planner logic, not model capability.

### 9. Seed naming conventions

`IsPullback` reused `IsProduct`'s field names (`p1`/`p2`,
`factor_exists`/`factor_unique`) because the coordinator instructed it to. Cheap
insurance against the four-mutually-incompatible-`Functor`s problem Qwen had.

### 10. Repair loops pay — but rank them after defect #1

The only Sonnet failure so far was a one-line `universe` omission: exactly the
"single-retry-fixable" class that motivated `26Q3-HARN-14`. Worth building, but
removing the cause (defect #1) beats repairing the symptom.

## What this does NOT show

- **Not a capability verdict on its own.** The Sonnet team got clean prompts
  with exact signatures and no repair budget (8/9); the box got noisier
  assignments *with* two repair attempts (10/71). The gap is partly capability
  and partly harness, and this experiment does not separate them. Lessons 1–9
  are harness and would lift Qwen too.
- **Not a CVFN number.** Everything here is N0 by construction — see the goal
  note above.
- **Not the same pipeline.** No planning panel, no peer review, no repair loop,
  one-shot per generation.

## Open questions for later

- Evolving prompts over generations (deferred — relates to
  the soft-prompt-genome arm, which is gated on Phase C).
- A goal outside Mathlib's coverage, so CVFN can be nonzero at all.
