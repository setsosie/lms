### 26Q3-HARN-20: Gate 4 reports decisive N1 for textbook theorems in a bespoke API

**User Story**: As the calibration program, I want Gate 4 to withhold a novelty
claim when its search could not have matched, so that CVFN counts genuinely
novel results rather than well-known theorems restated against a from-scratch
API.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH — CVFN is the number the whole Q3 program turns on |
| **Status** | 🔄 IN PROGRESS — Part 1 merged (#63), Part 2 in review |
| **Branch** | Part 1: `26Q3-HARN-20-part1-stages-that-cannot-match`; Part 2: `26Q3-HARN-20-novelty-gate-n1-false-positive` |
| **Dependencies** | 26Q3-HARN-19 (wires Gate 4 into the run at all) |
| **PR Size Target** | <400 lines per part |
| **Parts** | 2 (split 2026-10-08; points unchanged) |

---

#### Context

> Found 2026-08-21 during the Sonnet-team simulation
> (`docs/LESSONS_FROM_SONNET_SIMULATION.md`). Graded through the real
> `RealLeanVerifier` against Mathlib rev `fe3134f0`.

The **Yoneda Lemma**, proved over a hand-rolled bundled `Category`, was
classified by Gate 4 as:

```json
{"level": "N1", "confidence": 0.9, "needs_review": false,
 "evidence": ["not found by: name, loogle, exact_probe, semantic"],
 "stages_available": ["name", "loogle", "exact_probe", "semantic"],
 "stages_unavailable": []}
```

**All four search stages ran.** This is not degraded coverage. Confidence 0.9 is
the maximum the ladder assigns to N1, and `needs_review: false` means it would
**not** route to D4 — it would count toward CVFN unchallenged.

The classification was run *with* the informal statement "Yoneda lemma: natural
transformations Hom(X,-) => F correspond to elements of F(X)" supplied, so the
semantic backend had the words "Yoneda lemma" and still returned nothing above
threshold.

Three further textbook results scored N1 in the same run:
`yonedaEmbedding_fullyFaithful`, `pullback_pasting`, and
`isProduct_unique_up_to_iso` (INCONCLUSIVE).

#### Root cause

`NoveltyClassifier._verdict` (`lms/novelty/__init__.py`) infers novelty from
**absence of a match**, and scales confidence by how many stages were
*available* — not by whether a match was *possible*. When the artifact's
vocabulary is disjoint from Mathlib's (different names, different types,
different structure), every stage necessarily comes up empty, and the ladder
reads that unanimous silence as strong evidence of novelty.

The inference "no stage matched → novel" is invalid whenever the search could
not have matched in principle.

This bites hardest on exactly the goals the program uses: `stacks-ch4-phase1`
sets `forbidden_imports: ['Mathlib.CategoryTheory']`, mandating a from-scratch
API — so **every** artifact it produces is structurally unmatchable, and a CVFN
computed over it is inflated by construction.

#### Part 1 (2026-10-08): only a search that ran may vote "absent"

Found while preparing the Sprint 3 N1-density run (runbook Step 7), where the
same mechanism would have biased the slice decision. The root cause above,
made mechanical: a stage that **could not have matched** still reported
`available=True` with no hits, so the classifier counted it as a search that
ran and found nothing. Five places did this:

| Stage | Case | Before | After |
|---|---|---|---|
| `exact_probe` | statement does not elaborate in pure Mathlib | absence vote — or a spurious N0, when error recovery leaves a goal `rfl` closes | `available=False`, `statement did not elaborate` |
| `exact_probe` | first declaration is not a theorem (31/52 of the Gate A control) | absence vote | `available=False` |
| `exact_probe` | timed out | absence vote | `available=False` |
| `name`, `loogle` | no parseable declaration name (11/52 of the control) | absence vote | `available=False` |
| `name`, `loogle` | the name is a label (`ant_c06_…`, every ANT arc draft) | absence vote | stage excluded via the arc's `names_are_labels` |

Also: the probe's `Try this:` regex now accepts the `[apply]`-tagged form newer
toolchains print, and its cache key is versioned so pre-fix outcomes are not
replayed. The density report carries `stages_run` and `max_n1_confidence`; on
the ANT arcs two stages cap N1 at 0.6, so decisive N1 is 0 **by construction**
and the report says so instead of printing it as a measurement.

Expected effect on this card's case: a Yoneda proof over a hand-rolled `Category`
cannot elaborate against pure Mathlib, so its probe would no longer vote; with three
empty stages it reads N1 at 0.75 and routes to D4 instead of counting as novel.
That is the mechanism-level fix. It is **not** the regression test below:
the artifact itself is not in the repo's fixtures.

Part 1 does not change `_N1_CONFIDENCE_BY_STAGES` or any threshold, per the
decision gate below.

#### Part 2 (2026-10-09): a statement no search can match is never novel

**Reading of "the Mathlib namespace where its concept would live."** The
classifier classifies an artifact's *first* declaration, so names declared
later in the same artifact never reach its header. The vocabulary a Mathlib
search cannot see therefore arrives by import: the Yoneda artifact imported
`Category`, `Functor` and `NatTrans` from `LMS.Foundation` instead of from
`Mathlib.CategoryTheory`. `lms/novelty/vocabulary.py` reads every imported
project module (any root other than Mathlib, Lean's own, and their
dependencies) from the classifier's `project_dir`, and finds which of its
declarations the statement's header uses. If any are used, or a project module
cannot be read, an all-empty search is INCONCLUSIVE with the reason as
`evidence[0]`, never N1. A match still wins: N0 evidence is unaffected.

**Informal scoring.** `named_results` pulls eponyms from the informal
statement ("Yoneda lemma", "Nakayama's lemma", "Kummer–Dedekind theorem"). A
semantic hit whose name carries the eponym scores 0.7: plausible, never
decisive. An eponym the Lean header already uses as vocabulary is dropped, so
"Dedekind's theorem" over `IsDedekindDomain` does not match every Dedekind
hit. On the ANT arcs this rule can touch only core-13, core-14 and core-20,
all of them noted as expected N0 or SCHEMATIC. The flagged N1 candidate ram-17
is untouched, and no arc draft imports a project module, so DoD 1's
measurement does not move.

Both changes only ever turn an N1 into INCONCLUSIVE. Neither can manufacture an
N0 or an N1.

**CVFN report.** It reads `goal.json` from the run directory. A goal that
forbids any `Mathlib.*` import is reported `unmeasurable` with the reason, and
no number is computed. A run with no `goal.json` says the check did not run.
Also fixed here: the denominator counted *every* verified N1, including those
below the gate's decisive threshold. It now counts what Gate 4 counts, and
reports the rest as awaiting D4 review.

**The fixture is a reconstruction.** The Sonnet run's output directory was not
kept. `tests/fixtures/novelty/yoneda_bespoke_api.json` restates the statement
over the same vocabulary, keeps the recorded verdict verbatim, and adds real
LeanSearch hits for the same informal statement (recorded 2026-10-09).

**Known limits.** Transitive project imports are not followed. An eponym only
matches when Mathlib's declaration name carries it: Nakayama's lemma is
`Submodule.eq_bot_of_le_smul_of_le_jacobson_bot`. A goal whose
`allowed_imports` whitelist excludes a Mathlib area is not flagged; only
`forbidden_imports` is.

#### Acceptance criteria

- [x] An artifact that does not import the Mathlib namespace where its concept
      would live cannot receive a decisive N1. It reports INCONCLUSIVE and
      routes to D4. *(Part 2, read as above.)*
- [x] Regression test: the Yoneda artifact from this run (kept in the card's
      fixtures) does **not** classify as decisive N1. *(Part 2,
      reconstructed.)*
- [x] `cvfn_report` (when it exists) refuses to compute a CVFN over a run whose
      goal carries `forbidden_imports` covering the relevant Mathlib area, or
      reports it explicitly flagged as unmeasurable. *(Part 2.)*
- [x] Novelty scored on the informal statement as well as the Lean source, so a
      named theorem is recognisable regardless of the API it is written against.
      *(Part 2.)*
- [x] `verify_26Q3-HARN-20.sh` asserts behaviour in pytest, not inline Python.
      *(Part 1.)*

Part 2 carries the first four criteria. Part 1 covers the first one's
mechanism for every case it can detect without knowing which Mathlib
namespace a concept "would live" in.

#### Decision gates

- **Do not** simply lower N1 confidence globally — that would suppress genuine
  N1 on goals that *do* build on Mathlib, which is the case CVFN actually needs
  to detect.
- Whether to keep `forbidden_imports` goals at all is a **separate** question
  (they are a legitimate harness shakedown; they are just not a novelty
  measurement). Do not resolve it in this card.
