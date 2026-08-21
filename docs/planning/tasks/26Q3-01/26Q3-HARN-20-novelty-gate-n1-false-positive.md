### 26Q3-HARN-20: Gate 4 reports decisive N1 for textbook theorems in a bespoke API

**User Story**: As the calibration program, I want Gate 4 to withhold a novelty
claim when its search could not have matched, so that CVFN counts genuinely
novel results rather than well-known theorems restated against a from-scratch
API.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH — CVFN is the number the whole Q3 program turns on |
| **Status** | 📋 TODO |
| **Branch** | `26Q3-HARN-20-novelty-gate-n1-false-positive` |
| **Dependencies** | 26Q3-HARN-19 (wires Gate 4 into the run at all) |
| **PR Size Target** | <400 lines |
| **Parts** | single PR |

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

#### Acceptance criteria

- [ ] An artifact that does not import the Mathlib namespace where its concept
      would live cannot receive a decisive N1. It reports INCONCLUSIVE and
      routes to D4.
- [ ] Regression test: the Yoneda artifact from this run (kept in the card's
      fixtures) does **not** classify as decisive N1.
- [ ] `cvfn_report` (when it exists) refuses to compute a CVFN over a run whose
      goal carries `forbidden_imports` covering the relevant Mathlib area, or
      reports it explicitly flagged as unmeasurable.
- [ ] Novelty scored on the informal statement as well as the Lean source, so a
      named theorem is recognisable regardless of the API it is written against.
- [ ] `verify_26Q3-HARN-20.sh` asserts behaviour in pytest, not inline Python.

#### Decision gates

- **Do not** simply lower N1 confidence globally — that would suppress genuine
  N1 on goals that *do* build on Mathlib, which is the case CVFN actually needs
  to detect.
- Whether to keep `forbidden_imports` goals at all is a **separate** question
  (they are a legitimate harness shakedown; they are just not a novelty
  measurement). Do not resolve it in this card.
