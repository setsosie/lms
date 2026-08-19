### 26Q3-HARN-04: Novelty classifier (N0 / N1)

**User Story**: As the calibration measurement, I want each verified statement
labelled as "already in Mathlib" or "not", so that CVFN counts novel formalization
rather than re-derivation.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | CRITICAL |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q3-HARN-04-novelty-classifier` |
| **Dependencies** | 26Q3-HARN-01, 26Q3-HARN-02 |
| **PR Size Target** | <400 lines |

---

#### Context

`specs/faithfulness_protocol.md` §6.1 defines the novelty ladder:

| Level | Meaning | Program role |
|---|---|---|
| N0 | Re-proof of an existing Mathlib result | **calibration only, never claimed** |
| N1 | Known textbook statement absent from Mathlib | the bulk of the program (~11.6K stmts) |
| N2 | Connective lemma in no single source text | the p-adic bridge |
| N3 | Absent from sources and literature | the prize |

**Every artifact this project has ever produced is N0.** The category-theory
corpus re-derives Mathlib. The program's target of ~11.6–14K statements is
entirely N1+. Without this classifier there is no denominator for CVFN, because
"verified" and "verified-novel" are currently indistinguishable in the data.

This task also resolves the open decision in `calibration-program.md` §4: which
ANT slice to calibrate on. That must be chosen on **measured** N1 density, not on
recollection of Mathlib's contents.

---

#### Acceptance Criteria

- [ ] `classify_novelty(lean_decl) -> NoveltyResult{level, confidence, evidence}`
- [ ] Search strategy, in order, short-circuiting on a confident hit:
      1. exact/fuzzy **name** match in Mathlib (`lean_local_search`)
      2. **statement-shape** match via `lean_loogle` type-pattern query
      3. **provability-by-existing-lemma** probe: does `exact?` / `apply?` close
         the goal from Mathlib alone? (a strong N0 signal)
      4. semantic search fallback (`lean_leanfinder`, `lean_leansearch`)
- [ ] Emits `N0` (found), `N1` (not found), or `INCONCLUSIVE` with confidence
- [ ] `INCONCLUSIVE` and low-confidence `N1` route to D4 human review; they are
      **not** counted as novel without sign-off
- [ ] `evidence` records the matching Mathlib declaration name(s) for N0 — the
      claim must be auditable
- [ ] Rate limiting respected (loogle 3/30s, leansearch 90/30s,
      leanfinder 10/30s, state_search 6/30s); results cached on disk by
      statement hash so re-scoring archived runs is cheap
- [ ] N2/N3 are **out of scope** — they require a literature search, not a Mathlib
      search. Return `INCONCLUSIVE` and say so rather than guessing
- [ ] **Slice-selection mode**: `scripts/measure_n1_density.py <statements.json>`
      classifies a candidate list and reports N1 density, for the
      `calibration-program.md` §4 decision

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/novelty/__init__.py` | CREATE | `NoveltyResult`, `classify_novelty` |
| `lms/novelty/mathlib_search.py` | CREATE | lean-lsp MCP search strategies + cache |
| `lms/gates/novelty.py` | CREATE | Gate 4 wrapper |
| `lms/artifacts.py` | MODIFY | `novelty_level`, `novelty_confidence`, `novelty_evidence` |
| `scripts/measure_n1_density.py` | CREATE | Slice-selection tool |
| `tests/test_novelty.py` | CREATE | Unit tests with recorded search fixtures |

---

#### Implementation Notes

- Validation set, both directions:
  - **known N0** — take declarations from `lean/LMS/Categories/Basic.lean`,
    `Functor.lean`, `NatTrans.lean` (all re-derive Mathlib). The classifier must
    call these N0. (The card originally named `Yoneda.lean`, which does not
    exist in the corpus; `NatTrans.lean` substituted 2026-08-19.)
  - **known N1** — take the four novel theorems in
    `lean/LMS/Categories/Localization.lean` (Stacks tags 04VB, 04VD, 05Q2). The
    classifier must not find them in Mathlib.
  - Measure precision/recall on this set and report it. **A classifier whose
    error rate is unknown cannot certify a novelty claim.**
- Mathlib moves. Record the Mathlib revision with every classification; a stale
  N1 becomes an N0 when upstream lands it.

---

#### Decision Gates

- If measured accuracy on the validation set is poor (<80% either direction),
  ship it as an *advisory* signal that routes everything to D4 rather than as an
  automatic gate, and say so loudly in the sprint report. A wrong novelty label
  is worse than no label.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] Precision/recall on the validation set reported in the PR description
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
