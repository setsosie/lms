### 26Q4-HARN-13: Decide what a decisive N1 means at Gate 4

**User Story**: As the person reading CVFN, I want a decisive N1 to mean something I can count without review, so that library results restated under an agent's own name don't inflate the denominator.

> Context and draft criteria written by Claude on 2026-11-05 from the second pre-merge review of `26Q4-HARN-09`. The user asked for this card before Phase C but has not reviewed this text, and the choice in Acceptance Criteria is theirs.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING: option (a) chosen 2026-11-18 |
| **Branch** | `26Q4-HARN-13-decisive-n1-meaning-gate-4` |
| **Dependencies** | 26Q4-HARN-09 |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`26Q4-HARN-09` at `aebe660`):

- `default_novelty_classifier` (`lms/gates/novelty.py`) runs all four stages with `names_are_labels=False`, and `counts_as_novel` (`lms/accounting.py`) counts any N1 at confidence 0.8 or above without D4 review.
- Replaying recorded hits in that configuration, with the name, Loogle and `exact?` stages empty, core-13, -15, -17, -20 and ram-04, -05, -14, -17 all read N1 at 0.90 with no review. Those are arc statements under their label names, used here as stand-ins for an artifact whose name matches nothing. By hand reading (not D4), 6 of the 8 are in the library, ram-14 is close and ram-04 is open. core-13's own `notes` say "Expected N0".
- Label tokens enlarge the Jaccard union, so ram-04 and ram-14, which are near hits under the arc configuration, are not near hits here. A near hit still can't count, but fewer statements qualify as one. Docstrings and imports enlarge it too, because `_statement_tokens` reads the unstripped source: adding a capitalised docstring moved a near hit from N1 0.75 to N1 0.9, counted without review. None of the control's 19 N1s, all whole files, has a near hit.
- `data/ant_arcs/README.md` says name and Loogle silence on label names "is not evidence". Gate 4 counts both toward the four stages for any artifact.
- This is not a regression from `26Q4-HARN-09`. On `main`, the same statements read the same way or worse.

**Investigation**: on the recorded hits, no token rule separates these statements from the ones hand-read as open, so tightening the match rule won't fix this. The table is in the `26Q4-HARN-09` card.

---

#### Acceptance Criteria

- [x] The user has chosen, and the choice is recorded in this card. **Chosen 2026-11-18: (a), during calibration no N1 counts without D4 review.** The options were: (a) during calibration no N1 counts without D4 review; (b) an N1 counts only if a stage that can find a differently named statement ran, meaning `exact?` on a single declaration; (c) keep the current rule and report the count of unreviewed N1s beside CVFN — accepted via Outcome Demo
- [ ] `tests/test_novelty.py` pins the choice on the 8 statements above, replayed in Gate 4's configuration
- [ ] `lms.accounting.counts_as_novel` and `apply_novelty_gate` still agree on every case in those tests
- [ ] `uv run pytest -q` passes, and `uv run ruff check .` and `uv run mypy lms/` are clean

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/novelty/__init__.py` | MODIFY | What can read decisive |
| `lms/gates/novelty.py` | MODIFY | Gate 4's configuration, if the choice needs it |
| `lms/accounting.py` | MODIFY | Only if the choice changes what counts |
| `lms/arms.py` | MODIFY | The arm comparison reads the same D4 verdicts as the CVFN report (added 2026-11-18, when the user chose that a reviewed N1 counts) |
| `tests/test_novelty.py` | MODIFY | The 8 statements in Gate 4's configuration |
| `specs/faithfulness_protocol.md` | MODIFY | What a decisive N1 means, if the choice changes it |

---

#### Implementation Notes

- `counts_as_novel` reads only `novelty_level` and `novelty_confidence`, never `needs_review`. A rule that only sets a flag won't reach CVFN. Cap the confidence, as the near hit does.
- The recorded hits are in `tests/fixtures/novelty/semantic_hits_2026-11-04.json`.

---

#### Decision Gates

- If the choice would change a `TestRecordedFixtures` level → stop and surface it. Those are live recordings.
- If option (a) leaves Gate 4 unable to pass anything automatically → that is the choice working. Say so in the sprint doc, don't weaken it.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- `26Q4-HARN-14` owns telling an exhausted `exact?` from a miss.
- The match rule itself, chosen in `26Q4-HARN-09`.
- Name-stage false N0s.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-13.sh`.

---

#### Outcome Demo

**Where**: local
**Run**: `uv run pytest tests/test_novelty.py -k gate_4 -v`
**Expect**: the 8 statements read as the chosen rule says, and the user confirms that is what they want CVFN to count.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-13.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator

---

#### Notes

From the independent review at the final gate of `26Q4-HARN-09` (2026-11-06, at `3185b0e`), reproduced with a probe and not blocking:

- **Binder case can flip a verdict to decisive N1.** `extract_identifiers` drops a dotted name with a lowercase head. `theorem t (R : Type) [CommRing R] (P : Ideal R) (hP : P.IsPrime) : (P.comap (RingHom.id R)).IsPrime := sorry` with hit `Ideal.IsPrime.comap`, in Gate 4's configuration with the other stages empty, reads INCONCLUSIVE 0.5. Renaming `P` to `p` loses the `is` and `prime` tokens and reads N1 0.90, counted without review. "Renaming a binder changes no verdict" holds only for capital-to-capital renames.
- **Real-data impact:** none seen. Lowercasing every capital single-letter binder used dotted, across the 15 core and ramification statements in the recorded-hits fixture, flipped nothing in either configuration.
- **Built 2026-11-18, option (a)**, by changing what counts, not by capping the classifier's confidence (the Implementation Note's suggestion). The user chose this on 2026-11-18.
  - `apply_novelty_gate` never counts an N1 and sends every N1 to D4. `counts_as_novel` counts only N2/N3 D4 verdicts.
  - The classifier's confidences, the densities and the decisive line are unchanged. So the novelty control and the arc densities still measure what the search found, and D3's control bar keeps its meaning.
  - All 8 statements above still read N1 0.90 at Gate 4 and now count for nothing until D4.
- **Consequence for Gate 4 (Decision Gates: "the choice working"):** it can no longer pass anything on its own. Society only stamps the labels, so promotion and what a run keeps are unchanged.
- **The pre-merge review (2026-11-18) found that, as first built, a D4-reviewed N1 could never count.** On the user's choice, D4's verdict now comes from the run's `review_log.json`:
  - The format is an entry `{"statement_key": …, "novelty_verdict": "N0"…"N4"}`, in the file the CVFN report already read for review minutes. Keys are matched without the tag's quotes or spaces, and a verdict naming no level is ignored.
  - Until a review tool writes that file, D4 writes the entries by hand, one log per run directory; a paired comparison needs the verdict in every run's log. `26Q3-HARN-06`'s review view, unbuilt, plans faithfulness verdicts, so it will have to add `novelty_verdict`.
  - A verdict of N1–N3 counts and N0 does not.
  - The CVFN report and the arm comparison both read it. A statement with a verdict is no longer counted as awaiting review.
  - Until verdicts are written, the novel denominator, and every Phase C arm's distinct-statement count, read 0.
  - `measure_density` now queues decisive N1s for D4 too; its decisive count is unchanged.
