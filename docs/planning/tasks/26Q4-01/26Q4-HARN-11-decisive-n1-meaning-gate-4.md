### 26Q4-HARN-11: Decide what a decisive N1 means at Gate 4

**User Story**: As the person reading CVFN, I want a decisive N1 to mean something I can count without review, so that library results restated under an agent's own name don't inflate the denominator.

> Context and draft criteria written by Claude on 2026-11-05 from the second pre-merge review of `26Q4-HARN-09`. The user asked for this card before Phase C but has not reviewed this text, and the choice in Acceptance Criteria is theirs.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-11-decisive-n1-meaning-gate-4` |
| **Dependencies** | 26Q4-HARN-09 |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`26Q4-HARN-09` at `aebe660`):

- `default_novelty_classifier` (`lms/gates/novelty.py`) runs all four stages with `names_are_labels=False`, and `counts_as_novel` (`lms/accounting.py`) counts any N1 at confidence 0.8 or above without D4 review.
- Replaying recorded hits in that configuration, with the name, Loogle and `exact?` stages empty, core-13, -15, -17, -20 and ram-04, -05, -14, -17 all read N1 at 0.90 with no review. Those are arc statements under their label names, used here as stand-ins for an artifact whose name matches nothing. By hand reading (not D4), 6 of the 8 are in the library, ram-14 is close and ram-04 is open. core-13's own `notes` say "Expected N0".
- Label tokens enlarge the Jaccard union, so ram-04 and ram-14, which are near hits under the arc configuration, are not near hits here. A near hit still can't count, but fewer statements qualify as one.
- `data/ant_arcs/README.md` says name and Loogle silence on label names "is not evidence". Gate 4 counts both toward the four stages for any artifact.
- This is not a regression from `26Q4-HARN-09`. On `main`, the same statements read the same way or worse.

**Investigation**: on the recorded hits, no token rule separates these statements from the ones hand-read as open, so tightening the match rule won't fix this. The table is in the `26Q4-HARN-09` card.

---

#### Acceptance Criteria

- [ ] The user has chosen, and the choice is recorded in this card, between: (a) during calibration no N1 counts without D4 review; (b) an N1 counts only if a stage that can find a differently named statement ran, meaning `exact?` on a single declaration; (c) keep the current rule and report the count of unreviewed N1s beside CVFN — accepted via Outcome Demo
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

- `26Q4-HARN-12` owns telling an exhausted `exact?` from a miss.
- The match rule itself, chosen in `26Q4-HARN-09`.
- Name-stage false N0s.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-11.sh`.

---

#### Outcome Demo

**Where**: local
**Run**: `uv run pytest tests/test_novelty.py -k gate_4 -v`
**Expect**: the 8 statements read as the chosen rule says, and the user confirms that is what they want CVFN to count.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-11.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
