### 26Q4-HARN-15: Name search can't read N0 without checking the statement

**User Story**: As the person reading a novelty label, I want a matching name to stop proving a statement is already in the library, so that novel work with a common name doesn't read as a re-derivation.

> Context and draft criteria written by Claude on 2026-11-05 from the reviews of `26Q4-HARN-09`. The user asked for this card before Phase C but has not reviewed this text, and the choice in Acceptance Criteria is theirs.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-15-name-search-n0-needs-statement-check` |
| **Dependencies** | 26Q4-HARN-09 |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

False N0 is the error this classifier must not make. Name search makes it by design: a declaration with the same last name component reads N0 at 0.95, with no look at the statement.

**Current State** (`26Q4-HARN-09` at `3185b0e`):

- **Scoring:** `_score` (`lms/novelty/__init__.py:330-331`) gives a name hit 0.95, above `DECISIVE_CONFIDENCE`, so the ladder stops there. A Loogle hit with the same last name gives 0.9 (`:333-334`), also decisive, on a query that is only the quoted name.
- **Name source:** `StatementQuery.from_lean` (`lms/novelty/mathlib_search.py:264`) parses unstripped text, so a comment can supply the name. The `exact?` probe already strips comments. In `experiments/`, 10 of 870 distinct artifacts parse a different name that way, all `Op`. In `data/`, 0 of 93 do.
- **First declaration only:** the name comes from a file's first declaration, so a helper's name decides the file. 673 of those 870 artifacts have several declarations. `26Q4-HARN-09` closed this route for `exact?` but not for names.

**Investigation**: run through the real classifier with a real grep over a local library newer than the pin, other stages empty:

| Statement | Reads | Hit |
|---|---|---|
| `theorem add_comm (a b : Nat) : a + b = a` (false) | N0 0.95 | `add_comm` in `Mathlib.CategoryTheory.Abelian.NonPreadditive` |
| `theorem comp_assoc (p : Nat → Prop) : p 0` | N0 0.95 | `comp_assoc` in `Mathlib.AlgebraicGeometry.Birational.Composition` |
| `lemma aux …`, then `theorem novel_main …` | N0 0.95 | `aux` in `Mathlib.CategoryTheory.Presentable.Directed` |
| A docblock line `theorem of Dedekind…`, then `theorem brand_new_result …` | N0 0.95 | `of` in `Mathlib.AlgebraicGeometry.Morphisms.Basic` |

A stubbed Loogle answer `Nat.add_comm` on that false statement reads N0 0.9.

**What depends on it:**

- All 10 of the control's N0s at R4 came from name search. 8 of those 10 files have several declarations. With a forecast on local probes in which no `exact?` close counts as N0 on the control, removing name N0s could leave the control with none.
- The arcs don't depend on it. Their names are labels, so name search and Loogle don't run, and all 12 of their R4 N0s came from `exact?`.
- 9 of the 13 cases in `tests/fixtures/novelty/recorded_searches.json` read N0 through Loogle, and all 9 also accept INCONCLUSIVE, so `TestRecordedFixtures` allows the change.

---

#### Acceptance Criteria

- [ ] The user has chosen, and the choice is recorded in this card, between: (a) a name or Loogle hit reads INCONCLUSIVE with the hit as evidence, never N0; (b) it reads N0 only when Lean confirms the hit proves the statement; (c) option (a) now, with (b) as a later card — accepted via Outcome Demo
- [ ] `StatementQuery.from_lean` takes its name from comment-stripped source: a test in `tests/test_novelty.py` covers the docblock case above
- [ ] The false `add_comm` statement above can't read N0 from a name or Loogle hit: `tests/test_novelty.py` covers both stages
- [ ] A name hit on the first of several declarations can't read N0: `tests/test_novelty.py` covers `lemma aux`, then a second declaration
- [ ] `tests/test_novelty.py::TestRecordedFixtures` passes, including `test_no_known_n1_is_ever_called_n0`
- [ ] Runbook R4 and `data/novelty_control/README.md` say what the control's N0 count now means
- [ ] `uv run pytest -q` passes, and `uv run ruff check .` and `uv run mypy lms/` are clean

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/novelty/__init__.py` | MODIFY | What a name or Loogle hit can read |
| `lms/novelty/mathlib_search.py` | MODIFY | The name comes from stripped source; a Lean check, if chosen |
| `tests/test_novelty.py` | MODIFY | Tests above |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | R4: what the control's N0 means |
| `data/novelty_control/README.md` | MODIFY | Same |

---

#### Implementation Notes

- Option (b) needs the hit's full name, and name search returns only the bare name and module. `exact?` already confirms a statement the library proves, so (b) adds little beyond a single-declaration close, at the cost of another Lean run per hit.
- Stripping comments changes the name-stage and Loogle cache keys only for statements whose name changes.
- Mirror the semantic stage: a match is evidence for review, kept below `DECISIVE_CONFIDENCE`.
- `counts_as_novel` reads only `novelty_level` and `novelty_confidence`, so INCONCLUSIVE is enough to keep a statement out of CVFN.

---

#### Decision Gates

- If the chosen rule leaves the control with no N0, that is the rule working, but stop and show the numbers: Sprint DoD 2 says N0 must account for essentially all of it, and rewording that is the user's call.
- If a `TestRecordedFixtures` case would need its accepted levels changed → stop and surface it. Those are live recordings.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- `26Q4-HARN-13` owns what a decisive N1 means at Gate 4, and `26Q4-HARN-14` owns telling an exhausted `exact?` from a miss.
- The semantic stage's match rule, chosen in `26Q4-HARN-09`.
- `strip_comments` being fooled by `"/-"` in a proof body.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-15.sh`.

---

#### Outcome Demo

**Where**: local, then the GPU server as a round folder Claude prepares; the user runs it.
**Run**: `uv run pytest tests/test_novelty.py -v -k name`, then the round's `RUN_THIS.txt`, which re-runs `scripts/slurm/n1_density.sbatch` at the merged commit.
**Expect**: the four statements above no longer read N0. The control's N0 count changes as the chosen rule says, and each of its former name N0s now shows its hit as evidence for a reader. The arcs' N0 counts don't change.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-15.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator

---

#### Notes

A neighbouring defect from the independent review at the final gate of `26Q4-HARN-09` (2026-11-06, at `3185b0e`), reproduced with a probe and not blocking:

- **A label token leaks through an explicit universe list.** `extract_identifiers` keeps the trailing dot in `theorem Ant.core.thirteen.{u} ...`, so the identifier doesn't equal the name, and with `names_are_labels=True` and hit `Ant.core.thirteen` the statement reads INCONCLUSIVE 0.5. The same statement without `.{u}` reads N1 0.6. All 41 arc labels are lowercase and never extracted, so nothing real is affected. The fix is likely a one-line `rstrip(".")`.
