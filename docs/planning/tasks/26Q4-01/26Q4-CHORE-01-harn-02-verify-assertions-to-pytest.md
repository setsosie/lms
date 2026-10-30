### 26Q4-CHORE-01: Move the HARN-02 verify script's assertions into pytest

**User Story**: As a maintainer, I want the HARN-02 checks to run in pytest,
so that CI proves them on every commit and no check depends on an archive
that only exists on one machine.

> Context, AC and files drafted by Claude on 2026-10-20 while implementing;
> the Notes section is the 2026-10-09 evidence. Review them as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | LOW |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-CHORE-01-harn-02-verify-assertions-to-pytest` |
| **Dependencies** | None |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

> Summarize current state. Include investigation results (grep evidence).

**Current State** (`main` @ `0c3ad12`):
- `scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh` holds 13 inline-Python
  checks. Nine duplicate cases already in `tests/test_lean_extraction.py`:
  every cleaner case, the parser wiring, the raw capture and the fixture count.
- Four do not: `lean_code_raw` surviving `to_dict`/`from_dict`, and three
  checks on `scripts/reextract_lean_code.py` (sibling output, input
  byte-identical, stale verdicts reset). Nothing in `tests/` exercises that
  script.
- Two checks read `experiments/stacks_ch4_phase1/artifacts.json`, which is
  **not tracked** (`git ls-files experiments/stacks_ch4_phase1` is empty), so
  the script fails on any clean checkout, CI and the box included.

**Investigation**:
```bash
grep -c "uv run python" scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh   # 13
grep -rn "reextract" tests/                                             # nothing
```

---

#### Acceptance Criteria

> Every criterion needs something that can **accept** it — a command, or a human step in the
> Outcome Demo.

- [ ] `tests/test_reextract_lean_code.py` covers the four uncovered checks on
      a fixture built in `tmp_path`, plus "the script's own output is never an
      input" and "an unchanged payload keeps its verdict"
- [ ] The verify script has no inline Python
      (`grep -c "uv run python" … → 0`); behaviour runs through pytest
- [ ] No check reads an untracked archive
- [ ] `bash scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh` passes

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh` | MODIFY | Greps and pytest calls only |
| `tests/test_reextract_lean_code.py` | CREATE | The checks that had no test |

> This table is a fence the gate checks: `/pre-merge` treats a changed file not named here
> (tests and this task's own card/verify script excepted) as a blocking finding.

---

#### Implementation Notes

- Import the script the way `tests/test_measure_n1_density.py` does
  (`importlib.util.spec_from_file_location`).
- Do not delete a check without naming the test that covers it.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.
- An explicit "do NOT do X" in this card is binding even if X seems helpful.

---

#### Out of Scope

- The other `26Q3-*` verify scripts. Audit them separately if wanted.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-CHORE-01.sh`.

The script checks that the work exists and that its tests pass. It does not itself test.

---

#### Outcome Demo

**Where**: N/A. A test move with no user-visible behaviour; the verify
script's pass on a clean checkout is the demo.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-CHORE-01.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

> Found-work, verified against `main` @ `c06b79a` on 2026-10-09. Evidence only;
> the sections above are still the author's to write.

- `scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh` carries 13 inline-Python
  blocks (`grep -c "python -c\|python - <<\|uv run python"` → 13).
- The task-card schema's test-boundary rule says behavior belongs in `tests/`.
  The script should be `test -f` / `grep -q` / `uv run pytest` lines only.
- Check, before moving each block, whether an existing test in
  `tests/test_lean_extraction.py` already covers its assertion.
