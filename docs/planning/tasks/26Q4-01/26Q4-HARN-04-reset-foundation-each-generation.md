### 26Q4-HARN-04: Reset the foundation to its seed at every generation (the R9 arm)

**User Story**: As the Phase C experimenter, I want a run flag that returns `Foundation.lean` to its seed at the start of every generation, so that C9 against R9 at equal budget measures what the carried foundation is worth — the ratchet — rather than inferring it from reuse rate.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-04-reset-foundation-each-generation` |
| **Dependencies** | None. Required by ADR 0003 (Accepted 2026-10-30) for the R9 arm |
| **PR Size Target** | <250 lines (max 500) |
| **Parts** | single PR |

---

#### Context

`calculate_reuse_rate` (`lms/metrics.py`) shows that later artifacts *name*
earlier ones. It cannot show that carrying the foundation forward produces more
verified statements than not carrying it; that needs a run that does not carry
it. Committee generations already start from fresh contexts, so with the
foundation reset each generation the only cross-generation channels left are
the textbook, the dependency graph's statuses and the artifact library. ADR
0003 keeps all three in both arms, so C9 − R9 prices importable verified code.

**Current State**:
- `lms/society.py:298` `Society.reset_foundation()` installs the seed once per run; `lms/run.py:252` calls it before the first generation.
- `lms/society.py:246` `run_generation()` dispatches flat/iterative or committee mode and persists the foundation after any verifying generation.
- `FoundationFile.snapshot()` / `restore()` exist (`lms/foundation.py:714,723`), used by the HARN-22 rollback; `_write_and_build_foundation()` sets `_last_good_foundation` after a successful build (`lms/society.py:365`).
- No flag or attribute resets the foundation between generations.

**Investigation**:
```bash
grep -n "reset_foundation\|_last_good_foundation\|def snapshot\|def restore" lms/society.py lms/foundation.py lms/run.py
# society.py:298 reset_foundation; 186/365/368/375 rollback snapshot; foundation.py:714/723; run.py:252
grep -n "reset" lms/run.py
# only the run-start reset_foundation() call
```

---

#### Acceptance Criteria

- [ ] `Society.reset_foundation_each_generation: bool = False`
- [ ] `reset_foundation()` keeps the seeded snapshot on the society; when the flag is set, every `run_generation()` after the first restores that snapshot and rebuilds before dispatching, via `_write_and_build_foundation()`, so `_last_good_foundation` becomes the seed state too
- [ ] The per-generation restore does **not** call `goal.mark_seeded()` again or re-print the seed banner
- [ ] `lms/run.py` adds `--reset-each-generation` (store_true, help text naming ADR 0003); `metadata.json` records `"reset_each_generation": true|false` on every run
- [ ] Tests in `tests/test_foundation_persistence.py`: with the flag set, generation 1 starts with exactly the seed entries and `_derive_references` on code naming a generation-0 entry returns nothing; with the flag off, generation-0 entries carry into generation 1 (regression); the textbook and the dependency graph's statuses are unchanged by a reset; the flag is parsed and reaches the society (`tests/test_society.py` or the parser tests' existing location)
- [ ] All tests pass: `uv run pytest tests/test_foundation_persistence.py tests/test_society.py -v`
- [ ] `uv run ruff check lms/society.py lms/run.py` and `uv run mypy lms/society.py lms/run.py` clean
- [ ] A two-generation committee run on the box with the flag set shows a seed-only foundation entering generation 1 — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/society.py` | MODIFY | Flag, seed snapshot, restore at generation start |
| `lms/run.py` | MODIFY | CLI flag, wiring, metadata field |
| `tests/test_foundation_persistence.py` | MODIFY | Reset and regression tests |
| `tests/test_society.py` | MODIFY | Flag wiring test |
| `scripts/slurm/lms_run.sbatch` | MODIFY | `RESET_EACH_GENERATION=1` passes the flag, as `N_GROUPS` passes `--groups` |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-04.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Mirror the HARN-22 rollback: `snapshot()` / `restore()` plus `_write_and_build_foundation()`. Do not add a second write or build path.
- Mirror `--repair-attempts` for the CLI flag and how `run.py` copies it onto the society; mirror `prompt_overrides` for the metadata field.
- A resumed run (`--resume`) takes the flag from the command line like every other run flag; do not persist it into checkpoints in this PR.

---

#### Decision Gates

- If restoring the snapshot leaves `_definition_names` or `_claimed_concepts` holding generation-0 names, so a later redefinition is rejected as a duplicate → stop and surface. The fix touches `FoundationFile` semantics, not just a flag.
- If the dependency graph's DONE statuses turn out to make R9 meaningless — say, the planning panel refuses work whose prerequisites are DONE but unimportable → stop and report it. That is a finding about the ratchet, not a bug to route around.
- If resetting the textbook as well starts to look necessary → do not add it here; ADR 0003 fixes R9 as foundation-only.
- If the PR exceeds 500 lines → stop and split.

---

#### Out of Scope

- `26Q4-HARN-03` owns the comparison report that reads R9 runs.
- No reset of the textbook, the artifact library or the dependency graph.
- No change to the run-start `reset_foundation()` behaviour, the seed contents (`26Q3-HARN-23`) or the rollback (`26Q3-HARN-22`).
- No separate per-run `lean/` checkout; ADR 0003 runs arms sequentially.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-04.sh`.

---

#### Outcome Demo

**Where**: the box (you run it; Claude prepares the commands)
**Run**:
```bash
sbatch --export=ALL,AGENTS=3,GENERATIONS=2,N_GROUPS=3,ITERATIVE=0,RESET_EACH_GENERATION=1 scripts/slurm/lms_run.sbatch
# after it finishes, in the run directory:
grep -o '"reset_each_generation": [a-z]*' metadata.json
```
**Expect**: `"reset_each_generation": true`; the job log shows the seed banner once, at run start; no generation-1 artifact lists a generation-0 artifact in `references`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-04.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <250 lines changed (target) / <500 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
