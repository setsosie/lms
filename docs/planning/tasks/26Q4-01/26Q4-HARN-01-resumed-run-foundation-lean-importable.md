### 26Q4-HARN-01: A resumed run keeps its foundation where Lean can import it

**User Story**: As the operator of a run on the shared box, I want a job
that hits its Slurm walltime to resume with the foundation it had built, so
that a long committee or Phase C run is not lost to the scheduler.

> Context, AC, files and demo drafted by Claude on 2026-10-20 while
> implementing; the Notes section is the 2026-10-09 evidence. Review them as
> the author.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-01-resumed-run-foundation-lean-importable` |
| **Dependencies** | None |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

> Summarize current state. Include investigation results (grep evidence).

**Current State** (`main` @ `0c3ad12`):
- `Society.save` writes the foundation only to its own path, the Lean
  project's `lean/LMS/Foundation.{lean,json}`. Nothing writes
  `<run>/LMS/Foundation.json`, so the restore branch in `Society.load` is
  dead. A resumed society keeps the constructor's empty foundation: **no seed
  and no verified entries**. That is worse than the Notes below describe.
- The project copy does not survive either: `lms_run.sbatch` restores the
  tracked `lean/LMS/Foundation.*` when each job ends.
- `resume_experiment` never rewrites the project file, so a resumed
  generation imports whatever the last job left.
- `run_experiment` writes `metadata.json` only at the end of a run, and
  `resume_experiment` reads it first. A job killed at its walltime therefore
  cannot be resumed at all.
- Periodic checkpoints never write `goal.json`; committee mode refuses to
  resume without a goal.
- `lms_run.sbatch` has no resume path, and the CLI's default checkpoint
  interval (10) equals R6's generation count.

**Investigation**:
```bash
grep -rn '"LMS" / "Foundation' lms/   # only the dead read in Society.load
grep -n 'metadata.json' lms/run.py    # written after the loop, read first on resume
```

---

#### Acceptance Criteria

> Every criterion needs something that can **accept** it — a command, or a human step in the
> Outcome Demo.

- [ ] `Society.save` writes `foundation.json` (entries, artifact ids, seed)
      into the run directory on every checkpoint, including a seed-only one
- [ ] `Society.load` rebuilds the foundation from it at the project path,
      never under the run directory, and raises `ValueError` on a checkpoint
      without one
- [ ] `resume_experiment` writes and rebuilds the restored foundation before
      the first resumed generation, and stops if it does not compile
- [ ] `metadata.json` exists before generation 0; periodic checkpoints
      refresh it and write `goal.json`
- [ ] `lms_run.sbatch` takes `RESUME=1` and `CHECKPOINT_INTERVAL` (default 1),
      and refuses a fresh run over an existing checkpoint
- [ ] Runbook R6 says how to resume instead of "do not use `--resume`"
- [ ] A checkpoint taken mid-generation (the walltime case) resumes even when
      the interrupted generation's entries break the build: `load` marks the
      generation boundary as the last good state, so the existing rollback
      drops them; the resume is refused only if the boundary fails too
- [ ] `uv run pytest tests/test_resume_foundation.py -v` passes; 9 of its 10
      tests fail at `main`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | `save_metadata(path)`; `load(path, metadata_path=None)`; `snapshot_before(generation)` |
| `lms/society.py` | MODIFY | `FOUNDATION_SNAPSHOT`; save/load it; `restore_foundation()` |
| `lms/run.py` | MODIFY | Restore before resuming; metadata and goal at checkpoints |
| `scripts/slurm/lms_run.sbatch` | MODIFY | `RESUME`, `CHECKPOINT_INTERVAL`, overwrite guard |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | R6 resume instructions |
| `tests/test_resume_foundation.py` | CREATE | Behaviour tests |

> This table is a fence the gate checks: `/pre-merge` treats a changed file not named here
> (tests and this task's own card/verify script excepted) as a blocking finding.

---

#### Implementation Notes

- Mirror `reset_foundation`: the resume counterpart goes through
  `_write_and_build_foundation`, so a restored foundation that fails to
  build is reported the same way.
- Do not move the Lean file under the run directory. `import LMS.Foundation`
  resolves through Lake, and `reset_foundation`'s docstring explains why.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.
- An explicit "do NOT do X" in this card is binding even if X seems helpful.

---

#### Out of Scope

- Reconstructing a foundation for checkpoints written before this card.
  They are refused, not repaired: replaying `artifacts.json` would not
  reproduce rollbacks or gate-blocked promotions.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-01.sh`.

The script checks that the work exists and that its tests pass. It does not itself test.

---

#### Outcome Demo

**Where**: the box, only if a run hits its walltime (runbook R6)
**Run**: R6's `RESUME=1` command, then its `grep -m2` line.
**Expect**: `resume=1`, then `Continuing from generation N` with N the
generations the first job finished. The resumed generations verify against
the seed and earlier entries rather than failing on a missing `Category`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-01.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

> Found-work, verified against `main` @ `c06b79a` on 2026-10-09. Evidence only;
> the sections above are still the author's to write.

- `Society.load` (`lms/society.py:1752`) builds `foundation_path = output_dir / "LMS" / "Foundation.lean"`
  and calls `FoundationFile.load(foundation_path)`.
- `FoundationFile.load` (`lms/foundation.py:1214`) is `cls(path)`, so the resumed
  society's foundation now *lives* in the output directory. A fresh society
  uses the Lean project's file instead (`lms/society.py:214`, `:220`).
- Agents still write `import LMS.Foundation`, which Lean resolves inside the
  Lean project. After a resume, the society writes new declarations where Lean
  never looks.
- Why it matters: Phase C runs are hours long on a shared node with a
  walltime. The restore runbook's R6 tells the operator not to use `--resume`
  for this reason.
