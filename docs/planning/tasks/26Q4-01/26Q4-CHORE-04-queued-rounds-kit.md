### 26Q4-CHORE-04: Rounds at different commits can wait in the queue together

**User Story**: As the person running the box, I want to queue several experiments with one command and collect them with one command, so that the node keeps working through a night without me between rounds.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q4-CHORE-04-queued-rounds-kit` |
| **Dependencies** | None. `26Q4-CHORE-02` gives the run script the settings GPU rounds need |
| **Gate** | fast |
| **PR Size Target** | <800 lines (max 1200) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `4b01509`):
- Every round so far (R4, R4b, R5b–R5e) checked out its commit in the shared worktree at submit time and refused to run while any `lms-` job was queued. One round at a time, with a person between each.
- A run rewrites `lean/LMS/Foundation.lean` in that worktree, so two LMS jobs can't run at once (ADR 0003).
- A job killed at its time limit leaves the two tracked foundation files modified. The next round's submit script then refused the dirty worktree.
- Round scripts each carried their own copy of the same checks: setup present, commit on origin, tag free, queue empty.

**Investigation**:
```bash
grep -rn "an lms- job is queued or running" docs/ scripts/   # the refusal lived only in round scripts, outside the repo
```

---

#### Acceptance Criteria

- [ ] `scripts/slurm/queue/lms_q_lib.sh`: `q_submit` queues a job behind the newest `lms-` job (`afterany`), records it in `~/lms_<round>.jobs`, and cancels the round's earlier jobs if `sbatch` refuses one
- [ ] `q_pin` accepts a commit only if a branch on origin holds it; `q_round_guard` makes a second submit queue nothing; `q_tag_free` refuses to overwrite a run
- [ ] `scripts/slurm/queue/lms_q_job.sbatch` checks out the job's own commit when the job starts, after restoring the tracked foundation files, and refuses over any other tracked change
- [ ] A lock stops two LMS jobs sharing the worktree if one was submitted outside the kit
- [ ] `q_collect_begin` waits only for the round's own jobs; `q_copy_run` copies logs and run files with `$HOME` and `$SCRATCH` scrubbed
- [ ] `scripts/slurm/queue/lms_all_run.sh` runs every round's submit or collect from a plan file; a round that stops doesn't stop the next
- [ ] `tests/slurm_queue/` holds a fake server and the checks; `uv run pytest tests/test_queue_kit.py -q` runs them
- [ ] `docs/runbooks/2026-11-10-queued-rounds.md` describes the procedure
- [ ] Several rounds queued together on the box run in order, each at its own commit — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/slurm/queue/lms_q_lib.sh` | CREATE | Shared functions for a round's submit and collect scripts |
| `scripts/slurm/queue/lms_q_job.sbatch` | CREATE | The job: lock, restore, checkout, run |
| `scripts/slurm/queue/lms_all_run.sh` | CREATE | Every round from one plan file |
| `tests/slurm_queue/fakebox.sh` | CREATE | Fake server for testing rounds |
| `tests/slurm_queue/test_kit.sh` | CREATE | Checks for the library and the job |
| `tests/slurm_queue/test_all.sh` | CREATE | Checks for the plan runner |
| `tests/test_queue_kit.py` | CREATE | Runs the checks under pytest |
| `docs/runbooks/2026-11-10-queued-rounds.md` | CREATE | The procedure |
| `scripts/verify/26Q4-01/verify_26Q4-CHORE-04.sh` | CREATE | Verify script |

---

#### Notes

- LMS jobs still run one at a time. Two at once needs a `lean/` tree per run, which ADR 0003 deferred.
- The fake server's Slurm keeps a queue and does not model scheduling, so the first real use of each `sbatch` option is on the box.
