### 26Q4-CHORE-07: Three runs share one GPU, each in its own worktree

**User Story**: As the person queueing runs on the box, I want up to three LMS runs to share one GPU and one vLLM server, each in its own worktree, so that a batch gets about three runs per GPU-hour instead of one, without two runs ever sharing a foundation or a Lean tree.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔄 IN REVIEW |
| **Branch** | `26Q4-CHORE-07-three-runs-share-one-gpu` |
| **Dependencies** | `26Q4-CHORE-06` (lanes). Batch q5 runs on it |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500); actual ~1,130 added: the pack script copies `lms_run.sbatch`'s run steps (~120 lines) and the fake-server test is ~270 |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `68345cb`):
- The queue kit runs one LMS job per GPU, in up to four lanes (`26Q4-CHORE-06`). Each job starts its own vLLM server for one run.
- Batches q3 and q4 measured what one run uses: at most 3 requests at once and at most ~10% of one H100's KV cache. A one-GPU server gave 55 output tokens/s per request at 9 requests against 58 at 3 (q3's thr3). So a GPU serving one run is mostly idle.
- Two runs cannot share a worktree: each rewrites `lean/LMS/Foundation.lean` and builds in the Lean tree (ADR 0003).
- `scripts/slurm/lms_run.sbatch` always starts its own server, so a second run cannot use the first one's server through it.

**Investigation**:
```bash
grep -n 'start_server\|uv run python -m lms.run' scripts/slurm/lms_run.sbatch
grep -n '^q_lanes\|^q_submit\|^q_copy_run' scripts/slurm/queue/lms_q_lib.sh
```

---

#### Acceptance Criteria

- [x] `q_slots LANES SLOTS` makes slots 2..SLOTS (at most 3) in each lane where missing, each with its own Lean tree linking lane 1's library and its own `.venv`, and changes nothing when run again
- [x] `q_submit_pack` queues one job behind the lane's last LMS job and records one jobs-file line per run with its `slot=`; a pack file with two runs in one slot, a malformed knob or a missing slot is refused before anything is queued
- [x] The pack job readies slots 2 and 3 as slot 1 is readied, starts one server with `lms_run.sbatch`'s command and settings, and starts every run at once in its slot
- [x] Each run's `lms.run` call, directory and environment are the same as `lms_run.sbatch`'s for the same knobs, checked on the fake server against this checkout's real run script
- [x] A run whose slot or settings fail is left out and the others still run; with nothing to run the model is never loaded; a pack line that sets a server setting is refused
- [x] TERM stops every run and the server together
- [x] Collect reads each run from its slot and copies its own log and the pack's
- [x] `uv run pytest tests/test_queue_kit.py -q` covers each item above
- [x] The runbook describes packs

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/slurm/queue/lms_q_pack.sh` | CREATE | The pack job: slots, one server, the runs, teardown |
| `scripts/slurm/queue/lms_q_lib.sh` | MODIFY | `q_slots`, `q_slot_wt`, `q_submit_pack`, `q_job_log`, `q_pack_log`; slot-aware `q_tag_free`, `q_job_wt`, `q_copy_run` |
| `tests/slurm_queue/fakebox.sh` | MODIFY | `SLURM_JOB_NAME` for a job, and the pack script in the fake home |
| `tests/slurm_queue/test_pack.sh` | CREATE | Pack cases on the fake server |
| `tests/test_queue_kit.py` | MODIFY | Runs the pack cases; the pack script parses |
| `docs/runbooks/2026-11-10-queued-rounds.md` | MODIFY | A Packs section |
| `docs/planning/tasks/26Q4-01/26Q4-CHORE-07-three-runs-share-one-gpu.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-CHORE-07.sh` | CREATE | Verify script |

---

#### Notes

- The pack has its own copy of `lms_run.sbatch`'s run steps, because that script starts a server of its own. `test_pack.sh` runs both and compares them run for run, so a change to the run script that the copy misses fails CI. A round should also `q_pin_has` the lines the copy was made from (batch q5's go does).
- 160 GB and 24 CPUs per pack: each run builds its own foundation and verifies in its own Lean tree. The node has about 1.4 TB and 128 CPUs, so four packs fit.
- Three runs per GPU is what q3's speed curve supports. More has not been measured; the per-request speed falls slowly up to 72 requests (thr3), so a later batch could try more.
- ADR 0003's "no two Phase C runs execute concurrently against the same `lean/` tree" holds: every run has its own tree. Whether Phase C itself should run packed is a separate decision.
