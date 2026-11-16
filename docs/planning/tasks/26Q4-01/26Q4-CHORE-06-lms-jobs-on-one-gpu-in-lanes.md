### 26Q4-CHORE-06: LMS jobs run on one GPU each, in up to four lanes

**User Story**: As the person queueing runs on the box, I want each LMS job to take one GPU and up to four of them to run side by side, so that a batch uses the whole box (or leaves three GPUs free) instead of holding two GPUs while two sit idle, and so that no job lands on a GPU pair that hangs vLLM's start.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-CHORE-06-lms-jobs-on-one-gpu-in-lanes` |
| **Dependencies** | None. Batch q3 needs it |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `e9c7352`):
- The queue kit (`scripts/slurm/queue/`) runs every LMS job in one worktree, `~/code/lms-box`, one at a time, on two GPUs (`Q_GPU`). Over the q2 weekend that meant two GPUs busy and two often idle for about two days.
- Batch q2 settled the start-up hang: every TP=2 job on a GPU pair without an NVLink bridge hung (4 of 4, with q1's), and all 14 on the bridged pair served. `--disable-custom-all-reduce` never helped; `NCCL_P2P_DISABLE=1` served the one time it was tried, 14 minutes in. At TP=1 there is no NCCL.
- The model (Qwen3.6-27B, FP8, about 29 GiB of weights) fits on one H100 NVL with room for KV cache; the runs used 2–3% of the TP=2 cache.
- Two LMS jobs cannot share a worktree: each run rewrites `lean/LMS/Foundation.lean` and builds in the Lean tree.

**Investigation**:
```bash
grep -n 'Q_GPU\|q_tail\|lms_q.lock' scripts/slurm/queue/lms_q_lib.sh scripts/slurm/queue/lms_q_job.sbatch
grep -n 'PORT=\|TP_SIZE=' scripts/slurm/lms_run.sbatch
```

---

#### Acceptance Criteria

- [ ] `Q_GPU` is one GPU and `q_submit` passes `TP_SIZE=1` unless the job's variables set it; `Q_GPU_ALL` is all four
- [ ] `q_lanes N` makes lanes 2..N where missing (a worktree each, whose Lean tree links lane 1's packages and builds its own foundation), checks every lane, and changes nothing when run again
- [ ] `q_lane K` sends later jobs to lane K; each job waits only for the newest LMS job in its own lane, takes its own lane's lock, serves on its own port, and lanes 2–4 use their own vLLM cache
- [ ] A job in lanes 2–4 stops before running if Lake reports the shared library out of date
- [ ] The jobs file records each job's lane; collect reads each job from its lane; a tag holding a run in any lane is refused
- [ ] `uv run pytest tests/test_queue_kit.py -q` covers each item above on the fake server
- [ ] The runbook describes lanes

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/slurm/queue/lms_q_lib.sh` | MODIFY | Lanes, one-GPU default, per-lane tail, port, cache, collect |
| `scripts/slurm/queue/lms_q_job.sbatch` | MODIFY | Per-lane lock, Lake check in lanes 2–4 |
| `tests/slurm_queue/fakebox.sh` | MODIFY | `squeue %Z`, `lake` and `uv sync` stubs |
| `tests/slurm_queue/test_kit.sh` | MODIFY | Lane cases 19–21; one-GPU expectations |
| `tests/slurm_queue/test_all.sh` | MODIFY | The queued line names its lane |
| `docs/runbooks/2026-11-10-queued-rounds.md` | MODIFY | Rules and a Lanes section |
| `docs/planning/tasks/26Q4-01/26Q4-CHORE-06-lms-jobs-on-one-gpu-in-lanes.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-CHORE-06.sh` | CREATE | Verify script |

---

#### Notes

- A job's lane is read from its working directory (`squeue %Z`), so jobs queued before lanes existed count as lane 1 and the user's own cancel commands (`^lms`) still match every lane.
- Lanes share lane 1's built library read-only. The `lake build --no-build Mathlib` check is there because a Lake that judged it stale would rebuild it in place while other lanes read it; nothing on the box has tested whether a linked `packages` directory ever looks stale to Lake, so the first job in each lane should be short.
- 96 GB per job is a guess (q2 used 128 GB for two GPUs). `q_lanes` prints the node's memory so the batch can see whether four fit.
- The run script needs no change: the kit sets `TP_SIZE`, `PORT`, `VLLM_CACHE_ROOT` and any extra vLLM flags through the job's environment.
