### 26Q4-CHORE-05: A vLLM start that hangs in NCCL is retried, not fatal

**User Story**: As the person queueing runs on the box, I want `lms_run.sbatch` to stop a vLLM start that hangs and try again with the suspect transports turned off, and to log which GPUs the job got, so that a hang costs minutes instead of the whole job and the log says whether the GPU pair explains it.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-CHORE-05-vllm-start-survives-hung-nccl` |
| **Dependencies** | Stacked on #94, which carries the run-script changes of #93 and #95. Every GPU run after batch q1 needs it |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `317df86`, with #94 on top):
- In batch q1 (2026-11-11 to 11-13, Slurm node `fla1`, 4x H100 NVL), 4 of 18 TP=2 starts hung: jobs 412, 413, 418 and 427. The last vLLM line was `[pynccl.py:148] vLLM is using nccl==2.30.7`, then nothing until vLLM's own 600 s engine timeout.
- A healthy start logs `cuda_communicator.py ... Using ['CUSTOM', 'PYNCCL'] all-reduce backends` a second later and answers in 70-80 s. A cold start once took 195-212 s.
- The script started the server once and waited up to `READY_TIMEOUT` (900 s). Each hang cost the whole job, about 25 minutes, and the run never happened.
- Job 427 is the one hung job that logged its physical GPUs: IDX 0,2. R5e and R5d served on 0-1, which `nvidia-smi topo -m` shows as `NV12`; H100 NVL cards are NVLink-bridged in pairs.
- The script logs only `CUDA_VISIBLE_DEVICES`, not the physical cards, so the other three hangs cannot be matched to a pair.

**Investigation**:
```bash
grep -c "all-reduce backends" "$SCRATCH"/lms-runs/*/vllm.log   # 0 = a start that hung
grep -n "READY_TIMEOUT\|VLLM_PID\|CUDA_VISIBLE_DEVICES" scripts/slurm/lms_run.sbatch
```

---

#### Acceptance Criteria

- [ ] Before vLLM starts, the job prints the `GRES=...(IDX:...)` part of `scontrol show job -d` (under Slurm) and `nvidia-smi topo -m`; a failure of either does not stop the job
- [ ] A start that is alive but not answering after `START_TIMEOUT` seconds (default 360) has its whole process group stopped, TERM then KILL, and the next rung starts once the allocated GPUs hold <=1 GiB again (bounded wait)
- [ ] The rungs are `base` (today's command and environment, unchanged), `no-custom-ar` (+ `--disable-custom-all-reduce`) and `no-p2p` (+ `NCCL_P2P_DISABLE=1` for the server only); `START_LADDER` picks them, `START_LADDER=base` is the old single attempt, and an unknown rung stops the job before the server loads
- [ ] A server that dies during startup is fatal at once, with no further rung
- [ ] If every rung hangs, the job exits 1 with one FATAL line naming the rungs tried and the per-rung limit, plus the tail of the last attempt's log
- [ ] On success the job prints `SERVED_BY=<rung> after ~Ns` and `vLLM ready after ~Ns`, and still greps `GPU KV cache size`
- [ ] Each attempt writes `vllm.attempt-<rung>.log`; `vllm.log` names the log of the attempt that served
- [ ] The exit trap stops the process group of whichever attempt is current
- [ ] `READY_TIMEOUT` set without `START_TIMEOUT` is the per-rung limit
- [ ] `tests/test_run_script.py` runs the script against stub programs and covers each item above: `uv run pytest tests/test_run_script.py -q`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/slurm/lms_run.sbatch` | MODIFY | GPU and link logging, the start ladder, process-group teardown |
| `tests/test_run_script.py` | MODIFY | Stubs that hang, serve or die per rung; ladder and logging tests |
| `docs/planning/tasks/26Q4-01/26Q4-CHORE-05-vllm-start-survives-hung-nccl.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-CHORE-05.sh` | CREATE | Verify script |

---

#### Notes

- The unbridged-pair hypothesis is untested. The ladder's log is what tests it: `SERVED_BY=` next to the IDX line and the topology. A hang on a bridged pair, or a hang that `no-p2p` does not clear, counts against it.
- A rung other than `base` may change speed: custom all-reduce is the fast path for small messages, and with peer-to-peer off NCCL copies through host memory. Report the rung next to the run's results, and do not pool throughput across rungs without saying so.
- Worst case before giving up is three limits plus the stops, about 20 minutes at the defaults. A hang that the next rung clears costs about 6-7 minutes.
- A caller that still sets `READY_TIMEOUT=900` gets 900 s per rung, up to 45 minutes in all. Set `START_TIMEOUT` instead.
- vLLM's engine and workers are assumed to stay in the server's process group. If one ever leaves it, the GPU wait catches it and the job stops with a FATAL line rather than start a rung on top of it.
