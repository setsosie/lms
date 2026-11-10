### 26Q4-CHORE-02: The run script serves on a node with no CUDA toolkit, and takes a budget

**User Story**: As the person queueing runs on the box, I want `lms_run.sbatch` to start vLLM with the settings round R5e proved, ask for the GPU name this node has, and pass a token budget, so that every later run uses the tracked script instead of a patched copy.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q4-CHORE-02-vllm-flags-no-toolkit` |
| **Dependencies** | None. Every GPU run after R5e needs it: R6, R7 and the Phase C arms |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `4b01509`):
- vLLM 0.33.0 compiles its DeepGEMM and FlashInfer kernels at load. The node has no CUDA toolkit on its default paths, so R5b died in `find_cuda_toolkit` and R5d died in the sampler's first pass.
- Round R5e (job 352, 2026-11-09) served and answered requests with `VLLM_USE_DEEP_GEMM=0`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0` and `--gdn-prefill-backend triton`, at TP=2: 106, 245 and 696 tokens/s with 1, 3 and 9 requests at once. It ran a copy of the script patched with `sed`.
- The script's directive asks for `gpu:h100:4`. The restored box names the resource `gpu:nvidia_h100_nvl`, so a submit with no `--gres` override is refused.
- The script cannot pass `--max-tokens`, which ADR 0003's arms all need.
- Its readiness loop says "not ready after 600s" and really waits up to 20 minutes, because vLLM binds the port early and each `curl` waits out its 5 s limit.

**Investigation**:
```bash
grep -n "gres\|max-tokens\|VLLM_\|600s" scripts/slurm/lms_run.sbatch
```

---

#### Acceptance Criteria

- [ ] The three `VLLM_*` settings default to `0`, are exported before the server starts, and can each be overridden from the environment
- [ ] `--gdn-prefill-backend` reaches `vllm serve`, default `triton`, overridable through `GDN_PREFILL_BACKEND`
- [ ] `LMS_EXTRA_VLLM_ARGS` is appended to `vllm serve` as separate words
- [ ] `MAX_TOKENS=N` passes `--max-tokens N` to `lms.run`; unset passes nothing; a value that is not a positive whole number stops the job before the server loads
- [ ] The readiness wait is measured on the clock, limited by `READY_TIMEOUT` (default 900 s), and the failure message names the real limit
- [ ] The `--gres` directive is `gpu:nvidia_h100_nvl:4`
- [ ] `tests/test_run_script.py` runs the script against stub programs and covers each item above: `uv run pytest tests/test_run_script.py -q`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/slurm/lms_run.sbatch` | MODIFY | Settings, budget, extra flags, readiness wait, GPU name |
| `tests/test_run_script.py` | CREATE | Behaviour against stubs |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | R7 as a long run, and what to read from it |
| `docs/planning/tasks/26Q4-01/26Q4-CHORE-02-vllm-flags-no-toolkit.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-CHORE-02.sh` | CREATE | Verify script |

---

#### Notes

- The settings are measured at TP=2 only. TP=4 has not served on the restored box.
- Cutlass and Triton may be slower than the kernels that need the toolkit. Installing a toolkit under scratch is the alternative; the overrides exist for that case.
- The header comment about TP=2's 1,139,984-token KV cache predates the restore. R5e read 2,115,984 tokens at TP=2 with this model.
