### 26Q4-HARN-22: Each run samples from its own seed

**User Story**: As the person comparing arms, I want every run to draw its own sampling seed and record it, so that nine isolated runs are nine samples and not one sample repeated.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q4-HARN-22-each-run-samples-its-own-seed` |
| **Dependencies** | Stacked on `26Q4-HARN-04` (#94), which carries the provider changes of `26Q4-HARN-21` (#95). Phase C's I9 and S1 arms need it |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (branch `26Q4-HARN-04` @ `1a5d28f`):
- `OpenAIProvider.generate` sends no `seed`. vLLM then samples from its engine seed, 0 by default, in request order.
- In batch q1's arms rehearsal, each I9 run was one agent working alone against a vLLM server started fresh for its job. Eight of the nine ran, and all eight were identical: 11 calls, 294,028 prompt and 51,529 completion tokens, 345,557 in all. S1's first call matched them too. Committee runs differed (C9 2,223,347 tokens, C9b 2,143,492), because concurrent requests change the order in which samples are drawn.
- The I9 arm meant to be nine independent agents was one agent, counted eight times. ADR 0003's I9 arm, and any between-run variance, depend on this.

**Investigation**:
```bash
grep -n "create(" lms/providers/openai.py
grep -n "metadata: dict" lms/run.py
```

---

#### Acceptance Criteria

- [ ] `ProviderConfig.seed` set: every request carries `seed`, derived from it and different for every request in the process, within [0, 2^31); unset: no `seed` is sent
- [ ] `run_experiment` takes `sampling_seed`, draws one when it is None, gives it to every provider it builds, and records it as `sampling_seed` in metadata.json
- [ ] `resume_experiment` does the same and records `sampling_seed_resume`, leaving the original value alone
- [ ] `--seed N` reaches both; `LMS_SEED` is the fallback; a value that is not a whole number in range stops the run before any tokens are spent
- [ ] `uv run pytest tests/test_sampling_seed.py -q` covers each item above

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/config.py` | MODIFY | `seed` field, `SEED_SPACE` |
| `lms/providers/openai.py` | MODIFY | Per-request seed |
| `lms/run.py` | MODIFY | Draw, pass and record; `--seed`, `LMS_SEED` |
| `tests/test_sampling_seed.py` | CREATE | Behaviour |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-22-each-run-samples-its-own-seed.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-22.sh` | CREATE | Verify script |

---

#### Notes

- The run script needs no change: a batch job sets `LMS_SEED` in its environment, or leaves it unset and lets each run draw one.
- A seed fixes sampling, not the whole run. Requests that share a batch on the server still affect one another's numerics, so a re-run with the same seed is close to, not exactly, the original.
- Anthropic and Google providers send no seed; their APIs are not the local-server case this fixes.
