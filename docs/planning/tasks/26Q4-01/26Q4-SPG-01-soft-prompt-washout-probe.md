### 26Q4-SPG-01: Washout probe — does a soft prompt still change the model at our context lengths?

**User Story**: As the person deciding whether to evolve soft prompts, I want to measure how much a soft prompt shifts the model's next-token probabilities at real context lengths and positions, so that the arm can be reshaped or stopped after hours on one GPU, before any Lean run.

> Drafted by Claude on 2026-11-05 from a discussion with the user. The user accepted the thresholds for now. Review the rest as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-SPG-01-soft-prompt-washout-probe` |
| **Dependencies** | None to build. Runs in Sprint 8 |
| **PR Size Target** | <400 lines (max 800) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `1a9172e`):
- No measurement exists of a soft prompt's effect on any model we serve. `grep -ril "soft_prompt\|inputs_embeds" lms/ tests/` finds nothing.
- The closest work, [QD-LLM](https://arxiv.org/html/2605.09781v2), places 8 × 4096 values before the task tokens on short benchmark tasks and does not examine washout. LMS prompts can run far longer.
- Position matters beyond washout. vLLM's cached blocks chain on everything before them, so a different prefix at the very start of each agent's prompt stops agents sharing cached context. Just before the statement, the shared context stays cacheable and the soft tokens sit closer to where the model generates.
- `torch` and `transformers` are not project dependencies. The probe runs with `uv run --with`.

**Investigation**:
```bash
grep -n 'numpy\|torch\|transformers' pyproject.toml
# no hits
```

---

#### Scope

Forward passes only, on one GPU, with no sampling, no Lean and no serving engine. For each condition, compute the KL divergence of the next-token distribution at the last prompt position against the same prompt with no soft tokens.

| Factor | Levels |
|---|---|
| Context length | about 1K, 10K, 100K tokens, from real LMS prompts |
| Position | start of the prompt; just before the statement |
| Soft tokens | 2, 8 |
| Soft token source | one real token; 2-token blend; mean of two unrelated blends; mean of two relatives (one ancestor plus mutation) |

The baseline is natural rewording: the same KL for reworded versions of the instruction text. A soft prompt weaker than rewording a sentence has little leverage.

---

#### Acceptance Criteria

- [ ] `lms/soft_prompt_metrics.py` has `next_token_kl(p, q)` and `summarize_conditions(records)`, in numpy: `tests/test_soft_prompt_metrics.py` covers identical distributions (zero), a known value, and zero-probability entries
- [ ] `scripts/soft_prompt/washout_probe.py` takes `--model`, `--out` and `--seed`, and writes one JSON record per condition, including the model id and its training cutoff
- [ ] The script refuses a model with a missing training cutoff or one not before 2026-11-04
- [ ] The script also reruns the merge sizes from `scripts/research/soft_prompt_merge_sim.py` on the model's real embedding matrix and writes them to the same report, since that simulation assumes isotropic embeddings — accepted via Outcome Demo
- [ ] The report states the median soft-prompt KL divided by the median rewording KL, per context length and position — accepted via Outcome Demo
- [ ] `uv run pytest tests/test_soft_prompt_metrics.py -v` passes

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/soft_prompt_metrics.py` | CREATE | KL and report summary, numpy only |
| `scripts/soft_prompt/washout_probe.py` | CREATE | The probe |
| `tests/test_soft_prompt_metrics.py` | CREATE | Metric tests |
| `pyproject.toml` | MODIFY | Add `numpy`, as `26Q2-SPG-03` specifies |
| `scripts/verify/26Q4-01/verify_26Q4-SPG-01.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Use Hugging Face `inputs_embeds`. Keep model code in the script, so `lms/` gains no `torch` dependency.
- Use the served model, `Qwen/Qwen3.6-27B-FP8`. If FP8 weights will not load this way, see the first gate.
- Prompts should come from real LMS prompts with real Stacks content, not filler text. Washout on filler says little.

---

#### Decision Gates

- If the FP8 build cannot run with `inputs_embeds` → stop. Using an unquantized build is acceptable only if the report says so, and that is the user's call.
- If 100K tokens of context do not fit on one GPU → report the longest length that fits; do not change partitions or GPU counts without asking.
- **Reading the result (provisional; revisit once the first report shows the real range of these ratios):**
  - At 10K tokens, ratio ≥ 0.5 at either position → continue, at the stronger position.
  - Below 0.5 at both → stop before `26Q2-SPG-01`. Decide between per-layer prompts, steering vectors, or a write-up.
  - The start position much weaker than the statement position → the design moves off the start of the prompt, and the user decides.

---

#### Out of Scope

- `26Q2-SPG-01` owns serving and caching costs.
- `26Q2-SPG-02` owns whether different soft prompts solve different statements.
- No evolution, no Lean, no change to `lms/agent.py` or the providers.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-SPG-01.sh`.

---

#### Outcome Demo

**Where**: the GPU server, one GPU, as a round folder Claude prepares; the user runs it.
**Run**: the round's `RUN_THIS.txt`, which submits `scripts/soft_prompt/washout_probe.py` pinned to the merged commit.
**Expect**: one table per position with the ratio at each context length, plus the merge sizes on real embeddings. The result either continues the arm or stops it under the gates above.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-SPG-01.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <400 lines changed (target) / <800 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
