### 26Q4-HARN-30: Thinking is recorded, and its budget is a setting

**User Story**: As the person deciding how much the served model should think, I want every reply's thinking saved in the trace and its length set per run, so that I can read what the thinking tokens bought and compare runs that differ only in how much the model was allowed to think.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-30-thinking-is-recorded-and-budgeted` |
| **Dependencies** | None |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `e9c7352`):
- Batch q2 served `Qwen/Qwen3.6-27B-FP8`, which thinks before every answer. About 80% of output tokens were thinking, roughly half of every run's token budget.
- `GenerationResponse.thinking` holds the thinking (`26Q4-HARN-21`), but `ConversationLog.add_exchange` takes only the answer, so `conversations.json` has none of it. What half the budget bought cannot be read.
- Nothing can vary how much the model thinks. Qwen3.6 has no thinking levels, only on and off. vLLM 0.33.0 adds a per-request `thinking_token_budget`.
- Three q2 runs ended on `Request timed out.`. `OpenAIProvider` fixes the timeout at 300 s and keeps the SDK's two retries. A non-streamed reply arrives whole, so any reply longer than about 30,000 tokens (at ~100 tokens/s) waits 300 s three times, about 900 s, and then ends the run.
- `metadata.json` records the run's token budget and seed, but not the per-request cap, timeout or retries.

**Investigation**:
```bash
grep -rn "add_exchange" lms/                       # 7 call sites, none passes thinking
grep -n "AsyncOpenAI(" -A4 lms/providers/openai.py # timeout fixed, max_retries unset
grep -n "reasoning_content" lms/providers/openai.py
```

---

#### Acceptance Criteria

- [x] Every assistant turn in the trace carries a `thinking` field, the reply's thinking or `""`; `content` stays the answer. All seven call sites (propose, iterative attempt and both writeups, review, planning, working group) pass it
- [x] `conversations.json` round-trips `thinking`; a file without the field loads with `""`
- [x] Thinking reaches the trace whether the server returns it ahead of `</think>`, in `reasoning` (vLLM 0.33's reasoning parser) or in `reasoning_content` (older servers)
- [x] `ProviderConfig.thinking_budget` set: every OpenAI-compatible request sends `thinking_token_budget` in `extra_body`, a top-level field on the wire; None: nothing is sent
- [x] `--thinking-budget N` reaches `run_experiment` and `resume_experiment`; `LMS_THINKING_BUDGET` is the fallback; a value that is not a whole number 0 or more stops the run before any tokens are spent
- [x] `LMS_OPENAI_TIMEOUT` (seconds, > 0, finite) and `LMS_OPENAI_MAX_RETRIES` (≥ 0) reach the `AsyncOpenAI` client; unset, they stay 300 s and 2; a bad value fails at config load
- [x] `metadata.json` records `thinking_budget`, `request_max_tokens`, `request_timeout_s` and `max_retries` (per provider in a mixed society); a resume records them under `_resume` keys and leaves the originals alone
- [x] `uv run pytest tests/test_thinking_budget.py -q` covers each item above

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/config.py` | MODIFY | `thinking_budget`, `request_timeout_s`, `max_retries`; env parsing |
| `lms/providers/openai.py` | MODIFY | Budget in `extra_body`, timeout and retries on the client, read `reasoning` |
| `lms/traces.py` | MODIFY | `ConversationTurn.thinking`, save and load |
| `lms/agent.py` | MODIFY | Pass thinking at five call sites |
| `lms/planning.py` | MODIFY | Pass thinking |
| `lms/working_group.py` | MODIFY | Pass thinking |
| `lms/run.py` | MODIFY | `--thinking-budget`, `LMS_THINKING_BUDGET`, metadata |
| `.env.example` | MODIFY | Document the timeout and retries |
| `tests/test_thinking_budget.py` | CREATE | Behaviour |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-30-thinking-is-recorded-and-budgeted.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-30.sh` | CREATE | Verify script |

---

#### Notes

- **The server side is the batch kit's job, not this card's.** A budget needs vLLM started with `--reasoning-parser qwen3`. Longer thinking needs a larger `--max-model-len`, and a timeout sized to the longest reply: at least `LMS_OPENAI_MAX_TOKENS` divided by the measured tokens/s under the run's concurrency.
- **`thinking_token_budget` with no reasoning parser is refused, not ignored.** vLLM 0.33.0 (`vllm/v1/engine/input_processor.py`) raises `VLLMValidationError: thinking_token_budget is set but reasoning_config is not configured`, which the server returns as HTTP 400. The SDK does not retry a 400, so the run fails on its first request. With `--reasoning-parser qwen3` alone, the start and end markers come from the parser; `--reasoning-config` is optional.
- **Accepted values** (`vllm/sampling_params.py::validate_thinking_token_budget`): any integer 0 or more, or -1, which vLLM treats as unlimited. Floats, booleans and other negatives get a 400. The harness accepts 0 or more and uses None, not -1, for unlimited.
- **0 is a budget, not "thinking off".** The server forces `</think>` at once, so the model answers with next to no thinking. That is not the same prompt as `enable_thinking: false`. The budget counts from the last `<think>`, including any thinking tokens already in the prompt.
- **With a reasoning parser, vLLM 0.33 returns the thinking in `message.reasoning`.** It accepts `reasoning_content` only on input and renames it. The provider used to read only `reasoning_content`, so serving with the parser would have silently left `thinking` empty everywhere. It now reads `reasoning` first.
- **Usage counts thinking separately only with a parser.** vLLM 0.33 then fills `usage.completion_tokens_details.reasoning_tokens`, and `completion_tokens` still includes them. Without a parser, `completion_tokens_details` is null. This card does not record that count.
- **Only the OpenAI-compatible provider sends the budget**, and only it reads the timeout and retries. The Anthropic and Google providers ignore `thinking_budget`; metadata records the value the run was given.
- **`max_retries` covers more than timeouts.** The SDK also retries connection errors, 429 and 5xx. Setting it to 0 removes those retries too.
- **`LMS_THINKING_BUDGET` is read from the process environment**, like `LMS_SEED`, before `.env` is loaded. A batch job sets it in the job's environment. The budget is not kept in checkpoints, so `--resume` takes it from its own command line or environment.
- **Traces grow.** Thinking is most of the output, so `conversations.json` is several times larger than before.
