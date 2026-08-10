### 26Q3-INFRA-02: Make the per-request token cap configurable

> **Card for issue #17**, filed 2026-07-31 and confirmed against the Phase B
> serve command on 2026-08-10. Worked around in
> `docs/infrastructure/cluster-runbook-calibration.md` Step 4 by raising the
> served context window to 131072. The workaround holds for Gate B; it does not
> survive contact with a model whose native window is smaller, and it leaves the
> harness asking for 64k completions it will never use.

**User Story**: As an operator serving a local model, I want the per-request
completion cap to follow the endpoint I am serving, so that the harness does not
ask a 65k-context server for 64k completion tokens on every call.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-INFRA-02-per-request-token-cap` |
| **Dependencies** | 26Q3-INFRA-01 (merged, #18) |
| **PR Size Target** | <150 lines |

---

#### Context

`ProviderConfig.max_tokens` defaults to `DEFAULT_MAX_TOKENS = 64_000`
(`lms/config.py:11`), whose docstring says it is "limited by Claude Opus 4.5's max
output tokens". Nothing overrides it per request:

- `Config.from_env` never reads an env var for it — the three `ProviderConfig`
  constructions at `lms/config.py:65,73,83` all take the dataclass default.
- `lms/agent.py` calls `provider.generate(messages, system_prompt)` with no
  `max_tokens`, so `OpenAIProvider.generate` falls through to
  `self.config.max_tokens` and sends `max_completion_tokens=64000`.
- `run.py --max-tokens` is a *budget* ceiling for the whole run
  (`Society.max_tokens`), not a per-request cap. The names collide; the meanings
  do not.

vLLM rejects a request when `prompt_tokens + max_tokens > max_model_len`. Serving
Qwen3-Coder-30B-A3B at `--max-model-len 65536` therefore 400s on any prompt over
~1,536 tokens — which is every agent prompt. The failure surfaces on the first
generation and reads like a malformed request rather than a config mismatch.

A Claude-shaped constant is the wrong thing to size a self-hosted server around,
and [[local-first-llm-policy]] makes self-hosted the default path.

---

#### Acceptance Criteria

- [ ] `Config.from_env` reads `LMS_OPENAI_MAX_TOKENS` (and the Anthropic/Google
      equivalents) onto `ProviderConfig.max_tokens`; blank is treated as unset,
      matching the `or`-not-default convention already used in that function
- [ ] `DEFAULT_MAX_TOKENS` keeps its current value so hosted-provider behavior is
      unchanged when the var is absent
- [ ] The comment on `DEFAULT_MAX_TOKENS` stops asserting a Claude limit as if it
      were a global one
- [ ] A non-integer or non-positive value fails loudly at config load, not at the
      first generation
- [ ] Tests: env → `ProviderConfig.max_tokens`; unset keeps the default; the
      value reaches `max_completion_tokens` in the request payload (extend the
      `_StubEndpoint` in `tests/test_local_serving.py`, which already captures
      request payloads)
- [ ] `docs/infrastructure/cluster-runbook-calibration.md` Step 5 sets
      `LMS_OPENAI_MAX_TOKENS`, and Step 4's `--max-model-len` note is reduced to
      a sizing guideline rather than a workaround

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/config.py` | MODIFY | Env read + validation for `max_tokens` |
| `tests/test_local_serving.py` | MODIFY | Assert the cap reaches the payload |
| `docs/infrastructure/cluster-runbook-calibration.md` | MODIFY | Steps 4 and 5 |
| `scripts/verify/26Q3-01/verify_26Q3-INFRA-02.sh` | CREATE | Verification script |

---

#### Decision Gates

- Do **not** auto-detect the cap by querying `/v1/models` for `max_model_len`.
  It is one more thing to fail at startup for a value the operator already knows,
  and it does not exist on every OpenAI-compatible server.
- If per-*call-site* caps turn out to be wanted (a short cap for review turns, a
  long one for proof generation), that is a separate card. This one only makes
  the single existing cap configurable.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-INFRA-02.sh` exits 0
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
