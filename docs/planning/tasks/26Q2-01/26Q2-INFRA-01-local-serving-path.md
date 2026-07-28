### 26Q2-INFRA-01: Local OpenAI-compatible serving path

**User Story**: As the LMS team running local-first, I want agents to query a
self-hosted OpenAI-compatible endpoint (vLLM / SGLang / Ollama), so that the
formalization pipeline and the genome arm both run on local LLMs with no model
API spend — APIs reserved as a deliberate escape hatch.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | CRITICAL |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q2-INFRA-01-local-serving-path` |
| **Dependencies** | None |
| **PR Size Target** | <300 lines |

---

#### Context

> Local-first execution is a project policy (2026-06-18): default to self-hosted
> OSS models, fall back to model APIs only on a defined roadblock. The harness
> already abstracts providers, so this is a small enablement change, not a rewrite.

**Current State**:
- `lms/providers/openai.py:24-27` builds `AsyncOpenAI(api_key=..., timeout=...)`
  with **no `base_url`** → cannot target a local server.
- `lms/config.py:14-20` `ProviderConfig` has `api_key`, `model`, `max_tokens` —
  **no `base_url` field**.
- vLLM/SGLang/Ollama all expose an OpenAI-compatible `/v1` endpoint, so the
  existing `OpenAIProvider` can serve a local model once it accepts a base URL.

**Investigation**:
```bash
grep -rn "base_url" lms/                 # → (none)
grep -n "AsyncOpenAI" lms/providers/openai.py   # → line 24, no base_url kwarg
```

---

#### Acceptance Criteria

- [ ] `ProviderConfig` gains an optional `base_url: str | None = None` field
- [ ] `Config.from_env` reads `LMS_OPENAI_BASE_URL` into the OpenAI provider config
- [ ] `OpenAIProvider.__init__` passes `base_url` to `AsyncOpenAI` when set; when
      unset, behavior is unchanged (hosted OpenAI)
- [ ] A local model is usable by setting `LMS_OPENAI_BASE_URL` +
      `LMS_OPENAI_MODEL` + a dummy/`EMPTY` `OPENAI_API_KEY` (vLLM accepts any key)
- [ ] `tests/test_local_serving.py` covers: base_url threads into the client,
      unset base_url leaves the hosted default, env var → config round-trip
- [ ] Tests pass: `uv run pytest tests/test_local_serving.py -v`
- [ ] Type + lint clean: `uv run mypy lms/config.py lms/providers/openai.py && uv run ruff check lms/config.py lms/providers/openai.py`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/config.py` | MODIFY | Add `base_url` to `ProviderConfig`; read `LMS_OPENAI_BASE_URL` |
| `lms/providers/openai.py` | MODIFY | Pass `base_url` to `AsyncOpenAI` when present |
| `tests/test_local_serving.py` | CREATE | Unit tests |
| `scripts/verify/26Q2-01/verify_26Q2-INFRA-01.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Mirror the existing optional-field style in `ProviderConfig`; do not add a new
  provider class — reuse `OpenAIProvider` pointed at a local `/v1`. (Ollama and
  SGLang are also OpenAI-compatible, so one path covers all three.)
- Do NOT remove or break the Anthropic/Google providers — they stay as the
  escape-hatch route. This PR only *adds* the local option.
- Keep the genome arm's serving concern (`inputs_embeds`/`prompt_embeds`,
  `26Q2-SPG-01`) out of scope — that's a different, lower-level path; this card
  is plain chat-completions against a local endpoint.

---

#### Decision Gates

- If pointing `OpenAIProvider` at a local endpoint needs more than base_url +
  key tweaks (e.g. tool-calling / response-format incompatibilities) → stop,
  document the gap, propose a thin local provider subclass; do not silently
  fork the provider.
- If the local model can't follow the agent prompts well enough to be testable
  → that's a *capability* finding for the shakedown, not a blocker for this
  plumbing task; ship the plumbing, log the finding.

---

#### Out of Scope

- `26Q2-SPG-01` owns soft-embedding serving — not here.
- No model selection / routing logic (the API escape-hatch router) in this PR;
  that's a follow-up once a roadblock is actually hit.

---

#### Verification Script

See `scripts/verify/26Q2-01/verify_26Q2-INFRA-01.sh`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q2-01/verify_26Q2-INFRA-01.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened, <300 lines, tests included
