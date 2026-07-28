### 26Q3-INFRA-01: Local OpenAI-compatible serving path

> **Carryover from Sprint 1** (`26Q2-INFRA-01`, 0/24 points delivered). Scope
> unchanged; rationale updated — this is now also the lever for the Aug 21 API
> fallback in `docs/planning/calibration-program.md` §3 Phase B.

**User Story**: As the calibration program, I want agents to target any
OpenAI-compatible endpoint, so that the same harness runs against the cluster's
vLLM server or against a hosted API by changing configuration only.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-INFRA-01-local-serving-path` |
| **Dependencies** | 26Q3-CHORE-01 |
| **PR Size Target** | <300 lines |

---

#### Context

**Current State** (re-verified 2026-07-24, unchanged since the card was written):
- `lms/providers/openai.py:24-27` builds `AsyncOpenAI(api_key=..., timeout=...)`
  with **no `base_url`**.
- `lms/config.py:15-20` `ProviderConfig` has `api_key`, `model`, `max_tokens` —
  no `base_url`.
- `grep -rn "base_url" lms/` → no hits.

vLLM / SGLang / Ollama all expose an OpenAI-compatible `/v1`, so `OpenAIProvider`
serves a local model once it accepts a base URL.

**Why it matters more now than in Sprint 1**: this is the single switch that makes
the cluster-vs-API decision a config change. The calibration program pre-commits
to falling back to API models if the cluster isn't ready by Aug 21; that
contingency is only cheap if this lands first.

---

#### Acceptance Criteria

- [ ] `ProviderConfig` gains `base_url: str | None = None`
- [ ] `Config.from_env` reads `LMS_OPENAI_BASE_URL`
- [ ] `OpenAIProvider.__init__` passes `base_url` to `AsyncOpenAI` when set;
      unset → unchanged hosted-OpenAI behavior
- [ ] A local model is usable via `LMS_OPENAI_BASE_URL` + `LMS_OPENAI_MODEL` +
      a dummy `OPENAI_API_KEY` (vLLM accepts any key)
- [ ] `tests/test_local_serving.py`: base_url threads into the client; unset
      leaves the hosted default; env → config round-trip
- [ ] Anthropic/Google providers untouched — they remain the escape-hatch route

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/config.py` | MODIFY | `base_url` on `ProviderConfig`; env read |
| `lms/providers/openai.py` | MODIFY | Pass `base_url` when present |
| `tests/test_local_serving.py` | CREATE | Unit tests |
| `scripts/verify/26Q3-01/verify_26Q3-INFRA-01.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Reuse `OpenAIProvider`; do not add a provider class. One path covers vLLM,
  SGLang, and Ollama.
- Model-routing logic (the escape-hatch router) stays out of scope until a real
  roadblock appears.

---

#### Decision Gates

- If pointing at a local endpoint needs more than `base_url` + key handling
  (tool-calling or `response_format` incompatibilities), stop, document the gap,
  and propose a thin subclass. Do not silently fork the provider.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-INFRA-01.sh` exits 0
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
