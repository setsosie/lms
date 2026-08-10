### 26Q3-HARN-10: Foundation names need opening, not just importing

**User Story**: As an agent handed a verified `Category`, I want to be able to
refer to it by name, so that the foundation is usable rather than merely
present.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | CRITICAL |
| **Status** | ✅ DONE |
| **Branch** | `26Q3-HARN-10-foundation-names-need-opening` |
| **Dependencies** | 26Q3-HARN-09 |
| **PR Size Target** | <200 lines |

---

#### Context

Found in `experiments/shakedown_3x3_c` on 2026-08-10 — the first run in which
the foundation actually reached the next generation (26Q3-HARN-09 confirmed
working on the box: December's mock corpus gone, this run's own `Category`
written to disk).

All five generation-1 and generation-2 artifacts wrote `import LMS.Foundation`.
All five failed with the same error:

```
error(lean.unknownIdentifier): Unknown identifier `Category`
```

`FOUNDATION_HEADER` wraps every entry in `namespace LMS.Foundation`, so the
verified definition is `LMS.Foundation.Category`. **The module resolved; the
name never did.** `get_context_for_agent` told agents to `import` and nothing
else.

This was true in every earlier run as well. It only became visible when
`autoImplicit` was turned off in 26Q3-HARN-09 — before that, the unresolved
name was silently auto-bound as an implicit variable and the failure surfaced
somewhere else entirely as a type mismatch on `Functor C`, which read as the
agent's mathematics being wrong.

Two further defects in `AGENT_SYSTEM_PROMPT_V2_GOAL` (v2.5) pointed the same
way, and contradicted the corrected context:

1. *"All VERIFIED artifacts accumulate in `LMS.Foundation` (imported
   automatically)."* Nothing imports it for the agent.
2. The block labelled **RIGHT (Using existing - WILL SUCCEED)** contained no
   import at all, and declared `def Cone ... where` — which is not valid Lean
   for a declaration with fields. An example that cannot compile is worse than
   no example.

---

#### Acceptance Criteria

- [x] `FoundationFile.get_preamble()` returns both the `import` and the `open`
- [x] `get_context_for_agent` hands agents the preamble, not just the import
- [x] The context states *why* the `open` is required — an instruction without
      its reason gets dropped under pressure
- [x] `NAMESPACE` is a single constant, tested against what `save()` writes, so
      the two cannot drift
- [x] Goal system prompt no longer claims the foundation is auto-imported
- [x] Goal system prompt's "WILL SUCCEED" example compiles (`structure`, not
      `def`, and carries the import/open preamble)
- [x] Prompt version bumped 2.5.0 → 2.6.0 rather than edited in place — prompt
      versions are recorded per run in `metadata.json`, so editing content
      under a fixed version makes two runs incomparable

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | `NAMESPACE`, `get_preamble`, context text |
| `lms/prompts.py` | MODIFY | `AGENT_SYSTEM_PROMPT_V2_6_GOAL`, registry |
| `tests/test_foundation_namespace.py` | CREATE | 7 tests |

---

#### Implementation Notes

- `open` rather than dropping the namespace or `export`ing to the root. The
  foundation transitively imports Mathlib, where `Functor` is already taken;
  putting foundation names at the root would trade `unknown identifier` for
  ambiguity errors. Namespaced entries plus an explicit `open` keeps the
  collision surface opt-in.
- `get_import_statement()` is unchanged and still returns only the import —
  it has an existing test and other readers.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `uv run pytest` clean — 498 passed
- [x] `uv run ruff check` / `uv run mypy` clean on the files this card touches
- [x] Verify script fails at the merge base
