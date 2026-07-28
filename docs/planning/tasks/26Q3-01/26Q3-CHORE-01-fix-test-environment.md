### 26Q3-CHORE-01: Fix test environment

**User Story**: As anyone running `uv run pytest`, I want the async tests to
actually run, so that the suite's pass/fail signal is trustworthy.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-CHORE-01-fix-test-environment` |
| **Dependencies** | None |
| **PR Size Target** | <30 lines |

---

#### Context

`uv run pytest` reports **44 failed, 304 passed**. Every failure is:

```
async def functions are not natively supported.
PytestConfigWarning: Unknown config option: asyncio_mode
```

`pytest-asyncio` is declared in `[project.optional-dependencies] dev`
(`pyproject.toml`), which `uv sync` does **not** install by default — only
`[dependency-groups] dev` (which contains just `pytest-cov`) is installed. So
`asyncio_mode = "auto"` is set but the plugin providing it is absent.

No production code is broken. The suite has simply been reporting a false red for
some time, which makes it useless as a regression signal for the rest of this
sprint.

---

#### Acceptance Criteria

- [ ] `pytest` and `pytest-asyncio` are in `[dependency-groups] dev` so `uv sync`
      installs them
- [ ] `uv run pytest` reports **0 failed** (11 skipped is fine)
- [ ] The duplicate declaration in `[project.optional-dependencies] dev` is
      removed or reconciled — one source of truth
- [ ] `uv.lock` regenerated and committed

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | MODIFY | Move test deps into `[dependency-groups] dev` |
| `uv.lock` | MODIFY | Regenerate |

---

#### Definition of Done

- [ ] `uv run pytest` → 0 failed
- [ ] `uv run ruff check` clean
