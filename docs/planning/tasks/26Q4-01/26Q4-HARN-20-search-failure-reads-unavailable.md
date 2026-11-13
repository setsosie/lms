### 26Q4-HARN-20: A search that fails reads as unavailable instead of raising

**User Story**: As a [role], I want [feature] so that [benefit].

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-20-search-failure-reads-unavailable` |
| **Dependencies** | None |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

> Summarize current state. Include investigation results (grep evidence).

**Current State**:
- TODO

**Investigation**:
```bash
# TODO
```

---

#### Acceptance Criteria

- [ ] TODO

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| TODO | | |

---

#### Implementation Notes

- TODO

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- `26Q4-HARN-14`: an `exact?` that ran out of budget.
- TODO

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-20.sh`.

---

#### Outcome Demo

TODO

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-20.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

Split out of `26Q4-HARN-14` on 2026-11-09 (user's call). The evidence, as recorded there from the independent review at the final gate of `26Q4-HARN-09` (2026-11-06, at `3185b0e`), reproduced with a probe, predating that PR and not blocking:

- **Some search failures raise instead of reading as unavailable.** A grep `subprocess.TimeoutExpired`, and a `ConnectionResetError` or `UnicodeDecodeError` from either HTTP backend. No verdict comes out wrong. In the density script, one reset ends the run, and Gate 4 catches it as an audit hole. It's the same gap as `26Q4-HARN-09`'s "a search that failed" criterion, in a form its tests don't cover.
