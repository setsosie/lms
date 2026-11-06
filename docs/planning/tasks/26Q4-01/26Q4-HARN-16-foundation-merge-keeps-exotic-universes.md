### 26Q4-HARN-16: Foundation merge keeps universes the header doesn't bind

**User Story**: As the ratchet measurement, I want an artifact that binds its own universe names to survive the merge into `LMS.Foundation`, so that one agent's choice of level names can't roll back a whole generation's verified work.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-16-foundation-merge-keeps-exotic-universes` |
| **Dependencies** | None |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**:
- `FoundationFile.add_artifact` drops every line starting `universe `, on the grounds that the header binds them. The header binds only `u v w`.
- The verifier keeps a candidate's own `universe` lines, so `universe u₁ u₂` verifies alone and then breaks the merged module.
- A failed merged build rolls back the whole generation, so other agents' valid work goes with it.

**Investigation**:
```bash
grep -n 'startswith("universe ")' lms/foundation.py
grep -n "test_exotic_universe_names_are_preserved" tests/test_verify_universes.py   # covers the verifier's wrapper, not the merge
```

---

#### Acceptance Criteria

- [ ] A saved foundation binds every universe name an artifact declared beyond the header, for that artifact's entries
- [ ] Two artifacts binding the same exotic name, and a later artifact naming that level explicitly (`def foo.{u₁}`), still compile together: the binding is scoped to its artifact, in a `section`
- [ ] Names the header binds are not rebound, and an artifact with none of its own gets no `section`
- [ ] A name the seed binds is not rebound
- [ ] A rollback drops the bindings along with the entries, and they survive `save_metadata`/`load`, including a checkpoint written before this change
- [ ] A comment is not read as a binding: neither `universe u₁ -- levels` nor a docstring line beginning `universe of discourse` adds a name
- [ ] The artifact's own names keep their declared order, because Lean orders level parameters by it: `universe v₁ u₁` stays `v₁ u₁`, or `tag.{0, 1}` silently means different types
- [ ] `universe u in` reads `in` as a keyword, not a level name
- [ ] Real Lean compiles the saved file for the cases above, in the CI job that has a toolchain

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | Entries carry their artifact's universe names; `save` scopes them |
| `tests/test_foundation_universes.py` | CREATE | Merge, rollback, persistence, comments |
| `tests/test_lean_real.py` | MODIFY | The saved file compiles under real Lean |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-16-foundation-merge-keeps-exotic-universes.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-16.sh` | CREATE | Verify script |

---

#### Implementation Notes

- The names live on `FoundationEntry`, so snapshot, restore, `snapshot_before` and checkpoints need no new state: what's bound follows from the entries kept.
- A `section` per artifact rather than hoisting to the header. A header binding would break a later artifact that names the level explicitly, with `a universe level named 'u₁' has already been declared`.
- Mixed lines aren't fully covered. `universe v₁ u` still merges as `.{u, v₁}`, because the header binds `u` first. That's the pre-existing header-order defect in `26Q4-HARN-19`.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- Name collisions and per-artifact rollback: `26Q4-HARN-17`.
- Reopening goal tags after a rollback: `26Q4-HARN-18`.
- The verifier's wrapper, which already keeps these lines.
- The order of header names, and of a mixed line: `26Q4-HARN-19`.
- Names on a continuation line (`universe u₁\n  u₂`), which fail loudly on `main` too.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-16.sh`.

---

#### Outcome Demo

**Where**: laptop, or any machine with a Lean toolchain
**Run**:
```bash
uv run pytest tests/test_lean_real.py -k TestFoundationUniverses -v
```
**Expect**: all pass, not skipped. On `main` the same file fails with `unknown universe level`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-16.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

Evidence from the 2026-11-06 simulation at `3185b0e`, written by Claude for the author of the sections above. The run is local and gitignored: `experiments/sim_sonnet_team_2026-11-06/`.

- **Defect:** `FoundationFile.add_artifact` drops every line starting `universe ` (`lms/foundation.py:813`). `strip_header_universes` (`lms/foundation.py:59`) already keeps names the header doesn't bind, and the verifier keeps them too, so an artifact can verify and then break the merged module.
- **Observed:** one generation 1 artifact declared `universe u₁ u₂`. The merged build failed with `unknown universe level` and all 9 of that generation's entries were rolled back, including two other agents' valid work.
- **Confirmed:** with `u₁ u₂` bound in the header, generations 0 and 1 merge and build with no errors (`diag2.py`).
- **Existing test gap:** `tests/test_verify_universes.py::test_exotic_universe_names_are_preserved` covers the verifier's wrapper, not `add_artifact`.
