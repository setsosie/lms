### 26Q4-HARN-19: The merged foundation means what each artifact verified alone

**User Story**: As the ratchet measurement, I want an artifact in `LMS.Foundation` to mean what Lean checked when it verified alone, so that a statement doesn't silently change once merged.

> Context and draft criteria written by Claude on 2026-11-06 from the reviews of `26Q4-HARN-16` and `-17`. The user hasn't reviewed this text, and the choice in Acceptance Criteria is theirs.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-19-merged-foundation-means-what-verified` |
| **Dependencies** | 26Q4-HARN-16, 26Q4-HARN-17 |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**: An artifact verifies alone, and is then written into one file with others. Several differences between those two contexts can change what it means, and the merged file can still compile. All of these were reproduced on core Lean 4.32.1 and predate the fixes above.

- **Header universe order.** The verifier binds names in the artifact's own order, and the merged file binds `u v w` first. `universe v u` with `abbrev mAt := m.{0, 1}` reads `(α : Type) (β : Type 1)` alone and `(α : Type 1) (β : Type)` merged, exit 0. `universe v u` with `Cat.{v, u}` verifies alone and fails the merged build. `lms/prompts.py` teaches `universe v u`, so agents are likely to write this. A mixed line, such as `universe v₁ u`, has the same defect.
- **Shadowing.** An artifact using core `id` gets an earlier artifact's `id` once merged, with no name of its own colliding. `def useId : Nat := id 1` reads 1 alone and 2 merged.
- **An own-line attribute on the last entry.** It stays with that entry and attaches to the next artifact's declaration, exit 0.
- **Text before the first declaration is discarded.** That includes a leading `open` or `variable`. Most cases fail the build loudly, but a dropped `open` can change what a name resolves to.
- **A copy reads the same in a different context.** `def g : Nat := k + 1` under `variable (k : Nat)` and under `def k : Nat := 5` is the same text with a different type. The second is skipped as a copy, so a later `def h := g` reads `Nat` alone and `Nat → Nat` merged, exit 0.
- **`open X in` or `set_option ... in` before a skipped copy stays behind.** It attaches to the next artifact's first declaration, exit 0, or breaks the file if it comes last.

Failures that aren't silent, also predating these fixes: a docstring line beginning `universe ` that closes its comment leaves it unterminated; an artifact opening with a bare `section` keeps a stray `end`; names on a continuation line (`universe u₁\n  u₂`) aren't read; `universe x in` with a later `def b.{x}` fails the merged build as already declared.

Neighbours for a card of their own, from the same reviews:
- **A kill during a merged build can leave a checkpoint that won't resume.** `current_generation` advances before the build, so the artifact being built counts as good, and `restore_foundation` then returns False. A retry after a failed build widens that window.
- **`lake build` has no timeout,** so a hung build holds a job until its walltime.
- **A gate-blocked or refused artifact's note enters the textbook under a `[VERIFIED]` success entry.**

---

#### Acceptance Criteria

- [ ] **Choose the approach (user's call):** give every artifact its own `section` with the universe context the verifier used, and no file-level `universe` line, or compare each statement's elaborated type alone and merged after the build, and drop those that differ. The second also catches shadowing, and costs more per build.
- [ ] The silent cases above either compile with the verified meaning or are dropped with a reason in the log
- [ ] Real Lean compiles the fixtures in the CI job that has a toolchain

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | Depends on the choice above |
| `tests/test_lean_real.py` | MODIFY | The fixtures above, as real builds |

---

#### Implementation Notes

- Measure first if the box allows: how often do verified artifacts in a real run hit these? Nobody has counted.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- Name collisions and rollback: `26Q4-HARN-17`.
- Whether an artifact's mathematics is faithful to the source: the D4 review.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-19.sh`.

---

#### Outcome Demo

**Where**: any machine with a Lean toolchain
**Run**:
```bash
uv run pytest tests/test_lean_real.py -k MergedMeaning -v
```
**Expect**: all pass, not skipped.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-19.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
