### 26Q3-HARN-09: Verified work must reach the next generation

**User Story**: As an agent in generation N, I want the definitions my
predecessors verified to actually be importable, so that building on prior work
is possible rather than merely encouraged.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | CRITICAL |
| **Status** | ✅ DONE |
| **Branch** | `26Q3-HARN-09-foundation-reaches-next-generation` |
| **Dependencies** | 26Q3-HARN-01, 26Q3-HARN-02 |
| **PR Size Target** | <250 lines |

---

#### Context

Found on the box 2026-08-10, in the first multi-agent run
(`experiments/shakedown_3x3`, 3 agents × 3 generations, 260K tokens).

Generation 1 verified a `Category` definition. Generations 2 and 3 read the
library, cited it correctly — all four downstream artifacts referenced
`definition-Category-d85ff885`, every citation resolving — and wrote
`import LMS.Foundation` to build on it.

They got a module that predated the entire run.

**Cause 1 — the foundation only reaches disk at a checkpoint.**
`foundation.save()` is called only from `Society.save()`, which `lms/run.py`
invokes when `(gen + 1) % checkpoint_interval == 0`. The default interval is 10.
On a 3-generation run the condition never fired, so the foundation lived in
memory for the whole experiment and every `import LMS.Foundation` resolved
against whatever was on disk beforehand.

**Cause 2 — nothing recompiles it.** `LeanProject.ensure_built` rebuilds only
when a *new import name* appears. `LMS.Foundation`'s name never changes while
its contents change every generation, so even a written file would be shadowed
by a stale `.olean`.

**Cause 3 — `autoImplicit` hid both.** Lean's default is `true`, so an
unresolvable identifier silently becomes an auto-bound implicit variable rather
than an error. What the run actually reported was:

```
error: Function expected at
  Category
but this term has type
  ?m.1
...
error: invalid use of explicit universe parameters, `Category` is a local variable
```

which reads as the agent's mathematics being wrong. The real fault was that
`Category` was never in scope.

**Cause 4 — agents could not see the interface.** `get_context_for_agent`
listed `structure Category` plus a regex-extracted field list (which produced
exactly one field, `Hom`). Nothing showed the declaration header
`structure Category (obj : Type u)`, so an agent wrote `C.Ob`, guessing that
objects were a field. They are a parameter.

**This is the cumulative-knowledge mechanism, and it had never once worked.**
Every prior run either had one generation, or fewer than ten. The collective
brain hypothesis has not yet been tested — the apparatus for it was broken.

---

#### Acceptance Criteria

- [x] Persisting the foundation is part of finishing a generation, not part of
      checkpointing — `run_generation` persists whenever the generation verified
      something
- [x] Persisting recompiles: `LeanProject.rebuild_changed_sources()` builds
      unconditionally rather than via the new-import heuristic
- [x] No rebuild is triggered by a generation that verified nothing
- [x] Safe when the verifier has no Lean project (`MockLeanVerifier`)
- [x] `autoImplicit` and `relaxedAutoImplicit` are `false` on the verification
      path, passed to `lean` itself — **not** as `leanOptions` in
      `lakefile.toml`, which `lake env lean <file>` does not apply to an
      arbitrary file
- [x] `FOUNDATION_HEADER` sets the same two options, so an entry cannot pass
      standalone verification and then fail the library build
- [x] `get_context_for_agent` shows the real declaration signature
- [x] End-to-end regression test through `run_generation`, not only unit tests
      of the helper — the helper test passes vacuously if the call is dropped

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/society.py` | MODIFY | `run_generation` wrapper + `persist_foundation` |
| `lms/lean/project.py` | MODIFY | `rebuild_changed_sources` |
| `lms/lean/real.py` | MODIFY | `STRICTNESS_FLAGS` on the lean invocation |
| `lms/foundation.py` | MODIFY | Header options; signature in agent context |
| `tests/test_foundation_persistence.py` | CREATE | 9 tests incl. end-to-end |
| `tests/test_lean_real_env.py` | MODIFY | Assert file position from the end |

---

#### Implementation Notes

- `autoImplicit false` is a deliberate strictness increase. Some code that
  compiled before will now fail — that is the point, and it matches Mathlib's
  own setting. The failures become legible instead of cascading.
- The `run_generation` → `_run_generation_impl` rename puts persistence on one
  path covering all five call sites rather than at each of the many returns.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `uv run pytest` clean — 487 passed
- [x] `uv run ruff check` / `uv run mypy` clean on the files this card touches;
      `lms/foundation.py:16` (`dataclasses.field` unused) and the four
      `lms/society.py` type errors are unchanged from the merge base
- [x] Verify script fails at the merge base
