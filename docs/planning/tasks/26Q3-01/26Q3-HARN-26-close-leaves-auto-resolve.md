### 26Q3-HARN-26: Closing leaves — a proved child resolves its parent sketch

**User Story**: As the calibration program, I want a verified proof of an open
leaf to close that leaf, and a parent whose leaves have all closed to be
re-verified and promoted without anyone re-proving it, so that a reduction made
in one generation and its pieces proved in later ones add up to a verified
statement — the ratchet the project exists to measure.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | HIGH — without it a sketch is a dead end and HARN-25 only relabels failures |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-26-close-leaves-auto-resolve` |
| **Dependencies** | `26Q3-HARN-25` (sketches, leaves), `26Q3-HARN-28` (statement pinning) |
| **PR Size Target** | <600 lines (max 1000) |
| **Parts** | 2/2 of the proof-sketch pair; Part 1 is `26Q3-HARN-25` |

---

#### Context

> Written 2026-09-05. Evidence below is against `main` @ `85ccf9e` plus the
> HARN-25 red tests on `26Q3-HARN-25-proof-sketches-open-leaves` @ `b9f0da7`.

After HARN-25 a group's conditional proof is recorded as `SKETCH`, its
`sorry`-bodied children become AVAILABLE leaves tagged `<parent>/<child>`, and
the parent is BLOCKED on them. Nothing closes the loop: a leaf can be assigned
and its artifact verified, but the leaf is a graph node like any other — the
generic DONE path marks it, the parent's `requires` are then all DONE and
`_recalculate_availability` makes the parent AVAILABLE again, and the next
generation is handed the parent *as a fresh task* with the sketch nowhere in
its prompt. The reduction is lost at the moment it should pay off.

Prove2Me (arXiv 2608.28433 §3): "it suffices to submit a local proof of that
lemma, after which the parent theorem auto-resolves." The parent was already
accepted conditionally; closing the last import makes it unconditional.

**Current State (post-HARN-25)**:
- `lms/dependency.py` — `add_open_leaves` creates leaves and blocks the parent;
  no reverse operation, no notion of "all leaves closed"
- `lms/society.py` — the DONE path in Phase 5 is tag-agnostic; nothing looks
  at `parent_tag` or `sketch_artifact_id`
- `lms/artifacts.py` — a `SKETCH` artifact keeps `open_children`; nothing ever
  shrinks that list
- `lms/foundation.py` — receives only `VERIFIED_LEAN` artifacts; a sketch's
  main theorem never reaches it

**Investigation**:
```bash
grep -n "parent_tag\|sketch_artifact_id" lms/society.py
# no hits (HARN-25 branch)
grep -n "def add_open_leaves" -A 30 lms/dependency.py | grep -n "unlocks"
# leaves unlock the parent; nothing consumes that on the society side
```

---

#### Scope

- Committee mode only.
- Closing a leaf on a pinned, verified child proof.
- Resolving a parent whose leaves are all closed: strip, re-verify, promote —
  or roll back with the error recorded.
- Not partial resolution (some leaves closed) beyond bookkeeping, not
  multi-level sketches beyond what recursion gives for free.

---

#### Acceptance Criteria

- [ ] `DependencyGraph.close_leaf(leaf_tag, artifact_id) -> str | None` marks
      the leaf DONE with its artifact and returns the parent tag when **all**
      of the parent's leaves are DONE, else None; it returns None without
      mutating for a non-leaf or unknown tag.
      `tests/test_dependency_leaves.py::test_close_leaf_reports_last_child` passes.
- [ ] `DependencyGraph.leaves_of(parent_tag) -> list[DependencyNode]` returns
      the parent's leaf nodes in creation order.
- [ ] `lms/sketch.py::strip_children(code: str, names: list[str]) -> str`
      removes the named `sorry`-bodied child declarations from a sketch and
      leaves everything else byte-for-byte. `tests/test_sketch.py::test_strip_children`
      passes.
- [ ] In committee Phase 5, a verified artifact for a leaf task (pinned to the
      leaf's statement, HARN-28) enters the foundation as today and additionally
      calls `close_leaf`; the sketch artifact's `open_children` drops the
      closed name. `tests/test_society_resolve.py::test_child_proof_closes_leaf`
      passes.
- [ ] When `close_leaf` returns a parent tag, the society **resolves** it in
      the same generation: the sketch artifact's code is stripped of its
      children, verified with `allow_sorry=False` against the current
      foundation, and on success a new artifact (type and tag of the parent,
      `references` including every child artifact id, `notes` naming the
      sketch id) is `VERIFIED_LEAN`, gated, added to the foundation, the parent
      task DONE, `goal.mark_formalized` called, and
      `GenerationResult.artifacts_resolved` incremented.
      `tests/test_society_resolve.py::test_last_child_resolves_parent` passes,
      including that the verifier saw no `sorry` and that the parent's artifact
      cites both children.
- [ ] If the stripped parent fails to verify, the parent task returns to
      AVAILABLE with the error in a `[RESOLVE FAILED]` textbook entry; the
      sketch artifact and the children stay as they are; nothing enters the
      foundation. `tests/test_society_resolve.py::test_failed_resolution_reopens_parent`
      passes.
- [ ] A resolved parent's prompt content for any later assignment (if it is
      reopened) includes the sketch code under "Prior reduction", so the
      reduction is never presented as a fresh task again.
      `tests/test_society_resolve.py::test_reopened_parent_shows_sketch` passes.
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-26.sh` holds greps and pytest
      calls only.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/dependency.py` | MODIFY | `close_leaf`, `leaves_of` |
| `lms/sketch.py` | MODIFY | `strip_children` |
| `lms/society.py` | MODIFY | leaf-close hook; `_resolve_parent`; `artifacts_resolved`; prior-reduction task content |
| `lms/artifacts.py` | MODIFY | shrink `open_children` on close |
| `tests/test_dependency_leaves.py` | MODIFY | `close_leaf`, `leaves_of` |
| `tests/test_sketch.py` | MODIFY | `strip_children` |
| `tests/test_society_resolve.py` | CREATE | committee path |

---

#### Implementation Notes

- The child proof is in the foundation under the child's name with the child's
  pinned signature (HARN-28), so the stripped parent resolves `aux` through
  the same import/open path every artifact uses (HARN-09/10/11). Resolution
  therefore runs *after* the generation's foundation persist, or the strip
  must be verified with the child code prepended — pick the former; the
  latter re-verifies code Lean already accepted.
- Resolution spends no model tokens. Record its wall-clock in the ledger under
  the parent's statement key with `outcome="resolve"` so per-statement cost
  (HARN-05) includes it.
- The gates run on the resolved artifact exactly as on any verified one; a
  gate block leaves the parent un-DONE with `_note_gate_block`, same as today.
- `close_leaf` and resolution must survive a resume: both read only from the
  graph and the library, which are checkpointed. Add nothing new to `save`.
- A resolved parent may itself be a leaf of a higher sketch; the hook is the
  same DONE path, so nesting needs no special case — add one test that a
  two-level chain resolves both levels in order, and stop there.

---

#### Decision Gates

- If HARN-28 is not merged → do not build a second matching mechanism here;
  wait or branch from it.
- If the stripped parent needs the child *bodies* (the sketch used `aux`'s
  definitional unfolding, not just its statement) → that is a failed
  resolution by design; record it, do not special-case.
- If the PR exceeds 1000 lines → split resolution (`_resolve_parent`) into a
  Part 2b and land leaf closing alone.

---

#### Out of Scope

- Generating sketches or statements (HARN-25, HARN-28, HARN-29).
- Choosing *which* open leaf to work first — the planner's priority score
  already favours nodes that unlock more.
- Consolidation or proof-size metrics over resolved parents (a metrics card,
  later).

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-26.sh`.

---

#### Outcome Demo

**Where**: local, mock provider, no Lean needed
**Run**:
```bash
uv run pytest tests/test_society_resolve.py -q -k "last_child_resolves_parent" -s 2>&1 | grep -E "artifacts_resolved|RESOLVED|passed"
```
**Expect**: one `[RESOLVED]` textbook line for the parent and `1 passed`. On
the box, a committee run on `stacks-kernel-track-b` whose summary shows
`artifacts_sketched > 0` in an earlier generation and `artifacts_resolved > 0`
in a later one is the first ratchet this project will have recorded.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-26.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <600 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
