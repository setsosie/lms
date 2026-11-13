### 26Q4-HARN-17: A name collision can't discard a generation's verified work

**User Story**: As the ratchet measurement, I want one artifact's conflict to cost only that artifact, and no entry to land bound to a declaration it wasn't verified against, so that a generation's verified work survives and what reaches the foundation is what Lean checked.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-17-name-collision-discards-generation-work` |
| **Dependencies** | None |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**:
- `add_artifact` skips any entry whose name exists, even when the two declarations differ, and keeps the artifact's other entries. Those then bind to the other agent's declaration, which isn't what they were verified against.
- A failed merged build rolls back the whole generation. The comment on `_write_and_build_foundation` defers per-artifact handling until partial generations prove common. In the team simulation, 2 of 3 generations hit it.
- `test_deduplicates_definitions_by_name` approves the silent skip: an artifact redefines `Category` with more fields, and its `Functor` is kept over the first one. That test changes here, deliberately.

**Investigation**:
```bash
grep -n "Skip if name already exists" lms/foundation.py
grep -n "deliberately blunt" lms/society.py
grep -n "foundation.add_artifact" lms/society.py   # three promotion paths
```

---

#### Acceptance Criteria

- [ ] An artifact declaring a name that already exists contributes nothing, and returns False: not that entry, and not its other entries. This holds against the seed, an earlier artifact, and a name repeated inside the artifact, which happens when the merge strips `namespace`
- [ ] There's no exception for an identical declaration: it's refused too
- [ ] A declaration is read as Lean reads it: an attribute, modifier or `(priority := ...)` can't hide a name, a whole name is one name (`f'`, `f₁`, `rep_α`, `φ`), and `class inductive P` is named `P`
- [ ] A refusal says why: the reason names the declaration, is on the artifact's notes, and is in the run log, on all three promotion paths
- [ ] A refusal keeps no state, so the same artifact can land once the name is gone, including after a rollback on resume
- [ ] A failed merged build retries the generation's artifacts one at a time, in order, and keeps each that builds
- [ ] After a partial rollback, the last-good state includes what was kept, the file on disk is what was kept, and the log names what was dropped and what was kept
- [ ] A generation with one artifact takes no extra build, because the failed build already named it
- [ ] `universe u in def f ...` on one line keeps `f`, and a comment on a `universe` line isn't read as code

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | Reject on a differing name; list and reinstate a generation's artifacts |
| `lms/society.py` | MODIFY | One promotion helper; per-artifact rollback |
| `tests/test_foundation_collisions.py` | CREATE | Rejection and identical duplicates |
| `tests/test_foundation_rollback.py` | MODIFY | Per-artifact rollback |
| `tests/test_foundation.py` | MODIFY | The test that approved the silent skip |
| `tests/test_lean_real.py` | MODIFY | What the merge writes compiles under real Lean |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-17-name-collision-discards-generation-work.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-17.sh` | CREATE | Verify script |

---

#### Implementation Notes

- **Refuse on the name alone, with no comparison of bodies.** Three designs that compared text were each reviewed and each let a statement land meaning something else: the commands after a copy, the universes it uses, and what's in scope all change what the same text means. An artifact that imports the foundation can't redeclare a name, because the verifier rejects it, so a copy only comes from agents in one generation, and the next generation can import it. Of the 9 verified artifacts in the one real run kept locally, none was an identical copy and 2 had a real collision. That's a small sample, from another model.
- Both mechanisms, because they catch different things. Refusal stops an entry binding to the wrong declaration even when the merged file still compiles, which no build can catch. Per-artifact rollback covers merged-build failures with any other cause.
- First-come wins, matching the class's existing rule. A refused artifact stays in the library as verified: Lean did accept it.
- Retries cost one build per artifact, only after a failed merged build.
- This can't catch every change of meaning. An artifact using core `id` gets a different `id` if an earlier artifact declares one, with no name of its own colliding. That and related cases are in `26Q4-HARN-19`.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- Reopening goal tags for rejected or rolled-back artifacts: `26Q4-HARN-18`. Until it lands, those tags stay DONE as they do today.
- Universe lines: `26Q4-HARN-16`.
- Retrying an artifact that was refused against one later rolled back. The tag reopens under `26Q4-HARN-18`, so a later generation can redo it.
- Text before an artifact's first declaration, which the merge discards: `26Q4-HARN-19`.
- An artifact closing more than one tag.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-17.sh`.

---

#### Outcome Demo

**Where**: laptop
**Run**:
```bash
uv run pytest tests/test_foundation_collisions.py tests/test_foundation_rollback.py -v
```
**Expect**: all pass. For the real case, the 2026-11-06 simulation's generations 0–2 build under Lean with `26Q4-HARN-16` also applied, and generation 2 keeps agent 0's proof in the foundation. Those artifacts are local and gitignored, so the PR records that replay.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-17.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

Evidence from the 2026-11-06 simulation at `3185b0e`, written by Claude for the author of the sections above. The run is local and gitignored: `experiments/sim_sonnet_team_2026-11-06/`.

- **Defect 1, silent skip:** `add_artifact` skips any entry whose name already exists (`lms/foundation.py:862`), even when the two declarations differ. Later entries from the skipped artifact then bind to the other agent's declaration.
- **Defect 2, blunt rollback:** a failed merged build rolls back the whole generation (`lms/society.py:399`). The comment at `lms/society.py:376` defers per-artifact rollback until partial generations prove common. Here 2 of 3 generations hit it, though only this card's cause is a collision; the other is `26Q4-HARN-16`.
- **Observed:** all three generation 2 agents proved Yoneda, and all 26 entries were rolled back. Agents 0 and 1 both defined `rep`, returning different types. The merge strips `namespace` (`lms/foundation.py:816`), and all three had chosen `namespace MyYoneda` anyway, so keeping namespaces wouldn't separate them.
- **Misleading error:** the build reports `Unknown identifier C`. Agent 1 merges fine alone. Its `variable` line disappears only in the collision, apparently because it travels with the skipped `rep` entry. With the line restored, agents 0 and 1 still fail on a type mismatch (`diag4.py`).
- **Confirmed:** agents 0 and 2 merge and build together, so per-artifact handling would have kept 2 of the 3 proofs (`diag4.py`).
- **Open choice for the criteria, drafted as both:** reject a colliding artifact at promotion, and roll back per artifact after a failed build. An identical re-declaration stays a harmless duplicate.
- **Replay under these criteria:** generation 2 keeps 1 proof of 3, not the 2 above. Agent 2's `yonedaFullyFaithful` differs from agent 0's, so it's refused. It built before only because the old skip dropped that one declaration and kept agent 2's other nine. All three proofs close the same tag and stay verified in the library, so the foundation loses a redundant copy.
- **Overlap:** shares `add_artifact` with `26Q4-HARN-16`, so merge that first.
