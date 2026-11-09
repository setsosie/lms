### 26Q4-HARN-18: Rollback reopens the goal tags it undid

**User Story**: As the ratchet measurement, I want a goal tag to read DONE only while the foundation holds work behind it, so that agents aren't told to build on declarations that don't exist and `goal.json` can't overstate progress.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-18-rollback-reopens-goal-tags` |
| **Dependencies** | 26Q4-HARN-17 |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**:
- All three promotion paths call `goal.mark_formalized` as they process results. The foundation build runs afterwards, and a failed build restores the foundation without touching the goal.
- The next generation's prompt then says those items exist in `LMS.Foundation`, and its system prompt says not to redefine `[DONE]` items.
- On a goal with real edges, the graph releases dependents of a tag with nothing behind it.
- The standard and iterative paths also mark a tag when a gate blocked the artifact. The committee path doesn't.
- `DependencyGraph.revert_done` exists for this and has no caller.
- A gate-blocked committee tag stays in progress for good: never offered again, with its dependents locked. On a goal with one available task, that ends the run's progress.

**Investigation**:
```bash
grep -n "mark_formalized" lms/society.py
grep -n "revert_done" lms/*.py
grep -n "foundation-yoneda-complete" lms/goals.py   # tags pre-marked with no artifact in any foundation
```

---

#### Acceptance Criteria

- [ ] After a generation, a tag backed only by artifacts the foundation doesn't hold reads `[TODO]`, whether they were rolled back, refused or gate-blocked
- [ ] A tag with one surviving artifact stays DONE, and the withdrawn id leaves its `artifact_ids`
- [ ] Seeded tags, and tags pre-marked with an artifact this run never saw, are untouched
- [ ] A refused artifact doesn't back its tag, even if it restated work the foundation holds under another tag, and it does once it lands under the same id
- [ ] A gate-blocked committee tag goes back to available, unless another group finished it
- [ ] The graph reopens the tag through `revert_done`, and its dependents lock again
- [ ] A resume that rolls back to the generation boundary reopens the same tags
- [ ] The log names every reopened tag
- [ ] An artifact refused for claiming a core concept under another name says why, as other refusals do

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/goals.py` | MODIFY | Withdraw artifacts from a goal; report the tags left with none |
| `lms/foundation.py` | MODIFY | Say whether the foundation holds an artifact; record the core-concept refusal |
| `lms/society.py` | MODIFY | Reconcile the goal and graph with the foundation |
| `tests/test_goal_reopens_tags.py` | CREATE | The criteria above |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-18-rollback-reopens-goal-tags.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-18.sh` | CREATE | Verify script |

---

#### Implementation Notes

- Reconcile once, after the build, instead of guarding each `mark_formalized`. The build is what decides, and it runs after all three paths have marked.
- An artifact is withdrawn only if the run's library knows it and the foundation doesn't hold it. "Not in the foundation" alone would reopen seeded tags and the `stacks-full-path` pre-marks.
- A mock committee run no longer advances past its first layer of tags: mock artifacts can't enter the foundation, so their tags reopen. The flat paths already never marked them.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- The library and textbook still show a rolled-back artifact as verified. That's true of Lean's verdict, and agents' citations of it are a separate question, so this card leaves it for a later one if it recurs.
- An artifact closing more than one tag.
- Per-artifact rollback and refusals themselves: `26Q4-HARN-17`.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-18.sh`.

---

#### Outcome Demo

**Where**: laptop
**Run**:
```bash
uv run pytest tests/test_goal_reopens_tags.py -v
```
**Expect**: all pass. On a real run, a generation whose artifact is refused or rolled back logs `[goal] reopened`, and the next generation's prompt shows that tag as `[TODO]`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-18.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

Evidence from the 2026-11-06 simulation at `3185b0e`, written by Claude for the author of the sections above. The run is local and gitignored: `experiments/sim_sonnet_team_2026-11-06/`.

- **Defect:** the iterative path calls `goal.mark_formalized` as each generation's results are processed (`lms/society.py:1061`; the standard path does the same at `:728`). `persist_foundation` runs after that, and a failed build restores the foundation without touching the goal (`lms/society.py:399`). The committee path has not been checked.
- **Observed:** the final `goal.json` marks 001D and 0019 DONE, but the foundation holds no product or Yoneda declaration. The next generation's prompt told agents those items already exist in `LMS.Foundation`, and its system prompt tells them not to redefine `[DONE]` items.
- **Related, in the same output:** the library and textbook still present rolled-back artifacts as verified successes, so later agents cited them. Whether that belongs here is a choice for the criteria.
- **Dependency:** set on `26Q4-HARN-17` because per-artifact rollback changes which tags need reopening.
- **Neighbor, not in this card:** an artifact can close only one tag, which is why 0015 and 001P stayed `[TODO]` after being proved.
