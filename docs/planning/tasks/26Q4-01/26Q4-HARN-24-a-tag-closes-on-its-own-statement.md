### 26Q4-HARN-24: A tag closes only on an artifact that is that tag's statement

**User Story**: As the person reading goal progress and the dependency graph, I want a verified artifact to close its group's tag only when it formalizes that tag's statement, so that progress counts real work and downstream tasks open only when what they build on exists.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING: the matching rule needs the user's decision |
| **Branch** | `26Q4-HARN-24-a-tag-closes-on-its-own-statement` |
| **Dependencies** | The rule (below). Option C needs `26Q3-HARN-28` (theorem cards and statement pinning) |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `317df86`):
- In committee mode, a verified artifact that passes the gates is promoted, and then **the tag the planning panel assigned to its group is marked DONE**. That happens both in the dependency graph (`update_status(group.config.task_tag, DONE, …)`) and in the goal (`mark_formalized(artifact.stacks_tag, …)`, where `stacks_tag` is that same assigned tag). Nothing checks that the artifact is that tag's statement.
- Batch q1, round r6 (`stacks-ch4-phase1`, 10 generations), closed two tags, and both were closed by the wrong statement. Tag 0014 (*Functor*) was closed by `CategoryDef`, a renamed copy of the seeded `Category`. Tag 0015 (*Natural transformation*) was closed by a `Functor`. The progress line read 33% when no tag had been earned.
- The early close of 0014 released 0015. Generations 4–8 then attempted natural transformations with no functor in the foundation, and `Functor C` resolved to Lean core's monad class.
- The duplicate gate saw `CategoryDef` but only as INCONCLUSIVE (routed to D4), so it did not block promotion.

**Investigation**:
```bash
grep -n "update_status(\s*group.config.task_tag" -A3 lms/society.py
grep -n "stacks_tag=group.config.task_tag" lms/society.py
```

---

#### The decision this card needs

What makes an artifact "that tag's statement"? Three candidate rules, cheapest first:

- **A. Name rule.** The artifact declares a top-level name matching the tag's concept (for example `Functor` for "Definition: Functor"), and nothing that redeclares a seed or foundation name. Cheap and deterministic, and it would have caught both r6 closures. It is brittle for theorems, whose names are free-form.
- **B. Review rule.** A reviewer call (the review committee, or the chair) answers "does this formalize tag X's statement?" before the tag closes. General, but it is a model's judgement, so it can be gamed, and every closure costs tokens.
- **C. Pinned statement.** Each tag carries a pinned Lean statement (`26Q3-HARN-28`), and an artifact closes the tag only if it proves or defines exactly that statement, checked by Lean. Robust, but it waits on statement pinning, which is Phase C's critical path anyway.

A reasonable path is A now (it covers definition tags, which is what the kernel tracks mostly are) and C when HARN-28 lands. Whatever the rule, a verified artifact that fails it should still be promoted to the foundation (it compiled, and it may be useful); it just must not close the tag.

---

#### Acceptance Criteria

To be written once the rule is chosen. At minimum:
- [ ] A verified artifact that is not the assigned tag's statement leaves the tag open in the graph and in the goal, and the run log says why
- [ ] r6's two cases, as fixtures (`CategoryDef` on 0014, `Functor` on 0015), close nothing
- [ ] The collect read-back reports closures refused under the rule

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/society.py` | MODIFY | Check before `update_status`/`mark_formalized` |
| `tests/test_tag_closure.py` | CREATE | r6's two cases and the chosen rule |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-24-a-tag-closes-on-its-own-statement.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-24.sh` | CREATE | Verify script |

---

#### Notes

- Iterative (non-committee) mode reads `stacks_tag` from the agent's own artifact header, so the same question applies there with a different source of the claim. Check it under the same rule.
- Results: `docs/results/2026-11-13-batch-q1.md`, section r6.
