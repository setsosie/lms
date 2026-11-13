### 26Q4-HARN-24: A tag closes only on an artifact that is that tag's statement

**User Story**: As the person reading goal progress and the dependency graph, I want a verified artifact to close its group's tag only when it formalizes that tag's statement, so that progress counts real work and downstream tasks open only when what they build on exists.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING: rule A chosen 2026-11-13 |
| **Branch** | `26Q4-HARN-24-a-tag-closes-on-its-own-statement` |
| **Dependencies** | None for rule A. A later move to option C needs `26Q3-HARN-28` (theorem cards and statement pinning) |
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

#### Decision (2026-11-13): rule A now, refined over time

Three rules were considered, cheapest first:
- **A. Name rule:** the artifact's declared names must match the tag's name. Cheap and deterministic. It catches both r6 closures, but it is fuzzy on theorem tags.
- **B. Review rule:** a reviewer call decides. General, but it is a model's judgement and costs tokens.
- **C. Pinned statement:** Lean checks the artifact against a pinned statement for the tag. Robust, but it waits on `26Q3-HARN-28`.

**Rule A is the one to build.** It moves toward C as statement pinning arrives.

**Rule A, as specified here.** Tag names come in two shapes:
- concept names, for example `Definition: Functor` in `stacks-ch4-phase1`;
- Stacks labels, for example `Definition: definition categories over C (Categories over categories)` and `Lemma: lemma composition cartesian (Fibred categories)` in the kernel tracks, where 32 of 60 tags are lemmas.

So the match is on words, not on whole names:
- **Tag words:** drop a leading `MILESTONE:` and the kind word (`Definition:`, `Lemma:`, …), drop a trailing parenthetical (the section title), lowercase, split on anything that is not a letter or digit, and drop kind words, articles and short prepositions (`definition`, `lemma`, `of`, `the`, `a`, `an`, `and`, `for`, `to`, `in`, `is`, `with`, `by`) and single letters.
- **Artifact words:** the names after `def`, `structure`, `class`, `abbrev`, `inductive`, `instance`, `theorem` and `lemma` in the artifact's Lean code, split on dots, underscores and camelCase, then lowercased.
- **A match** is a pair where one word is a prefix of the other and the shorter has at least 3 letters (`nat`/`natural`, `cat`/`categories`, `comp`/`composition`), or where the two share a prefix of at least 5 letters.
- **The tag closes only if at least one pair matches.** Otherwise the artifact is still promoted (it compiled and may be useful), the tag goes back to available, and the run log prints `[goal] <tag> not closed by <artifact>: declares <names>, none matches "<tag name>"`.

On r6's two cases: `CategoryDef` (category, def) against "Functor" has no pair, and `Functor` against "Natural Transformation" has no pair, so both are refused. A `NatTrans` against "Natural Transformation" matches.

Known limits, to work up over time: abbreviations shorter than 3 letters (`op` for opposite), synonyms (`Hom` for morphism), and theorem names that share no word with the Stacks label. Each such case can be added as an alias, and option C ends the guessing.

---

#### Acceptance Criteria

- [ ] In committee mode, a verified, promoted artifact closes its group's tag in both the dependency graph and the goal only if rule A matches; otherwise the tag returns to available and the run log prints the `[goal] … not closed by …` line
- [ ] The same rule applies where flat and iterative modes close a tag from the artifact's own `stacks_tag`
- [ ] Fixtures from r6: `CategoryDef` on 0014 and `Functor` on 0015 close nothing; a `Functor` on 0014 and a `NatTrans` on 0015 close their tags
- [ ] Kernel-track fixtures: a `CatOverC` structure on "definition categories over C" closes it; a lemma `comp_cartesian` on "lemma composition cartesian" closes it
- [ ] A refused closure still promotes the artifact to the foundation
- [ ] `tests/test_tag_closure.py` covers each item above

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
