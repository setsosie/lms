### 26Q3-HARN-28: Theorem cards — a task carries its exact Lean statement, and proofs are pinned to it

**User Story**: As the calibration program, I want every task in the graph to
carry one immutable Lean statement that a working group must prove as written,
so that a verified artifact is a proof *of the task* rather than of whatever
the group chose to state, and so that the ANT slice can be run as the theorem
cards it already is.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | HIGH — the statement layer of the DAG phase (`docs/planning/dag-phase.md`); the "exact signatures per task" lever that took the Sonnet simulation to 9/9 |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-28-theorem-cards-statement-pinning` |
| **Dependencies** | `26Q3-HARN-25` (the `DependencyNode.statement` field and `allow_sorry`); `26Q3-HARN-27` recommended first so the cards sit on a real DAG |
| **PR Size Target** | <600 lines (max 1000) |
| **Parts** | 1/2 — Part 2 is `26Q3-HARN-29` (agent-authored cards for nodes with no curated statement), not yet carded |

---

#### Context

> Written 2026-09-05. Evidence below is against `main` @ `85ccf9e` and the
> `data/ant-arc-statements` branch @ `d455f51`.

Today a working group receives a task's TeX or prose and returns a Lean file
of its own design. The statement it proves is whatever the scribe wrote: the
`Opposite` class of artifact (a vacuous definition, T2 cannot see it), a
`stacks_tag` of the scribe's choosing (`smoke_c/d` wrote `CAT-0013` for task
`0013`), or a correct construction of a different theorem. Verification says
the code compiles; nothing says it is the task.

Prove2Me's data model (arXiv 2608.28433 §3) fixes the statement first: a
theorem card is "a natural-language account of the mathematics behind the
statement", a preamble, and "the target statement itself formalized in Lean 4"
that "is required to terminate in a `:= by sorry` placeholder"; "every
statement and proof-sketch is persistent and cannot be edited once
submitted"; proofs are submitted *for* a statement. The ANT candidate arcs
already have this shape — each entry carries `informal`, `book_ref`, and a
`lean_statement` ending in `:= sorry` — but nothing turns an arc into a `Goal`,
so Phase C has no slice file.

**Current State**:
- `lms/goals.py` — `StacksDefinition` has `tag/section/name/content/formalized/artifact_ids`; no Lean statement
- `lms/dependency.py` — `DependencyNode.statement` exists only for sketch leaves (HARN-25); goal nodes have `statement=None`
- `lms/society.py` — committee Phase 5 verifies `artifact.lean_code` as returned; `update_status(tag, DONE)` on any successful verify that clears the gates, whatever the artifact declares
- `lms/working_group.py` — `_parse_artifact` reads `stacks_tag`, `name`, `lean` from the scribe; no notion of a required declaration
- `data/ant_arcs/{core,ramification}_arc.json` — 20 + 21 theorem cards with no goal loader

**Investigation**:
```bash
grep -n "lean_statement" lms/goals.py lms/dependency.py lms/society.py
# no hits
grep -n "statement" lms/dependency.py
# only the HARN-25 leaf field (on that branch); none on main
grep -c '"lean_statement"' data/ant_arcs/core_arc.json data/ant_arcs/ramification_arc.json
# 20, 21 — every arc entry is already a card
grep -n "def declaration_header" -A 12 lms/foundation.py
# the header-up-to-':=' scan already exists on FoundationEntry; it needs a free function
```

---

#### Scope

- Committee mode only (`run_generation_with_groups`).
- Curated statements: from the goal file. Statements are never *generated*
  here — a node without one behaves exactly as today. Generating them is
  `26Q3-HARN-29`.
- Pinning: the harness makes the verified declaration be the card's statement.
- The ANT arc → goal converter, so the Phase C slice can be a goal with cards.
- Not leaf closing or parent resolution (`26Q3-HARN-26`), not search.

---

#### Acceptance Criteria

- [ ] `StacksDefinition.lean_statement: str | None = None`; `Goal.save` writes
      it only when set, `Goal.load` reads it with `.get`; older `goal.json`
      files still load. `tests/test_goals.py::test_lean_statement_round_trip`
      passes.
- [ ] `DependencyGraph.from_goal` copies each definition's `lean_statement`
      into `DependencyNode.statement`. `DependencyGraph.set_statement(tag,
      statement) -> bool` sets a node's statement once and returns False
      without mutating if the node already has one or is unknown.
      `tests/test_dependency_cards.py::test_statement_is_immutable` passes.
- [ ] `lms/statement.py` exists with `statement_header(statement: str) -> str`
      (the declaration up to its first top-level `:=`, comments stripped,
      whitespace collapsed; built on `lms/gates/lean_source.py`), and
      `pin_statement(code: str, statement: str) -> tuple[str, str | None]`:
      finds the declaration in `code` whose name equals the card's, replaces
      its header with the card's header, and returns the new code; returns
      `(code, error)` with a message naming the missing declaration when no
      declaration of that name exists. `tests/test_statement.py` passes for:
      matching header (no-op), drifted binders (spliced), missing name
      (error), `:=` inside a binder default (header scan ignores it).
- [ ] In committee Phase 5, when the task node has a statement, the candidate
      is pinned before the import check and before every verify, including
      each repair attempt; a pinning error enters the repair loop as the Lean
      error would. `tests/test_society_cards.py::test_pinned_statement_is_what_lean_sees`
      passes: the verifier's recorded input contains the card's header
      verbatim and not the scribe's drifted one.
- [ ] A task with a statement is marked DONE only if the verified artifact
      contains the pinned declaration; `Artifact.pinned_to: str | None` records
      the task tag. `tests/test_society_cards.py::test_done_requires_pinned_declaration`
      passes.
- [ ] `Society._get_task_content(tag)` for a node with a statement renders the
      statement in a ```lean block above the task content, with the sentence
      "Prove exactly this declaration; do not change its name, binders or
      type." `tests/test_society_cards.py::test_task_content_shows_card` passes.
- [ ] `to_prompt_context` prints an AVAILABLE goal node's statement beneath it
      (the HARN-25 rendering, now for goal nodes too).
- [ ] `scripts/goals/arc_to_goal.py <arc.json> <out.json>` converts an ANT arc
      into a goal file: `id → tag`, `book_ref → section` (the `§n` number, else
      `0`), `name → name`, `informal → content` (with `book_ref` appended as
      the source line), `lean_statement → lean_statement`, `requires: []`
      (curated by hand afterwards; the file says so in its `description`).
      Pure conversion lives in `lms/goals.py::goal_from_arc(data: dict) ->
      Goal`; `tests/test_goals.py::test_goal_from_arc` passes on a two-entry
      fixture.
- [ ] Scribe prompt in `lms/working_group.py` states that when the task shows a
      declaration, the artifact must declare it with that exact name and
      signature and may add auxiliary declarations before it.
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-28.sh` holds greps and pytest
      calls only.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/statement.py` | CREATE | `statement_header`, `pin_statement` |
| `lms/goals.py` | MODIFY | `lean_statement` field + round trip; `goal_from_arc` |
| `lms/dependency.py` | MODIFY | copy statements in `from_goal`; `set_statement` |
| `lms/artifacts.py` | MODIFY | `pinned_to` field + serialization |
| `lms/society.py` | MODIFY | pin before verify/repair; DONE gated on the pinned declaration; task content shows the card |
| `lms/working_group.py` | MODIFY | scribe prompt text |
| `scripts/goals/arc_to_goal.py` | CREATE | CLI wrapper over `goal_from_arc` |
| `tests/test_statement.py` | CREATE | header + pinning |
| `tests/test_dependency_cards.py` | CREATE | statements on goal nodes, immutability |
| `tests/test_society_cards.py` | CREATE | committee path |
| `tests/test_goals.py` | MODIFY | round trip, arc conversion |

---

#### Implementation Notes

- One Lean scanner: `statement_header` reuses `extract_declarations` and
  `strip_comments` from `lms/gates/lean_source.py`, and the top-level `:=`
  scan should be lifted from `FoundationEntry._scan_for_assign` into
  `lean_source.py` rather than copied. `declaration_header` on
  `FoundationEntry` then calls the shared function.
- Pinning replaces the header **only**; the agent's body, and any auxiliary
  declarations, are untouched. If the agent's binders differed from the card,
  Lean reports the mismatch against the true statement and the scribe repairs
  against it — that is the point. Do not attempt a textual "does it match"
  verdict; Lean is the judge.
- A leaf statement (HARN-25, `<parent>/<child>`) is a full declaration and
  pins the same way. Nothing in this card is leaf-specific, and HARN-26 relies
  on that.
- Immutability is graph-level: `set_statement` mirrors `update_status`'s
  refusal to demote DONE. Statements already round-trip through
  `DependencyGraph.save/load` (HARN-25); a resumed run keeps them.
- The card shows the statement *with* its `:= sorry` tail so the group sees
  the whole declaration; the pinned header is everything before that `:=`.
- Do NOT touch `_run_generation_iterative`, `lms/prompts.py`, the gates, or
  the foundation file format.

---

#### Decision Gates

- If HARN-25 is not merged when this starts → branch from it, do not
  re-implement `statement`/`allow_sorry`.
- If `pin_statement` needs more than the declaration keyword, name and a
  bracket-aware `:=` scan to find the header (e.g. `where`-structures) →
  restrict pinning to `theorem`/`lemma`/`def` and record the restriction in
  the docstring; do not grow a parser.
- If the PR exceeds 1000 lines → the arc converter becomes its own 1-pt card.

---

#### Out of Scope

- `26Q3-HARN-29` owns generating a statement for a node that has none.
- `26Q3-HARN-26` owns closing leaves and re-verifying parents.
- Curating `requires` edges for the ANT slice — a D4-reviewer task on the
  winning arc, done by hand in the converted goal file.
- Faithfulness of the curated statements (D4 review, Phase D).

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-28.sh`.

---

#### Outcome Demo

**Where**: local, mock provider, no Lean needed
**Run**:
```bash
uv run python scripts/goals/arc_to_goal.py data/ant_arcs/core_arc.json goals/ant_core_arc.json && uv run python -c "from pathlib import Path; from lms.goals import Goal; from lms.dependency import DependencyGraph; g = DependencyGraph.from_goal(Goal.load(Path('goals/ant_core_arc.json'))); n = g.get_node('core-01'); print(n.statement[:60]); print(g.set_statement('core-01', 'theorem other : True := sorry'))"
```
**Expect**: the first line is the arc's `theorem ant_c01_integral_add …`
declaration; the second is `False` — the card cannot be overwritten.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-28.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <600 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
