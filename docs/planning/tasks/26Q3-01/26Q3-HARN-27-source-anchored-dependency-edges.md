### 26Q3-HARN-27: Source-anchored dependency edges — the task graph is a DAG, not a chain

**User Story**: As the calibration program, I want a goal's dependency graph to
carry the edges its source text actually states, so that the planning panel has
more than one task to allocate per generation and a population-size experiment
measures parallel groups on distinct statements rather than one group on one
statement.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH — a defect in every committee run to date, and the precondition for Phase C's 9-agent arm meaning anything; first card of the DAG phase (`docs/planning/dag-phase.md`) |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-27-source-anchored-dependency-edges` |
| **Dependencies** | None. Touches `lms/goals.py`, `lms/dependency.py`, the extractor and the two kernel goal files; none of #54–#57 touch those |
| **PR Size Target** | <400 lines of code (the regenerated goal JSON adds `requires` lines on top) |
| **Parts** | single PR |

---

#### Context

> Written 2026-09-05. Evidence below is against `main` @ `85ccf9e` and the
> Stacks Project clone under `references/stacks-project/` (untracked).

`DependencyGraph.from_goal` builds edges by section order: every node requires
**all** earlier nodes in its chapter, and the first node of a chapter requires
the last node of the previous one. That is a total order, so exactly one task
is AVAILABLE at any moment for the whole run. Measured on `main` for all seven
registered goals, `available_tasks()` at start has length 1 — Track B is 1
available / 39 blocked, Track A 1 / 19.

The consequences are already in the logs, unattributed: the chair has never had
two distinct tasks to hand out, so either one group forms or several groups
take the same tag (the `committee_real_b` case `update_status`'s docstring
records); `PlanningPanel._top_up` can never top up because the spare list is
empty by construction; `--agents 9` seats three researchers per group
(HARN-18) but only one statement is ever in play. "N agents converge on one
tag" (`stacks-restart-notes.md`) is the graph.

The Stacks source carries the real edges. Every tagged statement cites what it
is built from with `\ref{label}`, and the extractor already loads the clone's
`tags/tags` label → tag map. Resolving those references over the two kernel
tracks and keeping only edges to earlier entries in the curation order:

| Track | Statement-body refs | Proof refs | Edges kept | Roots at start | Forward refs dropped |
|---|---:|---:|---:|---:|---:|
| B (40) | 13 | 25 | 33 | **18** | `02XN→02XW`, `026D→026E` |
| A (20) | 1 | 4 | 5 | **16** | none |

**Current State**:
- `lms/goals.py` — `StacksDefinition` has no dependency field; `Goal.save/load`
  round-trip six fields plus the three import-policy fields
- `lms/dependency.py` — `from_goal` always calls `_infer_dependencies`; no
  explicit-edge path, no cycle check
- `scripts/goals/extract_stacks_goal.py` — `extract_file` drops `\ref`s with
  the rest of the body text, reads no `\begin{proof}` block, emits no edges
- `goals/stacks_kernel_track_{a,b}.json` — no `requires`

**Investigation**:
```bash
uv run python -c "from pathlib import Path; from lms.goals import Goal; from lms.dependency import DependencyGraph; g = DependencyGraph.from_goal(Goal.load(Path('goals/stacks_kernel_track_b.json'))); print(g.progress_summary())"
# Progress: 0/40 (0%) | Available: 1 | In Progress: 0 | Blocked: 39
grep -c '\\ref{' goals/stacks_kernel_track_b.json
# 19 — the references survive in the statement text and are never used
grep -n "requires" lms/goals.py scripts/goals/extract_stacks_goal.py
# no hits
```

---

#### Scope

- Explicit edges in the goal schema and in `from_goal`; inference stays as the
  fallback for goals that carry none.
- The extractor resolves `\ref`s (statement body and the proof that follows)
  to in-track tags and emits `requires`; both kernel goal files regenerated.
- Not statements, pinning, sketches, or any prompt change.

---

#### Acceptance Criteria

- [ ] `StacksDefinition.requires: list[str] | None = None`. `Goal.save` writes
      `requires` only when it is not None; `Goal.load` reads it with `.get`, so
      every existing `goal.json` still loads.
      `tests/test_goals.py::test_requires_round_trip` passes.
- [ ] `DependencyGraph.from_goal` runs in **explicit mode** when any definition
      has `requires is not None`: `node.requires` is the listed tags that exist
      in the goal (unknown tags dropped with one printed line naming them),
      `unlocks` is the inverse, `_infer_dependencies` is not called, and a
      definition with `requires=[]` is a root. Otherwise behaviour is
      unchanged and `test_from_goal_infers_section_dependencies` /
      `test_from_goal_chapter_dependencies` pass untouched.
      `tests/test_dependency_edges.py::{test_explicit_edges_replace_inference,
      test_unknown_requires_dropped, test_empty_requires_is_a_root}` pass.
- [ ] Explicit mode raises `ValueError` naming the cycle when the edges are not
      acyclic; a goal file with a cycle is a data defect, not a run.
      `tests/test_dependency_edges.py::test_cycle_is_a_goal_defect` passes.
- [ ] `scripts/goals/extract_stacks_goal.py` gains `resolve_refs(text: str,
      stem: str, label_to_tag: dict[str, str]) -> set[str]` (same-file labels
      are prefixed `<stem>-`, cross-file labels are used as written, unknown
      labels dropped); `extract_file` also captures the `\begin{proof}` block
      that follows a statement and records `refs` = union of the two; and
      `build_goal` sets each definition's `requires` to the sorted in-track
      tags that precede it in the curation order, printing any forward
      reference it drops. `tests/test_extract_stacks_goal.py` loads the script
      via `importlib` (as `tests/test_make_novelty_control.py` does) and passes
      on a small TeX fixture covering all four label cases plus one forward
      reference.
- [ ] `goals/stacks_kernel_track_a.json` and `_b.json` are regenerated by the
      script, and `tests/test_goal_files.py::test_kernel_goals_are_dags`
      asserts on the checked-in files: Track B has 18 available tasks at
      start, `02XO` requires exactly `02XK` and `02XL`, `02XN` requires nothing,
      `004B` requires `003V` and `02XN`; Track A has 16 available, `012W`
      requires `012V`, `0132` requires `012W`.
- [ ] `goals/README.md` schema block documents `requires` and the
      forward-reference rule.
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-27.sh` holds greps and pytest
      calls only.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/goals.py` | MODIFY | `requires` field + round trip |
| `lms/dependency.py` | MODIFY | explicit mode in `from_goal`; cycle check |
| `scripts/goals/extract_stacks_goal.py` | MODIFY | `resolve_refs`, proof capture, `requires` emission |
| `goals/stacks_kernel_track_a.json` | REGENERATE | `requires` per definition |
| `goals/stacks_kernel_track_b.json` | REGENERATE | `requires` per definition |
| `goals/README.md` | MODIFY | schema |
| `tests/test_goals.py` | MODIFY | round trip |
| `tests/test_dependency_edges.py` | CREATE | explicit mode, cycle |
| `tests/test_extract_stacks_goal.py` | CREATE | `\ref` resolution on a fixture |
| `tests/test_goal_files.py` | CREATE | the checked-in goal files are DAGs |

---

#### Implementation Notes

- Keep `_infer_dependencies` exactly as it is; it is the fallback for the
  Python-defined goals (`stacks-ch4-*`) and its tests are the spec. Explicit
  mode is a branch before it, not a rewrite.
- The proof block is read for its references only. Nothing from a proof enters
  `content`; the goal files stay statements-only, as `goals/README.md`
  promises.
- The forward-reference rule relies on the curation order being a topological
  sort, which the extractor's TRACKS comment already asserts. Print what is
  dropped so a mis-ordered curation table is visible at regeneration time.
- Track B's forward references: `02XN` (fibred category) cites `02XW`
  (presheaf of categories, §4.36) as a "see also"; `026D` (descent datum)
  cites lemma `026E` stated after it. Both are correctly dropped.
- `DependencyNode.to_prompt_context` and `progress_summary` need no change;
  with real edges they finally say something.
- Do NOT change availability semantics, `update_status`, `revert_done`, or the
  HARN-25 leaf code. Do NOT touch `lms/society.py`.

---

#### Decision Gates

- If the extractor cannot find `references/stacks-project/` → regenerate on
  the machine that has the clone; the tests on the checked-in JSON are what CI
  verifies, not the extraction.
- If explicit mode leaves a kernel track with fewer than 10 roots → the
  measurement above was wrong; stop and re-measure before landing.
- If the user decides this lands before the 2026-09-07 harness freeze → it
  jumps the queue ahead of #54–#57 and is the only card that does.

---

#### Out of Scope

- Edges for the ANT slice — hand-curated in the converted goal file
  (`26Q3-HARN-28`'s converter emits `requires: []`).
- Agent-extended edges (HARN-25 leaves, HARN-29 cards).
- Using proof references as *hints* in the working-group prompt — a later,
  separate prompt card if the DAG phase shows groups ignoring the foundation.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-27.sh`.

---

#### Outcome Demo

**Where**: local, no Lean, no model
**Run**:
```bash
uv run python -c "from pathlib import Path; from lms.goals import Goal; from lms.dependency import DependencyGraph; g = DependencyGraph.from_goal(Goal.load(Path('goals/stacks_kernel_track_b.json'))); print(g.progress_summary()); print(sorted(n.tag for n in g.available_tasks())); print(g.get_node('02XO').requires)"
```
**Expect**: `Available: 18 | In Progress: 0 | Blocked: 22`; a list of 18 tags
that includes `003Y`, `02XK`, `02XN` and excludes `02XO`; and
`['02XK', '02XL']`. On the box, the next committee run's planning-panel
output shows three distinct tags assigned in generation 0 for the first time.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-27.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened within the size target
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
