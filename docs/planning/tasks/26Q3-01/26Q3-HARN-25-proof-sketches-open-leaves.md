### 26Q3-HARN-25: Proof sketches — an artifact may leave named child lemmas open

**User Story**: As the calibration program, I want a working group to be able to
submit a proof that is complete except for named, `sorry`'d child lemmas, so
that a correct reduction by a weak model becomes open leaves in the task graph
instead of a failed artifact.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | HIGH — the harness mechanism that turned the FLT swarm's failed Claude-Code loop into a success, and the "capable but doesn't land" fix from the Sonnet lessons |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-25-proof-sketches-open-leaves` |
| **Dependencies** | None — rebases over #54–#57 when they merge (they touch `society.py`'s committee path) |
| **PR Size Target** | <600 lines (max 1000) |
| **Parts** | 1/2 — Part 2 is `26Q3-HARN-26` (`26Q3-HARN-26-close-leaves-auto-resolve.md`, carded 2026-09-05); the wider design is `docs/planning/dag-phase.md` |

---

#### Context

> Written 2026-09-04 after Anthropic's FLT formalization post and the Prove2Me
> paper (arXiv 2608.28433). Evidence below is against `main` @ `85ccf9e`.

Today an artifact is pass/fail as a whole. `RealLeanVerifier.verify` rejects
any code containing the substring `sorry` before Lean runs, the mock and MCP
verifiers do the same, and T4 fails on `\bsorry\b`. The dependency graph has
one node per goal definition (`DependencyGraph.from_goal`), and
`Society._get_task_content` raises on any tag not in the goal. A group that
correctly reduces its task to three lemmas it cannot prove therefore produces a
FAILED artifact, and nothing of the reduction survives.

This is the failure class the Sonnet simulation named
(`docs/LESSONS_FROM_SONNET_SIMULATION.md` §4c): both Yoneda attempts had the
correct construction and died on trailing mechanical steps. It is also, per
the Prove2Me paper §3, the mechanism that let the FLT swarm make progress: "a
proof that imports other theorems [is] a proof-sketch: it establishes the
target conditional on the imported statements, deferring their proofs to
separate submissions," and each open child becomes a new open problem others
can claim.

**Current State**:
- `lms/lean/real.py` — substring pre-check `if "sorry" in code` returns a
  failure before Lean runs; `lms/lean/mock.py`, `lms/lean/mcp.py` mirror it
- `lms/dependency.py` — `DependencyNode` carries tag/name/chapter/section only;
  no statement text, no parent link
- `lms/society.py` — `_get_task_content` raises `ValueError` for any tag not in
  `goal.definitions`; committee Phase 5 calls `self.verifier.verify(lean_code)`
  with no way to permit a conditional proof
- `lms/working_group.py` — researcher and scribe prompts say "Do NOT use
  `sorry`" unconditionally

**Investigation**:
```bash
grep -n '"sorry" in code' lms/lean/real.py lms/lean/mock.py lms/lean/mcp.py
# three hits — every verifier rejects on the substring before compiling
grep -n "statement\|parent_tag" lms/dependency.py
# no hits
grep -n "def _get_task_content" -A 16 lms/society.py
# raises for unknown tags (deliberately, 2026-08-19 smoke)
grep -c "Do NOT use \`sorry\`" lms/working_group.py
# 2
```

---

#### Scope

- Committee mode only (`run_generation_with_groups`) — the path run on the box.
- Recognising a sketch, verifying it conditionally, recording it as `SKETCH`,
  and turning its children into open leaves with exact statements that the
  planner can assign.
- Not closing a leaf, matching a child proof to its statement, or re-verifying
  the parent — that is Part 2.

---

#### Acceptance Criteria

- [ ] `lms/sketch.py` exists with `split_sketch(code: str) -> Sketch`; `Sketch`
      has `children: list[OpenChild]`, `is_sketch: bool`, `reason: str | None`;
      `OpenChild` has `name`, `keyword`, `statement` (the declaration's source
      including its `sorry` body).
- [ ] A declaration is an open child iff it is a `theorem`/`lemma` whose entire
      body is `sorry` (`:= sorry`, `:= by sorry`, multi-line `by` then `sorry`).
      `tests/test_sketch.py::test_open_child_forms` passes.
- [ ] A `sorry` anywhere else — inside a longer proof, in a `def`, in a
      `structure` field — makes `is_sketch` False with a reason.
      `tests/test_sketch.py::test_sorry_outside_child_is_not_a_sketch` passes.
- [ ] A sketch has at least one non-child declaration, and every child is
      referenced by name outside its own declaration; otherwise `is_sketch` is
      False. `tests/test_sketch.py::test_children_must_be_used` and
      `::test_statement_list_is_not_a_sketch` pass.
- [ ] `VerificationStatus.SKETCH` exists; `Artifact.verified` is False for it;
      `Artifact.open_children: list[str]` round-trips through
      `to_dict`/`from_dict`.
      `tests/test_sketch.py::test_sketch_status_never_counts_as_verified` passes.
- [ ] `LeanVerifier.verify(code, *, allow_sorry=False)` on all three verifiers;
      with `allow_sorry=True` the substring pre-check is skipped and Lean's own
      verdict is returned. `tests/test_lean.py::TestSketchVerification` passes
      (mock); the real verifier compiles a sketch with warnings only, exit 0 —
      accepted via Outcome Demo.
- [ ] `DependencyNode` gains `statement: str | None`, `parent_tag: str | None`,
      `sketch_artifact_id: str | None`, serialized and loaded.
      `DependencyGraph.add_open_leaves(parent_tag, sketch_artifact_id, children)
      -> list[str]` creates one AVAILABLE leaf per child with tag
      `<parent_tag>/<child_name>`, `requires=[]`, `unlocks=[parent_tag]`,
      appends the leaf tags to the parent's `requires`, sets the parent BLOCKED,
      and returns `[]` without mutating if the parent is DONE or unknown.
      `tests/test_dependency_leaves.py` passes.
- [ ] `to_prompt_context` prints an AVAILABLE leaf's exact statement beneath
      it. `tests/test_dependency_leaves.py::test_prompt_context_shows_leaf_statement`
      passes.
- [ ] `Society._get_task_content(leaf_tag)` returns the leaf's statement
      followed by the parent task's content; it still raises for unknown tags.
      `tests/test_society_sketch.py::test_task_content_for_leaf` passes.
- [ ] In committee Phase 5, a candidate with `is_sketch` is verified with
      `allow_sorry=True`; on success its status is `SKETCH`, it enters the
      library but not the foundation, `artifacts_verified` does not count it,
      `GenerationResult.artifacts_sketched` does, the parent tag is BLOCKED on
      the new leaves, and the textbook gets a `[SKETCH]` entry.
      `tests/test_society_sketch.py::test_sketch_registers_leaves_and_does_not_verify`
      passes.
- [ ] A sketch that fails to compile enters the existing repair loop exactly as
      a plain failure does. `tests/test_society_sketch.py::test_failed_sketch_repairs`
      passes.
- [ ] Researcher and scribe prompts in `lms/working_group.py` describe the
      sketch form (`sorry` only as the entire body of a named `theorem`/`lemma`
      that the main proof uses) and no longer forbid `sorry` unconditionally.
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-25.sh` holds greps and pytest
      calls only.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/sketch.py` | CREATE | `split_sketch`, `Sketch`, `OpenChild` |
| `lms/lean/interface.py` | MODIFY | `VerificationStatus.SKETCH`; `allow_sorry` kwarg on `verify` |
| `lms/lean/real.py` | MODIFY | skip the substring pre-check when `allow_sorry` |
| `lms/lean/mock.py` | MODIFY | same |
| `lms/lean/mcp.py` | MODIFY | same |
| `lms/artifacts.py` | MODIFY | `open_children` field + serialization |
| `lms/dependency.py` | MODIFY | leaf fields, `add_open_leaves`, context rendering |
| `lms/society.py` | MODIFY | committee Phase 5 sketch path; `_get_task_content` for leaves; `GenerationResult.artifacts_sketched` |
| `lms/working_group.py` | MODIFY | prompt text for the sketch form |
| `tests/test_sketch.py` | CREATE | splitter + status |
| `tests/test_dependency_leaves.py` | CREATE | graph leaves |
| `tests/test_society_sketch.py` | CREATE | committee path |
| `tests/test_lean.py` | MODIFY | `allow_sorry` on the mock |

---

#### Implementation Notes

- Build the splitter on `lms/gates/lean_source.py` (`extract_declarations`,
  `strip_comments`); do not add a second Lean scanner. A child's body is the
  text from its `:=` to the next declaration start; "entire body is sorry"
  means that text, stripped of comments and whitespace, is `sorry` or
  `by sorry`.
- Mirror `update_status`/`revert_done` for `add_open_leaves`: mutate, then
  `_recalculate_availability()`. Leaf tags use `/` so they cannot collide with
  Stacks tags, and the planner's tag validation still finds them in the graph.
- Keep `VERIFIED_LEAN` semantics untouched — a sketch is not verified.
  `_apply_gates` must not run on `SKETCH` (T4.sorry would fail it for the
  wrong reason).
- `allow_sorry` skips only the pre-check; Lean is the judge of whether the
  conditional proof compiles. Do not filter or suppress warnings.
- Do NOT touch the iterative path (`_run_generation_iterative`) or
  `lms/prompts.py`.

---

#### Decision Gates

- If #54–#57 merge mid-implementation and the committee Phase 5 region has
  moved → rebase; do not re-implement their changes here.
- If the real verifier turns out to treat `sorry` warnings as failure (a
  warning-as-error flag in `STRICTNESS_FLAGS`) → stop and surface; do not
  weaken the flags unilaterally.
- If the PR exceeds 1000 lines → split the society hook into Part 1b.
- Do NOT implement leaf closing or parent auto-resolution here, even though the
  hook is obvious.

---

#### Out of Scope

- `26Q3-HARN-28` owns pinning a proof to its task's statement (leaves
  included); `26Q3-HARN-26` owns marking the leaf DONE on a pinned child
  proof, re-verifying the parent with children stripped, and promoting it to
  `VERIFIED_LEAN` and the foundation.
- `26Q3-HARN-12` owns committee reachability and the review committee.
- `26Q3-HARN-05` owns cost accounting; a sketch's spend is recorded under its
  parent's statement key as today.
- No change to the novelty gate, T2/T4, or the foundation file format.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-25.sh`.

---

#### Outcome Demo

**Where**: local, with the Lean toolchain installed and `lean/` built
**Run**:
```bash
uv run python -c "import asyncio; from lms.lean.real import RealLeanVerifier; v = RealLeanVerifier(project_dir='lean'); code = 'import Mathlib\ntheorem aux (n : Nat) : n + 0 = n := by sorry\ntheorem main (n : Nat) : n + 0 + 0 = n := by rw [aux, aux]'; print(asyncio.run(v.verify(code, allow_sorry=True)).success, asyncio.run(v.verify(code)).success)"
```
**Expect**: `True False` — Lean accepts the conditional proof with a
`declaration uses 'sorry'` warning; the same code without `allow_sorry` is
rejected by the pre-check as before.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-25.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <600 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
