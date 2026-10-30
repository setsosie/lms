### 26Q4-HARN-06: Coordinator–worker mode — one persistent coordinator, fresh workers (the W9 arm)

**User Story**: As the organization-sweep experimenter, I want a run mode in which one long-lived coordinator assigns a node to each fresh worker, with exact signatures in each task, so that the structure that ran best in the Sonnet-team simulation is measured against the committee on the same model, slice and budget.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-06-coordinator-worker-mode` |
| **Dependencies** | `26Q3-HARN-27` (real edges); `26Q4-HARN-03` (reads its runs). Stage 1 of `docs/planning/org-structures.md`, after Phase C |
| **PR Size Target** | <500 lines (max 900) |
| **Parts** | single PR |

---

#### Context

The Sonnet-team simulation (2026-08-20/21) used "one long-lived coordinator
agent allocates three tasks per generation; three fresh researcher agents each
get one shot with no repair loop; no planning panel; no peer review", and
reached 9/9 tags in five generations against Qwen's committee at 2/9 in ten.
The model and the harness were confounded
(`docs/LESSONS_FROM_SONNET_SIMULATION.md`). Two of its lessons are
organizational:
- "Per-task specificity beats a global preamble" (§6): the coordinator pasted exact field lists into every task.
- The coordinator was dependency-aware, including about sibling invisibility (§8).

The book draft's chapter 12 cites a result that "models below a capability
threshold exhibit a reversal where rigid structure outperforms autonomy". That
makes this the arm most likely to suit Qwen.

**Current State**:
- `lms/planning.py:188` `PlanningPanel` is re-created every generation (`lms/society.py`, committee path), so no planner carries state across generations.
- `PlanningPanel._enforce_known_tags` (`lms/planning.py:312`) forces assignments onto real graph tags; the chair once invented tags.
- Workers can be today's `Agent.propose_iterative` (`lms/agent.py:302`) given a single-node goal view.

**Investigation**:
```bash
grep -n "PlanningPanel(" lms/society.py
# constructed inside run_generation_with_groups, once per generation
grep -rn "coordinator" lms/ | grep -v "#"
# no coordinator role exists
```

---

#### Acceptance Criteria

- [ ] `lms/run.py` adds `--coordinator`; it requires `--goal` and is rejected with `--groups` or `--best-of-k`
- [ ] One `Coordinator` object lives for the whole run. Its message history persists across generations, bounded by a token cap with the oldest turns dropped first; the cap and every drop are logged
- [ ] Each generation the coordinator sees the graph's available and done nodes, the foundation's agent context, and last generation's per-worker outcomes (verified, or the first line of the error). It emits one assignment per worker: a tag plus a free-text task
- [ ] Assignments are validated against AVAILABLE tags with the same rule as `_enforce_known_tags`. An unknown tag is replaced, not silently dropped, and every replacement is recorded
- [ ] Each worker is a fresh agent that runs `propose_iterative` on a single-node goal view, with the coordinator's task text inserted verbatim. Worker attempts come from `--max-attempts`
- [ ] No planning panel, no working groups, no review committee. Coordinator spend goes to the ledger's overhead bucket, as panel spend does
- [ ] `coordinator.json` records each generation's assignments, replacements and the coordinator's token spend; `metadata.json` records `"coordinator": true`
- [ ] `tests/test_coordinator_mode.py` covers: history persists across generations and is truncated at the cap; unknown tags are replaced and recorded; a worker prompt contains the coordinator's text verbatim; coordinator spend lands in overhead; flag exclusivity
- [ ] All tests pass: `uv run pytest tests/test_coordinator_mode.py tests/test_planning.py -v`
- [ ] `uv run ruff check` and `uv run mypy` clean on the touched files
- [ ] A two-generation run on the box shows the coordinator using its previous generation's outcomes — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/coordinator.py` | CREATE | Persistent coordinator, assignment parsing and validation |
| `lms/prompts.py` | MODIFY | `coordinator_system` and `coordinator_user` prompts, overridable via `--prompt-file` |
| `lms/society.py` | MODIFY | Dispatch |
| `lms/run.py` | MODIFY | Flag, metadata |
| `scripts/slurm/lms_run.sbatch` | MODIFY | `COORDINATOR=1` passes the flag |
| `tests/test_coordinator_mode.py` | CREATE | Behaviour tests (mock provider) |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-06.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Share the tag-validation rule with `PlanningPanel._enforce_known_tags` by extracting it into a function both call; do not copy it.
- Reuse the single-node goal view from `26Q4-HARN-05` if it has merged; otherwise create it here, and `-05` imports it.
- The coordinator prompt asks for exact signatures from the foundation context in every task, as the Sonnet coordinator did. That is its whole design; do not add role-play.

---

#### Decision Gates

- If matching the committee's verify budget needs a worker repair loop, which `propose_iterative` does not have → stop and ask. Workers' attempts versus the scribe's repair turns is an arm-design choice.
- If the coordinator's history cap forces dropping the foundation context rather than old turns → stop; the cap is wrong for the served context.
- If the coordinator needs to read workers' transcripts, rather than their outcomes, to work → stop and surface. That changes what the arm measures.

---

#### Out of Scope

- `26Q4-HARN-05`, `26Q4-HARN-07` and `26Q4-HARN-08` own the other new arms.
- No change to `PlanningPanel` behaviour beyond extracting the shared validation function.
- No coordinator training or prompt evolution (`26Q4-EVO-*`).

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-06.sh`.

---

#### Outcome Demo

**Where**: the box (you run it; Claude prepares the commands)
**Run**:
```bash
sbatch --export=ALL,AGENTS=3,GENERATIONS=2,ITERATIVE=1,COORDINATOR=1,GOAL=<slice goal> scripts/slurm/lms_run.sbatch
# in the run directory afterwards:
uv run python -c "import json;d=json.load(open('coordinator.json'));print(json.dumps(d,indent=1)[:3000])"
```
**Expect**: three assignments per generation, all on AVAILABLE tags; generation 1's assignments refer to generation 0's outcomes; no panel or review calls in the job log.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-06.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <900 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
