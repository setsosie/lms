### 26Q4-HARN-05: Best-of-k mode — k independent attempts per available node, Lean selects (the K9 arm)

**User Story**: As the organization-sweep experimenter, I want a run mode that spends the population's budget as k independent attempts on each of the available DAG nodes, so that "an exact verifier selecting among independent samples" is measured as its own organization against the single loop, the isolated null and the committee.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-05-best-of-k-mode` |
| **Dependencies** | `26Q3-HARN-27` (real edges, so more than one node is available); `26Q4-HARN-03` (reads its runs). Runs at stage 1 of `docs/planning/org-structures.md`, after Phase C |
| **PR Size Target** | <400 lines (max 800) |
| **Parts** | single PR |

---

#### Context

`docs/planning/org-structures.md` (2026-10-30) lists K9 as a stage-1 arm. In
flat and iterative goal mode every agent sees the whole goal and picks its own
target (`lms/agent.py` `propose_iterative`, `goal.to_prompt_context()`), so no
existing mode puts k independent attempts on one statement. The Sonnet-team
simulation hit k=2 and k=3 by accident and found that pass@1 "reports 'capable
but doesn't land' and 'not capable' identically" (`docs/LESSONS_FROM_SONNET_SIMULATION.md` §4c).

**Current State**:
- `lms/society.py:927` `_run_generation_iterative` runs every agent's `propose_iterative` against the full goal, through a local `verify_fn` that applies import restrictions and `_verify_admissible`.
- `lms/dependency.py:146` `DependencyGraph.available_tasks()` lists AVAILABLE nodes; `from_goal` builds the graph (chain-shaped until `26Q3-HARN-27`).
- `Goal.definitions` is a list (`lms/goals.py:126`); there is no single-node view.

**Investigation**:
```bash
grep -n "async def _run_generation_iterative\|async def verify_fn" lms/society.py
# 927, 945 (verify_fn is a closure, not a method)
grep -rn "best_of_k\|pass_at_k\|pass@k" lms/ tests/
# nothing
```

---

#### Acceptance Criteria

- [ ] `lms/run.py` adds `--best-of-k K` (int ≥ 2). It requires `--goal`, is rejected with `--groups`, and requires `--agents` divisible by K (an error otherwise, naming both numbers)
- [ ] Each generation takes the first `n_agents // K` AVAILABLE nodes by `priority_score` and gives each node K agents. Each agent runs `propose_iterative` against a single-node view of the goal, under the same `--max-attempts` as iterative mode
- [ ] Verified artifacts go through the existing gates, library, ledger and foundation path unchanged. A node becomes DONE when any of its K attempts verifies; the other verified attempts stay in the library as duplicates
- [ ] `best_of_k.json` in the run directory records, per generation and node: K, attempts that verified, pass@1 (verified ÷ K) and pass@k (any verified), and each attempt's final outcome as the ledger's `AttemptRecord` holds it
- [ ] `metadata.json` records `"best_of_k": K` (or `null`)
- [ ] `verify_fn` construction moves to a `Society` method used by both iterative and best-of-k mode; iterative behaviour is unchanged
- [ ] `tests/test_best_of_k.py` covers: allocation (9 agents, K=3 → three nodes × three agents); fewer available nodes than `n_agents // K` (the spare agents go to the next nodes in order, never two nodes short of K); DONE on any pass; pass@1 and pass@k on a fixture; flag validation errors
- [ ] All tests pass: `uv run pytest tests/test_best_of_k.py tests/test_society.py -v`
- [ ] `uv run ruff check` and `uv run mypy` clean on the touched files
- [ ] A short best-of-k run on the box produces a readable `best_of_k.json` — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/best_of_k.py` | CREATE | Allocation, single-node goal view, per-node record |
| `lms/society.py` | MODIFY | Dispatch, `verify_fn` as a method |
| `lms/run.py` | MODIFY | Flag, validation, metadata |
| `scripts/slurm/lms_run.sbatch` | MODIFY | `BEST_OF_K=` passes the flag |
| `tests/test_best_of_k.py` | CREATE | Behaviour tests (mock provider and verifier) |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-05.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Mirror `_run_generation_iterative` for result handling. Build the single-node view with `dataclasses.replace(goal, definitions=[defn])`, so the agent prompt is the existing `agent_user_goal` over one definition. No new prompt.
- Keep the mode out of `society.py` beyond the dispatch line; `society.py` is already the largest module.

---

#### Decision Gates

- If only one node is ever AVAILABLE (the chain graph, `26Q3-HARN-27` not merged) → stop. The mode would measure best-of-9 on one statement per generation; that is a different arm, and the doc must say so before it runs.
- If the single-node view changes what the agent prompt contains beyond the definitions list, such as the seed context or the import rules → stop and surface.
- If attempt outcomes are not recoverable per attempt from the ledger → stop; do not add a second outcome log.

---

#### Out of Scope

- `26Q4-HARN-06` (coordinator–worker), `26Q4-HARN-07` (swarm) and `26Q4-HARN-08` (provider pool) own the other new arms.
- No change to committee mode, to gates or to `cvfn_report`.
- No early stopping once a node has passed: all K attempts run, so pass@1 is unbiased.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-05.sh`.

---

#### Outcome Demo

**Where**: the box (you run it; Claude prepares the commands)
**Run**:
```bash
sbatch --export=ALL,AGENTS=6,GENERATIONS=2,ITERATIVE=1,BEST_OF_K=3,GOAL=<slice goal> scripts/slurm/lms_run.sbatch
# in the run directory afterwards:
uv run python -c "import json;d=json.load(open('best_of_k.json'));print(json.dumps(d,indent=1)[:2000])"
```
**Expect**: two nodes per generation with three attempts each; pass@1 and pass@k per node; one DONE node per passing set; `metadata.json` shows `"best_of_k": 3`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-05.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <400 lines changed (target) / <800 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
