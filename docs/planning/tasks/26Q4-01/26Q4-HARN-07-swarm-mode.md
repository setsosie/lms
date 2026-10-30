### 26Q4-HARN-07: Swarm mode — Prove2Me-style organization on the local DAG (the P9 arm)

**User Story**: As the organization-sweep experimenter, I want a run mode with no coordinator: each agent picks a node by a fixed reading order, reads a bounded set of foundation entries, and writes only verified statements back. That way Prove2Me's organization, which cannot be run as a hosted arm, is measured locally against the committee, the coordinator, best-of-k and the isolated null.

| Field | Value |
|-------|-------|
| **Story Points** | 5 (Part 1) + 3 (Part 2) |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-07-swarm-mode` (Part 2: `26Q4-HARN-07p2-swarm-comments`) |
| **Dependencies** | `26Q3-HARN-27`, `26Q3-HARN-28` (pinning: "first accepted proof of a card" needs a fixed statement), `26Q3-HARN-25` / `26Q3-HARN-26` (sketches and leaves); `26Q4-HARN-03`. Stage 2 of `docs/planning/org-structures.md` |
| **PR Size Target** | <600 lines per part (max 1000) |
| **Parts** | 0/2 — Part 1 mode, reading order, bounded reads, claim flag; Part 2 per-node comment channel |

---

#### Context

Prove2Me has no coordinator or scheduler; its paper rejects an "orchestrating
agent" (§4.1). Agents choose work through a reading order (milestones first, then the
graph, then search), with no claim or lock. Credit goes to "the earliest
`ACCEPTED` submission", and agents talk through per-mission discussion threads
(`references/prove2me/`, untracked; summarized in `org-structures.md`).
Prove2Me's server code is not public, so the organization is re-implemented
here.

The book draft's swarm definition supplies the conditions this mode must meet:
- no agent reads another agent's transcript;
- every interaction goes through the medium;
- each agent's reads are bounded by a k that "does not grow with N".

**Current State**:
- Flat and iterative agents read the whole goal (`goal.to_prompt_context()`), the full foundation context (`FoundationFile.get_context_for_agent()`, unbounded on purpose, `lms/society.py:123` comment) and the textbook (`lms/agent.py`, `textbook.get_for_context`).
- `DependencyNode.priority_score` (`lms/dependency.py:43`) orders nodes. Goals mark milestones only by a `"MILESTONE: "` name prefix (`lms/goals.py:378`); there is no structured milestone field.
- Committee mode already caps its prompt summary at `COMMITTEE_SUMMARY_MAX_ENTRIES = 10`.

**Investigation**:
```bash
grep -n "COMMITTEE_SUMMARY_MAX_ENTRIES\|def get_context_for_agent" lms/society.py lms/foundation.py
# society.py:127 = 10; foundation.py:1041 takes max_entries
grep -n "MILESTONE" lms/goals.py | head -3
# name prefix only
```

---

#### Acceptance Criteria

Part 1:
- [ ] `lms/run.py` adds `--swarm`, rejected with `--groups`, `--best-of-k` and `--coordinator`; `--read-k K` (default 10) and `--claim` (default off)
- [ ] Each generation, agent *i* takes a node by a fixed reading order: open sketch leaves first, then AVAILABLE nodes by `priority_score`, starting at offset *i* so agents spread deterministically. With `--claim`, a node one agent has taken is skipped by the agents after it in the same generation. Without it, agents may collide, as on Prove2Me
- [ ] An agent's context is its node's pinned theorem card plus at most `--read-k` foundation entries: the node's DAG prerequisites first, then the seed. No textbook, no library listing, no other agent's output. The entries actually read are recorded per attempt
- [ ] Verified statements are written through the existing gates and foundation path. When two agents verify the same card in one generation, the first to finish is promoted and the other is recorded as a duplicate, never promoted twice
- [ ] `swarm.json` records per generation: each agent's node, its read set (entry names), outcome, and whether it collided; `metadata.json` records `swarm`, `read_k` and `claim`
- [ ] `tests/test_swarm_mode.py` covers: reading order and offsets; claim on versus off; the read set never exceeds `--read-k` on a foundation of 50 entries; a duplicate is recorded and not promoted; no planning-panel, review or textbook call is made (mock provider records calls by prompt name)

Part 2:
- [ ] `--comments`: after an attempt, an agent may append one short note (token-capped) to its node's thread. The thread's last notes are read as part of the same `--read-k` budget, and authors are recorded
- [ ] Tests cover: comments count against `--read-k`; a note is attributed to its author; with `--comments` off no thread is read or written

Both parts:
- [ ] All tests pass: `uv run pytest tests/test_swarm_mode.py -v`
- [ ] `uv run ruff check` and `uv run mypy` clean on the touched files
- [ ] A two-generation swarm run on the box shows bounded read sets and recorded collisions — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/swarm.py` | CREATE | Reading order, read-set builder, claim, collision record (Part 2: threads) |
| `lms/society.py` | MODIFY | Dispatch |
| `lms/run.py` | MODIFY | Flags, metadata |
| `lms/prompts.py` | MODIFY | `swarm_agent_user`: card plus read set, overridable via `--prompt-file` |
| `scripts/slurm/lms_run.sbatch` | MODIFY | `SWARM=1`, `READ_K=`, `CLAIM=1` (Part 2: `COMMENTS=1`) |
| `tests/test_swarm_mode.py` | CREATE | Behaviour tests (mock provider and verifier) |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-07.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Build the read set from `FoundationFile.get_context_for_agent(max_entries=...)`'s renderer, restricted to named entries; do not write a third renderer (`_get_foundation_summary`'s docstring records what two did).
- Pinning (`26Q3-HARN-28`) defines "the same card". Collisions are counted per pinned declaration, not by text similarity.
- Reuse the single-node goal view from `26Q4-HARN-05` or `-06`, whichever merged first.

---

#### Decision Gates

- If `26Q3-HARN-28` has not merged → do not start; without pinning, two agents' proofs of "the same" node are not comparable.
- If the DAG's prerequisites for a node exceed `--read-k` → keep the closest by DAG distance and record the truncation. Do not raise k silently; k not growing with N is the point.
- If the milestone-first reading order is wanted → stop. There is no structured milestone field, and parsing `"MILESTONE: "` names is a goal-schema change, not this card.
- If agents cannot produce a proof without the textbook → that is a result for the arm; report it rather than re-enabling the textbook.

---

#### Out of Scope

- `26Q4-HARN-05`, `-06` and `-08` own the other new arms; `26Q3-HARN-29` / `-30` own agent-authored cards and search-before-propose.
- No hosted Prove2Me client (ADR 0002).
- No change to flat, iterative or committee modes, to gates or to `cvfn_report`.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-07.sh` (Part 1 checks; Part 2 adds its lines).

---

#### Outcome Demo

**Where**: the box (you run it; Claude prepares the commands)
**Run**:
```bash
sbatch --export=ALL,AGENTS=6,GENERATIONS=2,ITERATIVE=1,SWARM=1,READ_K=10,GOAL=<slice goal> scripts/slurm/lms_run.sbatch
# in the run directory afterwards:
uv run python -c "import json;d=json.load(open('swarm.json'));print(json.dumps(d,indent=1)[:3000])"
```
**Expect**: six agents per generation on deterministic nodes; every read set at most 10 entries; any collisions listed with one promoted proof; no textbook, panel or review calls in the job log.

---

#### Definition of Done

- [ ] All acceptance criteria checked off (Part 2's at Part 2)
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-07.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] Each PR <600 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
