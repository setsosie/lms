# Pull Request Plan

**Date**: 2025-12-18
**Goal**: Implement Interactive Society & Working Groups architecture

---

## Overview

The existing codebase (~11,600 lines) will be pushed to GitHub as the initial commit.
New features from the Interactive Society spec will be added in 5 incremental PRs.

---

## PR 0: Initial Push (Not a PR)

Push existing codebase to GitHub repository.

```bash
git remote add origin git@github.com:USERNAME/lms.git
git push -u origin main
```

**Files**: Everything currently tracked
**Lines**: ~11,600

---

## PR 1: Dependency Graph

Add task dependency tracking for goal-directed formalization.

### Files to Create

| File | Description | Est. Lines |
|------|-------------|------------|
| `lms/dependency.py` | DependencyGraph, DependencyNode, TaskStatus | ~180 |
| `tests/test_dependency.py` | Unit tests for dependency graph | ~220 |

### Key Classes

```python
class TaskStatus(Enum):
    DONE, IN_PROGRESS, AVAILABLE, BLOCKED

class DependencyNode:
    tag, name, chapter, section, requires, unlocks, status

class DependencyGraph:
    from_goal(), available_tasks(), update_status(), save(), load()
```

### Test Coverage

- [x] Create nodes, add to graph
- [x] Infer dependencies from section ordering
- [x] Calculate availability when dependencies change
- [x] Priority scoring (unlocks count)
- [x] Save/load from JSON
- [x] Integration with Goal class

**Total: ~400 lines**

---

## PR 2: Planning Panel

Add generation-level planning with Chair + Voting Members.

### Files to Create

| File | Description | Est. Lines |
|------|-------------|------------|
| `lms/planning.py` | PlanningPanel, PlanningSession, Vote | ~300 |
| `tests/test_planning.py` | Unit tests for planning | ~250 |

### Key Classes

```python
class Vote(Enum):
    APPROVE, REJECT, ABSTAIN

class WorkingGroupAssignment:
    group_id, task_tag, priority, guidance, backup_task

class PlanningPanel:
    run_session() -> list[WorkingGroupAssignment]
```

### Test Coverage

- [x] Chair proposes assignments
- [x] Members vote
- [x] Majority approval logic
- [x] Revision after rejection
- [x] Parse proposal/vote from LLM output
- [x] Integration with DependencyGraph

**Total: ~550 lines**

---

## PR 3: Enhanced Working Groups

Upgrade existing working_group.py with full conversation support.

### Files to Modify/Create

| File | Description | Est. Lines Changed |
|------|-------------|-------------------|
| `lms/working_group.py` | Add Role enum, multi-turn conversation | +150 |
| `tests/test_working_group.py` | Full test coverage | ~300 |

### Key Enhancements

```python
class Role(Enum):
    CHAIR, SCRIBE, RESEARCHER

class WorkingGroup:
    async run_session() -> Artifact
    _chair_opening()
    _discussion_round()
    _chair_summary()
    _scribe_finalize()
```

### Test Coverage

- [x] Role assignment
- [x] Multi-turn conversation flow
- [x] Blackboard updates from code blocks
- [x] Consensus detection
- [x] Artifact compilation by Scribe
- [x] Mock provider integration

**Total: ~450 lines**

---

## PR 4: Society Integration

Wire everything together in Society class.

### Files to Modify

| File | Description | Est. Lines Changed |
|------|-------------|-------------------|
| `lms/society.py` | Add run_generation_with_groups() | +120 |
| `lms/config.py` | Add working group config fields | +20 |
| `lms/run.py` | Add CLI flags | +30 |
| `tests/test_society.py` | Add working group tests | +150 |

### Key Changes

```python
# config.py
use_working_groups: bool = False
n_working_groups: int = 3
group_size: int = 3
max_turns_per_group: int = 5

# society.py
async def run_generation_with_groups(self, generation: int) -> GenerationResult

# run.py
parser.add_argument("--working-groups", action="store_true")
parser.add_argument("--n-groups", type=int, default=3)
```

### Test Coverage

- [x] run_generation_with_groups() execution
- [x] Parallel group execution
- [x] Verification and Foundation update
- [x] Textbook logging from groups

**Total: ~320 lines**

---

## PR 5: Role Prompts

Add specialized prompts for Working Group roles.

### Files to Modify

| File | Description | Est. Lines Changed |
|------|-------------|-------------------|
| `lms/prompts.py` | Add CHAIR, SCRIBE, RESEARCHER prompts | +80 |

### Prompts to Add

- `CHAIR_SYSTEM_PROMPT` - Facilitate, don't code
- `RESEARCHER_SYSTEM_PROMPT` - Propose and critique code
- `SCRIBE_SYSTEM_PROMPT` - Compile final artifact
- `PLANNING_CHAIR_SYSTEM_PROMPT` - Allocate work
- `PLANNING_MEMBER_SYSTEM_PROMPT` - Vote on proposals

**Total: ~80 lines**

---

## Summary

| PR | Description | Est. Lines | Dependencies |
|----|-------------|------------|--------------|
| 0 | Initial push | 11,600 | None |
| 1 | Dependency Graph | 400 | PR 0 |
| 2 | Planning Panel | 550 | PR 1 |
| 3 | Enhanced Working Groups | 450 | PR 0 |
| 4 | Society Integration | 320 | PR 2, PR 3 |
| 5 | Role Prompts | 80 | PR 3 |

**Total new code: ~1,800 lines**

---

## Execution Order

```
PR 0 (push) ─────┬───────────────────────> main
                 │
PR 1 (deps) ─────┴─> PR 2 (planning) ─────┐
                                          │
PR 3 (groups) ────────────────────────────┼─> PR 4 (integration)
                                          │
                                          └─> PR 5 (prompts) ──> main
```

PRs 1 and 3 can be developed in parallel.
PR 4 requires both PR 2 and PR 3.
PR 5 is a small follow-up.

---

## Branch Naming

```
main
├── feature/dependency-graph      (PR 1)
├── feature/planning-panel        (PR 2)
├── feature/working-groups-v2     (PR 3)
├── feature/society-integration   (PR 4)
└── feature/role-prompts          (PR 5)
```
