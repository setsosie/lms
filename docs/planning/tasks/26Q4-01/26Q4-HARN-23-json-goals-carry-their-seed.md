### 26Q4-HARN-23: A goal file names its own seed, and the kernel tracks start bare

**User Story**: As the person reading a run on a goal that builds on Mathlib, I want the run to start from the foundation the goal names, so that agents are not handed a second `Category` class that collides with Mathlib's.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-23-json-goals-carry-their-seed` |
| **Dependencies** | Stacked on `26Q4-HARN-04` (#94). The re-run of R7 and the arms rehearsal need it |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (branch `26Q4-HARN-04` @ `1a5d28f`):
- `Goal.seed` picks the generation-0 foundation, and `seed=""` means a bare one. Only in-code goals could set it: `Goal.save` never wrote the key and `Goal.load` never read it. Every `goals/*.json` goal therefore started from the shipped seed, a from-scratch `LMS.Foundation.Category` class.
- Both kernel tracks tell agents to build on Mathlib's CategoryTheory. In batch q1's arms rehearsal on track B (jobs 414–426), 52 of 73 artifacts imported both `LMS.Foundation` and Mathlib, and 24 failed with `failed to synthesize CategoryTheory.Category`. No arm verified anything.
- `stacks-ch4-phase1` forbids Mathlib's CategoryTheory, so for that goal the from-scratch seed is the only `Category`, which is correct.

**Investigation**:
```bash
grep -n "seed" lms/goals.py lms/society.py scripts/goals/extract_stacks_goal.py
```

---

#### Acceptance Criteria

- [ ] `Goal.save` writes `seed` when it is set and `seeded_tags` when non-empty; `Goal.load` reads both, and a file without them loads with the old defaults
- [ ] `goals/stacks_kernel_track_a.json` and `_b.json` carry `"seed": ""`, and a `Society` built on either has an empty seed source
- [ ] `scripts/goals/extract_stacks_goal.py` writes the same key, so regenerating the files keeps it
- [ ] `stacks-ch4-phase1` still starts from the `Category` seed
- [ ] `goals/README.md` describes the seed and no longer calls file-goal registration unapplied
- [ ] `uv run pytest tests/test_goal_seed.py -q` covers each item above

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/goals.py` | MODIFY | Round-trip `seed` and `seeded_tags` |
| `goals/stacks_kernel_track_a.json` | MODIFY | Bare foundation |
| `goals/stacks_kernel_track_b.json` | MODIFY | Bare foundation |
| `scripts/goals/extract_stacks_goal.py` | MODIFY | Emit the key; drop a stale note |
| `goals/README.md` | MODIFY | Seed, and current registration |
| `tests/test_goal_seed.py` | CREATE | Behaviour |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-23-json-goals-carry-their-seed.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-23.sh` | CREATE | Verify script |

---

#### Notes

- The JSON files were edited by hand to match what the extractor now writes, because the Stacks clone it reads is not on this machine. A test checks that the two agree.
- A run's saved `goal.json` now keeps its seed, so a resumed run starts from the same seed as the original.
- The kernel preamble still says `LMS.Foundation` is importable. With a bare seed it is, and it is empty at generation 0.
