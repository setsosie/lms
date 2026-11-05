### 26Q4-HARN-12: Committee prompts come from the registry, so overrides and recorded versions are true

**User Story**: As the person reading a committee run, I want the prompts agents saw to be the ones the run records, so that a bred prompt can reach a committee and metadata cannot claim a prompt that never ran.

> Drafted by Claude on 2026-11-05 from a check run that day. The user approved carding the fix. Review the rest as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-12-committee-prompts-through-registry` |
| **Dependencies** | None |
| **Gate** | full |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `1a9172e`):
- `lms/working_group.py:177,190,215` and `lms/planning.py:143,170` define committee prompts as module constants and pass those to the model.
- `lms/prompts.py:576-581` also registers `chair_system`, `researcher_system`, `scribe_system`, `planning_chair_system` and `planning_member_system`, and nothing outside `lms/prompts.py` reads them.
- `get_prompt("agent_system_goal")` appears only in `Agent.propose` and `Agent.propose_iterative` (`lms/agent.py:271`, `:363`), which the committee path does not call.
- **So an override cannot reach a committee.** Applying an override to all five keys changed the registry and left the constants as they were. A committee run with `--prompt-file` would still record that genome in `prompt_overrides`.
- **The recorded versions are already wrong for three keys.** The registry text and the running constant differ:

| Key | Registry text (chars) | Running constant (chars) |
|---|---:|---:|
| `chair_system` | 579 | 404 |
| `researcher_system` | 777 | 1069 |
| `scribe_system` | 733 | 1028 |
| `planning_chair_system` | identical | identical |
| `planning_member_system` | identical | identical |

**Investigation**:
```bash
grep -rn 'chair_system\|researcher_system\|scribe_system\|planning_chair_system\|planning_member_system' lms/ tests/ | grep -v '^lms/prompts.py'
# no hits
grep -n 'SYSTEM_PROMPT' lms/working_group.py lms/planning.py
```

---

#### Scope

The registry takes on the text committees actually run, and committee code reads the registry at call time. **A committee's prompts stay byte-for-byte the same**, so the Phase C arms are unaffected and this can merge at any point.

---

#### Acceptance Criteria

- [ ] The registry's `chair_system`, `researcher_system` and `scribe_system` hold the text of today's constants, with a new version string, since the old versions described text that never ran
- [ ] A byte-for-byte check of all five texts against hashes taken at `1a9172e` passes: `tests/test_committee_prompts.py::test_text_unchanged_from_running_constants`
- [ ] `lms/working_group.py` and `lms/planning.py` read these prompts with `get_prompt` at call time, and the module constants are gone
- [ ] An override of each key reaches the model call: `tests/test_committee_prompts.py::test_override_reaches_committee_call` passes for all five, using a mock provider
- [ ] Clearing overrides restores the base text: `tests/test_committee_prompts.py::test_clear_restores_base` passes
- [ ] `grep -rn '_SYSTEM_PROMPT = ' lms/working_group.py lms/planning.py` finds nothing
- [ ] The full suite passes: `uv run pytest -q`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/prompts.py` | MODIFY | Registry holds the running text, new versions |
| `lms/working_group.py` | MODIFY | Read prompts from the registry |
| `lms/planning.py` | MODIFY | Read prompts from the registry |
| `tests/test_committee_prompts.py` | CREATE | Text unchanged, overrides reach the call |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-12.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Mirror `Agent.propose`: call `get_prompt` where the prompt is used, not at import time, or overrides applied at startup will be missed.
- Take the hashes before changing anything, and put them in the test as literals.
- Any test importing the constants by name needs updating, not a compatibility alias.

---

#### Decision Gates

- If keeping the text byte-identical turns out to be impossible, for example because a constant is built at runtime → stop. A changed prompt is a harness change for Phase C, and that is the user's call.
- Do not choose the registry's text over the constants' on quality grounds, even where the registry reads better. Which text is better is an experiment.

---

#### Out of Scope

- Whether `agent_system_goal` should reach committee agents, which is a design question for a later prompt arm.
- `26Q4-EVO-03`'s genome stays `agent_system_goal` only in v0.
- Per-agent prompts or soft prefixes: `26Q4-SPG-03`.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-12.sh`.

---

#### Outcome Demo

**Where**: local, mock provider
**Run**:
```bash
uv run pytest tests/test_committee_prompts.py -v
```
**Expect**: the unchanged-text test and the five override tests pass, showing committees run the same words as before and an override now reaches them.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-12.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <300 lines changed (target) / <500 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
