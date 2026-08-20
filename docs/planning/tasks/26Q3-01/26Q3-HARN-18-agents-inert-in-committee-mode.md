# 26Q3-HARN-18: `--agents` is inert in committee mode

**User Story**: As the calibration program, I want society size to actually
change the committee's working population, so that agent-count experiments
(the collective-brain axis) measure something.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-18-agents-inert-in-committee-mode` |
| **Dependencies** | None |
| **PR Size Target** | <150 lines |
| **Parts** | single PR |

---

#### Context

`committee_6x10` (box, 2026-08-20, post-#51): `--agents 6` produced the same
`Created: 3` and the same ~115-140K tokens/generation as `--agents 3`. The
committee path's cast was fixed — planning panel (chair + 3 members), each
working group exactly chair + scribe + 1 researcher
(`WorkingGroupConfig.members_per_role` default), and society agents used only
as round-robin reviewers (one review per artifact, so agents beyond
`n_artifacts` never generated a token). The `members_per_role` machinery,
serialization, and multi-researcher discussion loop all existed and were
tested — nothing wired `n_agents` into it.

#### Change

`Society._committee_members_per_role()`: researchers per group =
`max(1, n_agents // n_groups)`; chair and scribe stay fixed overhead. 3
agents / 3 groups reproduces the old cast exactly; 6/3 seats 2 researchers
per group.

#### Acceptance criteria

- [ ] 3 agents / 3 groups → `{CHAIR: 1, SCRIBE: 1, RESEARCHER: 1}` (old
      behavior preserved bit-for-bit).
- [ ] 6 agents / 3 groups → 2 researchers per group.
- [ ] A 2-researcher session consumes one more provider turn per discussion
      round (both researchers speak) — proven end-to-end.
- [ ] Full suite green.
