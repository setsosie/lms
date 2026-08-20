### 26Q3-HARN-15: A same-tag failure clobbers a task's DONE status

**User Story**: As the calibration program, I want a verified task to stay
DONE, so that generations are never re-assigned solved work that can only
fail by redefinition collision.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-15-done-status-clobber` |
| **Dependencies** | None (evidence from the HARN-14 demo run) |
| **PR Size Target** | <300 lines |
| **Parts** | single PR |

---

#### Context

> Investigated 2026-08-19 from `experiments/committee_real_b` (the HARN-14
> outcome demo, 4×H100). Line numbers from the `26Q3-HARN-14-*` branch.

**The evidence.** `Category-715b6d9c` verified in **generation 1** and set
tag 0013 to DONE. Yet generations 2 and 3 were re-assigned 0013 and failed
with the only error they could produce:

```
gen 2  Category-f90794b5  failed  `LMS.Foundation.Category` has already been declared
gen 3  Category-0308f9c6  failed  `LMS.Foundation.Category` has already been declared
```

Only at generation 4 did DONE stick (`Category-2406ee24`, a single artifact
on the tag) — which is why the in-run progress read "first verified at gen 4"
when the artifact record says gen 1. **~3 of the run's 8 generations were
spent re-solving or colliding with a solved task**; Functor work started at
gen 5 instead of gen 2.

**The mechanism.** `DependencyGraph.update_status` (`lms/dependency.py:89-97`)
assigns unconditionally — no transition rules. The committee path resets a
group's tag to AVAILABLE on *every* failure branch: no-artifact
(`lms/society.py:1039`), empty code (`:1054`), review REJECT (`:1163`),
import restriction (`:1199`), verify failure (`:1284`). When several groups
share one tag — guaranteed early, since the goal graph gates everything
behind 0013 — Phase 5 processes their artifacts sequentially, and a failed
artifact *after* the verified one demotes DONE back to AVAILABLE. In
`committee_real_b` gen 1 the order was exactly that: failed
(`Category-2d7eb10a`) → verified (`Category-715b6d9c`, DONE) → failed
(`Category-8793433b`, **DONE → AVAILABLE**).

**Investigation**:
```bash
grep -n "self.nodes\[tag\].status = status" lms/dependency.py
# 94: unconditional assignment, no monotonicity
grep -c "TaskStatus.AVAILABLE" lms/society.py
# 5 committee-path demotion sites
```

---

#### Acceptance Criteria

- [ ] A DONE task stays DONE: `update_status(tag, AVAILABLE)` on a DONE node
      is a no-op (demotion refused in ONE place — the graph, not five call
      sites): `uv run pytest tests/test_dependency.py -k done_is_terminal -q`
- [ ] The committee_real_b gen-1 sequence is pinned: same-tag artifacts
      processed verified-then-failed leave the tag DONE:
      `uv run pytest tests/test_society.py -k same_tag_failure_after_verify -q`
- [ ] Legitimate rollback stays possible via an explicit, separate path
      (e.g. `revert_done(tag)`), not via `update_status` — the foundation
      roll-back-on-broken-build plan (`foundation.py:131-134`) will need it.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/dependency.py` | MODIFY | refuse DONE demotion in `update_status`; explicit `revert_done` |
| `tests/test_dependency.py` | MODIFY | terminal-DONE + revert tests |
| `tests/test_society.py` | MODIFY | same-tag verified-then-failed sequence |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-15.sh` | CREATE | landing checks |

---

#### Implementation Notes

- Fix the graph, not the call sites: five demotion sites exist today and
  more will be added; a transition rule in `update_status` covers all of
  them and every future caller.
- Do NOT silently ignore other transitions — only DONE→(anything but DONE)
  is refused. IN_PROGRESS→AVAILABLE (task released) stays legal.
- Log the refused demotion (one line) so a clobber attempt is visible in
  run output rather than silent.

---

#### Decision Gates

- If any existing test legitimately depends on DONE→AVAILABLE, stop and
  surface it — that test encodes the bug or a rollback need, and which one
  is the user's call.
- Standard gates: no scope absorption, stop if PR target exceeded, explicit
  "do NOT"s are binding.

---

#### Out of Scope

- Panel/assignment logic (why several groups share one tag) — that is
  allocation behaviour, not status integrity.
- Foundation rollback implementation (`lake build` after append) — this card
  only leaves it a door (`revert_done`).
- The redefinition-collision error class itself — with DONE terminal, solved
  tags are never re-assigned and the class disappears.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-15.sh`. Must fail at the merge
base — measure discrimination after committing.

---

#### Outcome Demo

**Where**: mahpiya (4×H100), human-run.
**Run**: re-run the committee_real_b config; **Expect**: no
"has already been declared" failures; once a tag verifies, the next
generation's assignments move to its dependents.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-15.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened within size target
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or `N/A` and why)
