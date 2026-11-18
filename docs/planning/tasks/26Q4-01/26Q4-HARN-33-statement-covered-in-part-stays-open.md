### 26Q4-HARN-33: A statement covered in part stays open, and the next group extends it

**User Story**: As the person measuring whether work ratchets, I want a verified artifact that covers only some of its statement's parts to stay in the foundation while its statement stays open, with the next group told what is there and what is missing, so that a good partial attempt becomes a step later work builds on instead of a closed task or a discarded one.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-33-statement-covered-in-part-stays-open` |
| **Dependencies** | `26Q4-HARN-32` (the coverage read); `26Q4-HARN-24` (tag closure by the name rule), which edits the same closure code, so it lands first |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `e9c7352`):
- **Status is binary.** `TaskStatus` (`lms/dependency.py:22`) has DONE, IN_PROGRESS, AVAILABLE and BLOCKED. A verified artifact that passes the gates is promoted, and its group's tag becomes DONE (`lms/society.py`, the `update_status(group.config.task_tag, TaskStatus.DONE, …)` call after `_promote_to_foundation`).
- **The goal closes with it.** `Goal.mark_formalized` (`lms/goals.py:214`) sets `formalized = True`.
- **DONE is final.** A DONE tag is not demoted (`[graph] refusing demotion`), and `available_tasks()` (`lms/dependency.py:147`) never offers it again.
- **No one is asked to finish.** Groups receive the statement's full text (`task_content` from `_get_task_content`), but nothing tells a later group that a partial version exists and what it lacks.

Evidence: batch q4, run s3.
- `CategoryFibredInGroupoids` (generation 4) closed 003T while covering only the opening and condition (2) of Definition 4.35.1 (D4, 2026-11-18: partial credit).
- 003T was never offered again, so the five remaining generations never had the chance to add condition (1).
- In q4, 10 later artifacts used a kept entry and 1 of them verified. A partial entry plus an explicit "add (1)" brief is the case where building on prior work is the obvious next step.
- 19 of track B's 40 statements have two or three numbered parts.

**Investigation**:
```bash
grep -n "class TaskStatus" -A8 lms/dependency.py
grep -n "TaskStatus.DONE, artifact.id" -B12 -A6 lms/society.py
grep -n "def mark_formalized\|def withdraw" -A12 lms/goals.py
grep -n "refusing demotion" lms/dependency.py
```

---

#### Scope

**PARTIAL is a new status between AVAILABLE and DONE.** It applies when a verified, promoted artifact, which closes its tag under `26Q4-HARN-24`'s name rule, has a `26Q4-HARN-32` coverage read with at least one part `missing`.
- **The artifact stays.** It stays promoted in the foundation, so later work can import and extend it.
- **The tag stays open.** The tag becomes PARTIAL, not DONE, and the goal records the missing parts.
- **PARTIAL tags are offered again,** ahead of AVAILABLE ones, and the group's brief says what exists and what is missing.
- **A later artifact on the same tag** that verifies with no part missing makes it DONE. One that covers more makes the missing list shorter.
- **PARTIAL does not release dependents.** A tag whose `requires` includes a PARTIAL tag stays BLOCKED, as with an unfinished tag today.

---

#### Acceptance Criteria

- [ ] `TaskStatus.PARTIAL = "partial"` exists. `available_tasks()` returns PARTIAL nodes ahead of AVAILABLE ones. PARTIAL does not count as done for dependents or for `progress()`. Covered by `tests/test_partial_status.py::test_partial_*`.
- [ ] Take an artifact that verifies, passes the gates and the name rule, and whose `coverage` lists a `missing` part. Its tag becomes PARTIAL, not DONE. The goal entry records `partial: true` and `missing: ["(1)"]`, and `formalized` stays false. The run log prints `[goal] 003T partial: CategoryFibredInGroupoids covers opening, (2); missing (1)`. Covered by `test_003t_partial`.
- [ ] Coverage of `None` (not read) closes the tag as today. The run log prints `[goal] 003T closed without a coverage read`, and the summary counts these. Covered by `test_unread_coverage_closes_as_before`.
- [ ] Coverage with no part missing closes the tag as today (DONE, `formalized: true`). Covered by `test_full_coverage_closes`.
- [ ] A group assigned a PARTIAL tag gets this in its brief (`guidance`): the partial entry's name, its kind and signature line from the foundation, the parts it covers and the parts missing, and the instruction to declare a new name that builds on it (for example `structure IsFibredInGroupoids extends CategoryFibredInGroupoids C where …`) rather than redeclare it. Covered by `test_partial_brief`.
- [ ] A later artifact on a PARTIAL tag is judged on its own coverage. With no part missing, the tag becomes DONE and both artifacts are listed in the goal entry. If it is missing fewer parts than the entry it extends, the tag stays PARTIAL with the shorter list. Covered by `test_partial_then_done`, `test_partial_narrows`.
- [ ] A PARTIAL tag that is offered 3 times without its missing list getting shorter is no longer offered, and stays PARTIAL. The run log says so. Covered by `test_partial_offer_cap`.
- [ ] Rollback and withdrawal (`Goal.withdraw`) treat a PARTIAL tag like a DONE one: withdrawing its only artifact reopens it as AVAILABLE. Covered by `test_partial_withdraw`.
- [ ] The run summary lists PARTIAL tags with their covered and missing parts. The round's "built on" read counts an artifact that extends a PARTIAL entry (it names the entry and declares a new name) as built on. Covered by `test_summary_lists_partial`.
- [ ] All tests pass: `uv run pytest tests/test_partial_status.py -v`, and the full suite stays green.
- [ ] On a track-B run with this card and `26Q4-HARN-32`, at least one multi-part tag goes PARTIAL and is offered again — accepted via Outcome Demo.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/dependency.py` | MODIFY | `TaskStatus.PARTIAL`, ordering in `available_tasks()`, dependents and `progress()` |
| `lms/goals.py` | MODIFY | `mark_partial(tag, artifact_id, missing)`; save/load `partial` and `missing`; `withdraw` reopens PARTIAL |
| `lms/society.py` | MODIFY | Closure decision after promotion (committee and flat paths); the offer cap; build the brief for a PARTIAL tag |
| `lms/planning.py` | MODIFY | Carry a PARTIAL tag's brief into the assignment's `guidance` |
| `lms/run.py` | MODIFY | PARTIAL lines in the end-of-run summary |
| `tests/test_partial_status.py` | CREATE | The criteria above |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-33-statement-covered-in-part-stays-open.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-33.sh` | CREATE | Verify script |

---

#### Implementation Notes

- **Put the closure decision in one helper**, `_close_or_partial(tag, artifact)`, called where DONE is set today, after `26Q4-HARN-24`'s name check. Do not scatter coverage checks.
- **Build the brief from data:** the foundation entry, the coverage dict and the statement's part labels (`statement_parts` from `26Q4-HARN-32`). Do not change the working-group prompts' text. The brief travels in `guidance`, as the planning panel's guidance does today.
- **Keep the name rule.** The extending artifact must declare a new name. The promotion refusal of a redeclared name (q3: 7 refused) is unchanged; the brief is what steers away from it.
- **Offer cap:** a per-tag counter in the dependency graph, saved with it, so a resumed run keeps it.

---

#### Decision Gates

- If `26Q4-HARN-24` has not merged, or its closure code differs from what this card assumes → stop and rebase onto it. Do not reimplement the name rule here.
- If ordering PARTIAL ahead of AVAILABLE starves untouched tags on a fixture (every generation goes to PARTIAL tags) → stop and surface the ordering choice to the user. The cap of 3 is the card's answer, but the user decides if it is not enough.
- If releasing dependents of a PARTIAL tag turns out to be needed for track B to progress (a dependent's statement needs only the covered parts) → stop. That is the user's call, not a silent change.
- If the change exceeds the PR Size Target → stop and split, status and closure first, then the brief.

---

#### Out of Scope

- `26Q4-HARN-32` owns the coverage read and the statement split; this card consumes `artifact.coverage`.
- `26Q4-HARN-24` owns the name rule. `26Q4-HARN-17` owns what happens on a name collision. `26Q4-HARN-19` owns what the merged foundation means.
- No partial credit score (fraction or weight) in CVFN or the gate histogram. How D4 records partial credit is a separate decision.
- No change to the working-group session, the review prompt, or the generators' prompts.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-33.sh`.

---

#### Outcome Demo

**Where**: the server, through a round folder (runbook rule); read locally
**Run**:
```bash
uv run pytest tests/test_partial_status.py -v
grep -h "\[goal\] .* partial:\|closed without a coverage read" <round>/back/lms_<round>/*/logs/run.log
```
**Expect**:
- The tests pass.
- A track-B run of 10 generations at a pin holding this card, `26Q4-HARN-32` and `26Q4-HARN-24` prints at least one `[goal] <tag> partial:` line.
- The same tag is assigned again in a later generation, with the brief naming the missing parts.
- The summary lists the PARTIAL tags.
- Whether the later group's artifact extends the partial entry is the measurement, not a pass condition.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-33.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator

---

#### Notes

- **Added 2026-11-18 from batch q4,** with `26Q4-HARN-32`.
- **Sprint 5 found-work,** attempted after the committed scope; if it does not land by 2026-11-27 it carries to Sprint 6.
- **This card changes what Phase C reads** (reuse and the ratchet). If it lands after the harness freeze (Sprint 5 DoD 3), Phase C runs without it unless the freeze moves. That choice is the user's.
