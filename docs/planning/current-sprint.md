# Sprint 5: Make the slice decision, and freeze the harness

**Dates**: 2026-11-16 → 2026-11-27 (two weeks; 2026-11-26 is a holiday)
**Quarter**: Q4 2026 · folder `26Q4-01`
**Program**: `docs/planning/calibration-program.md` — Phase B close, Phase C entry
**Sprint Goal**: Turn Sprint 4's measurements into **a written slice
decision**, and settle the rules that change the number Phase C reads, so the
harness can freeze at a named commit and Sprint 6 can run Phase C.

**Status**: 🔄 ACTIVE

> **Phase C is not in this sprint.** Gate B, the calibration run, `26Q3-HARN-25`
> and `-28` are the Sprint 6 pre-lock in `upcoming-sprints.md`. They start
> once DoD 1 and DoD 3 below are done.

> **Every sprint runs something on the server.** This sprint's are batch q2
> (built 2026-11-13, runs the weekend of 2026-11-14) and the §7 density run.
> The user runs the runbooks and sends the output back.

---

## Sprint 4 close-out (26Q4-01): 38/58 pts, measured, not decided

| Metric | Value |
|--------|-------|
| Dates | 2026-10-09 → 2026-11-13 |
| Delivered | **38** (66% of 58; committed 3/3) |
| Carried here | 10 pts across 4 tasks; 10 more (HARN-25, -28) carried to Sprint 6 |
| N1 density | ✅ measured. Slice decision ❌ not written; waits on the D4 read |

Full record: [`archive/sprint-04-26Q4-01.md`](archive/sprint-04-26Q4-01.md).

Every number Sprint 4's goal named exists: densities for core, ramification
and Neukirch Ch. I §1–§6, the control run, and a CVFN denominator from R5e.
The decision they feed is a human read of 37 statements, which Sprint 4 never
sized. It is this sprint's first row.

Velocity: **24.3 pts/sprint** over four sprints (0, 19, 40, 38). Sprints 3
and 4 committed 10 and 3; the rest was found-work. This sprint commits 11,
near the measured committed-scope rate, not the mean.

## Sprint 5 Summary

| Metric | Value |
|--------|-------|
| Committed points | 11 |
| Server runs | batch q2 (4 rounds) + §7 density, unpointed |
| Human read | D4 queue, 37 + 10 statements, unpointed |
| Stretch | 0 |
| Window | 2026-11-16 → 2026-11-27 |
| Track | Phase B close: slice decision, harness freeze |
| Status | 🔄 ACTIVE |

## The decision (the sprint goal)

| Step | What | Owner | DoD | Status | Notes |
|------|------|-------|-----|--------|-------|
| D1 | D4 read of the §1–§6 queue (37 statements) | user | 1 | 🔄 IN PROGRESS | Sheet: `references/neukirch-ch1/d4_queue_s01-s06.md` (untracked), ordered Group 3, 2, 1. The classifier never recognises a special case (Group 2: 0 N0), and 13 of its 22 N1 labels are wrong |
| D2 | §7 drafts elaborate and get a density, on the server | user runs | 1 | 🔲 PENDING | 10 statements, `data/ant_arcs/neukirch_ch1/s07_dirichlet_unit_theorem.json`. The local pinned library is gone, and the user chose not to rebuild it, so this runs on the box |
| D3 | Restate or keep DoD 2's control bar | user | 2 | 🔲 PENDING | The control reads N0 10/52, INCONCLUSIVE 38, decisive N1 0. Either the bar becomes "decisive N1 = 0", or `26Q4-HARN-15` changes what the control can read |
| D4 | Write the slice decision | Claude drafts, user decides | 1 | 🔲 PENDING | Names the arc and its statements, or records "~zero, go to Phase E". The calendar for Sprints 6–8 follows from it |

## Carryover from Sprint 4

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q4-HARN-13: Decide what a decisive N1 means at Gate 4 | 2 | HIGH | HARN | 🔲 PENDING | — | Before the freeze. Library results can read decisive N1 at Gate 4. The choice of rule is the user's |
| 26Q4-HARN-15: Name search can't read N0 without checking the statement | 3 | HIGH | HARN | 🔲 PENDING | — | Before the freeze. All 10 control N0s came from name search, so this bears on D3. The choice of rule is the user's. The card's Notes record a neighbouring `.{u}` label-token leak from #82's review |
| 26Q4-HARN-19: The merged foundation means what each artifact verified alone | 3 | HIGH | HARN | 🔲 PENDING | — | Before the freeze. Universe order, shadowing and dropped text can change a statement's meaning with the merged file still compiling. The choice of approach is the user's |
| 26Q4-HARN-24: A tag closes only on an artifact that is that tag's statement | 2 | HIGH | HARN | 🔲 PENDING | — | From r6 (batch q1). Rule A (name match) chosen 2026-11-13; spec in the card |
| 26Q3-HARN-25: Proof sketches become open leaves | 5 | MEDIUM | HARN | ⏭ Sprint 6 | — | Not committed here. Phase C preparation |
| 26Q3-HARN-28: Theorem cards and statement pinning | 5 | MEDIUM | HARN | ⏭ Sprint 6 | — | Not committed here. Holds the ANT arc → goal converter; no ANT goal exists without it |

## Planned

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q4-HARN-20: A search that fails reads as unavailable instead of raising | 1 | MEDIUM | HARN | 🔲 PENDING | — | Card on main since #101; its user story and criteria are still template text |

**Committed: 11 pts** (HARN-13 2, -15 3, -19 3, -24 2, -20 1).

## Server runs

| Step | What | DoD | Status | Notes |
|------|------|-----|--------|-------|
| q2 | Batch q2: r7b, arms2, thr2, hang, at pin `05aebc22` on `batch/q2-pins` | 4 | 🔲 QUEUED | Built 2026-11-13; runs the weekend of 2026-11-14. Delete `batch/q2-pins` once collected |
| R7 | Three groups on three tags (`stacks-kernel-track-b`), as q2's r7b | 4 | 🔲 QUEUED | Both q1 jobs hung in NCCL; r7b has `26Q4-CHORE-05`'s start ladder |
| arms2 | Phase C arms rehearsal: C9, I9 ×9, S1, R9 with per-run seeds | 4 | 🔲 QUEUED | Checks that `26Q4-HARN-22` gives distinct I9 runs and `-23` stops the track-b collision |
| D2 | §7 density run | 1 | 🔲 PENDING | See the decision table |

## Definition of Done

1. **The slice decision is written down** (Sprint 4's DoD 5). It names the
   chosen arc and its statements from the D4 read, or records "both ~zero,
   Phase C does not start, go to Phase E". The calendar for Sprints 6–8
   follows from it.
2. **DoD 2's control bar is settled**: kept and met, or restated with a reason
   (for example, decisive N1 = 0), in the slice decision.
3. **The harness freezes at a named commit.** `26Q4-HARN-13`, `-15`, `-19` and
   `-24` merged, or explicitly deferred past Phase C with the reason written
   down. The freeze SHA is the one Phase C runs.
4. **Batch q2 is read.** R7 has a result or a diagnosed failure; arms2 shows
   whether the I9 runs now differ.

Read the gate histogram's *shape*, not its total.

## Risk register

| Risk | Mitigation | Task |
|------|------------|------|
| The D4 read is the user's time, and the holiday week shortens the window | It is row D1, with an owner. Group 3 comes first, so a partial read still says the most | D1 |
| Rule choices for `-13`, `-15`, `-19` stall the freeze | Each card states its options. Ask early in the sprint; a choice deferred past Phase C is allowed if written down | DoD 3 |
| Found-work displaces the decision, as runs were displaced in Sprints 3 and 4 | New cards go to Sprint 6 unless they block DoD 1 or the freeze | — |
| q2 hangs in NCCL again | `26Q4-CHORE-05`'s ladder (base → no-custom-ar → no-p2p); the hang round tries every GPU pair | q2 |
| §7 can't elaborate locally | It runs on the server; do not rebuild the local library without the user's go-ahead (7.2 GB, disk 95% full) | D2 |

## Deferred — not this sprint

| Task | Points | Why |
|------|--------|-----|
| Gate B + Phase C (S1 / C9 / I9 / R9 + frontier control, ADR 0001, 0003) | TBD | Sprint 6 pre-lock. Entry-gated on DoD 1 and DoD 3 |
| 26Q3-HARN-25, 26Q3-HARN-28 | 10 | Sprint 6, with Phase C |
| 26Q3-HARN-26: close leaves and auto-resolve parents | 5 | Sprint 6 pre-lock (`docs/planning/dag-phase.md`) |
| 26Q3-HARN-06: D4 side-by-side review view | 3 | Needed before Phase D |
| 26Q3-HARN-08: agents emit Lean 3, not Lean 4 | 2 | **Re-evidence or close.** Still no evidence the defect exists |
| 26Q4-HARN-10: gated best-practice layer | 5 (placeholder) | Depends on `26Q3-HARN-28` |

Task cards follow the shared task-card template (`templates/task-card.md` in
the planning tooling). Cards for this sprint live under
`docs/planning/tasks/26Q4-01/`.

## Sync Log

- **2026-11-13** — Sprint 4 closed at 38/58 pts (committed 3/3), measured but
  not decided, archived to `archive/sprint-04-26Q4-01.md`. Sprint 5 opened
  2026-11-16 → 2026-11-27 at 11 committed points: the slice decision and the
  before-freeze cards. Gate B + Phase C move to the Sprint 6 pre-lock with
  `26Q3-HARN-25` and `-28`.

---

*Last Updated: 2026-11-13*
