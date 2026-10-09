# Sprint 4: Get the number Sprint 3 was for

**Dates**: TBD (opened 2026-10-09; set at planning)
**Quarter**: Q4 2026 by calendar · the carried card lives in folder `26Q3-01`
**Program**: `docs/planning/calibration-program.md` — Phase B, completion
**Sprint Goal**: Run the measurement Sprint 3 built but never ran, on the
restored box: **measured N1 density for both ANT Ch. I candidate arcs, the
Gate A control, and a CVFN denominator from a real run.** The density decides
whether Phase C starts at all. If both arcs measure ~zero, it does not.

**Status**: 🔄 ACTIVE

> **Phase C is not in this sprint.** Gate B, the two-arm calibration run and
> the DAG-phase cards moved to the Sprint 5 pre-lock in `upcoming-sprints.md`.
> All of them are entry-gated on DoD 1 below, and none of them can be sized
> honestly until its number exists.

> **Every sprint runs something on the server.** This sprint's server runs are
> the sprint goal: runbook R4 (DoD 1 and 3) and R5 (DoD 4). The user runs the
> runbook and pastes the output back.

---

## Sprint 3 close-out (26Q3-01): 40/43 pts, goal not met

| Metric | Value |
|--------|-------|
| Dates | 2026-08-10 → 2026-09-04 (closed 2026-10-09) |
| Delivered | **40** (93% of 43; committed 10/10) |
| Carried here | 3 pts across 1 task, plus DoD items 1, 3 and 4 |
| N1 density | ❌ not measured. The 2026-08-19 reading is void (#63) |

Full record: [`archive/sprint-03-26Q3-01.md`](archive/sprint-03-26Q3-01.md).

All ten committed points landed by 2026-08-19, and 30 points of found-work
followed. The measurement itself never ran. The box time went to committee
runs, and then the box was unavailable 2026-09-08 → 2026-10-08.

Velocity: **19.7 pts/sprint** over three sprints (0, 19, 40). Sprint 3's 40
is mostly found-work, so commit to what this sprint actually needs, not to
the mean.

## Sprint 4 Summary

| Metric | Value |
|--------|-------|
| Committed points | 3 |
| Server runs | 3 DoD items, unpointed |
| Stretch | 0 |
| Carryover | everything here is Sprint 3 carryover |
| Window | TBD |
| Track | Phase B: get the slice decision |
| Status | 🔄 ACTIVE |

## Carryover from Sprint 3

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q3-HARN-20: Gate 4 withholds N1 when its search could not have matched | 3 | HIGH | HARN | 🔲 PENDING (Part 2 of 2) | Part 1: #63, merged 2026-10-09 | Part 2: the import-namespace rule, the Yoneda regression fixture, flagging `forbidden_imports` runs in the CVFN report, and scoring the informal statement. DoD 1 does not wait on it; Part 1 is what the density run needed |

## Server runs (the sprint goal)

Runbook: [`docs/runbooks/2026-10-08-box-restore.md`](../runbooks/2026-10-08-box-restore.md).
These rows carry no points; their code was counted with `-04` and `-05` in
Sprint 3. They are still what this sprint is for.

| Step | What | DoD | Status | Notes |
|------|------|-----|--------|-------|
| R0 | Inventory (read-only) | — | 🔲 PENDING | Paste the output back before anything else; the box was restored to an earlier state |
| R1–R3 | Scratch routing, Lean toolchain + library build, Python env + suite | — | 🔲 PENDING | Install the pinned toolchain, not `stable`; `lean/.lake` may dangle |
| R4 | N1 density over both arcs + the Gate A control, one CPU batch job (`scripts/slurm/n1_density.sbatch`) | 1, 3 | 🔲 PENDING | Expect some non-SCHEMATIC drafts to fail at the pin; repair them in one PR and re-run |
| R5 | CVFN denominator from a real run | 4 | 🔲 PENDING | R5a if a run's `attempts.json` survived the restore, else R5b (a small real run) |

## Definition of Done

1. **N1 density measured for both ANT Ch. I arcs** at the pinned build, from
   runbook R4. Record per arc: the upper density, the D4 queue,
   `stages_run` and `max_n1_confidence`. On the arcs, decisive N1 is 0 by
   construction (two search stages cap N1 at 0.6), so the slice decision reads
   the upper density and the D4 queue, not a decisive count.
2. **Gate A control run**, from the same batch job: every artifact in the
   committed control arc (`data/novelty_control/gate_a_control_arc.json`)
   classified. N0 must account for essentially all of them.
3. **A CVFN denominator exists**: tokens and wall-clock attributed per
   statement, including failed attempts, on at least one real run (R5).
4. **`26Q3-HARN-20` Part 2 merged.**
5. **The slice decision is written down.** It names the chosen arc, or
   records "both ~zero, Phase C does not start, go to Phase E", and the
   calendar re-plan for Sprints 5–7 follows from it.

Read the gate histogram's *shape*, not its total.

## Risk register

| Risk | Mitigation | Task |
|------|------------|------|
| Found-work displaces the runs again, as it did all of Sprint 3 | No new card enters this sprint until R4 has run. New defects card to Sprint 5 unless they block R4 or R5 | — |
| The restored box is missing pieces (home contents, toolchain, library cache) | R0 is read-only and comes first. Re-provision from R1–R3, not from memory | R0–R3 |
| Arc drafts fail to elaborate at the pin | 28/41 elaborated against a newer build; every failure was already SCHEMATIC. Repair any non-SCHEMATIC failure in one PR and re-run R4 | R4 |
| Both arcs measure ~zero N1 | **That is the sprint succeeding.** CVFN is undefined at current scope; go to Phase E early. Do not widen the slice until something scores | DoD 5 |
| No run's `attempts.json` survived the restore | R5b: a small real run produces one | R5 |
| A third external result invites a third reframe | The pre-commitment exists to survive news. Get the number first | — |

## Deferred — not this sprint

| Task | Points | Why |
|------|--------|-----|
| Gate B + Phase C (two-arm calibration run, ADR 0001) | TBD | Sprint 5 pre-lock. Entry-gated on DoD 1 |
| 26Q3-HARN-25 / -26 / -27 / -28: proof sketches and the DAG phase | 5 / 5 / 3 / 5 | Sprint 5 pre-lock (`docs/planning/dag-phase.md`) |
| 26Q3-HARN-06: D4 side-by-side review view | 3 | Needed before Phase D, whose dates are TBD |
| 26Q3-HARN-08: agents emit Lean 3, not Lean 4 | 2 | **Re-evidence or close.** Still no evidence the defect exists |

Task cards follow the shared task-card template (`templates/task-card.md` in
the planning tooling). Cards for this sprint live under
`docs/planning/tasks/26Q3-01/`.

## Sync Log

- **2026-10-09** — Sprint 3 closed at 40/43 pts (committed 10/10), goal not
  met, archived to `archive/sprint-03-26Q3-01.md`. Sprint 4 opened with dates
  TBD and scope cut to the Sprint 3 server runs plus `26Q3-HARN-20` Part 2.
  The pre-locked Gate B + Phase C content moves to Sprint 5; Sprints 5–7 are
  TBD until DoD 1's number exists. The 2026-09-30 verdict date has passed and
  needs a re-plan.

---

*Last Updated: 2026-10-09*
