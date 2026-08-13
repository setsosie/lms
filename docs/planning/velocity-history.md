# Velocity History — lms

<!--
One row per CLOSED sprint. Maintained by `/retro --write`. Read by `/sprint-forecast`.

Columns:
  - Sprint:    The sprint identifier as it appears in the archive (e.g., "Sprint 7", "Sprint 4 (Q2)", "Org Sprint 3").
  - Dates:     Start → end as absolute ISO dates (YYYY-MM-DD → YYYY-MM-DD).
  - Type:      One of {code, mixed, experiment}.
                 code       — no GPU/experiment workload; pure software work
                 mixed      — code + some experiments
                 experiment — primary deliverable is experiment results
  - Planned:   Sum of points planned at sprint start.
  - Completed: Sum of points actually delivered (PR merged or artifact produced).
  - Carryover: Points that rolled to the next sprint.
  - GPU-Hours: Compute spent during the sprint (— if not tracked by this repo).
  - Retro:     Link to the retro file for this sprint (relative path). Retros are
               local-only and gitignored, so this link resolves on the machine that
               ran `/retro --write` and nowhere else. This file is the tracked
               record of the sprint's numbers; the retro is not.

Idempotency: re-running `/retro --write` for the same sprint number updates the
existing row in place. Do not hand-edit unless you know what you're doing —
`/sprint-forecast` reads this file as its primary velocity input.

Sort: oldest → newest. New rows append to the bottom.
-->

| Sprint | Dates | Type | Planned | Completed | Carryover | GPU-Hours | Retro |
|--------|-------|------|---------|-----------|-----------|-----------|-------|
| Sprint 1 | 2026-06-16 → 2026-06-27 | code | 24 | 0 | 3 | — | — |
| Sprint 2 | 2026-07-27 → 2026-08-12 | code | 41 | 19 | 22 | — | [link](retros/sprint-02-retro.md) |

<!--
Repo-specific notes, so `/sprint-forecast` isn't misled:

- Sprint 1 delivered nothing. It was scaffolded and never worked — the one task
  marked IN PROGRESS had a PENDING hard dependency, so there was no executable
  first step. A real zero, not a missing measurement.
- Sprint 2's `Planned` is 41, the table at close, not the 24 committed at sprint
  start. 17 pts were added mid-sprint, all real defects found by running the
  harness on the cluster. Completed + Carryover reconciles against 41.
- Sprint 2's end date is its actual close (2026-08-12), not its nominal one
  (2026-08-07). 19 pts took 17 days, not 12.
- GPU-Hours is untracked repo-wide. Cluster time was spent this sprint
  (shakedown runs on the 4×H100) but no task records it, so the column stays —.
  Both sprints classify as `code`: no task is experiment-tagged and 0% of
  delivered points are experiment points.
-->

<!-- Generated/maintained by /retro --write. Last updated: 2026-08-12 -->
