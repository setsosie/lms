# Sprint archive

Closed sprints, one file per sprint. Naming: `sprint-<NN>-<quarter-folder>.md`,
where `<NN>` is the project-internal sprint number and `<quarter-folder>` matches
the task-card folder under `docs/planning/tasks/`.

Created 2026-08-12 during the Sprint 2 close. Sprint 1 was archived
retroactively at the same time — it was never formally closed, and its close-out
lived inside `current-sprint.md`.

## Sprints

| # | Folder | Sprint | Dates | Planned | Delivered | % | File |
|---|--------|--------|-------|--------:|----------:|--:|------|
| 1 | `26Q2-01` | ANT shakedown + local serving | 2026-06-16 → 2026-06-27 | 24 | 0 | 0% | [sprint-01-26Q2-01.md](sprint-01-26Q2-01.md) |
| 2 | `26Q3-01` | Make the harness incapable of lying | 2026-07-27 → 2026-08-07 | 41 | 19 | 46% | [sprint-02-26Q3-01.md](sprint-02-26Q3-01.md) |
| 3 | `26Q3-01` | Find out whether a calibratable slice exists | 2026-08-10 → 2026-09-04 | 43 | 40 | 93% | [sprint-03-26Q3-01.md](sprint-03-26Q3-01.md) |

## Velocity tracking

| Sprint | Planned | Delivered | % | Notes |
|--------|--------:|----------:|--:|-------|
| 1 | 24 | 0 | 0% | Scaffolded, never worked. No executable first step — the one task marked IN PROGRESS had a PENDING hard dependency |
| 2 | 41 | 19 | 46% | Planned at 24; 17 pts added mid-sprint, all real defects found by running on the box. Ran 5 days past its end date |
| 3 | 43 | 40 | 93% | Committed 10, all delivered by day 10; 33 pts of found-work added, 30 delivered. **Goal not met**: the DoD server runs never ran. Closed 35 days late, across a month with no box |

**Average delivered: 19.7 pts/sprint** over 3 sprints (0, 19, 40).

Three uneven data points do not make a forecast. `/sprint-forecast` should be
treated as indicative only until at least Sprint 4 closes. Sprint 3's 40 is
mostly found-work over a 61-day actual window; its committed scope was 10.
Sprint 2's 46% is also measured
against a denominator that grew 71% mid-sprint — against the *original* 24-pt
plan it delivered 11 pts of planned work, also 46%.

## Summary

| Metric | Value |
|--------|-------|
| Sprints closed | 3 |
| Total points planned | 108 |
| Total points delivered | 59 |
| Overall delivery rate | 55% |

## Key milestones

| Date | Milestone | Sprint |
|------|-----------|--------|
| 2026-08-10 | **The Lean oracle went live.** Mathlib + vLLM up on the 4×H100 box; first end-to-end run against a real verifier | 2 |
| 2026-08-10 | **First `verified_lean` artifact in project history** — `verifier.kind: real`, `mathlib_rev` pinned. Gate B-minus green | 2 |
| 2026-08-10 | **First observed cross-generational reuse** — a gen-2 agent cited a gen-1 result and its import elaborated (`shakedown_3x3_d`) | 2 |
| 2026-08-10 | **The mock verifier can no longer write `verified`** — the root cause of every retracted roadmap number is closed | 2 |
| 2026-08-19 | **All five machine gates exist in code** — T2/T4, novelty N0/N1, per-statement cost | 3 |
| 2026-08-20 | **Committee mode runs end to end on the box** (Checkpoint 8, `committee_smoke_e`); first verified committee reuse, uncited (`committee_yolo_a`) | 3 |
| 2026-09-09 | **The repo runs CI for the first time**, test suite and real Lean verifier | 3 |

Not yet reached: **Gate A** (archived mock-verified run must re-score to ~0
verified novel statements), **Gate B** (one artifact through all five machine
gates), and **a measured N1 density** for either ANT arc. Gate A and the
density are Sprint 4's server runs.

---

*Last Updated: 2026-10-09*
