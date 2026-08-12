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

## Velocity tracking

| Sprint | Planned | Delivered | % | Notes |
|--------|--------:|----------:|--:|-------|
| 1 | 24 | 0 | 0% | Scaffolded, never worked. No executable first step — the one task marked IN PROGRESS had a PENDING hard dependency |
| 2 | 41 | 19 | 46% | Planned at 24; 17 pts added mid-sprint, all real defects found by running on the box. Ran 5 days past its end date |

**Average delivered: 9.5 pts/sprint** over 2 sprints.

Two data points do not make a forecast. `/sprint-forecast` should be treated as
indicative only until at least Sprint 4 closes. Sprint 2's 46% is also measured
against a denominator that grew 71% mid-sprint — against the *original* 24-pt
plan it delivered 11 pts of planned work, also 46%.

## Summary

| Metric | Value |
|--------|-------|
| Sprints closed | 2 |
| Total points planned | 65 |
| Total points delivered | 19 |
| Overall delivery rate | 29% |

## Key milestones

| Date | Milestone | Sprint |
|------|-----------|--------|
| 2026-08-10 | **The Lean oracle went live.** Mathlib + vLLM up on the 4×H100 box; first end-to-end run against a real verifier | 2 |
| 2026-08-10 | **First `verified_lean` artifact in project history** — `verifier.kind: real`, `mathlib_rev` pinned. Gate B-minus green | 2 |
| 2026-08-10 | **First observed cross-generational reuse** — a gen-2 agent cited a gen-1 result and its import elaborated (`shakedown_3x3_d`) | 2 |
| 2026-08-10 | **The mock verifier can no longer write `verified`** — the root cause of every retracted roadmap number is closed | 2 |

Not yet reached: **Gate A** (archived mock-verified run must re-score to ~0
verified novel statements) and **Gate B** (one artifact through all five machine
gates). Both carry into Sprint 3.

---

*Last Updated: 2026-08-12*
