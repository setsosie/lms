# Sprint 3: Green Gate B — one artifact through all five machine gates

**Dates**: 2026-08-10 → 2026-08-21
**Quarter**: Q3 2026 · folder `26Q3-01`
**Program**: `docs/planning/calibration-program.md` — Phase B
**Sprint Goal**: One artifact clears all five machine gates end-to-end on the
cluster, with per-statement cost recorded. That is Gate B, and it is what makes
a CVFN numerator possible at all.

**Status**: 🔄 ACTIVE

> **Hard checkpoint: 2026-08-21.** If Gate B is not green by that date, the
> pre-committed API fallback fires — Phase C runs on API models with a hard $300
> cap and the local-first policy is formally suspended for the calibration only.
> This is a pre-commitment made on 2026-07-24, not a decision to make in the
> moment. See `calibration-program.md` §3, Phase B contingency.

> **Execution note**: the cluster is already up. Phase B day 1 is done — Mathlib
> compiles, vLLM serves `lms-generalist` on the box, and the loop closes. The
> remaining work is **local Python + Lean-LSP that Claude executes**; your cost
> is review plus running the Phase B smoke test on the box. See
> `docs/infrastructure/cluster-runbook-calibration.md`.

---

## Sprint 2 close-out (26Q3-01) — 19/41 pts, goal not met

| Metric | Value |
|--------|-------|
| Dates | 2026-07-27 → 2026-08-07 (closed 2026-08-12) |
| Planned points | 24, grown to 41 by mid-sprint additions |
| Delivered | **19** (46%) |
| Carried here | 22 pts across 7 tasks (+3 stretch) |
| Gate A | ❌ not run |

Full record: [`archive/sprint-02-26Q3-01.md`](archive/sprint-02-26Q3-01.md).

What landed: the Lean oracle went live and produced the first `verified_lean`
artifact in project history, and the cumulative-knowledge mechanism worked for
the first time ever (`-09`, `-10`, `-11`). What didn't: all three originally-
planned CRITICAL gate tasks — `26Q3-HARN-03`, `-04`, `-05`, 13 pts — were never
started, so the harness still cannot tell a formalization from a Mathlib
re-export. **Those three are the entire reason this sprint exists.**

Velocity: 9.5 pts/sprint over two sprints (0, then 19).

## Sprint 3 Summary

| Metric | Value |
|--------|-------|
| Committed points | 15 |
| Stretch | 8 |
| Carryover | 15 of 15 committed pts are Sprint 2 carryover |
| Track | Phase B — stand up the oracle, close the gates |
| Execution | Local Claude-executed work + one smoke test you run on the box |
| Status | 🔄 ACTIVE |

**Scope was cut deliberately.** 22 pts carried out of Sprint 2 against an
observed 9.5 pts/sprint velocity. Committing all 22 would repeat Sprint 2's
mistake of a plan nobody defends. The four committed tasks are exactly the ones
on the Gate B critical path; everything else is stretch or deferred.

## Committed — the Gate B critical path

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q3-HARN-13: Verify an artifact in the namespace it will be stored in | 2 | HIGH | HARN | 🔲 PENDING | card in #30, no impl | **Do this first — cheapest, and it is a false negative in the oracle.** `real.py:194` verifies at top level; `foundation.py:138` stores inside `namespace LMS.Foundation`. Suppresses the CVFN numerator directly |
| 26Q3-HARN-03: T2 / T4 machine gates | 5 | CRITICAL | HARN | 🔲 PENDING | — | Gate 2 (no `sorry` / new `axiom` / `native_decide`) + Gate 3 (non-vacuity; reject trivial `example`). Move the `sorry` check **post-compile**. T2 must catch axiom-free `structure Category` and Mathlib re-definitions |
| 26Q3-HARN-04: Novelty classifier (N0 / N1) | 5 | CRITICAL | HARN | 🔲 PENDING | — | Gate 4. Mathlib name search + `exact?`/`loogle` via lean-lsp MCP. **Every artifact produced to date is N0.** Also unblocks the Phase C slice decision |
| 26Q3-HARN-05: Per-statement cost accounting | 3 | HIGH | HARN | 🔲 PENDING | — | Tokens + wall-clock per statement including failed attempts. **This is the CVFN numerator.** Fixes `society.py:356` counting `len(response.attempts)` as artifacts created |

## Stretch

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q3-CAL-01: Select the ANT slice by measured N1 density | 3 | HIGH | CAL | 🔲 PENDING | — | **Card not yet written.** Hard-depends on `-04`. Resolves the open decision in `calibration-program.md` §4 — the committed Ch. I core arc is probably ~all N0, which would make CVFN undefined. Run the classifier over both candidate arcs; do not decide from memory of Mathlib |
| 26Q3-HARN-12: Make the committee architecture reachable + review stage | 3 | HIGH | HARN | 🔲 PENDING | card + verify script uncommitted | `PlanningPanel` / `WorkingGroup` / `DependencyGraph` exist and pass tests but no CLI path reaches them. Iterative mode hardcodes `reviews_total=0`, so the collective and the feedback loop are mutually exclusive. **Wire, don't rebuild** |
| 26Q3-INFRA-02: Per-request token cap is configurable | 2 | HIGH | INFRA | 🔲 PENDING | — | Issue #17. Hardcoded Claude-shaped 64k `max_tokens`, no env lever. Worked around in the runbook |

## Deferred — not this sprint

| Task | Points | Why |
|------|--------|-----|
| 26Q3-HARN-08: Agents emit Lean 3, not Lean 4 | 2 | **Re-evidence before building.** `shakedown_3x3_d` showed zero Lean 3 syntax and zero missing-Mathlib errors across 4/4 artifacts. The card may describe a defect that no longer exists |
| 26Q3-HARN-06: D4 side-by-side review view | 3 | Not needed until Phase D (2026-09-14). Build it in Sprint 4 |

## Definition of Done

The sprint is done when all four hold:

1. **Gate B green** — the Phase B smoke test passes: 1 agent, 1 statement, real
   verification, one artifact through all five machine gates, on the cluster.
2. **Gate A run** — carried from Sprint 2, and it blocks Phase C. Re-run the
   archived `experiments/stacks_ch4_phase1/artifacts.json` through the rebuilt
   pipeline. **It must now report ~0 verified novel statements** (it currently
   reports 48/52 = 92%). If the rebuilt gates still score that run highly, the
   gates do not work and Phase C does not start.
3. **Secondary check** — `experiments/run_20251218_105831` (15 agents, 8.99M
   tokens) reports its ~2 verified artifacts as **N0**, with a populated
   gate-failure histogram over the other 73.
4. **One real CVFN reading exists** — even if the numerator is 0, the
   denominator and the gate-failure histogram are recorded per statement.

Read the gate histogram's *shape*, not its total — it is the graded signal, and
it is what the deferred evolutionary arm would eventually optimize against.

## Risk register

| Risk | Mitigation | Task |
|------|------------|------|
| **15 committed pts against a 9.5 pts/sprint velocity, with 9 days left in the window** | Deliberate. Aug 21 is a fixed external date, and the $300 API fallback is the pre-committed mitigation if it slips. `-13` first so the cheapest numerator unblock lands early | — |
| Novelty classifier is unreliable (Mathlib search is fuzzy) | Report N0/N1 with a confidence field; low-confidence routes to D4 human review rather than being counted | 26Q3-HARN-04 |
| Non-vacuity checking is undecidable in general | Implement the tractable subset: reject `example` with no new named declaration, reject statements whose hypotheses are unsatisfiable by a witness search. Log what it can't decide | 26Q3-HARN-03 |
| Gate A is skipped again | It was Sprint 2's exit criterion and went unrun. It is now a numbered DoD item that blocks Phase C | — |
| Mid-sprint box findings displace the critical path, as in Sprint 2 | New defects get carded and go to Sprint 4 unless they block Gate B. The four committed rows are not negotiable against found-work | — |
| Zero N1 in every candidate slice | That is a **result**, not a failure — "CVFN undefined, thesis untestable at current capability." Proceed to Phase E and write it up | 26Q3-CAL-01 |

## Next sprint (Sprint 4, 2026-08-24 → 2026-09-11) — pre-lock

Phase C of the calibration program — **the calibration run itself**.

- One fixed ~20-statement ANT slice, selected by measured N1 density
  (`26Q3-CAL-01`).
- Three population sizes: **1 agent, 3 agents, 9 agents**. Same slice, same
  per-config token budget, same model. Yields CVFN *and* the population-size
  comparison from the same spend.
- **Budget: 2M tokens per config, 6M total, hard-capped.** If a config exhausts
  its budget at zero verified statements, that is a result — record and stop.
- **Gate C**: ≥1 statement clears all five machine gates in at least one config.
- Carries `26Q3-HARN-06` (D4 review view) — Phase D starts 2026-09-14.

## Sync Log

- **2026-08-12** — Sprint 2 closed at 19/41 pts and archived to
  `archive/sprint-02-26Q3-01.md`; Sprint 1 archived retroactively to
  `archive/sprint-01-26Q2-01.md`. Sprint 3 opened at 15 committed + 8 stretch,
  scoped down from 22 pts of carryover to the Gate B critical path.
  Corrected during the close: `26Q3-HARN-11` was recorded 🔄 IN PROGRESS with
  "#28 (open)" by the same-day sync, but #28 merged at 20:47 — an hour before
  the sync commit landed. Counted as delivered (3 pts, 16 → 19).
  `26Q3-HARN-13` stays carryover: #30 merged its card and verify stub only.

---

*Last Updated: 2026-08-12*
