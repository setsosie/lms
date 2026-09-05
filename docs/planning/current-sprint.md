# Sprint 3: Find out whether a calibratable slice exists

**Dates**: 2026-08-10 → 2026-09-04
**Quarter**: Q3 2026 · folder `26Q3-01`
**Program**: `docs/planning/calibration-program.md` — Phase B
**Sprint Goal**: Answer the open decision in `calibration-program.md` §4 with a
measurement, not a guess: **which ANT slice has enough N1 density to calibrate
against, and is that number greater than zero?** If it is zero everywhere, the
program ends early and honestly, and Phase C's 6M tokens are never spent.

**Status**: 🔄 ACTIVE

> **Light sprint, long window.** 10 committed points over 26 days. Capacity
> — not scope — is the binding constraint this cycle: the week straddling
> August and September is expected to be near-idle, and the two weeks before it
> are thin. The point total reflects what will actually get done, not what the
> calendar could theoretically hold. Sprint 2 delivered 19 pts in three bursts
> separated by 12 dead days; planning around real capacity is the correction.

> **Every sprint runs something on the server.** This sprint's server run is the
> novelty classifier over both ANT candidate arcs on the 4×H100 (DoD item 1). A
> sprint with no run on the box is a sprint whose assumptions went unchecked —
> every consequential defect found in Sprint 2 came from running the thing, not
> from reading it.

---

## Scope change 2026-08-12: this sprint cannot green Gate B

Gate B requires one artifact through **all five** machine gates, which requires
`26Q3-HARN-03` (T2/T4, 5 pts) *and* `26Q3-HARN-04` (novelty, 5 pts) *and*
`26Q3-HARN-13` (2 pts) — 12 pts before any cost accounting. At a 10-pt ceiling,
Gate B does not fit. Rather than commit 12 and miss, the sprint is re-pointed at
the single highest-information question available for 10.

**Gate B moves to Sprint 4.** `26Q3-HARN-03` moves with it.

Why the slice decision is the right thing to buy instead: if the candidate arcs
are ~entirely N0 — which `calibration-program.md` §4 explicitly suspects, because
Minkowski theory, class-number finiteness and Dirichlet units are all in Mathlib
— then **CVFN is undefined by construction** and no amount of gate work changes
that. Learning this before Phase C saves the 6M-token run. Learning it after
wastes the quarter.

## ⚠️ The 2026-08-21 checkpoint needs your decision

The pre-commitment made 2026-07-24 reads: *if Gate B is not green by 2026-08-21,
Phase C runs on API models with a hard $300 cap.* On this plan **Gate B will not
be green on 2026-08-21**, so the fallback fires by its own terms.

But it was written against the wrong risk. The fallback swaps the *serving stack*
— it is the mitigation for "the cluster is unavailable, FLAIME has P0 priority."
The cluster is up and the oracle works. The actual constraint is **bandwidth**,
and API models do not add bandwidth. Firing it as written would spend $300 and
change nothing.

Recommendation: re-point the checkpoint at **2026-09-04** (this sprint's end) and
restate it as a *decision* rather than a *swap* — at Sprint 3 close, if the N1
density measurement has not happened, the Sep 30 verdict is no longer reachable
and the program should be re-planned or shelved rather than rushed. Keep the $300
as available-on-demand for hard sub-tasks, not date-triggered.

**This is your pre-commitment to change, not mine.** Recorded here as an open
decision; nothing has been edited in `calibration-program.md`.

> **Resolved 2026-09-04** — `decisions/0001-phase-c-model-harness-control.md`.
> The checkpoint is re-pointed as a *control*, not a swap: Phase C runs the
> local model and one frontier model on the same slice and harness, and the
> $300 pays for the frontier arm. `calibration-program.md` §3 is amended.

## Sprint 2 close-out (26Q3-01) — 19/41 pts, goal not met

| Metric | Value |
|--------|-------|
| Dates | 2026-07-27 → 2026-08-07 (closed 2026-08-12) |
| Delivered | **19** (46%) |
| Carried here | 22 pts across 7 tasks (+3 stretch) |
| Gate A | ❌ not run |

Full record: [`archive/sprint-02-26Q3-01.md`](archive/sprint-02-26Q3-01.md).
Retro: `retros/sprint-02-retro.md` (local-only).

The Lean oracle went live and produced the first `verified_lean` artifact in
project history; the cumulative-knowledge mechanism worked for the first time
ever. But all three originally-planned CRITICAL gate tasks — `-03`, `-04`, `-05`,
13 pts — were never started, and 17 pts of found-work displaced them.

Velocity: 9.5 pts/sprint over two sprints (0, then 19).

## Sprint 3 Summary

| Metric | Value |
|--------|-------|
| Committed points | 10 |
| Stretch | 0 — deliberately none |
| Carryover | all 10 committed pts are Sprint 2 carryover |
| Window | 26 days, low expected capacity |
| Track | Phase B — resolve the slice decision, keep the oracle honest |
| Status | 🔄 ACTIVE |

**No stretch rows this sprint.** Stretch is what let Sprint 2's scope grow 71%
mid-flight. If the committed 10 land early, pull `26Q3-HARN-03` forward from
Sprint 4 as an explicit decision, not as a row that was sitting there.

## Committed

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q3-HARN-13: Verify an artifact in the namespace it will be stored in | 2 | HIGH | HARN | 🔲 PENDING | card in #30, no impl | **Do this first.** `real.py:194` verifies at top level; `foundation.py:138` stores inside `namespace LMS.Foundation`. A false negative in the oracle — it suppresses the CVFN numerator directly, and it is the cheapest thing on the board |
| 26Q3-HARN-04: Novelty classifier (N0 / N1) | 5 | CRITICAL | HARN | 🔲 PENDING | — | Gate 4. Mathlib name search + `exact?`/`loogle` via lean-lsp MCP. **Every artifact produced to date is N0.** Its acceptance test is DoD item 1 — the classifier must run on the box over both candidate arcs, not just pass unit tests |
| 26Q3-HARN-05: Per-statement cost accounting | 3 | HIGH | HARN | 🔲 PENDING | — | Tokens + wall-clock per statement including failed attempts. **This is the CVFN denominator**, and it is what makes the Sprint 4 calibration run interpretable. Fixes `society.py:356` counting `len(response.attempts)` as artifacts created |

## Definition of Done

1. **The server run happened** — the novelty classifier ran on the 4×H100 over
   *both* ANT Ch. I candidate arcs (the committed core arc: integrality →
   Minkowski → class number → units; and the ramification arc: extensions of
   Dedekind domains, Hilbert ramification theory, different and discriminant).
   **Measured N1 density is recorded for each.** This resolves
   `calibration-program.md` §4 and is the input Sprint 4's calibration run needs.
2. **`26Q3-HARN-13` landed** — artifacts verify in the namespace they are stored
   in, so the oracle stops emitting false negatives.
3. **Gate A run to the extent `-04` allows** — every artifact in the archived
   `experiments/stacks_ch4_phase1/artifacts.json` classified N0/N1. It currently
   reports 48/52 verified; the N0 count must account for essentially all of them.
   The T2/T4 half of Gate A defers to Sprint 4 with `-03`.
4. **A CVFN denominator exists** — tokens and wall-clock attributed per
   statement, including failed attempts, on at least one real run.

Read the gate histogram's *shape*, not its total.

## Risk register

| Risk | Mitigation | Task |
|------|------------|------|
| **The Sep 30 verdict date is now at genuine risk** | Phase C compresses to Sprint 4 (2026-09-07 → 2026-09-18), Phase D to 2026-09-21 → 2026-09-25, Phase E to 2026-09-28 → 2026-09-30. There is no slack left. If Sprint 3 slips past 2026-09-04, the verdict date moves — say so then rather than compressing D and E further | — |
| Both candidate arcs measure ~zero N1 | **That is the sprint succeeding, not failing.** It means CVFN is undefined at current scope, and the honest move is Phase E early: write up "thesis untestable at current capability" and shelve. Do not respond by widening the slice until something scores | 26Q3-HARN-04 |
| Novelty classifier is unreliable (Mathlib search is fuzzy) | Report N0/N1 with a confidence field; low-confidence routes to D4 human review rather than being counted. On a *density* measurement, systematic bias matters more than per-item error — report the confidence distribution, not just the mean | 26Q3-HARN-04 |
| Found-work displaces the committed 10, as it displaced Sprint 2's 13 | New box defects get carded to Sprint 4 unless they block DoD item 1. Three committed rows is few enough to hold in mind — that is the point of a light sprint | — |
| The near-idle straddle week absorbs the whole sprint | Front-load: `-13` is 2 pts and unblocks the oracle; `-04` is the only row whose acceptance test needs the box. Aim to have DoD item 1 done before 2026-08-28 | — |

## Next sprint (Sprint 4, 2026-09-07 → 2026-09-18) — pre-lock

Back to normal length. Phase C — **the calibration run** — plus the Gate B work
this sprint could not fit.

- `26Q3-HARN-03` (5) — T2/T4 machine gates, carried from Sprint 3's scope cut.
  Gate B cannot go green without it.
- **Gate B** — one artifact through all five machine gates, end-to-end on the box.
- **Phase C** — the slice chosen by Sprint 3's measured N1 density, run as
  two arms on one frozen harness (ADR 0001, 2026-09-04): Qwen at **1 and 9
  agents**, 2M tokens each, hard-capped; one frontier control at 9 (then 1 if
  budget remains), capped at $300. ~~1, 3 and 9 agents on one model~~. This is
  the first direct test of the project's core hypothesis that population size
  matters, with the model-vs-harness attribution the single-model design
  could not give.
- **Gate C** — ≥1 statement clears all five machine gates in at least one config.
  Zero across all configs is a legitimate, publishable go/no-go outcome.
- `26Q3-HARN-25` (5) — proof sketches: open child lemmas become leaves in the
  task graph. Candidate; does not gate Phase C (harness frozen 2026-09-07).
- **The DAG phase** (`docs/planning/dag-phase.md`, 2026-09-05): `26Q3-HARN-27`
  (3, source-anchored dependency edges — **decide before 2026-09-07** whether
  it enters the frozen baseline; every goal's graph is a chain today),
  `26Q3-HARN-28` (5, theorem cards + statement pinning + ANT arc → goal
  converter), `26Q3-HARN-26` (5, closing leaves, Part 2 of `-25`). `-28`/`-26`
  are the post-freeze harness variant, run on the same slice and budget after
  the two frozen arms.

Sizing note: `-03` plus Gate B plus Phase C already fills a normal sprint; the
DAG-phase cards are 13 points on top and will not all fit. The 3-agent config
was cut by ADR 0001. Decide the rest at Sprint 4 planning with Sprint 3's N1
numbers in hand.

## Deferred — not this sprint

| Task | Points | Why |
|------|--------|-----|
| 26Q3-HARN-03: T2 / T4 machine gates | 5 | Moved to Sprint 4 with Gate B. Cut to fit the 10-pt ceiling |
| 26Q3-HARN-12: committee architecture reachable + review stage | 3 | Real (iterative mode has no peer-review phase at all), but it does not block CVFN. Card and verify script sit uncommitted in the working tree |
| 26Q3-INFRA-02: per-request token cap configurable | 2 | Issue #17. Worked around in the runbook; revisit if Phase C hits it |
| 26Q3-HARN-08: agents emit Lean 3, not Lean 4 | 2 | **Re-evidence or close.** `shakedown_3x3_d` showed zero Lean 3 syntax across 4/4 artifacts — the card may describe a defect that no longer exists |
| 26Q3-HARN-06: D4 side-by-side review view | 3 | Needed before Phase D (2026-09-21 under the revised calendar). Sprint 4 |
| 26Q3-CAL-01: select the ANT slice | 3 | **Absorbed into this sprint's DoD item 1.** The card is unnecessary — it was always `-04`'s acceptance test |

## Sync Log

- **2026-09-05** — **The DAG phase** designed (`docs/planning/dag-phase.md`)
  as the harness-side response to Prove2Me and the Stacks dependency
  structure. Finding while designing it: `DependencyGraph.from_goal` infers a
  total order, so every registered goal has exactly **one** AVAILABLE task at
  any time — no committee run has ever allocated two distinct statements in a
  generation, and "N agents converge on one tag" was the graph, not the
  agents. The Stacks source's `\ref` cross-references give the real edges
  (Track B and A go from 1 root to 18 and 16). Three cards: `26Q3-HARN-27`
  (edges, 3), `26Q3-HARN-28` (theorem cards + pinning + arc → goal converter,
  5), `26Q3-HARN-26` (closing leaves, 5, Part 2 of `-25`). Sprint 4 pre-lock
  above amended to ADR 0001's two-arm design. **Open for the user**: whether
  `-27` lands before the 2026-09-07 harness freeze. Sprint 3 not closed in
  this entry.

- **2026-09-04** — Response to Anthropic's FLT formalization post (2026-09-03)
  and the Prove2Me paper. Two decision records opened in
  `docs/planning/decisions/`: **0001** (Accepted) re-points the 2026-08-21
  checkpoint as a model × harness control — Phase C becomes Qwen at {1, 9} plus
  one frontier arm at {9, then 1}, the $300 funds the control, Sep 30 stands;
  **0002** (Proposed) restates the research question as measurement on open
  models and drops the three-text program as a production target. README and
  CLAUDE.md reframed accordingly. New card `26Q3-HARN-25` (proof sketches, 5
  pts) added to the Sprint 4 pre-lock as a candidate. Sprint 3 not closed in
  this entry.

- **2026-08-12 (second entry)** — Sprint 3 rescoped after a capacity review.
  Window extended 2026-08-21 → **2026-09-04**; committed points cut 15 → **10**;
  stretch rows removed entirely. `26Q3-HARN-03` (5) deferred to Sprint 4, which
  moves **Gate B to Sprint 4** — 10 pts cannot cover the five machine gates.
  Sprint goal re-pointed from "green Gate B" to "resolve the §4 slice decision
  with a measurement", the highest-information 10 points available.
  Standing rule adopted: **every sprint runs something on the server.** This
  sprint's is DoD item 1.
  Two things the user should decide: the 2026-08-21 API-fallback pre-commitment
  (see the flagged section above — it will fire by its own terms and would not
  help), and whether the 2026-09-30 verdict date survives a Phase C that now
  starts 2026-09-07.

- **2026-08-12** — Sprint 2 closed at 19/41 pts and archived to
  `archive/sprint-02-26Q3-01.md`; Sprint 1 archived retroactively to
  `archive/sprint-01-26Q2-01.md`. Sprint 3 opened.
  Corrected during the close: `26Q3-HARN-11` was recorded 🔄 IN PROGRESS with
  "#28 (open)" by the same-day sync, but #28 merged at 20:47 — an hour before
  the sync commit landed. Counted as delivered (3 pts, 16 → 19).
  `26Q3-HARN-13` stays carryover: #30 merged its card and verify stub only.

---

*Last Updated: 2026-08-12*
