# Upcoming sprints & program roadmap

> ## ⚠️ SUSPENDED 2026-07-24 — do not cite the numbers below
>
> A repo audit found that this roadmap's governing inputs — **~50 statements per
> WC cycle** and **~400–500K tokens per cycle**, both marked "observed" in
> *Fixed quantities (high confidence)* — derive from `experiments/stacks_ch4_phase1`,
> a run verified by `MockLeanVerifier`'s regex rather than by Lean. Every run that
> used a real verifier landed at **0–6% verification** and **0.0 reuse**, and
> produced only N0 (already-in-Mathlib) content. The 15-agent run spent **9.0M
> tokens for ~2 verified artifacts**.
>
> Consequently the derived quantities — ~230–280 WC cycles, the cost figures, all
> three timeline scenarios, and the "~3–4 months" bottom line — are **unsupported**.
> They are not known to be wrong; they are known to rest on nothing.
>
> Evidence: `docs/planning/2026-07-24-feasibility-assessment.md`.
> Replacement plan: `docs/planning/calibration-program.md` — 10 weeks to one
> measured cost-per-verified-faithful-novel statement (CVFN), which regenerates
> this file's inputs or kills the program.
>
> **The phased roadmap and critical path below remain structurally sound** — the
> dependency order (kernel → ART+CNF → Stacks) is unaffected by the calibration
> error. It is only the rates and dates that are suspended.

Forward planning for LMS. The committed, parsed sprint is always
`current-sprint.md`; this file is the broad multi-sprint roadmap and the
next-sprint pre-lock. Generated 2026-06-18.

> **Execution policy (applies to the whole roadmap): local-first.** All agents
> run on self-hosted OSS models; model APIs are a deliberate escape hatch used
> only when local hits a defined roadblock. See `current-sprint.md` →
> *Execution environment*. Consequence: **$ ≈ GPU box wall-clock (≈ $0 if you own
> the hardware); the binding constraint is throughput, not token price.** API
> spend is incurred only on hard sub-tasks routed out by the escape hatch.

---

## The honest headline

Total **effort** is fairly well-known; **wall-clock is not yet calibrated** —
and calibrating it is exactly what the ANT shakedown (`26Q2-ANT-01`) is for.
Until one real WC cycle on the *local* stack reports tokens + wall-clock + lemma-
reuse, every calendar date below is an estimate on assumed throughput. Fixed
quantities are stated precisely; time quantities are scenario ranges.

## Fixed quantities (high confidence) — ⚠️ two rows retracted

| Quantity | Value | Source |
|---|---|---|
| Novel statements after Mathlib (all 3 texts) | ~11,600–14,000 | committee synthesis — **stands** (independent of the calibration error) |
| ~~Statements per WC cycle~~ | ~~~50~~ | ❌ **RETRACTED** — from a mock-verified run; real-verifier runs produced 0–2 per run, all N0 |
| ~~**→ Total WC cycles**~~ | ~~**~230–280**~~ | ❌ **RETRACTED** — derived from the retracted row above |
| ~~Tokens per WC cycle~~ | ~~~400–500K~~ | ❌ **RETRACTED** — real-verifier runs ranged 446K–8.99M per run |
| Formalization cost — **local-first** | ≈ box wall-clock (~$0 if owned; ~$1.4–3.4K if renting @ $12/hr) | cost-model config B |
| Formalization cost — *if it were all API* | ~$2,000–3,000 (Opus) | reference only |
| Genome arm, full 500-gen run (self-hosted) | ~$1,700 / ~142 wall-hrs | `scratch/cost_model.py` |

## Phased roadmap (gated milestones)

| Phase | Scope | WC cycles | Gate to advance |
|---|---|---|---|
| **0 — Shakedown** *(current)* | INFRA-01 local serving → ANT Ch. I core arc on local LLM | 2–3 | Pipeline metrics healthy (not math coverage) → **calibrates everything below** |
| **1 — Shared kernel** | Spectral sequences + fibred cats + core algebra (3-way parallel) | ~7–10 (~350–500 stmts) | Spectral sequences compile = acid test; unblocks all 3 texts |
| **2 — ART + CNF** | 1,383 + 800 stmts (post-Mathlib); p-adic bridge shared | ~30–45 | CNF needs OCR of `cnf_2nd.pdf` first |
| **3 — Stacks** | ~85% of the corpus, the long tail | ~180–220 | bulk parallel formalization; watch for the reuse phase transition |
| **∥ Genome arm** | SPG: INFRA serving → diversity gate → evolution → QD | independent | each gate; off the formalization critical path |

## Timeline scenarios — ⚠️ VOID

> All three scenarios below multiply the retracted "~230–280 cycles" figure by an
> assumed concurrency. Both factors are unmeasured, so the products carry no
> information. Retained only to show what was assumed. **The `~3–4 months`
> headline came from this table and should not be repeated anywhere.**
>
> Note also that the "high lemma-reuse" premise of the Aggressive column has
> **measured reuse of 0.0** in every real-verified run to date. The phase
> transition has not merely failed to fire; at a ~3% per-statement success rate
> there is not yet enough verified material for reuse to be a meaningful quantity.

Free variable: **parallel WCs × wall-clock per cycle**, both of which the
shakedown measures. The project thesis (population size matters) argues for high
parallelism — and local-first means parallelism is capped by GPU box capacity,
not by API rate limits or budget.

| Scenario | Parallel WCs | Effective rate | ~230–280 cycles → |
|---|---|---|---|
| **Conservative** | 3–4 concurrent, slow compile-debug loops | ~8–12 cycles/wk | **~6–9 months** |
| **Moderate** | 6–8 concurrent | ~20–30 cycles/wk | **~3–4 months** |
| **Aggressive** | 12+ concurrent, high lemma-reuse | ~40–50 cycles/wk | **~6–10 weeks** |

**The wildcard is the phase transition.** Core hypothesis: lemma-reuse rate jumps
discontinuously past a critical mass. If it fires during Stacks (Phase 3, the
tail), the aggressive column becomes realistic and the back ~180 cycles go much
faster than the front. If it never fires (Tasmania effect), conservative holds.
**Phase 3 is where both the science and the schedule are decided.**

Local-first caveat on the schedule: if the local model is materially weaker than
Opus, expect *more* compile-debug iterations per cycle (slower wall-clock) and
more escape-hatch routing on hard statements. The shakedown will quantify this
penalty directly — it's the first thing its metrics will show.

## Critical path

```
INFRA-01 ─→ ANT shakedown ─→ Shared kernel (spectral seq) ─→ ART+CNF ─→ Stacks (85%, the tail)
[local serve]  [calibrates]      [unblocks all 3]                        [phase transition?]
                    │
   genome: INFRA serving → SPG-01 → SPG-02 gate ─→ evolution   (parallel, off critical path)
```

- **Nearest hard dependency:** spectral sequences (zero Mathlib coverage) gates
  all three texts — first real math risk after the shakedown.
- **Queue early (async):** OCR of `cnf_2nd.pdf` — blocks the CNF half of Phase 2.
- **Not the constraint:** money. **The constraints are wall-clock throughput, the
  reuse phase-transition, and local-model capability** (escape hatch absorbs the last).

## Pre-locked next sprint (Sprint 2 — draft) — ✅ RESOLVED 2026-07-24

This section offered three branches. **The third one fired.**

> *"If shakedown reveals a pipeline problem → a fix-the-pipeline sprint instead;
> do not advance to the shared kernel until metrics are healthy."*

The audit found the pipeline problem before the shakedown ran: verification
itself was not trustworthy, so "metrics are healthy" could not have been assessed.
Sprint 2 is therefore the fix-the-pipeline sprint — see `current-sprint.md`.
Credit where due: this file's own gating logic called it correctly.

Deferred branches, unchanged and still queued behind a green calibration:

- Phase 1 shared kernel: `SK-01` spectral sequences (acid test), `SK-02` fibred
  categories + descent, `SK-03` core algebra — 3-way parallel.
- Genome arm `SPG-01/02/03` — pushed to Q4. Its fitness function *is* proof
  success; evolving against a signal that is ~0 for every genome is a flat
  landscape by construction.

## Bottom line — ⚠️ REPLACED 2026-07-24

~~If the shakedown proves the pipeline on local LLMs and the reuse transition fires,
**~3–4 months** is the realistic target for the full three-text program, with a
credible path to ~6–10 weeks and a downside of ~6–9 months if reuse stays linear
or the local model needs heavy escape-hatch support.~~

**Current bottom line**: there is no defensible timeline for the three-text
program, because there is not yet a single measured data point on the cost of a
verified *novel* statement. The Q3 calibration program exists to produce exactly
one number (CVFN) by **2026-09-30**. Three outcomes, all acceptable:

| Outcome | Consequence |
|---|---|
| CVFN decent | Re-derive this file's rates from real inputs and restart the program |
| CVFN bad but finite | Pipeline works, is uneconomic → scope to one tractable target, publish the calibration |
| CVFN undefined (zero N1) | Thesis untestable at current capability → write it up and shelve |

Until then this file forecasts nothing. See `docs/planning/calibration-program.md`.
