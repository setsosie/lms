# Upcoming sprints & program roadmap

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

## Fixed quantities (high confidence)

| Quantity | Value | Source |
|---|---|---|
| Novel statements after Mathlib (all 3 texts) | ~11,600–14,000 | committee synthesis |
| Statements per WC cycle | ~50 | observed |
| **→ Total WC cycles** | **~230–280** | derived |
| Tokens per WC cycle | ~400–500K | observed |
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

## Timeline scenarios (estimate — pending shakedown calibration)

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

## Pre-locked next sprint (Sprint 2 — draft)

Promote into `current-sprint.md` at Sprint 1 close. Contents depend on shakedown
outcome:

- **If shakedown green** → open Phase 1 shared kernel: `26Q2-SK-01` spectral
  sequences (acid test), `26Q2-SK-02` fibred categories + descent, `26Q2-SK-03`
  core algebra — 3-way parallel.
- **Carryover (likely):** `26Q2-SPG-02` diversity gate (cluster), `26Q2-SPG-03`
  genome representation if not finished.
- **If shakedown reveals a pipeline problem** → a fix-the-pipeline sprint instead;
  do not advance to the shared kernel until metrics are healthy.

## Bottom line

If the shakedown proves the pipeline on local LLMs and the reuse transition fires,
**~3–4 months** is the realistic target for the full three-text program, with a
credible path to ~6–10 weeks and a downside of ~6–9 months if reuse stays linear
or the local model needs heavy escape-hatch support. This range gets replaced
with a real projection once `26Q2-ANT-01` reports its first local cycle metrics.
