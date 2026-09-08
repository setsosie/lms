# 0001 — Phase C runs a model × harness control, not a single-model population sweep

**Status**: Accepted
**Date**: 2026-09-04
**Deciders**: setsosie (solo)
**Tags**: scope, calibration, experiment-design

## Context

Phase C of the calibration program (`calibration-program.md` §3) was designed
as one ANT slice × {1, 3, 9} agents on the local Qwen model, and a
pre-commitment dated 2026-07-24 said that if the cluster was not serving by
2026-08-21, Phase C would run on API models under a $300 cap. The cluster is
serving; Sprint 3 flagged the pre-commitment as firing for the wrong reason.

On 2026-09-03 Anthropic reported a Lean formalization of Fermat's Last Theorem
by a swarm of frontier-model agents in 11 days. Their first harness, a Claude
Code multi-agent loop, failed the way this project's committee runs fail
(agents lost the project's state); the same model succeeded once the work moved
to the Prove2Me platform's statement DAG. The Prove2Me paper (arXiv 2608.28433
§7) explicitly leaves "holding the model fixed and varying only the harness" to
future work. The Sonnet-team simulation of 2026-08-20
(`docs/LESSONS_FROM_SONNET_SIMULATION.md`) reached 9/9 tags in five generations
against Qwen's 2/9 in ten, with model and harness confounded.

A Qwen-only Phase C therefore produces a number that cannot be attributed: a
bad CVFN reads as "the model is too weak" and "the harness is broken"
identically.

## Decision

Phase C runs the same slice, the same harness, and the same per-statement
accounting on two arms: the **local arm**, Qwen3.6-27B-FP8 at 1 and 9 agents,
and a **control arm**, one frontier API model at 9 agents (then 1 if budget
remains). The $300 API budget pays for the control arm. The 2026-09-30 verdict
date stands. The local-first policy is unchanged: the API arm is a control, not
the execution path.

## Alternatives considered

- **Original design, Qwen at {1, 3, 9}** — rejected: the result cannot separate
  model capability from harness quality, which after 2026-09-03 is the only
  question left open.
- **Fire the $300 fallback as written (swap Qwen for an API model)** — rejected:
  it was written for an unavailable cluster; the cluster is up, and a swap
  discards the weak-model result the project exists to produce.
- **No API arm** — rejected: without a frontier reference on the same slice, a
  zero from Qwen is uninterpretable.
- **Run the calibration as a Prove2Me mission** — rejected for Phase C: a
  shared, hosted platform gives no isolation of population size, no verifier
  provenance, and none of this project's gates. Its data model is borrowed
  instead (task `26Q3-HARN-25`).

## Consequences

**What this enables:**
- A CVFN with an attribution: model effect, population effect, and their
  interaction on one slice. It is the experiment the Prove2Me paper defers.

**What this costs:**
- The 3-agent local config is dropped (4M local tokens instead of 6M).
- Up to $300 of API spend; the control arm's 1-agent config is conditional on
  budget remaining.

**What this commits us to:**
- Per-statement cost accounting (`26Q3-HARN-05`) on both arms before Phase C
  starts, with tokens recorded per arm so the arms compare on tokens, not only
  dollars.
- Both arms through the same `Society` committee path, harness frozen at the
  2026-09-07 state of `main`.
- The Gate 4 caveat (`26Q3-HARN-20`) reported identically for both arms.

## Related

- `docs/planning/calibration-program.md` §3 Phase C, §6, §7 — amended 2026-09-04
- `docs/planning/upcoming-sprints.md` — Sprint 4 pre-lock
- [[0002]] — what the project is measuring now
- Anthropic, "Formalizing Fermat's Last Theorem" (2026-09); Chen, Marwaha, Lu,
  Yuen, Peng, "Prove2Me: An Open Collaborative Platform for Scaling Math
  Formalization" (arXiv 2608.28433)
