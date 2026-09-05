# LMS: LLM Mathematical Society

A measurement instrument for collective formalization: does coordination
structure let a population of *weak* LLM agents accumulate Lean-verified
mathematics, and does what they accumulate ratchet or merely pile up?

## What changed in September 2026

On 2026-09-03 Anthropic reported that a swarm of frontier-model agents produced
a Lean 4 proof of Fermat's Last Theorem in 11 days (13M lines, ~30K theorems,
Lean's three standard axioms). Their first harness — a Claude Code multi-agent
loop — failed the way this project's runs fail: agents lost track of the
project's state. The same model succeeded once the work moved onto Prove2Me's
immutable statement DAG, with proof sketches and search-before-propose.

Whether LLM collectives with a proof assistant *can* formalize research
mathematics is therefore no longer this project's question. See
`docs/planning/decisions/0002-lms-measures-collective-formalization.md`.

## The question that survives

Drawing on Henrich's collective-brain theory — population size and network
structure over individual capability — with Lean as the cultural ratchet:

1. **Structure vs capability.** Held to one slice and one harness, how much of
   the outcome is the model and how much is the coordination structure? The
   Prove2Me paper leaves "hold the model fixed, vary the harness" to future
   work; this project runs it on open models on owned hardware.
2. **Ratchet or pile.** The FLT proof is over five times Mathlib's size with no
   consolidation. Do reuse rate, duplication, and proof size per theorem
   improve over generations, or only accumulate?
3. **Minimum viable culture.** Below what population does accumulation fail?
   The metric is `potential_ratchet_failure`, after Tomasello's cultural
   ratchet — failure to accumulate, not loss of existing knowledge.

## What it measures

- **CVFN** — cost per verified-faithful-novel statement, through six gates:
  compiles under real Lean 4 + Mathlib; no `sorry`, new axiom, or
  `native_decide`; non-vacuous; novel relative to Mathlib; source-anchored;
  human sign-off.
- **Gate-failure histogram** — read its shape, not its total.
- **Reuse, duplication, and proof size over generations.**
- **Model × harness attribution** — the same slice on a local open model and a
  frontier control arm (`docs/planning/decisions/0001-phase-c-model-harness-control.md`).

## Status

Calibration program, 2026-07-24 → 2026-09-30
(`docs/planning/calibration-program.md`). The Lean oracle is live on a 4×H100
cluster; CVFN is 0 to date — every verified artifact so far re-derives material
Mathlib already has. Sprint records live in `docs/planning/`.

## Execution

Local-first: agents run on self-hosted open models (Qwen3.6-27B-FP8 via vLLM);
a frontier API is a control arm only. Python via `uv`; Lean 4 + Mathlib in
`lean/`.

## Related

- [llm_parl](../llm_parl) — parent project on deliberative multi-agent systems
- Anthropic, *Formalizing Fermat's Last Theorem* (2026-09); Chen, Marwaha, Lu,
  Yuen, Peng, *Prove2Me: An Open Collaborative Platform for Scaling Math
  Formalization* (arXiv 2608.28433)
