# CLAUDE.md

## Project Overview

**LMS (LLM Mathematical Society)** is a measurement instrument: it tests whether coordination structure lets a population of *weak* LLM agents accumulate Lean-verified mathematics, and whether what they accumulate ratchets (reuse, consolidation) or merely piles up.

**Core idea**: Multiple LLM agents collaborate to formalize mathematics in LEAN, with the proof assistant serving as a perfect verification oracle. Drawing from Joseph Henrich's collective brain theory, we hypothesize that population size and network structure matter more than individual agent capability.

## Research question (revised 2026-09-04)

On 2026-09-03 Anthropic reported a Lean 4 proof of Fermat's Last Theorem by a
swarm of frontier-model agents in 11 days via the Prove2Me statement DAG; their
own Claude-Code multi-agent harness had failed first, with the same model.
"Can LLM collectives + Lean formalize" is answered elsewhere. What this project
measures, on open models and owned hardware:

1. **Structure vs capability** — same slice, same harness, local model vs a
   frontier control arm (`docs/planning/decisions/0001-phase-c-model-harness-control.md`).
2. **Ratchet or pile** — reuse, duplication, proof size over generations.
3. **Minimum viable culture** — `potential_ratchet_failure` vs population size.

The three-text program (Stacks / ART / CNF) is **not** an LMS production
target; see `docs/planning/decisions/0002-lms-measures-collective-formalization.md`
(Proposed). Do not pitch LMS as a formalization engine.

## Key Concepts

### Collective Brain Theory (Henrich)
- Human innovation is collective, not individual
- Larger, connected populations → more innovation
- Below critical mass, knowledge *decays* — Henrich's Tasmanian case (2004),
  an interpretation contested since (Vaesen et al. 2016, PNAS)
- LLM societies may exhibit similar dynamics

**Naming**: the metric is `potential_ratchet_failure`, after Tomasello's cultural
ratchet — *not* "the Tasmania effect". It measures failure to accumulate, not
loss of existing technology. Use "ratchet failure" in code and prose.

### The LEAN Advantage
- Humans: weak verification (social, slow, error-prone), strong intuition
- LLMs + LEAN: **perfect verification** (oracle), weaker intuition
- This trade-off may favor collective approaches

### Phase Transitions
Watch for discontinuous jumps in:
- Lemma reuse rate
- Proof complexity
- d(proofs)/dt

## Experimental Parameters

| Parameter | Current Hypothesis |
|-----------|-------------------|
| Agent count | 6 preferred over 3 (network size matters) |
| Context per agent | 16M tokens (shorter lives, more agents) |
| Verification | LEAN 4 |
| Target | Bounded formalization goal (e.g., textbook chapter) |

## Directory Structure

```
lms/
├── README.md           # Project overview
├── CLAUDE.md           # This file
├── lms/                # Python package (avoids mutmut issues with src/)
├── docs/               # Documentation
├── experiments/        # Experimental runs and results
└── lean/               # LEAN project files
```

## Development Notes

- This is a **prototype** for quick iteration
- Prioritize simplicity over completeness
- Document observations about collective dynamics
- Track metrics for phase transition detection

## Related Work

- Parent project: [llm_parl](../llm_parl)
- Research notes: `llm_parl/docs/COLLECTIVE_MATHEMATICAL_INTELLIGENCE.md`
- Vision: `llm_parl/docs/COLLECTIVE_INTELLIGENCE_VISION.md`
- Decision records: `docs/planning/decisions/`
- External results the 2026-09-04 reframe responds to: Anthropic, "Formalizing
  Fermat's Last Theorem" (2026-09); Chen et al., "Prove2Me" (arXiv 2608.28433)
