# CLAUDE.md

## Project Overview

**LMS (LLM Mathematical Society)** is a prototype exploring whether LLM collectives with formal verification can achieve emergent mathematical reasoning.

**Core idea**: Multiple LLM agents collaborate to formalize mathematics in LEAN, with the proof assistant serving as a perfect verification oracle. Drawing from Joseph Henrich's collective brain theory, we hypothesize that population size and network structure matter more than individual agent capability.

## Key Concepts

### Collective Brain Theory (Henrich)
- Human innovation is collective, not individual
- Larger, connected populations → more innovation
- Below critical mass, knowledge *decays* (Tasmania effect)
- LLM societies may exhibit similar dynamics

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
