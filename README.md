# LMS: LLM Mathematical Society

A prototype experiment testing whether LLM collectives with formal verification (LEAN) can achieve mathematical reasoning beyond individual model capabilities.

## Hypothesis

Drawing from Joseph Henrich's collective brain theory: network size and connectivity matter more than individual capability. LLM societies with perfect verification (LEAN oracle) may exhibit emergent collective intelligence through:

- Parallel exploration (many agents try different directions)
- Accumulated scaffolding (verified lemmas, intermediate results)
- Recombination (combining proven results in novel ways)

## Key Questions

1. **Minimum viable culture**: What's the smallest agent population that can sustain knowledge accumulation?
2. **Phase transitions**: Does mathematical progress show discontinuous jumps when crossing population/scaffolding thresholds?
3. **Population vs. longevity trade-off**: Is 6 agents × 16M tokens better than 3 agents × 32M tokens?

## Experimental Design

**Goal**: Formalize a bounded mathematical target (e.g., Chapter 1 of a textbook) in LEAN

**Agents**: Heterogeneous LLMs with:
- Basic reasoning ability
- LEAN tool use (invoke prover, interpret results)
- Access to shared lemma library

**Metrics**:
- Proof complexity over time (depth of proof trees)
- Lemma reuse rate (citations to prior agent work)
- Time-to-proof for comparable difficulty
- Derivative rate d(proofs)/dt (looking for inflection points)

## Theoretical Foundation

- **Henrich's Collective Brain**: Human innovation emerges from interconnected populations, not individual genius
- **The Tasmania Effect**: Isolated populations lose technology; there's a critical mass for knowledge maintenance
- **LEAN as Cultural Ratchet**: Perfect verification prevents backward slippage, like writing preserved mathematics

## Status

**Prototype** - Early exploration phase

## Related

- [llm_parl](../llm_parl) - Parent project on deliberative multi-agent systems
- [COLLECTIVE_MATHEMATICAL_INTELLIGENCE.md](../llm_parl/docs/COLLECTIVE_MATHEMATICAL_INTELLIGENCE.md) - Full research notes
