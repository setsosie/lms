# Future Research Directions

## RL/Coevolution of Culture and Genetics

**Idea**: Hook LMS up to an RL engine to explore coevolution of culture (accumulated verified theorems) and genetics (agent policy/weights).

**Key insight**: With API-based models, we can only evolve prompts/culture. To evolve "genetics" (actual model behavior), we need **open-source weights** that can be fine-tuned.

**Architecture sketch**:
- **Genetics**: Agent weights/LoRA adapters that get selected/mutated based on verification success
- **Culture**: Foundation file - accumulated verified theorems that persist across generations
- **Selection pressure**: LEAN verification (perfect oracle)
- **Fitness signal**: Verification rate, lemma reuse, novel theorem creation

**Phase transitions to watch for**:
- Jump in lemma reuse rate
- Emergence of novel theorems from building blocks
- Collective capability exceeding individual agent limits

**Requirements**:
- Open-source base model (Llama, Mistral, etc.)
- LoRA/adapter training infrastructure
- Population of diverse agent variants
- Cultural memory (Foundation file) + genetic memory (model weights)

**Related work**:
- Henrich's collective brain theory
- Cultural evolution algorithms
- Neuroevolution + LLMs

---

*Added: 2024-12-17*
