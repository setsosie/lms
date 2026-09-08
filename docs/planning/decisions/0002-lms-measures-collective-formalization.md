# 0002 — LMS measures collective formalization; it does not produce it

**Status**: Proposed
**Date**: 2026-09-04
**Deciders**: setsosie (solo)
**Tags**: scope, research-question

## Context

Until 2026-09-03 the project stacked two goals on one apparatus: test whether
coordination structure lets a collective of weak LLM agents accumulate
verified mathematics (the Henrich hypothesis), and, if so, formalize three
texts (Stacks, ART, CNF; ~12–14K statements). The Fermat's Last Theorem
formalization and the Prove2Me paper settle the first half of the first goal —
collectives with Lean can formalize research-level mathematics — and price the
second goal at consumer scale: missions of 17K–151K Lean lines closed by 3–9
agents for $200–600 on subscriptions. Buzzard's assessment: "mathematically
this work tells us essentially nothing"; what it shows is autoformalization
capability. Prove2Me publishes no reuse statistics and no failure rates, and
cites ~43% statement faithfulness in prior work.

## Decision

The project's deliverable is **measurement on open models**: cost per
verified-faithful-novel statement, the gate-failure histogram, reuse and
duplication over generations, and the attribution of outcomes to model versus
harness. The research question is restated as: *does coordination structure
substitute for individual model capability, and does a verified swarm ratchet
(reuse, consolidation) or merely pile (bloat)?* The three-text program is no
longer an LMS production target.

## Alternatives considered

- **Keep the three-text program as the goal** — rejected: a solo researcher on
  a 4×H100 cannot outrun consumer subscriptions on a hosted platform, and the
  texts' formalization no longer needs this apparatus.
- **Shut the project down** — rejected: the surviving question is unclaimed,
  cheap to run here, and the one this hardware is uniquely placed for.
- **Become a Prove2Me client (point local agents at its API)** — deferred:
  hosted, shared missions break population isolation, and the server's
  openness is unverified. Borrow the data model instead ([[0001]],
  `26Q3-HARN-25`).

## Consequences

**What this enables:**
- A defensible niche: the measurement layer Prove2Me lacks, applied to weak
  models.

**What this costs:**
- The three-text roadmap's production framing, the shared-kernel build order
  as an LMS deliverable, and the CNF OCR work lose their reason to exist here.
  `goals/*.json` remain as slices and test material.

**What this commits us to:**
- README and CLAUDE.md reframed (same PR as this record).
- If the texts are to be formalized, doing it as a Prove2Me captain, with this
  project's gates as the contribution.
- The soft-prompt-genome arm stays gated on dynamic range in proof success,
  unchanged.

**Open for the deciders:** whether to retire the three-text program outright or
park it. This record stays Proposed until that is answered.

## Related

- [[0001]]; `docs/planning/calibration-program.md` §5; `specs/three_text_synthesis.md`
- Anthropic, "Formalizing Fermat's Last Theorem" (2026-09); Prove2Me
  (arXiv 2608.28433); K. Buzzard, Xena blog, 2026-09
