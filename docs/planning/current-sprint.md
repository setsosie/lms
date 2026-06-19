# Sprint 1: Pipeline shakedown + stand up the soft-prompt-genome arm

**Dates**: Jun 16 - Jun 27, 2026
**Quarter**: Q2 2026
**Sprint Goal**: Validate the formalization pipeline on the Neukirch ANT Ch. I core
arc (exit on pipeline metrics, not math), and in parallel stand up the
soft-prompt-genome research arm through its first falsifiable gate — does a
sparse soft-prefix induce more proof-attempt diversity than a temperature-only
baseline at matched compute? **All work runs on local LLMs** (see Execution
environment below).

**Status**: 🔄 ACTIVE

> **Bootstrap note (2026-06-18):** this is the first sprint doc in the repo —
> numbering starts at Sprint 1 / folder `26Q2-01`. Dates are a starting
> assumption; renumber/rebase as needed. Generated when pulling `lms` up to the
> IndigiGenius canonical sprint schema (`claude-skills/docs/SPRINT-SCHEMA.md`).

## Sprint Summary

| Metric | Value |
|--------|-------|
| Planned points | 24 |
| Carryover | None (first sprint) |
| Delivered to date | 0 |
| Tracks | Infrastructure · Formalization (ANT shakedown) ∥ Soft-Prompt Genome (gated) |
| Execution | **Local-first** — self-hosted OSS models; APIs are an escape hatch |
| Status | 🔄 ACTIVE |

## Infrastructure Track

Enables local-first execution. Must land before the shakedown runs locally.

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q2-INFRA-01: Local OpenAI-compatible serving path | 3 | CRITICAL | INFRA | 🔲 PENDING | — | Add `base_url` so agents hit a local vLLM/SGLang `/v1`; unblocks ANT-01 on local LLMs |

## Formalization Track

The active "what do we formalize" work. Source: `specs/ant_shakedown.md`.

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q2-ANT-01: ANT Ch. I core-arc shakedown | 8 | HIGH | ANT | 🔄 IN PROGRESS | — | Integrality → Minkowski → class number → units; 2–3 WC cycles, exit on pipeline metrics. **Runs on local LLM** (depends on 26Q2-INFRA-01) |

## Soft-Prompt Genome Track

Parallel "how do we generate diverse minds" research arm. Source:
`specs/soft_prompt_genome_work_plan.md` (phased + **gated** — do not start a
phase until the prior gate is green). Phases 0–1 are cluster/serving work the
user drives; SPG-03 (genome representation) is local code and can start now.

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q2-SPG-01: Serving feasibility (Phase 0) | 3 | HIGH | SPG | 🔲 PENDING | — | `inputs_embeds`/`prompt_embeds` at batch; **user-driven (cluster)**. Gate → SPG-02 |
| 26Q2-SPG-02: Diversity-knob critical gate (Phase 1) | 5 | CRITICAL | SPG | 🚫 BLOCKED | — | Blocked by SPG-01. Soft-prefix diversity > temperature baseline at matched compute. Gate → evolution |
| 26Q2-SPG-03: Genome representation + mutation (Phase 2 prep) | 5 | HIGH | SPG | 🔲 PENDING | — | Local code, TDD. Sparse k-hot soft prefix, Gaussian mutation, serialize. No serving needed |

### Queued (gated, not in this sprint's committed scope)

Reference rows — not counted in planned points. Promote once SPG-02 is green.

| Task | Points | Priority | Epic | Status | Blocked By | Notes |
|------|--------|----------|------|--------|-----------|-------|
| SPG Phase 3: Evolutionary loop under LEAN fitness | — | — | SPG | ⏸️ DEFERRED | SPG-02 | Flat-landscape risk → partial-proof credit / subgoal curriculum |
| SPG Phase 4: MAP-Elites QD archive | — | — | SPG | ⏸️ DEFERRED | SPG Phase 3 | Behavior descriptor for proof strategies is an open question |
| SPG Phase 5: Ablations + QD-LLM differentiation | — | — | SPG | ⏸️ DEFERRED | SPG Phase 3 | Baseline = QD-LLM (arXiv:2605.09781); e-/m-geodesic mutation ablation |

## Risk register

| Risk | Mitigation | Task |
|------|------------|------|
| Soft prefix too weak vs long shared math context | Per-layer prompts (P-tuning v2) → steering vectors (RepE/ActAdd/CAA) | SPG-02 |
| `prompt_embeds` unsupported at batch on target engine | Fall back to activation-steering genome (forward hook only) | SPG-01 |
| ANT shakedown stalls on math, not pipeline | Exit criterion is pipeline metrics; descope math, not the workflow test | 26Q2-ANT-01 |
| Novelty overclaim (QD-LLM exists) | Differentiate on LEAN fitness + collective framing; dedicated 2025–26 NTP search before any paper claim | SPG Phase 5 |
| Local model too weak to close proofs | Defined escape-hatch trigger routes the hard sub-task to an API model (see below) | 26Q2-ANT-01 |

## Execution environment (policy, 2026-06-18)

**Local-first.** Default to self-hosted OSS models for *all* agents — the
formalization committees/WCs and the genome population both run on one frozen
self-hosted model (full-box continuous batching; cost-model config **B**). This
makes API spend ~$0 and unifies both tracks on one serving stack (the genome arm
*requires* self-hosting anyway). The binding constraint becomes box wall-clock,
not token price.

**Escape hatch — APIs only on a defined roadblock.** Reserve model APIs for when
local hits a wall. Trigger (route only the *blocked sub-task*, log it, continue):

- a WC cycle fails to close its proofs after a bounded number of local attempts
  with no Lean-progress, **or**
- a statement is flagged as exceeding local-model capability.

Then route that sub-task to an API model (cost-model config **C**, "robust mix"
= mostly local + targeted API). This is surgical, not a wholesale switch; the
router itself is a follow-up task, built only once a real roadblock appears.

Enablement: `26Q2-INFRA-01` adds the local serving path. Anthropic/Google
providers stay wired as the escape-hatch route.
