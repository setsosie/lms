### 26Q2-SPG-01: Soft-prefix serving feasibility (Phase 0)

**User Story**: As the LMS team, I want to confirm we can feed per-agent soft
embeddings at batch on the target self-hosted engine, so that the whole
soft-prompt-genome arm rests on a serving path that actually exists before we
build evolutionary machinery on top of it.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | — (no code deliverable; runbook + recorded result) |
| **Dependencies** | None |
| **PR Size Target** | n/a (investigation + version pins, no app code) |

---

#### Context

> Phase 0 of `specs/soft_prompt_genome_work_plan.md`. The cheapest blocking
> question: does `inputs_embeds` (HF) / `prompt_embeds` (vLLM/SGLang) work on the
> chosen model, and does it compose with **continuous batching** at population
> scale (~50 distinct prefixes in one batch)?

**Current State**:
- No engine selected / version-pinned for the genome arm.
- Cost model (`scratch/cost_model.py`) assumes one frozen model, full-box batching.

> **Execution boundary**: this runs on the user's self-hosted GPU box / cluster.
> Claude does **not** execute it — Claude prepares the exact runbook and records
> the outcome here; the user runs the commands.

---

#### Acceptance Criteria

- [ ] `inputs_embeds`/`prompt_embeds` confirmed working on the target model (a
      short generation from supplied embeddings produces coherent output)
- [ ] Composes with continuous batching: ~50 distinct soft prefixes in one batch,
      throughput hit measured and recorded (tokens/s vs token-prefix baseline)
- [ ] Engine + serving path decided; **version pins recorded** in this card
- [ ] Result (pass / fall-back-to-steering) written into the **Gate outcome**
      section below

---

#### Runbook (user runs on the cluster)

> Claude fills these with concrete, engine-specific commands once the target
> model + engine are named. Placeholder skeleton:

```bash
# 1. inputs_embeds smoke test (HF Transformers)
#    uv run python -c "..."   # build (k,d) embeds, pass inputs_embeds=, generate
# 2. vLLM / SGLang prompt_embeds at batch
#    serve model; submit ~50 requests each with a distinct prompt_embeds prefix
# 3. throughput delta vs token-only prefixes; record tokens/s
```

---

#### Decision Gate

- **Gate**: soft-prefix serving works at batch with acceptable throughput.
- **If green** → unblock `26Q2-SPG-02` (diversity-knob test).
- **If blocked** (`prompt_embeds` unsupported / batch-incompatible) → switch the
  genome phenotype to **activation-steering vectors** (forward hook only, no
  `prompt_embeds`); re-scope SPG-02 to the steering genome. STOP and surface
  before changing the plan.

#### Gate outcome

_(record here after the run: engine, version pins, throughput delta, decision)_
