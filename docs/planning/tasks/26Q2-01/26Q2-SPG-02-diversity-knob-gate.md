### 26Q2-SPG-02: Diversity-knob critical gate (Phase 1)

**User Story**: As the LMS team, I want to know whether a sparse soft prefix
actually induces *distinct, useful* proof strategies (vs being washed out by the
long shared math context), so that we only build the evolutionary loop if the
diversity knob demonstrably beats a temperature-only baseline.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | CRITICAL |
| **Status** | 🚫 BLOCKED |
| **Branch** | — (experiment; harness wiring may produce code) |
| **Dependencies** | `26Q2-SPG-01` (serving path), `26Q2-SPG-03` (genome representation), `26Q4-HARN-11` (probes) |
| **PR Size Target** | <500 lines (any harness glue) |

---

#### Context

> Phase 1 of `specs/soft_prompt_genome_work_plan.md` and the **suggested first
> experiment** in `specs/soft_prompt_genome.md` §8. The minimal falsifiable test,
> no evolution yet. This is the gate the whole arm lives or dies on.

> **Execution boundary**: inference runs on the user's self-hosted box. Claude
> prepares the descriptor definition, the harness, and the analysis; the user
> runs the generation.

**Amended 2026-11-05 — the metric is solves, not text.** The first version
measured distinct tactics and lemma orderings, but random noise can look diverse
while solving nothing. The gate now reads what diversity is for: whether
different soft prompts solve different statements, on `26Q4-HARN-11`'s dev probes.

---

#### Acceptance Criteria

- [ ] Treatment: N agents with sparse-random soft prefixes (k=2–5, from SPG-03),
      at the position `26Q4-SPG-01` found stronger
- [ ] Baseline: same N agents, **temperature/seed-only** diversity, no prefix,
      **matched on output tokens**
- [ ] Run on `26Q4-HARN-11`'s dev probes, one attempt per agent per probe
- [ ] Reported: distinct probes solved by each population, each agent's mean
      solve rate, and the correlation of per-probe solves across agents
- [ ] Gate decision recorded below

---

#### Decision Gate — **CRITICAL**

- **Gate**: the soft-prefix population solves more distinct probes than the
  baseline at matched output tokens, without a drop in per-agent solve rate that
  cancels the gain.
- **If green** → build the evolutionary loop (`26Q4-SPG-03`).
- **If red** → switch genome to **steering vectors** (RepE / ActAdd / CAA) and
  re-run this phase before proceeding. **If both fail → stop and write it up**
  (a clean negative result; do not keep escalating). STOP and surface either way.

#### Gate outcome

_(record here: N, descriptor values treatment vs baseline, compute parity check,
decision)_

---

#### Out of Scope

- No evolution / selection / fitness here — pure single-generation diversity test.
- Genome representation is `26Q2-SPG-03`; do not reimplement it here.
