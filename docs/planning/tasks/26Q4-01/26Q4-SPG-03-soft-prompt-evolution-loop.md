### 26Q4-SPG-03: Soft-prompt evolution loop, selected by Lean on the probe set

**User Story**: As the person testing whether evolution can find useful soft prompts, I want a loop that mutates, merges and selects soft prompts by Lean results on fixed probes, with controls, so that a gain can be credited to selection rather than to noise or chance.

> Drafted by Claude on 2026-11-05 from a discussion with the user, who wants evolution on soft prompts as the main experiment and gradient tuning as an option. Budgets and thresholds are proposals. Review as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-SPG-03-soft-prompt-evolution-loop` |
| **Dependencies** | `26Q4-HARN-11`, `26Q4-EVO-02`, `26Q4-EVO-03` (population file), `26Q4-SPG-01` green, `26Q2-SPG-01` green, `26Q2-SPG-02` green, `26Q4-SPG-02` |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**: nothing in `lms/` can put a soft prefix in front of an agent. `26Q4-EVO-01` is the only prompt injection point, and it holds text for a whole process.

**Serving**: vLLM's [docs](https://docs.vllm.ai/en/latest/features/prompt_embeds.html) accept `prompt_embeds` parts in Chat Completions messages, in any position relative to text. So only the soft tokens travel as tensors.

**Search budget**: QD-LLM ran 500 evaluations over 32,768 values. At a 100K output-token cap (a placeholder until `26Q4-HARN-11`'s baseline), one probe per evaluation is at most 50M output tokens and 20 per evaluation at most 1B. The one recorded serving rate, 1,240 tokens/s, batched and counting prompts too, makes those about 11 hours and 9 days, so both are optimistic. See `26Q4-SPG-02` for why genome size matters.

**What is scored**: the spec's aim is diverse minds, and Lean picks winners, so an agent that solves one rare statement is worth more than one that repeats the others. Population fitness is therefore **coverage**: distinct dev probes solved at equal output tokens. A genome's selection score is its **marginal coverage**, the probes lost if it is removed, with mean PAR-2 cost breaking ties. Per-probe solve vectors also serve as the behaviour descriptor the spec left open.

---

#### Scope

| Arm | Start | Merge | Purpose |
|---|---|---|---|
| A | One ancestor plus mutation | Rank-weighted mean, 5 parents | The user's averaging idea, in the regime where it holds its size |
| B | Unrelated 2-token blends | Per-token, 5 parents | Keeps spread with unrelated parents |
| C | As A, on a small genome | As A | Tests genome size: 2 soft tokens in the top 16 principal components of the embedding matrix |
| Null | As A | None, random survivors | Mutation with selection removed |
| Control | No soft prompt | None | The same agents, varied by seed only |

The merge is applied, then mutation. Run arms one at a time, and do not start the next until the first has passed the repeatability gate.

---

#### Acceptance Criteria

- [ ] Soft prefixes reach the model per agent through the provider's chat request: `tests/test_soft_prompt_requests.py::test_each_agent_sends_its_prefix` passes with a mock provider
- [ ] Run metadata records each agent's soft-prefix hash beside `prompt_overrides`, so a run accounts for what every agent saw: `tests/test_soft_prompt_requests.py::test_metadata_records_prefix_hash` passes
- [ ] `lms/soft_prompt_evolution.py` has `marginal_coverage(solves)` and `select_survivors(records)`: `tests/test_soft_prompt_evolution.py` covers a specialist beating a duplicate of a stronger genome, ties broken by cost, and an all-zero population
- [ ] The loop reads dev probes only and refuses a held-out split: `tests/test_soft_prompt_evolution.py::test_selection_never_reads_heldout` passes
- [ ] Survivors are re-evaluated each generation and ranked on their mean: `tests/test_soft_prompt_evolution.py::test_survivors_reevaluated` passes
- [ ] Every generation logs population size and spread from `population_stats`, plus output tokens; a run stops and reports if spread falls below a set floor: `tests/test_soft_prompt_evolution.py::test_stops_on_collapse` passes
- [ ] The population file resumes after a kill, losing only the in-flight evaluation: `tests/test_soft_prompt_evolution.py::test_resume` passes
- [ ] `scripts/soft_prompt/evolve.py` runs one generation per invocation, and `--confirm` evaluates the winner, generation 0's best, the null and the control on held-out probes, never changing the ranking
- [ ] A claim that evolution helped rests only on held-out paired results against generation 0's best and the control — accepted via Outcome Demo
- [ ] Model id and training cutoff are recorded, and a cutoff that is missing or not before 2026-11-04 is refused through `lms/probes.py::require_clean_cutoff`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/soft_prompt_evolution.py` | CREATE | Coverage, selection, generation step |
| `lms/providers/openai.py` | MODIFY | Add `prompt_embeds` parts to a chat request |
| `lms/providers/base.py` | MODIFY | Carry an optional soft prefix on a message |
| `scripts/soft_prompt/evolve.py` | CREATE | One generation per invocation |
| `tests/test_soft_prompt_evolution.py` | CREATE | Selection and loop tests |
| `tests/test_soft_prompt_requests.py` | CREATE | Request and provenance tests |
| `scripts/verify/26Q4-01/verify_26Q4-SPG-03.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Reuse `lms/breeder/population.py`, `lms/genome_merge.py`, `lms/probes.py` and the paired tests in `lms/arms.py`.
- A soft prompt is tied to the exact served weights, so pin the evaluated model for the whole study and record its revision.
- Soft prompts cannot be read or screened, so the held-out split is the only defense against one encoding answers. Keep dev and held-out sections separate, as `26Q4-HARN-11` requires.

---

#### Decision Gates

- **Repeatability**: before generation 1, evaluate generation 0 twice. If the variance between genomes does not exceed the variance within them → stop and report. Selection would act on noise.
- If the control's coverage is near zero on dev probes → stop. There is nothing to select on.
- If arm A shows no gain over the null on dev probes at its budget → stop arm A and report it before spending on B or C. Do not extend the budget without asking.
- If arm C beats arms A and B → stop and surface it; the study should move to small genomes.
- If `prompt_embeds` chat parts behave differently in the pinned vLLM version → stop and return to `26Q2-SPG-01`.
- If the change exceeds the PR Size Target → split provider work from the loop.

---

#### Out of Scope

- `26Q4-SPG-04` owns gradient tuning. Tuned prompts join as a start population only after that card.
- `26Q4-EVO-03` owns text prompts. The text × soft comparison happens after this loop.
- Activation steering and per-layer prompts.
- MAP-Elites archives, until a plain run shows a reason.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-SPG-03.sh`.

---

#### Outcome Demo

**Where**: the GPU server, as round folders Claude prepares; the user runs them.
**Run**: the round's `RUN_THIS.txt`: serve the pinned model, run the repeatability check, then generations of arm A, then `--confirm`.
**Expect**: a coverage curve against output tokens per arm, size and spread per generation, and a held-out table comparing the winner with generation 0's best, the null and the control. A null result is reported as a result.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-SPG-03.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
