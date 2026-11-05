### 26Q4-EVO-03: Promptbreeder loop — strong-model mutation, run-per-generation

**User Story**: As the LMS project, I want a small evolutionary loop where a
strong model breeds agent prompts and each LMS run scores one genome, so that
prompt quality improves by selection against the Lean oracle instead of by
hand-editing `prompts.py`.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-EVO-03-promptbreeder-loop` |
| **Dependencies** | 26Q4-EVO-01 (landed), 26Q4-EVO-02 |
| **PR Size Target** | <500 lines |

---

#### Context

This is the text-space v0 of the evolution arm (decision 2026-08-28,
superseding-as-precursor the soft-prompt SPG arm, which stays queued behind
it). Design premises:

- **Breeder ≠ population.** The breeder is the strongest, newest open model
  that fits on the server and meets the cutoff rule below, used only for
  mutation/crossover calls. The evaluated model is whatever the runs serve.
  The two need not run together: swap them between breeding generations, so
  the breeder can use more of the box for a short time.
- **One run = one fitness evaluation** via `26Q4-EVO-02`. Population size is
  therefore the budget knob: at ~6–8 genomes × 1 short capped run each,
  a breeding generation costs on the order of one committee smoke run.
- **Start plain, add the paper's tricks later.** Promptbreeder's
  self-referential layer (evolving the mutation-prompts themselves) and
  QD/diversity maintenance are deliberately v1+; the paper's ablations say
  plain evolution captures much of the value, and a working plain loop is
  the platform to measure the tricks against.

The genome is the JSON object `--prompt-file` accepts — in v0, override
`agent_system_goal` only. Everything else (`review_system`, the committee
role prompts) stays at base so fitness differences are attributable to the
one prompt under selection.

**Amended 2026-10-30 — selection inflates the scores it selects on.** As first
written, the loop evaluated only unevaluated genomes, so a survivor kept the
single score it was selected for. The maximum of several noisy scores
overstates its genome's true fitness, so survivors carried an upward bias into
every later generation, and the final "evolution beat the hand-written prompt"
comparison was made on those same inflated scores. The criteria below now
separate the **selection pool** (evaluations used to choose) from **fresh
confirmation** (evaluations never used to choose), and the decision gates add a
repeatability check: if run-to-run noise is as large as the differences between
genomes, selection acts on noise, which the ~0-score gate below does not catch.

**Amended 2026-11-05 — a strong breeder can answer the test.** It can solve a
statement itself and write the proof into the prompt, so fitness would measure
the breeder. Fresh evaluations on the same statements catch noise, not this. So
selection reads `26Q4-HARN-11`'s dev probes only, the breeder never sees a
held-out statement, and "evolution beat the control" is claimed only on
held-out probes. Output-token budgets also make long prompts free, so length is
capped. And the breeder is local: Axiom's Stacks library was released on
2026-11-04, so a model with training data from that date on may have seen it
and could pass it into a prompt.

---

#### Acceptance Criteria

- [ ] `lms/breeder/population.py`: population state as one JSON file —
      genomes with id, content, parent ids, operator that produced them,
      fitness records per evaluation, generation number. The file **is** the
      resume point: killing the loop and restarting from it loses nothing
      but any in-flight evaluation
- [ ] `lms/breeder/operators.py`: two operators, each one breeder-model
      call — *direct mutation* (genome + its fitness/failure summary →
      variant) and *crossover* (two genomes + scores → recombination).
      Failure summaries come from the evaluation run's artifacts
      (verification errors, gate rejections), truncated to a budget
- [ ] Breeder calls go through the existing provider abstraction, against a
      locally served model — no new client code, and the breeder's token spend
      is tracked and reported per generation
- [ ] The breeder's model id and documented training cutoff are recorded in the
      population file, and a cutoff that is missing or not before 2026-11-04 is
      refused through `lms/probes.py::require_clean_cutoff`
- [ ] The breeder is chosen when this card starts: survey open models that fit
      the box and meet the cutoff, then score each finalist's generation-0
      variants on dev probes — accepted via a note in `docs/planning/`
- [ ] Failure summaries given to the breeder come from dev probes only:
      `tests/test_breeder_loop.py::test_breeder_never_sees_heldout` passes
- [ ] A variant naming a probe's tag, declaration or statement, or quoting its
      proof, is refused before it runs:
      `tests/test_breeder_loop.py::test_leak_screen_refuses_probe_content` passes
- [ ] A variant over a fixed length cap is refused before it runs:
      `tests/test_breeder_loop.py::test_length_cap` passes
- [ ] `scripts/breeder/breed.py`: one *breeding generation* per invocation —
      evaluate every genome in the population once more (via `26Q4-EVO-02`,
      sequential is fine in v0) **including survivors**, rank on the mean of
      each genome's selection-pool evaluations, truncation-select survivors,
      breed replacements, write the population file, print a one-screen
      generation report (best/median mean fitness, evaluations per genome,
      spend). Loop = invoke repeatedly; cron/sbatch-friendly
- [ ] Every fitness record carries `pool: "selection" | "confirmation"`;
      ranking reads only `selection` records
- [ ] `scripts/breeder/breed.py --confirm K` runs K fresh evaluations **on
      held-out probes** each of the current best genome, generation 0's best
      and the hand-written control, on a token budget stated separately from
      the breeding budget, records them as `confirmation`, and reports each
      genome's selection-pool mean beside its confirmation mean. "Evolution beat
      the control" is claimed only from the confirmation means, and if
      generation 0's best ties the winner, one rewrite was enough and selection
      added nothing
- [ ] Tests: ranking ignores `confirmation` records; a survivor's mean updates
      when it is re-evaluated; `--confirm` never changes the ranking
- [ ] Seed population: current `agent_system_goal` base content plus N-1
      breeder-generated variants of it, so generation 0 contains the
      hand-written prompt as the control that evolution must beat
- [ ] Selection, population size, and per-evaluation caps are flags with
      conservative defaults (population 6, keep top 3, caps from EVO-02's
      pinned settings)
- [ ] Tests: population round-trip and resume; selection math; operator
      prompt construction (breeder model mocked); a full generation against
      a mocked evaluator — no live breeder calls, no runs
- [ ] `docs/planning/` note or card update recording the first real
      breeding-generation result on the box, whatever it shows

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/breeder/population.py` | CREATE | Population state, lineage, resume |
| `lms/breeder/operators.py` | CREATE | Mutation + crossover via breeder model |
| `scripts/breeder/breed.py` | CREATE | One breeding generation per invocation |
| `tests/test_breeder_loop.py` | CREATE | Population/selection/operator tests |
| `scripts/verify/26Q4-01/verify_26Q4-EVO-03.sh` | CREATE | Verification script |

---

#### Decision Gates

- If every genome in generation 0 solves ~no dev probes, stop after two
  breeding generations and report — that is the flat-landscape outcome, and the
  honest response is easier probes or improving the base prompt by hand first,
  not burning budget on selection with no gradient.
- If no model that fits the box has a documented cutoff before 2026-11-04 → stop
  and ask. Do not assume a cutoff from a release date.
- **Repeatability gate** (before breeding generation 1): run EVO-02's
  `--repeats 2` on every generation-0 genome. If the variance between genome
  means does not exceed the mean within-genome (run-to-run) variance, stop and
  report: the scores carry no heritable signal at this evaluation size, and the
  response is a larger or more uniform evaluation, not more generations.
- If dev scores rise and held-out scores do not, the prompts have learned the
  dev probes. Stop and report; do not add generations.
- Multi-prompt genomes (evolving review/committee prompts jointly) are v1;
  they multiply attribution noise and are exactly the "complexity over
  time" axis this design defers.

---

#### Out of Scope

- Self-referential mutation-prompt evolution and QD archives (v1+).
- Soft-prompt genomes — the SPG arm (`26Q2-SPG-01/02/03`, `26Q4-SPG-01/02/03/04`)
  is queued separately.
- Any coupling to the in-run `VotingSystem` prompt machinery.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-EVO-03.sh` exits 0
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
- [ ] One real breeding generation on the box, its report checked in
