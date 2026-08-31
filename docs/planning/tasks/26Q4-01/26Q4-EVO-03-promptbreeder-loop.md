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

- **Breeder ≠ population.** The breeder is one strong model (the Anthropic
  escape-hatch config) used only for mutation/crossover calls — token cost
  is rounding error next to evaluations. The evaluated model is whatever the
  runs serve (local-first).
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
- [ ] Breeder calls go through the existing provider abstraction
      (`create_provider("anthropic", ...)`) — no new client code, and the
      breeder's token spend is tracked and reported per generation
- [ ] `scripts/breeder/breed.py`: one *breeding generation* per invocation —
      evaluate unevaluated genomes (via `26Q4-EVO-02`, sequential is fine in
      v0), truncation-select survivors, breed replacements, write the
      population file, print a one-screen generation report (best/median
      fitness, spend). Loop = invoke repeatedly; cron/sbatch-friendly
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

- If every genome in generation 0 scores ~0 *including partial credit*, stop
  after two breeding generations and report — that is the flat-landscape
  outcome, and the honest response is to shrink the evaluation slice or
  improve the base prompt by hand first, not to burn budget on selection
  with no gradient.
- If the breeder model starts producing prompts that game partial credit
  (e.g. maximizing trivially-compilable output), that is a fitness-shaping
  bug in EVO-02's weights — fix it there, do not patch operators.
- Multi-prompt genomes (evolving review/committee prompts jointly) are v1;
  they multiply attribution noise and are exactly the "complexity over
  time" axis this design defers.

---

#### Out of Scope

- Self-referential mutation-prompt evolution and QD archives (v1+).
- Soft-prompt genomes — the SPG arm (`26Q2-SPG-01/02/03`) remains queued
  separately.
- Any coupling to the in-run `VotingSystem` prompt machinery.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-EVO-03.sh` exits 0
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
- [ ] One real breeding generation on the box, its report checked in
