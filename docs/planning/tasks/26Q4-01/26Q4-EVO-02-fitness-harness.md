### 26Q4-EVO-02: Fitness harness — one genome, one score, from real runs

**User Story**: As the promptbreeder loop, I want a single command that takes
a prompt-genome file and returns a fitness record computed from actual LMS
run outputs, so that selection acts on measured proof success per token, not
on a proxy.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-EVO-02-fitness-harness` |
| **Dependencies** | 26Q4-EVO-01 (landed); 26Q4-HARN-11 (probe set and PAR-2 cost) |
| **PR Size Target** | <400 lines |

---

#### Context

Fitness is the expensive and the fragile half of promptbreeding here:

- **Expensive**: measured whole runs ranged 446K–8.99M tokens. The harness must
  make evaluations *small and uniform*: 1 agent, fixed probes, a hard
  output-token cap.
- **Fragile**: at a ~3% per-statement verified rate, raw verified-count
  fitness is ~0 for most genomes — the same flat-landscape argument that
  deferred the SPG arm.

**Amended 2026-11-05 — the score has no partial credit.** This card first
scored gated statements per million tokens, with weights for compiling short of
the gates. Weights can be gamed, and the human gate cannot run inside a loop.
The score is now `26Q4-HARN-11`'s PAR-2 cost on fixed pinned probes: output
tokens to the first verified proof, or twice the cap if unsolved. The gradient
comes from probes of graded difficulty, each one pass or fail. Novelty is not
scored, so fitness works even where CVFN is undefined. Budgets are in output
tokens, so the score uses `completion_tokens`, not `CostLedger.total_tokens`,
which includes prompts. An artifact can keep `verified_lean` after a gate
failure, so solves read `gate_results`.

---

#### Acceptance Criteria

- [ ] `lms/breeder/fitness.py` exists with `FitnessRecord` (dataclass,
      JSON round-trip) carrying at minimum: genome hash, probe file hash,
      split, cap, per-probe solved and output tokens, and a scalar `fitness`
- [ ] The scalar is the negative mean of `lms/probes.py::par2_cost` over the
      probes; `tests/test_breeder_fitness.py::test_fitness_is_negative_mean_cost`
      passes, and records with a different cap or probe hash refuse to compare
- [ ] Compiling short of the gates earns nothing:
      `tests/test_breeder_fitness.py::test_no_partial_credit` passes
- [ ] `scripts/breeder/evaluate_genome.py` runs one genome over a split as a
      subprocess, through `26Q4-HARN-11`'s attempt runner, with its settings
      recorded in the output, and scores the result
- [ ] Scoring saved attempts is a separate entry point from running them
      (`--score-only <dir>`), so results can be re-scored without new tokens
- [ ] The harness refuses to score attempts whose recorded `prompt_overrides`
      hash does not match the genome file (provenance check, loud error)
- [ ] `--repeats N` runs the same genome N times and reports per-run scores
      plus the mean — run-to-run variance is the noise floor selection has
      to clear, and it must be visible, not averaged away silently
- [ ] Tests use fixture attempt records, no live runs or GPUs

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/breeder/__init__.py` | CREATE | Package |
| `lms/breeder/fitness.py` | CREATE | `FitnessRecord`, scoring from attempt records |
| `scripts/breeder/evaluate_genome.py` | CREATE | Run + score one genome |
| `tests/test_breeder_fitness.py` | CREATE | Scoring on fixture attempt records |
| `scripts/verify/26Q4-01/verify_26Q4-EVO-02.sh` | CREATE | Verification script |

---

#### Decision Gates

- Do not put the evaluation loop inside `Society`. The harness invokes
  subprocesses precisely so a genome cannot leak state between evaluations and
  a crashed run cannot take the breeder with it.
- If a genome needs partial credit to show any signal, stop and report it: that
  is the flat-landscape result, not a reason to add weights.

---

#### Out of Scope

- Selection, mutation, population state — `26Q4-EVO-03`.
- Probes and PAR-2 cost — `26Q4-HARN-11`.
- Any change to how attempts execute — the harness only invokes and reads them.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-EVO-02.sh` exits 0
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
- [ ] One real evaluation run on the box: a genome file → fitness record,
      end to end, with the token cap respected
