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
| **Dependencies** | 26Q4-EVO-01 (landed); a slice choice from Sprint 3's N1-density measurement (for the default goal); 26Q3-HARN-05 `attempts.json` (landed) |
| **PR Size Target** | <400 lines |

---

#### Context

Fitness is the expensive and the fragile half of promptbreeding here:

- **Expensive**: one evaluation = one LMS run, and measured runs ranged
  446K–8.99M tokens. The harness must make evaluations *small and uniform*:
  1 agent, a fixed statement slice, a hard token cap
  (`--max-tokens`, plus `LMS_OPENAI_MAX_TOKENS` per request via
  `26Q3-INFRA-02`).
- **Fragile**: at a ~3% per-statement verified rate, raw verified-count
  fitness is ~0 for most genomes — the same flat-landscape argument that
  deferred the SPG arm. The score must carry partial credit so selection has
  a gradient before the first full success.

Everything the score needs is already on disk after a run:

- `metadata.json` — `analysis.verified_artifacts`, `prompt_overrides`
  (which genome actually ran — refuse to score a run whose recorded
  overrides don't match the genome being evaluated).
- `artifacts.json` — per-artifact status and gate verdicts (T2/T4,
  `26Q3-HARN-03`) for partial credit.
- `attempts.json` — the `26Q3-HARN-05` cost ledger: per-statement tokens and
  wall-clock including failed attempts. **This is the denominator.**

---

#### Acceptance Criteria

- [ ] `lms/breeder/fitness.py` exists with `FitnessRecord` (dataclass,
      JSON round-trip) carrying at minimum: genome id/hash, run dir,
      tokens spent, statements attempted, compile successes, gate passes,
      gated-verified count, and a scalar `fitness`
- [ ] The scalar is gated-verified statements per million tokens, with
      documented partial-credit weights for compiles-short-of-gates —
      weights in one place, trivially editable, recorded in the output so
      two records are comparable only when scored the same way
- [ ] `scripts/breeder/evaluate_genome.py` runs
      `python -m lms.run --prompt-file <genome> ...` as a subprocess with
      pinned evaluation settings (1 agent, fixed goal slice, fixed
      generation count, hard token cap — all overridable by flags, all
      recorded in the output), then scores the run dir
- [ ] Scoring a run dir is a separate entry point from running one
      (`--score-only <run_dir>`), so existing runs can be re-scored when
      weights change without re-spending tokens
- [ ] The harness refuses to score a run whose `metadata.json`
      `prompt_overrides` hash does not match the genome file (provenance
      check, loud error)
- [ ] `--repeats N` runs the same genome N times and reports per-run scores
      plus the mean — run-to-run variance is the noise floor selection has
      to clear, and it must be visible, not averaged away silently
- [ ] Tests use fixture run dirs (checked-in miniature `metadata.json` /
      `artifacts.json` / `attempts.json`), no live runs or GPUs

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/breeder/__init__.py` | CREATE | Package |
| `lms/breeder/fitness.py` | CREATE | `FitnessRecord`, scoring from run dirs |
| `scripts/breeder/evaluate_genome.py` | CREATE | Run + score one genome |
| `tests/test_breeder_fitness.py` | CREATE | Scoring on fixture run dirs |
| `scripts/verify/26Q4-01/verify_26Q4-EVO-02.sh` | CREATE | Verification script |

---

#### Decision Gates

- If the Sprint 3 N1-density measurement has not landed by the time this is
  built, default the slice to the Gate A control arc — a known corpus is
  fine for *relative* fitness; do not block on the slice decision.
- Do not put the evaluation loop inside `Society`. The harness invokes runs
  as subprocesses precisely so a genome cannot leak state between
  evaluations and a crashed run cannot take the breeder with it.
- If partial-credit weighting turns into a research question, stop — ship
  the simplest monotone version (gated > compiled > attempted) and card the
  ablation.

---

#### Out of Scope

- Selection, mutation, population state — `26Q4-EVO-03`.
- Any change to how runs execute — the harness only invokes and reads them.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-EVO-02.sh` exits 0
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
- [ ] One real evaluation run on the box: a genome file → fitness record,
      end to end, with the token cap respected
