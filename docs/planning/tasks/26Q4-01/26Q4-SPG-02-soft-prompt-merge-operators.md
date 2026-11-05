### 26Q4-SPG-02: Soft-prompt merge operators — averaging, multiple parents, per-token crossover

**User Story**: As the person evolving soft prompts, I want several merge operators with any number of parents, starting from simple averaging, so that merging can be compared directly and a population cannot quietly collapse to one point.

> Drafted by Claude on 2026-11-05 from a discussion with the user, who asked to start from simple averaging and to explore other merges and more than two parents. Review as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-SPG-02-soft-prompt-merge-operators` |
| **Dependencies** | `26Q2-SPG-03` (`lms/genome.py`, `numpy`) |
| **PR Size Target** | <400 lines (max 800) |
| **Parts** | single PR |

---

#### Context

**Evidence**: `scripts/research/soft_prompt_merge_sim.py`, on synthetic embeddings with no model and no Lean. Real embeddings are not isotropic, so `26Q4-SPG-01` repeats the size check on the real matrix.

**Unrelated parents collapse under averaging; relatives do not.** Size and spread against a real token, after 20 generations with no selection, starting from 24 genomes of 8 soft tokens:

| Operator | Unrelated start: size, spread | One ancestor: size, spread |
|---|---:|---:|
| Mutation only | 0.84, 0.84 | 0.82, 0.46 |
| Mean, 2 parents | 0.27, 0.14 | 0.70, 0.14 |
| Mean, 5 parents | 0.22, 0.11 | 0.69, 0.11 |
| Mean, 2 parents, norm kept | 0.83, 0.14 | 0.82, 0.14 |
| Per-token, 2 parents | 0.84, 0.44 | 0.82, 0.35 |
| Per-token, 5 parents | 0.82, 0.56 | 0.83, 0.36 |

- With no mutation, every mean operator ends at spread 0.00: the whole population becomes one point.
- Keeping the norm restores size but not spread.
- Because `e = pᵀE` is linear, a mean of soft tokens is the same as a mean of their simplex points. Each merge widens the active set, moving toward the dense interior where the spec's probe measured collapse.

**Averaging was still the best search operator in a toy test.** Final cosine with a hidden target after 500 noisy evaluations (noise 0.05), 3 parents:

| Genome size | Mutation | Crossover | Mean |
|---:|---:|---:|---:|
| 16 | 0.831 | 0.865 | 0.905 |
| 256 | 0.228 | 0.263 | 0.319 |
| 2,048 | 0.046 | 0.039 | 0.054 |
| 32,768 | 0.003 | 0.003 | 0.002 |

**More parents helped, up to a point** (same test, noise 0.05, 10 offspring):

| Genome size | 1 | 2 | 3 | 5 | 5, rank-weighted | 10, no selection |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.824 | 0.880 | 0.905 | 0.922 | 0.925 | 0.013 |
| 256 | 0.170 | 0.269 | 0.319 | 0.317 | 0.359 | -0.001 |

The last column has as many parents as offspring, which removes selection. The toy's landscape is smooth, which Lean's will not be, so it shows mechanics and the size effect, not expected results.

**Genome size mattered more than any operator.** At QD-LLM's size, 32,768 values, 500 noisy evaluations found essentially nothing here. So `26Q4-SPG-03` also tests a small genome.

---

#### Acceptance Criteria

- [ ] `lms/genome_merge.py` has `merge_mean(parents, weights=None)`, `merge_mean_keep_norm(parents, centroid, weights=None)`, `merge_per_token(parents, *, rng)`, and `rank_weights(n)`, and each accepts two or more parents
- [ ] Operators act on `(k, d)` arrays and refuse parents of different shapes, or fewer than two: `tests/test_genome_merge.py::test_refuses_bad_parents` passes
- [ ] `merge_mean` with equal weights equals the arithmetic mean, and weights must sum to 1: `tests/test_genome_merge.py::test_mean_and_weights` passes
- [ ] `merge_per_token` takes each soft token whole from one parent: `tests/test_genome_merge.py::test_per_token_inherits_whole_tokens` passes
- [ ] `merge_mean_keep_norm` keeps each token's distance from the centroid at its parents' mean: `tests/test_genome_merge.py::test_keep_norm` passes
- [ ] `rank_weights` is positive, decreasing and sums to 1: `tests/test_genome_merge.py::test_rank_weights` passes
- [ ] `lms/genome_merge.py::population_stats(pop, centroid, token_radius)` returns size and spread, so a run can log both: `tests/test_genome_merge.py::test_population_stats` passes
- [ ] Unrelated parents under repeated means lose spread faster than under per-token merges, as a regression on the finding above: `tests/test_genome_merge.py::test_mean_collapses_unrelated_parents` passes
- [ ] Children record operator and parent ids, so lineage is complete: `tests/test_genome_merge.py::test_lineage_recorded` passes

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/genome_merge.py` | CREATE | Merge operators and population stats |
| `tests/test_genome_merge.py` | CREATE | Behaviour tests |
| `scripts/verify/26Q4-01/verify_26Q4-SPG-02.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Plain functions over numpy arrays, in the idiom of `scripts/research/soft_prompt_merge_sim.py`. No torch, no serving code, no operator class hierarchy.
- `rank_weights` follows the log-rank weights CMA-ES uses for recombination; the sim's `search` has the formula.

---

#### Decision Gates

- If `26Q2-SPG-03` has not merged → branch from it; do not duplicate the genome type.
- If an operator seems to need model access → stop. Merging must remain arithmetic on arrays.
- Do not add an operator beyond these without a reason from a run. More operators split a small evaluation budget.

---

#### Out of Scope

- `26Q4-SPG-03` owns selection, fitness and which operators get real budget.
- Simplex-versus-embedding mutation geometry stays with the spec's later ablation.
- Merging text prompts: `26Q4-EVO-03`'s breeder does that.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-SPG-02.sh`.

---

#### Outcome Demo

**Where**: local
**Run**:
```bash
uv run pytest tests/test_genome_merge.py -v && uv run --no-project --with numpy python scripts/research/soft_prompt_merge_sim.py
```
**Expect**: tests pass, and the simulation prints the tables above: means collapse unrelated parents, per-token merges keep spread, and relatives keep their size under means.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-SPG-02.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <400 lines changed (target) / <800 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
