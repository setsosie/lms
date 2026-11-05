### 26Q4-SPG-04: Gradient-tuned soft prompt, as an option alongside evolution

**User Story**: As the person evolving soft prompts, I want to tune a soft prompt by gradient on verified Lean proofs and compare it with no soft prompt, so that I can see what this control surface can do with a directed search, and optionally start evolution from it.

> Drafted by Claude on 2026-11-05. The user approved gradient tuning as an option, with evolution as the main interest, so this does not gate `26Q4-SPG-03`. Review as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | LOW |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-SPG-04-gradient-tuned-soft-prompt` |
| **Dependencies** | `26Q4-HARN-11`, `26Q4-SPG-01` green, `26Q2-SPG-01` green |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**: no tuning code exists, and `lms/` has no `torch` dependency. `docs/planning/org-structures.md` does not train anything.

- Prompt tuning (Lester et al. 2021, arXiv:2104.08691) learns soft-prompt embeddings by gradient on a frozen model. One backward pass gives a direction across all values, which random mutation cannot do at this budget (`26Q4-SPG-02`).
- Its objective is a proxy, the likelihood of known proofs, not solved statements. So a tuned prompt that fails on probes is strong evidence against this surface's leverage, not proof.
- Soft prompts can hold content. Our literature review cites work where one hidden-state vector drives a model to reproduce target text (Subramani et al. 2022, arXiv:2205.05124). The tuning corpus therefore decides what the result means.

---

#### Scope

Tune one soft prompt on a filtered corpus, then compare it with no soft prompt on dev probes, then held-out. Optionally produce several, from different parts of the corpus, as a start population for `26Q4-SPG-03`.

---

#### Acceptance Criteria

- [ ] `lms/tuning_corpus.py::filter_corpus(proofs, probes)` drops every proof of a dev or held-out statement, and any proof naming one: `tests/test_tuning_corpus.py::test_probe_proofs_excluded` passes
- [ ] The corpus uses material available before 2026-11-04 and no Axiom source: `tests/test_tuning_corpus.py::test_refuses_excluded_source` passes
- [ ] `scripts/soft_prompt/tune.py` optimizes only the prefix values, with model weights frozen, and records model revision, corpus hash, loss curve and final prefix hash
- [ ] The model's training cutoff is recorded, and one missing or not before 2026-11-04 is refused
- [ ] Tuned-versus-none is compared per probe on dev, then held-out, with the same output-token cap and the paired test in `lms/arms.py` — accepted via Outcome Demo
- [ ] The report repeats the comparison on the served build if tuning used different weights — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/tuning_corpus.py` | CREATE | Corpus filtering |
| `scripts/soft_prompt/tune.py` | CREATE | Tuning job |
| `tests/test_tuning_corpus.py` | CREATE | Filtering tests |
| `scripts/verify/26Q4-01/verify_26Q4-SPG-04.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Keep `torch` in the script, run with `uv run --with`, so `lms/` gains no dependency.
- We serve an FP8 build, and gradients may need unquantized weights. If so, tune there and confirm the effect on the served build.
- A tuned prompt fits parameters to proofs, so it is closer to training than to culture. In any comparison with Phase C it counts as a changed model, not an organizational or cultural gain.

---

#### Decision Gates

- If a tuned prompt shows held-out gains but the corpus filter was changed after seeing them → discard the result and re-run.
- If gradients cannot run on the available weights without new infrastructure → stop and ask; this card is optional.
- If tuned-versus-none shows no lift → report it, and do not treat that as ending `26Q4-SPG-03`, which also tests diversity. The user decides.

---

#### Out of Scope

- Training or modifying model weights.
- `26Q4-SPG-03` owns evolution; this card supplies at most a start population.
- `26Q4-EVO-03`'s text prompts.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-SPG-04.sh`.

---

#### Outcome Demo

**Where**: the GPU server, as a round folder Claude prepares; the user runs it.
**Run**: the round's `RUN_THIS.txt`: tune, serve the model, then run tuned and none on dev probes, and on held-out probes if dev shows a gain.
**Expect**: a loss curve, a per-probe table of solves and output tokens, and a paired result on held-out probes, whichever way it falls.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-SPG-04.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
