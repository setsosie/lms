### 26Q4-HARN-08: Local provider pool and solve correlation — heterogeneous populations (the H9 arm)

**User Story**: As the organization-sweep experimenter, I want to assign agents round-robin from a pool of local OpenAI-compatible endpoints, and to measure how correlated agents' per-statement solves are, so that "a heterogeneous population covers more of the DAG because its members fail on different statements" is a measured claim rather than an assumption.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | LOW |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-08-local-provider-pool` |
| **Dependencies** | `26Q4-HARN-03` (extends `lms/arms.py`). Stage 3 of `docs/planning/org-structures.md` |
| **PR Size Target** | <350 lines (max 700) |
| **Parts** | single PR |

---

#### Context

The book draft (chapter 12) bounds a population's effective size by 1/ρ, where
ρ is the correlation of members' errors, and calls heterogeneity "the only
lever because it is the only thing that lowers ρ". It also carries the caveats:
temperature spread inside one model is not the lever, and "Self-MoA
outperforms standard MoA" (chapter 13). The isolated heterogeneous arm needs no
pool: alternate the served model across I9's nine runs. An *interacting*
heterogeneous arm needs agents from different endpoints in one society.

**Current State**:
- `Society` already takes `providers: list[BaseLLMProvider]`, one per agent (`lms/society.py:129`).
- `lms/run.py` builds such a list only for `--mixed`, one agent per *API vendor* (`config.available_providers()`), so two local models are unreachable.
- Agent ids are `agent-{i}-{provider.name}` (`lms/society.py`), and `name` is a class attribute: `"openai"` for every OpenAI-compatible endpoint (`lms/providers/openai.py`). Two local models would produce indistinguishable ids.
- `metadata.json` records one `model` (`lms/run.py:421`).

**Investigation**:
```bash
grep -n "mixed_providers\|available_providers" lms/run.py lms/config.py
# --mixed only, vendor-keyed
grep -n "name: str = " lms/providers/*.py
# base.py "base", openai.py "openai": a class attribute, not per instance
```

---

#### Acceptance Criteria

- [ ] `lms/run.py` adds `--provider-pool <pool.json>`, rejected with `--mixed` and `--groups`. The pool file is a list of `{name, base_url, model, max_tokens}`; a missing field, a duplicate `name` or an empty list is an error at load naming the entry
- [ ] Agents are assigned round-robin; agent ids carry the pool entry's `name`, so `agent-3-qwen` and `agent-4-prover` are distinguishable
- [ ] `metadata.json` records the pool and the agent→entry map; every artifact's author id identifies its model
- [ ] `lms/arms.py` adds `solve_correlation(run_dirs: list[Path], statements: list[str])`: the mean pairwise correlation ρ̂ of per-statement solve indicators across runs, the Kish effective size N/(1+(N−1)ρ̂), and the count of statements no run solved. It reports, rather than divides by, a run with zero variance
- [ ] `scripts/compare_arms.py` prints ρ̂, the effective size and the all-unsolved count per pooled arm
- [ ] `tests/test_provider_pool.py` covers: pool validation errors; round-robin assignment; distinct agent ids; metadata. `tests/test_arms.py` covers `solve_correlation` on a hand-computed fixture, including a zero-variance run
- [ ] All tests pass: `uv run pytest tests/test_provider_pool.py tests/test_arms.py tests/test_config.py -v`
- [ ] `uv run ruff check` and `uv run mypy` clean on the touched files
- [ ] A two-model flat run on the box records both models per agent — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/config.py` | MODIFY | `load_provider_pool(path) -> list[ProviderConfig]` with per-entry names |
| `lms/providers/openai.py` | MODIFY | Accept an instance name; the class default stays `"openai"` |
| `lms/run.py` | MODIFY | Flag, provider list, metadata |
| `lms/arms.py` | MODIFY | `solve_correlation` |
| `scripts/compare_arms.py` | MODIFY | Print ρ̂, effective size, all-unsolved |
| `tests/test_provider_pool.py` | CREATE | Pool tests |
| `tests/test_arms.py` | MODIFY | Correlation tests |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-08.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Mirror `--mixed`'s construction of the `providers` list; do not add a second society constructor path.
- `ProviderConfig` gains an optional `name` defaulting to the vendor name, so every existing config is untouched.
- Serving two models on the box is a runbook matter (`docs/runbooks/`), not code: Claude writes the commands, you run them.

---

#### Decision Gates

- If the box cannot serve two models at the context Phase C uses → stop and report the measured limit. Do not lower the context for one arm only.
- If committee or coordinator modes turn out to need per-member providers for the comparison → stop; `WorkingGroup` takes one provider, and that is a separate change.
- If ρ̂ needs more than per-statement solve indicators, such as per-attempt correlations → stop and surface; that is a design question.

---

#### Out of Scope

- `26Q4-HARN-05`, `-06` and `-07` own the other new arms.
- No per-role or per-member providers in committee mode.
- No temperature-spread "heterogeneity" (the book draft says it is not the lever).

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-08.sh`.

---

#### Outcome Demo

**Where**: the box (you run it; Claude prepares the commands)
**Run**:
```bash
# after serving two models per the runbook, with pool.json naming both:
uv run python -m lms.run --iterative --agents 4 --generations 1 --goal <slice goal> \
  --verifier real --provider-pool pool.json --output experiments/pool_smoke
grep -o '"agent-[0-9]-[a-z0-9_-]*"' experiments/pool_smoke/metadata.json | sort -u
```
**Expect**: four agent ids alternating between the two pool names; `metadata.json` maps each to its model.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-08.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <350 lines changed (target) / <700 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
