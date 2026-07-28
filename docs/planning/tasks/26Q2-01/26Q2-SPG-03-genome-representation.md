### 26Q2-SPG-03: Soft-prompt genome representation + mutation

**User Story**: As the LMS evolution loop, I want a compact, serializable
soft-prefix genome with sparse k-hot init and gradient-free mutation, so that
per-agent "minds" can be represented and varied on a frozen model without any
fine-tuning.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q2-SPG-03-genome-representation` |
| **Dependencies** | None (pure representation; no serving — can start in parallel with SPG-01) |
| **PR Size Target** | <500 lines |

---

#### Context

> Phase 2 of `specs/soft_prompt_genome_work_plan.md`, the part that needs no
> cluster. Implements the genotype from `specs/soft_prompt_genome.md` §2–§3: a
> per-agent sparse soft prefix `e = pᵀE`, where `p` lives on the vocabulary
> simplex `Δ^{V-1}` and k-hot sparsity (k=2–5) avoids the centroid collapse the
> probe measured (uniform Dirichlet → `1/√V` ≈ 0.0026 diversity; k=2 → 0.71).

**Current State**:
- `scratch/embedding_collapse_probe.py` demonstrates the sparsity fix (numpy,
  verified to reproduce the spec's table).
- `lms/agent.py` (611 lines) is the per-agent surface a genome would later feed.
- No genome module exists: `grep -ril "genome" lms/ tests/` → empty.
- `numpy` is **not** a project dependency yet (`pyproject.toml` lists
  anthropic/openai/google-generativeai/python-dotenv only); the probe used
  `uv run --with numpy`.

**Investigation**:
```bash
grep -ril "genome\|soft_prompt\|soft.prefix" lms/ tests/   # → (none)
grep -n "numpy" pyproject.toml                              # → (none)
```

---

#### Acceptance Criteria

> Each criterion verifiable with a single command returning exit code 0.

- [ ] `numpy>=2.0` added to `[project].dependencies` in `pyproject.toml`
- [ ] `lms/genome.py` exists with `SoftPrefixGenome`
- [ ] `SoftPrefixGenome` carries a genotype of shape `(k, V)` simplex rows (or an
      equivalent sparse `(k, active)` representation) and exposes `k`, `vocab_size`
- [ ] `SoftPrefixGenome.random_khot(k: int, vocab_size: int, active: int, *, rng) -> SoftPrefixGenome`
      constructs a k-hot sparse genome with `active` nonzero tokens per soft token
- [ ] `SoftPrefixGenome.to_embedding(E: np.ndarray) -> np.ndarray` returns the
      `(k, d_model)` soft prefix `pᵀE`
- [ ] `SoftPrefixGenome.mutate(sigma: float, *, rng) -> SoftPrefixGenome` applies
      Gaussian mutation and **re-projects** rows back onto the simplex (nonneg, sum-to-1)
- [ ] `SoftPrefixGenome.to_dict()/from_dict()` round-trip preserves the genome
      exactly (for persistence alongside run state)
- [ ] `tests/test_genome.py` covers: k-hot sparsity (exactly `active` nonzeros),
      simplex invariant after init **and** after mutate, embedding shape `(k, d)`,
      serialization round-trip, and a collapse-regression check (uniform-dense init
      has radius ratio < a k=2 genome) mirroring the probe
- [ ] Tests pass: `uv run pytest tests/test_genome.py -v`
- [ ] Type check passes: `uv run mypy lms/genome.py`
- [ ] No lint errors: `uv run ruff check lms/genome.py`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/genome.py` | CREATE | `SoftPrefixGenome` representation + ops |
| `tests/test_genome.py` | CREATE | Unit tests (TDD red first) |
| `pyproject.toml` | MODIFY | Add `numpy>=2.0` dependency |
| `scripts/verify/26Q2-01/verify_26Q2-SPG-03.sh` | CREATE | Verification script |

---

#### Implementation Notes

> Mirror the existing `@dataclass` config style in `lms/working_group.py`
> (`WorkingGroupConfig`) and the numpy idioms already in
> `scratch/embedding_collapse_probe.py` — do not invent a new abstraction layer.

- Keep it a plain dataclass + free functions or small methods; no torch, no
  serving code, no agent wiring in this PR.
- Sparsity is the whole point — store rows sparsely or assert the nonzero count;
  a dense `(k, 150000)` float array per agent is acceptable for tests but note it.
- `mutate` must preserve the simplex (project: clip negatives to 0, renormalize);
  an unprojected Gaussian step leaves `Δ^{V-1}` and breaks `e = pᵀE` semantics.
- The simplex-vs-embedding mutation question (e-/m-geodesic, spec §7.5) is an
  **ablation for Phase 5** — implement simplex-coords mutation here, leave a
  `# TODO(SPG-05): embedding-space mutation arm` marker, do not build both.

---

#### Decision Gates

> When reality contradicts this card, STOP and surface — do not silently pivot.

- If `numpy>=2.0` conflicts with an existing pin or CI → stop, report the conflict,
  propose the compatible version; do not downgrade other deps unilaterally.
- If representing the genome cleanly requires touching `lms/agent.py` or the
  serving path → stop; that plumbing is **SPG Phase 2 proper / SPG-02**, not this card.
- If the change exceeds the PR Size Target → stop and split (e.g. serialization
  into a follow-up), don't power through.

---

#### Out of Scope

- `26Q2-SPG-01` owns serving (`inputs_embeds`/`prompt_embeds`) — no engine code here.
- `26Q2-SPG-02` owns the diversity measurement — no LEAN/agent-run code here.
- SPG Phase 3 owns evolution (selection, fitness, crossover) — `mutate` only here.
- No activation-steering genome (the robustness-hedge phenotype) in this PR.

---

#### Verification Script

See `scripts/verify/26Q2-01/verify_26Q2-SPG-03.sh`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q2-01/verify_26Q2-SPG-03.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened, <500 lines, tests included with implementation
