### 26Q3-HARN-08: Agents emit Lean 3, not Lean 4

**User Story**: As an agent, I want the prompt to tell me which Lean I am
writing and what I may import, so that my output has a chance of compiling
against the pinned Mathlib.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | CRITICAL |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-08-lean4-syntax-prompt` |
| **Dependencies** | 26Q3-HARN-02 |
| **PR Size Target** | <150 lines |

---

#### Context

Found on the box, 2026-08-10, in the first Gate B-minus smoke runs against
`lms-generalist` (Qwen3-Coder-30B-A3B-Instruct). Both runs produced Lean 3.

Run 1, `theorem fundamental_theorem_of_algebra`:

```
begin
  -- This is a classic result ...
  sorry
end
```

Run 2, `theorem euler_totient_product_formula`:

```
φ n = n * ∏ p ∈ (nat.prime_factors n), (1 - 1/p)
```

Three defects, one cause:

1. **`begin ... end` is Lean 3 tactic syntax.** Lean 4 is `:= by`. Run 2 got
   `:= by` right and still used `nat.prime_factors` (Lean 3 snake_case naming;
   Lean 4 is `Nat.primeFactors`), so this is not a stable per-run coin flip.
2. **No `import` line at all.** `Polynomial ℂ` and `φ` resolve to nothing
   without `import Mathlib...`, so even syntactically valid output cannot
   elaborate.
3. **`AlgebraicallyClosedField` is not a Mathlib name** (Mathlib 4 has
   `IsAlgClosed`; Mathlib 3 had `is_alg_closed`).

Cause: the non-goal system prompt (`lms/prompts.py:50-78`, used whenever
`--goal` is absent) says "LEAN 4" three times but never distinguishes it from
Lean 3, never mentions Mathlib, and never mentions imports. Lean 3 / mathlib3
dominated public Lean corpora for years, so absent an explicit instruction the
base model's prior is Lean 3.

Note this card was deliberately **excluded** from 26Q3-HARN-02, whose
implementation notes read: "Do not try to make the prompt stop emitting YAML
block scalars instead — the parser must be robust to what models actually emit.
Prompt changes can come later and independently." HARN-02 made the parser
robust; this card is the independent prompt change.

**Until this lands, no run can produce a verified artifact**, so every
verification rate measured before it is a measurement of the prompt.

---

#### Acceptance Criteria

- [ ] The non-goal system prompt states Lean 4 explicitly and names the Lean 3
      constructs to avoid (`begin`/`end`, `snake_case` lemma names, `λ x, ...`
      comma-lambda)
- [ ] The prompt requires an `import Mathlib...` line on any artifact carrying
      Lean code, and states that the pinned toolchain is `v4.27.0-rc1`
- [ ] A new prompt version is added rather than edited in place, following the
      existing `PromptVersion` convention in `lms/prompts.py`
- [ ] The goal-directed prompt (`AGENT_SYSTEM_PROMPT_V2_GOAL`) gets the same
      treatment, or is shown by test to already carry it
- [ ] A cheap static check flags Lean 3 syntax in `lean_code` before the
      artifact reaches the verifier, recording the reason distinctly from a
      compiler rejection (coordinate with 26Q3-HARN-03 so the two gates share a
      rejection vocabulary)
- [ ] Tests assert the prompt text contains the Lean 4 / import instructions,
      so a future prompt edit cannot silently drop them

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/prompts.py` | MODIFY | New non-goal + goal system prompt versions |
| `tests/test_prompts.py` | CREATE/MODIFY | Assert the instructions are present |

---

#### Implementation Notes

- Do not fix this by post-processing `begin`/`end` into `by` — a translator
  that half-works would make failures harder to read, not easier. The prompt
  states the requirement; the compiler enforces it.
- Worth re-measuring after landing: if a 30B model still emits Lean 3 at a high
  rate with an explicit instruction, that is a model-selection finding for
  [[model-selection-2026q2]], not a prompt bug.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `uv run pytest` clean
- [ ] `uv run ruff check` / `uv run mypy` clean on the files this card touches
- [ ] A 1-agent smoke run on the box produces Lean 4 syntax with an import line
      (verified rate may still be 0 — that is a separate question)
