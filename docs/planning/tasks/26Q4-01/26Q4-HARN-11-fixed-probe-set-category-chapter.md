### 26Q4-HARN-11: Fixed probe set from the Stacks category chapter, scored by penalized cost

**User Story**: As the person comparing prompts and soft prompts, I want a fixed set of pinned statements running from easy to hard, with a dev and a held-out split, so that a change is scored by Lean on the same statements every time and no human has to review each candidate.

> Drafted by Claude on 2026-11-05 from a discussion with the user. The user chose the category chapter and setting the token cap from this card's baseline run. Review the rest as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-11-fixed-probe-set-category-chapter` |
| **Dependencies** | `26Q3-HARN-28` (`lms/statement.py`, pinning) |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `1a9172e`):
- No fixed test set exists. Every fitness idea so far reads whole-run outcomes, which vary with which statements a run happened to attempt.
- The Stacks tag file has 216 lemmas, definitions, theorems and propositions in `categories`, and the 2-Yoneda statement `004B` is in `goals/stacks_kernel_track_b.json`.
- `lms/goals.py:383` labels the Yoneda milestone `0019`. The tag file gives `0019` as `categories-example-group-groupoid` and `001P` as `categories-lemma-yoneda`. Probes use the tag file's labels.
- `26Q3-HARN-28` pins statements in committee mode only, and its card says not to touch the iterative path. A probe attempt therefore calls `pin_statement` itself.
- `lms/lean/real.py:265` verifies through a temporary file, so an attempt that imports no foundation may not need the fixed `Foundation.lean` path.
- `lms/accounting.py:92` keeps `completion_tokens` per record. `CostLedger.total_tokens` includes prompt tokens, and budgets are in output tokens.

**Investigation**:
```bash
grep -cE '^[0-9A-Z]{4},categories-(lemma|definition|theorem|proposition)' references/stacks-project/tags/tags
# 216
grep -nE '^(0019|001P|004B),' references/stacks-project/tags/tags
grep -n "tempfile" lms/lean/real.py
```

---

#### Scope

**Scoring.** For genome `g` and probe `p`, `cost(g, p)` is the output tokens to the first verified proof of the pinned statement, or twice the cap if unsolved (PAR-2, from SAT solver competitions). Fitness is the negative mean cost. Verified means the pinned header compiled under `RealLeanVerifier` and no gate in `gate_results` failed, so there is no partial credit: the gradient comes from probes of graded difficulty.

**Starting point.** Probes use Mathlib's definitions, with named results blocked where a probe would otherwise be one line.

---

#### Acceptance Criteria

- [ ] `data/probes/categories_v1.json` holds candidates with `tag`, `section`, `informal`, `lean_statement` (ending `:= by sorry`), `band` (`easy`, `medium`, `hard`), `split` (`dev`, `heldout`) and `forbidden_constants`
- [ ] Every `lean_statement` elaborates at the pinned library build: `uv run pytest tests/test_probes.py -m lean` passes where Lean is available
- [ ] `001P`, the general Yoneda statement and `004B` are in the held-out split and flagged `milestone: true`, so their results are reported separately
- [ ] Dev and held-out probes come from different sections of the chapter, so a dev proof does not state a held-out result: `tests/test_probes.py::test_splits_share_no_section` passes
- [ ] `lms/probes.py` has `ProbeSet.load`, which refuses overlapping splits or a duplicate tag: `tests/test_probes.py::test_load_refuses_overlap` passes
- [ ] `lms/probes.py::par2_cost` returns tokens on success and `2 × cap` on failure: `tests/test_probes.py::test_failure_costs_twice_cap` passes, including an early failure that spent few tokens
- [ ] A proof that compiles but has a failed gate counts as unsolved, even if its status reads `verified_lean`: `tests/test_probes.py::test_gate_failure_is_not_a_solve` passes
- [ ] A proof whose term uses a `forbidden_constants` entry counts as unsolved: `tests/test_probes.py::test_forbidden_constant_is_not_a_solve` passes
- [ ] The attempt runner pins the statement before every verify, including repair attempts: `tests/test_probes.py::test_pinned_header_reaches_verifier` passes
- [ ] Token counts use `completion_tokens` only: `tests/test_probes.py::test_cost_uses_output_tokens` passes
- [ ] `lms/probes.py::require_clean_cutoff` refuses a model whose recorded training cutoff is missing or not before 2026-11-04: `tests/test_probes.py::test_cutoff_refused` passes
- [ ] Each baseline record carries the model id, training cutoff, prompt version, probe file hash and cap
- [ ] `scripts/probes/run_baseline.py` runs one genome over a split for `--attempts K` at `--cap N`, and takes `--dev-only` so a loop cannot read held-out probes by accident
- [ ] The baseline run keeps probes the hand-written prompt solves sometimes, drops those it always or never solves (milestones excepted), and records the chosen cap in this card — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `data/probes/categories_v1.json` | CREATE | The probe statements |
| `lms/probes.py` | CREATE | Loading, split checks, PAR-2, attempt runner, cutoff check |
| `scripts/probes/run_baseline.py` | CREATE | One genome over one split |
| `tests/test_probes.py` | CREATE | Behaviour tests |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-11.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Reuse `pin_statement` from `lms/statement.py`, `RealLeanVerifier`, and the pairing helpers in `lms/arms.py`. Do not build a second verifier or statistics module.
- Run each genome in a subprocess, as `26Q4-EVO-02` requires, so a crash or leaked state cannot affect the next genome.
- Choose the cap after the baseline: run at a generous cap, then take the smallest cap that keeps about 90% of the solves.
- Keep Axiom's Stacks library out of every path an agent can read. It is not a probe source, and it does not go under `references/`.

---

#### Decision Gates

- If the attempt runner turns out to need `lean/LMS/Foundation.lean` → stop and surface it. Attempts then run one at a time, and the wall-clock estimates change.
- If a transitive check of `forbidden_constants` also rejects ordinary basic lemmas → stop and show which, rather than loosening it to direct constants alone.
- If fewer than 10 probes per split survive the baseline → stop. This is the flat-landscape result, and the response is an easier band or a decision about the arm, not partial credit.
- If `26Q3-HARN-28` has not merged → wait; do not reimplement pinning.
- If the change exceeds the PR Size Target → split the statement file from the code.

---

#### Out of Scope

- `26Q4-EVO-02` owns the fitness record and `26Q4-EVO-03` the loop; this card supplies the probes and the cost.
- Agents writing their own definitions, the second Yoneda starting point: later, with a bridge to Mathlib's definitions.
- Comparing results against Axiom's library.
- Correcting the `0019` label in `lms/goals.py`.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-11.sh`.

---

#### Outcome Demo

**Where**: the GPU server, as a round folder Claude prepares; the user runs it.
**Run**: the round's `RUN_THIS.txt`, which submits `scripts/probes/run_baseline.py` for the hand-written prompt on every candidate, pinned to the merged commit.
**Expect**: a per-probe table of solves and output tokens, Yoneda's hit rate reported separately, a chosen cap, and a statement on whether enough probes sit between always and never solved. The user reviews each kept probe card once for faithfulness.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-11.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
