# 26Q3-HARN-13: Verify an artifact in the namespace it will be stored in

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q3-HARN-13-verify-in-stored-namespace` (card's name was taken by #30's head) |
| **Dependencies** | none (independent of 26Q3-HARN-11 / #28) |
| **PR Size Target** | <250 production lines (max 500), tests and this card excluded |
| **Parts** | single PR |

---

#### Context

**The verification oracle rejects work that would be legal where it lands.**

An artifact is verified in one namespace context and stored in a different one:

- `RealLeanVerifier.verify()` does `temp_path.write_text(code)`
  (`lms/lean/real.py:194`) — the agent's code **verbatim, at top level**.
- `FoundationFile.FOUNDATION_HEADER` opens `namespace LMS.Foundation`
  (`lms/foundation.py:138`), and `add_artifact` strips the artifact's own
  `import` / `universe` / `namespace` / `end` lines before appending.

So a declaration is checked as `Functor` and stored as `LMS.Foundation.Functor`.
Any agent that picks a name Lean core or an imported library already binds at
top level fails verification for a collision **that does not exist at the
destination**.

**Observed, `shakedown_3x3_e`, 2026-08-11** (`definition-Functor-ef55362e`,
gen 2, a well-formed 8-line structure with no `namespace` line of its own):

    verify_9d6c6d67.lean:6:10: error: `Functor` has already been declared
    verify_9d6c6d67.lean:13:17: error(lean.unknownIdentifier): Unknown constant `Functor.id`

`Functor` is bound by Lean core. The second error is a knock-on: once the
`structure` is rejected, its projections do not exist either.

**Why this matters beyond hygiene.** This is a **false negative in the oracle**,
so it suppresses the CVFN numerator directly — the go/no-go number counts
verified novel statements, and this discards verified-worthy ones. The exposed
surface is every name core or an imported library binds: `Functor`, `Option`,
`Prod`, `Sum`, `Quotient`, `Set`, `Or`, `And`, and so on — precisely the
vocabulary a category-theory or algebra goal reaches for first.

**Rate is unmeasured.** It cost 1 of 7 artifacts in the single run where it was
noticed. That is one observation, not a rate, and this card does **not** assume
one — AC-5 requires measuring it over the corpus before and after. (26Q3-HARN-11
shipped a card claiming `4.2% → 0` against a check that could not fail; the
correction is in that card's measurement table. Do not repeat it here.)

---

#### Acceptance Criteria

- [x] **AC-1** A declaration whose name collides with a Lean core binding at
      top level — `structure Functor …` with no `namespace` line — verifies
      successfully, because it is elaborated inside the namespace it will be
      stored in. (`test_core_name_collision_verifies`, real compiler)
- [x] **AC-2** The wrapper is applied **after** the code's `import` lines, not
      before. Lean rejects an `import` inside a `namespace`, so a naive prefix
      turns every artifact into a syntax error.
      (`split_imports` hoist; `test_imports_hoisted_above_namespace`)
- [x] **AC-3** Code that already carries its own `namespace Foo … end Foo`
      still verifies; nesting inside `LMS.Foundation` is legal and must not
      regress. (`test_own_namespace_still_verifies`, real compiler)
- [x] **AC-4** The namespace used by the verifier and the one used by
      `FOUNDATION_HEADER` come from **one** constant. Two string literals that
      must agree is how this asymmetry arose. (`FOUNDATION_NAMESPACE` in
      `lms/foundation.py`; header, footer, `FoundationFile.NAMESPACE`, and the
      verifier wrap all consume it; the verify script rejects a second literal
      in `real.py`)
- [x] **AC-5** The change is **measured, not asserted**: see the measurement
      table below. Before: **24** artifacts across 7 archived runs. After: not
      re-runnable locally (see AC-6); the recorded runs are immutable history.
- [x] **AC-6** No artifact that verified before this change fails after it.
      Re-verifying the corpus is **not tractable on this checkout** — every
      real agent artifact imports Mathlib, and no Mathlib build exists here
      (that is a box job; the runbook owns it). Checked instead: the full
      suite (543 passed, incl. every pre-existing `RealLeanVerifier` test now
      running through the wrap) and real-compiler spot checks
      (`test_plain_theorem_still_verifies`, `test_genuine_error_still_fails`).

##### AC-5 measurement

Census of `verification_error` matching `has already been declared` over
`experiments/*/artifacts.json` (local checkout, 2026-08-19):

| Run | Artifacts with collision error |
|---|---|
| `run_20251216_220545` | 7 |
| `run_20251217_121700` | 3 |
| `run_20251218_084146` | 1 |
| `run_20251218_092910` | 1 |
| `run_20251218_105831` | 2 |
| `run_20251218_130736` | 2 |
| `test_mcp` | 8 |
| **Total (before)** | **24** |

The card's observed instance (`definition-Functor-ef55362e`,
`shakedown_3x3_e`) lives on the box and is not in this census — 24 is a
lower bound. The **after** number must come from the next box run's
histogram: the wrap makes this error class unreachable for core-name
collisions, so the expected count is 0; state the measured value in the
run report rather than here.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/lean/real.py` | MODIFY | Wrap the temp file body in the storage namespace, after imports (`:194`) |
| `lms/foundation.py` | MODIFY | Expose the namespace as one shared constant; `FOUNDATION_HEADER` consumes it (`:138`) |
| `tests/test_verify_namespace.py` | CREATE | AC-1..AC-4 |
| `docs/planning/tasks/26Q3-01/26Q3-HARN-13-verify-store-namespace-asymmetry.md` | MODIFY | This card |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-13.sh` | MODIFY | Verification |

> This table is a fence `/pre-merge` checks, not a suggestion: a changed file
> not named here (tests and this task's own card/verify script excepted) is a
> blocking finding.

---

#### Implementation Notes

`add_artifact` already strips `import` / `universe` / `namespace` / `end` from
stored code. The verifier needs the *same* split — hoist the imports, wrap the
remainder — so reuse that logic rather than writing a second copy of it. A
second copy is the bug this card exists to remove.

**This narrows the gap; it does not close it.** A single entry elaborated
inside `namespace LMS.Foundation` is still not the same as the whole foundation
compiled together — cross-entry collisions and ordering only surface in
`persist_foundation()`'s rebuild, which already exists and should stay the
final authority.

**Cheaper alternative, deliberately not chosen:** tell agents in the goal
prompt to avoid core names. That is one prompt line, but it treats a harness
defect as an agent-behaviour problem and leaves the oracle still disagreeing
with the store. Worth adding *as well*, not *instead*.

---

#### Decision Gates

Stop and surface — do not pivot — if:

- Wrapping changes the result for any artifact that previously verified
  (AC-6 fails). That means the two contexts differ in ways beyond naming, and
  the fix needs rethinking rather than forcing.
- The import hoist cannot be shared with `add_artifact` without restructuring
  it. Restructuring `add_artifact` is out of scope here.
- The change exceeds the PR Size Target.

---

#### Out of Scope

- The goal-prompt instruction about core names (separate, additive).
- Anything about *what agents do* with the foundation they are shown — that is
  the falsified-hypothesis follow-up from 26Q3-HARN-11, not this.
- Restructuring `add_artifact` or `_extract_entries`.

---

#### Verification Script

`scripts/verify/26Q3-01/verify_26Q3-HARN-13.sh` — structure checks only.
Behaviour assertions live in `tests/test_verify_namespace.py`.

---

#### Outcome Demo

**Where**: any checkout (no cluster needed — this is verifier-local).
**Run**: verify a bare core-name declaration through `RealLeanVerifier` and
confirm it now passes, then re-run the corpus error census from AC-5.
**Expect**: `structure Functor … ` with no `namespace` line verifies, and the
`has already been declared` count drops to 0 with no previously-verifying
artifact regressing.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-13.sh` exits 0, and exits 1 at
      the merge base
- [ ] `uv run ruff format`, `uv run ruff check` clean on touched files; `mypy`
      not worse than the pre-existing count
- [ ] PR opened within the size target
- [ ] Tests included with implementation
- [ ] Outcome Demo run, with the AC-5 numbers stated
