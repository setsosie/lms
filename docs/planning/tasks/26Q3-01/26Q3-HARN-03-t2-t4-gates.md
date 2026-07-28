### 26Q3-HARN-03: T2 non-vacuity and T4 axiom/sorry gates

**User Story**: As the calibration measurement, I want statements that compile but
prove nothing to be rejected automatically, so that CVFN counts mathematics rather
than typechecking tricks.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | CRITICAL |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-03-t2-t4-gates` |
| **Dependencies** | 26Q3-HARN-01, 26Q3-HARN-02 |
| **PR Size Target** | <400 lines |

---

#### Context

`specs/faithfulness_protocol.md` §4 defines T1–T4. None are implemented.

The December runs show why T2 matters. Artifacts recorded as verified include:

```lean
example {C : Type*} [CategoryTheory.Category C] (X : C) : X ⟶ X := 𝟙 X
```

This typechecks, is anonymous, introduces nothing, and is offered as a
formalization of "Definition: Category" (Stacks tag 0013). Under any honest
accounting it is worth zero. It is exactly the degenerate strategy an agent
optimizing "get the verifier to say yes" will find.

Gates implemented here (numbering from `calibration-program.md` §2):

- **Gate 2 (T4)**: no `sorry`, no newly-introduced `axiom`, no `native_decide`.
- **Gate 3 (T2)**: non-vacuity.

---

#### Acceptance Criteria

**T4 — axiom/sorry audit**

- [ ] Rejects source containing `sorry` (already partially in `RealLeanVerifier`;
      lift into a named gate with a structured reason)
- [ ] Rejects any new `axiom` declaration in agent-authored code
- [ ] Rejects `native_decide` (compiler-trusting, not kernel-checked)
- [ ] Post-compile check via `#print axioms <name>` (lean-lsp `lean_verify`):
      the axiom set must be a subset of
      `{propext, Classical.choice, Quot.sound}`; anything else is a failure with
      the offending axiom named

**T2 — non-vacuity**

- [ ] Rejects artifacts whose Lean code introduces **no named declaration** —
      i.e. `example`-only submissions
- [ ] Rejects declarations whose statement is alpha-equivalent to an existing
      Mathlib declaration's statement (delegate the search to 26Q3-HARN-04)
- [ ] For a theorem with hypotheses, attempt a satisfiability witness: flag
      (do not silently pass) statements whose hypotheses are contradictory,
      making the theorem vacuously true
- [ ] Undecidable cases are recorded as `INCONCLUSIVE` and routed to D4 human
      review — never counted as passing

**Reporting**

- [ ] Each artifact carries `gate_results: list[GateResult]` with
      `{gate, passed, reason, detail}`
- [ ] A `gate_failure_histogram` is emitted per run — the failure distribution is
      a primary output of the calibration, not debug output

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/gates/__init__.py` | CREATE | Gate protocol + registry |
| `lms/gates/axioms.py` | CREATE | T4 |
| `lms/gates/vacuity.py` | CREATE | T2 |
| `lms/artifacts.py` | MODIFY | `gate_results` field |
| `lms/society.py` | MODIFY | Run gates after verification |
| `lms/metrics.py` | MODIFY | Gate-failure histogram |
| `tests/test_gates_axioms.py`, `tests/test_gates_vacuity.py` | CREATE | Unit tests |

---

#### Implementation Notes

- T2's general form is undecidable. Build the tractable subset and be explicit
  about the boundary — `INCONCLUSIVE` is a first-class outcome, not a bug.
- Use the `lean-lsp` MCP (`lean_verify`, `lean_declaration_file`) rather than
  reimplementing axiom-set walking.
- Gates run **after** compilation, on the compiled declaration names.

---

#### Decision Gates

- If `#print axioms` needs a full build per artifact and that is too slow to run
  in the loop, batch gate-checking into a post-generation pass and log the
  latency. Do not drop the gate.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] A regression test asserts the trivial `example` above is **rejected**
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
