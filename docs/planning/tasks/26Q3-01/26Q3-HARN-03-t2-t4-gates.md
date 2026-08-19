### 26Q3-HARN-03: T2 non-vacuity and T4 axiom/sorry gates

**User Story**: As the calibration measurement, I want statements that compile but
prove nothing to be rejected automatically, so that CVFN counts mathematics rather
than typechecking tricks.

| Field | Value |
|-------|-------|
| **Story Points** | 5 |
| **Priority** | CRITICAL |
| **Status** | 🔄 IN PROGRESS |
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

- [x] Rejects source containing `sorry` (already partially in `RealLeanVerifier`;
      lift into a named gate with a structured reason) — `T4.sorry`,
      word-boundary match on comment-stripped source
- [x] Rejects any new `axiom` declaration in agent-authored code —
      `T4.axiom_decl`, offending names in `detail`
- [x] Rejects `native_decide` (compiler-trusting, not kernel-checked) —
      `T4.native_decide`
- [x] Post-compile check via `#print axioms <name>`: the axiom set must be a
      subset of `{propext, Classical.choice, Quot.sound}`; anything else is a
      failure with the offending axiom named — `T4.axiom_audit`. Implemented by
      appending `#print axioms` to the artifact and compiling under the
      verifier's own `lake env lean` (the lean-lsp MCP is not importable from
      the harness; the subprocess route reuses the exact verification
      environment). No toolchain → INCONCLUSIVE, never a pass

**T2 — non-vacuity**

- [x] Rejects artifacts whose Lean code introduces **no named declaration** —
      i.e. `example`-only submissions — `T2.named_declaration`
- [x] Alpha-equivalence to an existing Mathlib statement: **delegated to
      26Q3-HARN-04** via an injectable `duplicate_checker`; until wired,
      `T2.duplicate` reports INCONCLUSIVE (routed to D4) rather than
      pretending it searched
- [x] For a theorem with hypotheses, attempt a satisfiability witness —
      `T2.hypothesis_satisfiability`: explicit `False` and textual `P`/`¬P`
      hypothesis pairs FAIL; otherwise an `example : ∃ <telescope>, True`
      witness probe runs, and no-witness-found is INCONCLUSIVE (flagged),
      never a silent pass
- [x] Undecidable cases are recorded as `INCONCLUSIVE` and routed to D4 human
      review — never counted as passing (`GateResult.passed` is strict;
      `Artifact.gates_passed` requires gates to have run)

**Reporting**

- [x] Each artifact carries `gate_results: list[GateResult]` with
      `{gate, outcome, passed, reason, detail}`, serialized in artifacts.json
- [x] A `gate_failure_histogram` (and a separate `gate_inconclusive_histogram`)
      is emitted per run via `analyze_library` / `print_analysis`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/gates/__init__.py` | CREATE | Re-exports + `default_gate_runner` (kept minimal so 26Q3-HARN-04's `novelty.py` merges cleanly) |
| `lms/gates/base.py` | CREATE | Gate protocol, `GateResult`/`GateOutcome`, `GateRunner`, `LeanProbeRunner` |
| `lms/gates/lean_source.py` | CREATE | Best-effort Lean source scanning (declarations, namespaces, theorem binders) |
| `lms/gates/axioms.py` | CREATE | T4 |
| `lms/gates/vacuity.py` | CREATE | T2 |
| `lms/artifacts.py` | MODIFY | `gate_results` field + strict `gates_passed` |
| `lms/society.py` | MODIFY | `_apply_gates` after verification in all three paths |
| `lms/metrics.py` | MODIFY | Gate-failure + gate-inconclusive histograms |
| `tests/test_gates_axioms.py`, `tests/test_gates_vacuity.py` | CREATE | Unit tests |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-03.sh` | CREATE | Structure checks + pytest delegation |

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

- [x] All acceptance criteria checked off
- [x] A regression test asserts the trivial `example` above is **rejected**
      (`test_trivial_example_rejected`)
- [x] `uv run pytest` clean (576 passed, 43 new). Neither ruff nor mypy is a
      project dependency; via `uvx`, `lms/gates/` + both test files are
      ruff-clean and mypy-clean. `lms/society.py` carries 4 pre-existing ruff
      findings and 3 pre-existing mypy findings at the merge base — left
      untouched to keep the diff minimal for the parallel HARN-05/-12 work
      in the same file
