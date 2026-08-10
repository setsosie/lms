### 26Q3-HARN-07: Verifier must invoke Lean with the project environment

**User Story**: As the calibration harness, I want `RealLeanVerifier` to verify
Lean code that imports Mathlib, so that Gate B measures the pipeline rather than
a missing search path.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | CRITICAL |
| **Status** | ✅ DONE |
| **Branch** | `26Q3-HARN-07-verifier-lean-path` |
| **Dependencies** | 26Q3-CHORE-01 |
| **PR Size Target** | <120 lines |

---

#### Context

Found on the H100 box, 2026-07-28, during runbook Step 2c.

`RealLeanVerifier.verify` invokes the compiler as:

```python
proc = await asyncio.create_subprocess_exec(
    self.lean_path,          # bare `lean`
    str(temp_path),
    ...                      # no env=, no cwd=
)
```

`lms/lean/real.py:111-116`. Nothing puts the project's
`.lake/packages/*/build/lib/lean` directories on `LEAN_PATH`, so **the verifier
can only check import-free Lean.** Any snippet with `import Mathlib...` fails
with `unknown module prefix 'Mathlib'` — an environment error, indistinguishable
in the result from a genuine proof failure.

This was masked because the runbook's only smoke tests imported nothing:

- `theorem cal_smoke (n : Nat) : n + 0 = n := by simp` — no imports, passes
- `theorem nonsense : 1 = 2 := by rfl` — no imports, correctly rejected
- `theorem bad ... := by sorry` — never reaches Lean at all; caught by the
  `if "sorry" in code` substring test at `real.py:86`

Every real agent proof imports Mathlib. Left unfixed, Gate B reports **zero**
verified statements for reasons having nothing to do with the models, and the
CVFN numerator is structurally zero.

The correct invocation is known to work — it is what produced the Step 2c axiom
audit on the box:

```bash
cd ~/code/lms/lean && lake env lean "$SCRATCH/axcheck.lean"
```

`lake env` exports the package's `LEAN_PATH` and runs the given command inside
it.

---

#### Acceptance Criteria

- [ ] When `project_dir` is set, `verify()` invokes `lake env lean <file>` with
      `cwd` = the Lean project directory
- [ ] When `project_dir` is **not** set, behavior is unchanged (bare
      `self.lean_path`), so the existing no-project tests keep passing
- [ ] A snippet importing Mathlib verifies successfully against the real project
- [ ] A *false* theorem that imports Mathlib is still **rejected** — the fix must
      not turn the oracle into a yes-machine
- [ ] `lake` is located the same way `lean` is (`_find_lean`'s PATH + `ELAN_HOME`
      logic), with a clear error if absent
- [ ] Tests assert the command vector and `cwd` via a mocked subprocess, so they
      run on a machine with no Lean toolchain

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/lean/real.py` | MODIFY | `lake env lean` when a project is configured |
| `tests/test_lean_real_env.py` | CREATE | Mocked-subprocess tests for the command vector |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-07.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Do not set `LEAN_PATH` by hand. `lake env` is the supported mechanism and
  stays correct when the dependency set changes.
- Keep the bare-`lean` path for `project_dir=None`; `tests/test_lean_real.py`
  depends on it and it is the right behavior for import-free snippets.
- Out of scope: the `if "sorry" in code` substring gate (`26Q3-HARN-03`) and the
  transitive-`sorryAx`-through-import hole. Both are real, both are that card.

---

#### Decision Gates

- If `lake env lean` proves too slow per-call to be viable at Gate C volume,
  stop and measure before optimizing — a persistent `lake serve` / LSP session
  is a much larger change and needs its own card.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-07.sh` exits 0
- [ ] `uv run pytest` → 0 failed
- [ ] Runbook Checkpoint 3c passes on the box (currently documented as an
      expected failure — flip it to expected-pass in the same PR)
