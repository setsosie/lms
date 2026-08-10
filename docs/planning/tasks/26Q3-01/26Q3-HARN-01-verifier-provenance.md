### 26Q3-HARN-01: Verifier provenance — mock can never mark an artifact verified

**User Story**: As someone forecasting a multi-thousand-statement program from
experiment data, I want it to be structurally impossible for heuristically-checked
artifacts to be recorded as verified, so that no future roadmap is calibrated on a
regex.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | CRITICAL |
| **Status** | ✅ DONE |
| **Branch** | `26Q3-HARN-01-verifier-provenance` |
| **PR** | #14 |
| **Verify** | `scripts/verify/26Q3-01/verify_26Q3-HARN-01.sh` |
| **Dependencies** | 26Q3-CHORE-01 |
| **PR Size Target** | <250 lines |

---

#### Context

> This is the root cause of `docs/planning/2026-07-24-feasibility-assessment.md`.
> The three-text roadmap's "~50 statements per WC cycle / 400–500K tokens" figure
> came from a run whose 92% verification rate was produced by
> `MockLeanVerifier`'s regex. Nothing in the artifact record distinguished it from
> a real Lean result.

**Current State**:
- `lms/run.py:493` — `--verifier` defaults to `"mock"`.
- `lms/lean/mock.py` — accepts anything matching
  `^\s*(theorem|lemma|def|axiom|example|structure|inductive|class)\s+`.
- `lms/artifacts.py:35` — `Artifact.verified: bool`. One boolean, no provenance.
- `lms/society.py:333` — `artifact.verified = result.success`, regardless of which
  verifier produced `result`.
- No experiment `metadata.json` in `experiments/` records a verifier field.

**The fix is a type change, not a warning.** A boolean cannot carry provenance, so
replace it with a status enum and make the mock physically unable to emit the
verified value.

---

#### Acceptance Criteria

- [x] `VerificationResult` (`lms/lean/interface.py`) carries a `verifier_id: str`
      and `verifier_kind: Literal["mock", "real", "mcp"]`
- [x] New `VerificationStatus` enum on `Artifact`:
      `UNVERIFIED | VERIFIED_HEURISTIC | VERIFIED_LEAN | FAILED`
- [x] `MockLeanVerifier` can only produce `VERIFIED_HEURISTIC`; asserted by a test
      that fails if the mock ever yields `VERIFIED_LEAN`
- [x] `Artifact.verified` becomes a read-only property → `status is VERIFIED_LEAN`,
      so every existing call site keeps working but tightens to real-Lean-only
- [x] `ArtifactLibrary.get_verified()` returns only `VERIFIED_LEAN`
- [x] `metadata.json` gains `verifier: {kind, id, lean_version, mathlib_rev}`
- [x] Backward-compatible load: `Artifact.from_dict` maps a legacy
      `verified: true` with no status field to `VERIFIED_HEURISTIC` — **never** to
      `VERIFIED_LEAN`. Historical runs must not be silently promoted
- [x] `lms/run.py` prints a loud banner when running with the mock verifier
- [x] Tests: mock→heuristic, real→lean, legacy-load demotion, metadata round-trip

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/lean/interface.py` | MODIFY | `verifier_kind`/`verifier_id` on result; `VerificationStatus` enum |
| `lms/lean/mock.py` | MODIFY | Emit `VERIFIED_HEURISTIC` only |
| `lms/lean/real.py`, `lms/lean/mcp.py` | MODIFY | Emit `VERIFIED_LEAN`; report toolchain version |
| `lms/artifacts.py` | MODIFY | `status` field, `verified` property, serialization |
| `lms/society.py` | MODIFY | Set status from result; metadata block |
| `tests/test_verifier_provenance.py` | CREATE | Unit tests |

---

#### Decision Gates

- If changing `Artifact.verified` from field to property breaks many call sites,
  keep the property and add `status` alongside — do **not** leave a settable
  boolean that can be assigned `True` by anything but a real verifier.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `uv run pytest` clean — 431 passed
- [~] `uv run ruff check` / `uv run mypy` — **no new findings vs `main`**, not
      absolutely clean. The repo carries 14 ruff and 18 mypy findings that
      predate this branch; 5 pre-existing ruff errors in touched test files and
      8 pre-existing mypy errors in `society.py` are fixed here. Absolute
      cleanliness is its own chore, not this card's scope.
- [x] PR opened (#14), tests included
- [x] Verify script `scripts/verify/26Q3-01/verify_26Q3-HARN-01.sh` — PASS on
      branch, 11 of 12 checks FAIL at the merge base (the twelfth is the
      full-suite regression guard, which passes at base by design)
