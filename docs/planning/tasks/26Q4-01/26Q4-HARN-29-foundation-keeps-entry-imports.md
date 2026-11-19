### 26Q4-HARN-29: The foundation keeps the imports its entries verified with

**User Story**: As the ratchet measurement, I want each verified artifact to reach `LMS.Foundation` with the imports it verified with, so that a generation's verified work isn't rolled back by a merged build that lacks them.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-29-foundation-keeps-entry-imports` |
| **Dependencies** | None |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (main @ `e9c7352`):
- `FoundationFile.add_artifact` drops every `import` line from a verified artifact before merging its body, on the assumption that the header covers what entries need. The header imports five modules: `Mathlib.Tactic.Common`, `Mathlib.Logic.Basic`, `Mathlib.Logic.Function.Basic`, `Mathlib.Data.Nat.Basic` and `Mathlib.Algebra.Ring.Basic`. Artifacts verify alone with their own imports, and the merged foundation is then built without them, so it fails and the generation's additions are rolled back. `set_seed` drops a seed's imports the same way.
- Batch q2 (server, 2026-11-16) had 15 verified artifacts, and all 15 fit this. The 12 importing `Mathlib.CategoryTheory.Category.Basic` or `Mathlib.CategoryTheory.Functor.Basic`, and the 1 importing `Mathlib.Algebra.Group.Subgroup.Basic`, were rolled back after `[foundation] build FAILED`. The 2 importing only `Mathlib.Algebra.Group.Defs`, which the header reaches transitively, were kept. One run verified the same definition 8 times and ended with an empty foundation.
- The kernel-track goals start from a bare seed (`"seed": ""`) and build on Mathlib's own `Category`, so every category artifact needs an import the foundation discards.
- A failed build logs only `Lake build warning: error: build failed`. `LeanProject.build` prints the first 200 characters of stderr, which hold only Lake's summary, while Lean's errors arrive on stdout.
- `get_context_for_agent` always says Mathlib's category API doesn't exist here. That was true of the from-scratch `Category` seed. It's false for goals built on Mathlib, and more so once entries carry category imports.

**Investigation**:
```bash
grep -n "split_imports" lms/foundation.py
grep -n "Lake build warning" lms/lean/project.py
grep -n "category API does NOT exist" lms/foundation.py
```

---

#### Acceptance Criteria

- [x] Each foundation entry keeps the modules its artifact imported, other than `LMS.Foundation` and the header's own. An indented import counts, a trailing comment isn't a module, and an import line inside a comment imports nothing
- [x] The saved `Foundation.lean` imports the header's modules, then the sorted union of the seed's and the entries' imports, all above `universe u v w` and the namespace
- [x] The imports round-trip through `Foundation.json`. A file written before this change still loads, with no extra imports
- [x] A rollback after a failed build, a resume from a checkpoint and a reset to the seed write exactly the imports of the entries that are left, plus the seed's
- [x] A seed's own imports are kept the same way, including a seed of imports alone
- [x] A failed build prints Lean's error lines from stdout and stderr: each error at a file position with the lines continuing its message, at most 20 lines of 300 characters
- [x] The warning that Mathlib's category API is absent appears only when the seed declares its own `Category` and no foundation import is under `Mathlib.CategoryTheory`. The agent context lists the modules `import LMS.Foundation` brings
- [x] `uv run pytest tests/test_foundation_imports.py -q` covers each item above

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | Per-entry and seed imports, derived header, persistence, agent context |
| `lms/lean/project.py` | MODIFY | Print Lean's errors when a build fails |
| `tests/test_foundation_imports.py` | CREATE | Behaviour |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-29-foundation-keeps-entry-imports.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-29.sh` | CREATE | Verify script |

---

#### Notes

- The header is derived from the entries that are left, as `26Q4-HARN-16` did for universes, so a rollback, a reset and a resume need no import bookkeeping of their own.
- Related to `26Q4-HARN-19` (the merged foundation means what each artifact verified alone). The merged file imports the union, so an entry is elaborated with modules it didn't verify with. A module's instances, simp lemmas or global notation can change what an earlier entry means, or make it ambiguous. This card fixes the case where the merge had too few imports, not that one.
- No real-Lean test. CI's `pytest (real Lean verifier)` job runs bare `lean` on core-only code with no Mathlib and no `lake build`, so a foundation that imports `Mathlib.CategoryTheory` can't be built there. The workstation has no built Mathlib either. The next server batch is the end-to-end check: category artifacts should stay in the foundation, and any failed build should log Lean's errors.
