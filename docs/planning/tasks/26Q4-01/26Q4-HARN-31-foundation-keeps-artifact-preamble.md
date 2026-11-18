### 26Q4-HARN-31: The foundation keeps the `open` and `variable` lines its entries verified under

**User Story**: As the ratchet measurement, I want each verified artifact to reach `LMS.Foundation` with the `open` and `variable` lines it verified under, so that a verified definition isn't rolled back by a merged build that can't parse or name its binders.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-31-foundation-keeps-artifact-preamble` |
| **Dependencies** | `26Q4-HARN-29` (stacked on its branch) |
| **Gate** | fast |
| **PR Size Target** | <300 lines (max 500) |
| **Parts** | single PR |

---

#### Context

**Current State** (`26Q4-HARN-29` @ `65b97bf`):
- `FoundationFile._extract_entries` slices an artifact from each declaration to the next. Whatever sits above the first declaration belongs to no entry and never reaches `Foundation.lean`. In practice that is the artifact's `open` and `variable` lines.
- Batch q3 (server, 2026-11-17) ran six track-B runs and verified 25 artifacts; the foundation kept none. All 25 opened with `open CategoryTheory`, and 12 also declared their binders with `variable`, such as `variable (p : S ⥤ C)`. The merged module failed for every one of them:
  - `Unknown identifier S` and `Unknown identifier p.obj`: the binders were gone.
  - `expected token` at `⥤` and `≫`: that notation is scoped to the `CategoryTheory` namespace.
  - `unexpected identifier; expected 'lemma'`: a field docstring stranded after that parse error.
- A replay of the merge at the batch's pin wrote `def IsCartesian {X Y : S}` at line 36, column 25. The run's first build error was `Foundation.lean:36:25: Unknown identifier S`.
- Each rollback reopened the tag, so one run verified `IsCartesian` (02XK) 9 times. C9 and R9 again could not be told apart, because neither foundation kept anything.
- Track A kept 4 entries from 2 artifacts; its artifacts have no lines above their first declaration.
- `add_artifact` drops every line that begins with `end `, so a named `section` loses its `end` and leaves the merged module unbalanced. It also reads command keywords from raw lines, so a docstring line that begins with `end ` or `namespace ` is dropped.

**Investigation**:
```bash
grep -n "def _extract_entries" -A40 lms/foundation.py
grep -n 'startswith("end ")' lms/foundation.py
```

---

#### Acceptance Criteria

- [x] An artifact's lines above its first declaration, without comments and blank lines, are kept on each of its entries. `save` writes them right after the artifact's `section` and before its entries
- [x] Each artifact that has such lines, or universes beyond the header's, gets a `section` of its own, so its `variable`s don't reach another artifact's entries. An artifact with neither is written as before
- [x] An artifact's namespaces go with the `end`s that close them; its sections stay with theirs. A section it leaves open is closed after its entries, and an `end` with nothing open is dropped
- [x] Commands are recognised with block comments blanked, so a comment line that reads like a command stays in the comment
- [x] The lines round-trip through `Foundation.json`. A file written before this change still loads, with no lines. A rollback takes them away with their entries
- [x] The agent context lists an artifact's lines `under:` its first entry shown, with one note on how `variable` binders become leading arguments. The note appears only when some entry has such lines
- [x] Every one of batch q3's 27 verified artifacts reaches the saved file with no line lost, other than imports, `universe` lines and namespaces (`tests/fixtures/foundation_preamble/q3_verified.json`)
- [x] Core-Lean stand-ins compile in the merged module: a `variable` binder, an `open`, two artifacts reusing a variable name, and named and unclosed sections (`tests/test_lean_real.py::TestFoundationPreamble`)

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | Keep each artifact's preamble, balance its scopes, write it in a section, show it to agents |
| `tests/test_foundation_preamble.py` | CREATE | Behaviour |
| `tests/fixtures/foundation_preamble/q3_verified.json` | CREATE | Batch q3's 27 verified artifacts |
| `tests/test_lean_real.py` | MODIFY | Core-Lean compile checks |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-31-foundation-keeps-artifact-preamble.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-31.sh` | CREATE | Verify script |

---

#### Notes

- No Mathlib test. CI's real-Lean job and the workstation have core Lean only. The end-to-end check is on the server: rebuild a foundation from the 25 q3 track-B artifacts and `lake build` it, then run a short track-B round.
- Merged in run order under this change, track B's 11 verified artifacts give 4 entries: `IsCartesian`, `IsCartesianOver`, `IsStronglyCartesian` and `CategoryOver`. The 7 others repeat a name the foundation then has and are refused at promotion.
- An artifact's namespace is still dropped, so a definition written in `namespace FiberedCategories` lands as `LMS.Foundation.IsCartesian`. A reference inside the artifact to `FiberedCategories.IsCartesian` would not resolve. None of the 27 q3 artifacts makes one.
- The agent context shows the `variable` lines, not the elaborated signature. Each binder a definition uses becomes a leading argument, which an agent has to work out from those lines.
