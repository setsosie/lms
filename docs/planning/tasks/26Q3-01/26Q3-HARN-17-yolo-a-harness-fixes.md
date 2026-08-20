# 26Q3-HARN-17: committee_yolo_a post-mortem harness fixes

**User Story**: As the calibration program, I want the four harness defects
committee_yolo_a exposed fixed before the next box run, so that a 20-gen run
measures the collective instead of re-measuring known harness gaps.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-17-yolo-a-harness-fixes` |
| **Dependencies** | None (supersedes the self-report half of HARN-16) |
| **PR Size Target** | <400 lines |
| **Parts** | single PR |

---

#### Context

`experiments/committee_yolo_a` (box, 2026-08-20): 100 committee generations,
266 artifacts, 2 verified, 10.4M tokens, then a 55-generation stall. The
post-mortem attributed every failure to four harness gaps (full record in the
session memory `committee-yolo-a-first-verified-reuse`):

1. **Reuse invisible despite HARN-16 being live.** The run started ≈6 min
   after the `2cee54c` pull; the citation prompt ran on all 100 generations
   and 0/266 artifacts carried a reference — while ~97 failures were reuse
   *attempts* and 0014 verifiably built on 0013. Scribe self-report is
   refuted as a mechanism.
2. **Hallucinated import paths.** 6 artifacts died at line 1 on module names
   that do not exist (`Mathlib.Data.Equiv.Basic` — mathlib3's layout,
   `Mathlib.CategoryTheory.NaturalTransformation`, `LMS.Foundation.Category`).
   Phase 1 declares no import policy, so the wired `validate_code` check was
   a no-op; and an *initial* violation hard-failed with zero repair turns.
3. **Mathlib-prior API collisions (~97 failures)** against the foundation's
   fields (`Category C`, `.Hom`, bare `Ob`, `C.id x`) — with the full API
   already rendered in the prompt. Model-side, but the context can push back
   harder.
4. **Idle groups.** 17/100 generations ran 1 group instead of 3 (~13% of
   capacity): partial assignments were never topped up; only the
   zero-survivors case fell back to default allocation.

#### Changes

- `society.py`: `_derive_references` — references derived mechanically from
  the artifact's code against foundation entry names (comments stripped,
  self-definitions excluded); scribe citations only add. Import violations
  (initial or repaired) now enter the same scribe repair loop as compile
  errors, and an all-blocked artifact fails with the restriction message.
- `agent.py::_clean_lean_code`: `import LMS.Foundation.X` rewritten to the
  umbrella `import LMS.Foundation` (the only module that exists), deduped.
- `goals.py`: `validate_imports` is dot-aware and its rejection message
  carries the full allowed list (it is fed to the repair turn verbatim);
  `STACKS_CHAPTER_4_PHASE_1` now declares `allowed_imports` /
  `forbidden_imports`.
- `foundation.py::get_context_for_agent`: explicit anti-prior warning — the
  Mathlib API for a same-named concept does not exist here; use exactly the
  fields printed.
- `planning.py::_top_up`: final assignments are topped up to `n_groups` from
  unassigned available tasks, on both the approved and revised paths.

#### Acceptance criteria

- [ ] `_derive_references` links 0014-shaped code to a 0013-shaped foundation
      entry with no scribe citation (end-to-end test).
- [ ] The three committee_yolo_a hallucinated import paths are rejected by
      Phase 1's `validate_code` with a message that lists the legal imports.
- [ ] An initial-draft import violation gets repair turns; verifier is never
      called on offending code.
- [ ] A 1-assignment planning outcome with 3 available tasks yields 3
      assignments with distinct tags and group ids.
- [ ] Full suite green.
