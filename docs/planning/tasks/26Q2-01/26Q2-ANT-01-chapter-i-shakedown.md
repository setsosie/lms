### 26Q2-ANT-01: ANT Ch. I core-arc shakedown

**User Story**: As the LMS team, I want to run the full committee→scoping→
formalization→Lean-compile pipeline end-to-end on a real, bounded math target,
so that we validate the *workflow* (not the math) before committing to the
three-text program.

**Points**: 8 | **Priority**: HIGH | **Status**: 🔄 IN PROGRESS | **Branch**: `26Q2-ANT-01-chapter-i-shakedown`

**Full spec**: `specs/ant_shakedown.md` (decided 2026-06-09). This card is the
sprint-table pointer; the spec is the source of truth.

**Scope**: Neukirch ANT Ch. I core arc — integrality → Minkowski → class number
→ units. 2–3 working-group cycles.

**Acceptance Criteria**:
- [ ] Core-arc goals formalized on Mathlib; novel files compile with 0 errors
- [ ] Pipeline metrics captured per WC cycle (tokens, wall-clock, lemma reuse,
      sorry count) — the **exit criterion is pipeline health, not math coverage**
- [ ] Retrospective written (mirror `docs/stacks_session_1_retrospective.md`)

**Exit / stop if**: the workflow proves out on metrics → continue through the
book. If it stalls on *math* rather than *pipeline*, descope the math, not the
workflow test, and surface. Do not silently expand beyond Ch. I core arc.

**Verification**: novel Lean files compile (`lake build` green, 0 `sorry` in
novel files); see `specs/ant_shakedown.md` for the per-cycle metric checklist.
