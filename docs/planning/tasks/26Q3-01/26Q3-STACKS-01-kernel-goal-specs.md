### 26Q3-STACKS-01: Shared-kernel goal specs (Tracks A/B) from Stacks TeX

**User Story**: As the first committee run, I want a source-anchored Stacks
kernel goal the harness can load, so that a real collective run targets the
three-text program's actual Phase 1 material instead of the Ch. 4 warm-up goal.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q3-STACKS-01-kernel-goal-specs` |
| **Dependencies** | None (docs/data only). The committee-run demo depends on 26Q3-HARN-12 + a box run |
| **PR Size Target** | <500 lines of hand-written content (generated JSON excluded) |

---

#### Context

The user de-deferred Stacks *preparation* on 2026-08-19 ("see how far we can
get on the stacks work"). The deferral of Stacks *formalization spend* until
CVFN exists still stands — this card produces goal data and docs only.

The three-text synthesis (`specs/three_text_synthesis.md`, working tree) puts
Phase 1 of the shared kernel at three parallel tracks: A spectral sequences,
B fibred categories + descent, C core algebra. Track B is the natural first
committee target: it starts exactly where the compiled WC-3 corpus
(`lean/LMS/Categories/`) stops — M2's report calls fibred categories "the
critical next step" bridging WC-3's 2-categorical work to the sites/stacks
layer.

Existing goals are Python literals in `lms/goals.py` with hand-paraphrased
content. These goal files are instead **extracted from the Stacks Project TeX
source with real tags** (`references/stacks-project/`, untracked clone;
`tags/tags` is the tag authority), so every statement is auditable at
`stacks.math.columbia.edu/tag/<tag>` — the same source-anchoring Gate 5 (T3)
will demand of artifacts.

#### Acceptance Criteria

- [x] `goals/stacks_kernel_track_b.json` — 40 statements, Stacks Ch. 4
      §32–41 + Ch. 8 §3 (descent data), verbatim TeX statements, real tags,
      dependency-ordered, 3 milestones (02XX, 004B, 026E)
- [x] `goals/stacks_kernel_track_a.json` — 20 statements, spectral sequences,
      Stacks Ch. 12 §20–25, milestones 012W and 0132
- [x] Both files round-trip through the current `Goal.load` and render via
      `to_prompt_context()` (verified: 40/20 defs, 0% progress, ~32K/~16K chars)
- [x] Generator script `scripts/goals/extract_stacks_goal.py` (curation table
      = the reviewable artifact; regeneration is deterministic)
- [x] `goals/README.md` documents schema, provenance, and the exact
      registration patch `lms/goals.py` needs
- [ ] Registration patch applied (post-HARN-12 follow-up; see README) so
      `--goal stacks-kernel-track-b` resolves
- [ ] Track C curation table added (TODO, unscheduled)

#### Files Created

| File | Action | Purpose |
|------|--------|---------|
| `goals/stacks_kernel_track_b.json` | CREATE | Track B goal (generated) |
| `goals/stacks_kernel_track_a.json` | CREATE | Track A goal (generated) |
| `goals/README.md` | CREATE | Schema, provenance, registration patch |
| `scripts/goals/extract_stacks_goal.py` | CREATE | Extractor + curation tables |
| `docs/planning/stacks-restart-notes.md` | CREATE | What changed since Phase 1 was planned |

#### Implementation Notes

- **No `lms/` changes in this branch** — `26Q3-HARN-12` is rewiring `lms/`
  in parallel; the one-hunk registration fallback in `get_goal` lands after it
  merges.
- Statement bodies are statements only, never proofs; two of 44 are trimmed at
  40 lines with an explicit marker.
- Honesty constraints observed: no statements-per-cycle or $-per-statement
  figures anywhere (the mock-derived roadmap numbers are retracted); novelty is
  phrased as *measured by the N0/N1 classifier (26Q3-HARN-04)*, never asserted.
  Mathlib has partial `CategoryTheory.FiberedCategory.*` coverage — some Track B
  statements will classify N0, and that is data, not failure.

#### Out of Scope

- Any Lean proving against these goals (deferred until CVFN exists).
- Sites/topoi (Phase 2B) and stacks formalism (Phase 3B).
- Track C extraction.
- The novelty measurement itself (26Q3-HARN-04 owns it; these files are a
  natural second input for `scripts/measure_n1_density.py` after the ANT arcs).

#### Verification Script

N/A — data/docs card. The generator is its own check: it hard-fails on any
curated tag missing from the source, and `Goal.load` round-trip is asserted in
the README's verified claim.

#### Outcome Demo

**Where**: mahpiya (4×H100), after 26Q3-HARN-12 + registration patch merge.
**Run**: the HARN-12 `--groups` invocation with `--goal stacks-kernel-track-b`.
**Expect**: a committee run whose planning panel allocates *distinct* Track B
tags across working groups, starting from the WC-3 corpus as foundation.

#### Definition of Done

- [ ] Goal files + generator + README merged
- [ ] Registration follow-up merged (post-HARN-12) and `--goal
      stacks-kernel-track-b` resolves on the box
- [ ] First committee run against Track B recorded in `experiments/`
