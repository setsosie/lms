### 26Q3-CHORE-02: Rename the "Tasmania effect" metric to ratchet failure

**User Story**: As a reader of a run report, I want the accumulation warning to
be named for what it measures and to fire only when it means something, so that
I read it instead of ignoring it.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | MEDIUM |
| **Status** | ✅ DONE |
| **Branch** | `26Q3-CHORE-02-rename-ratchet-failure` |
| **Dependencies** | — |
| **PR Size Target** | <150 lines |

---

#### Context

`potential_tasmania_effect` was named after Henrich (2004), "Demography and
Cultural Evolution: How Adaptive Cultural Processes Can Produce Maladaptive
Losses — The Tasmanian Case" (*American Antiquity*). Henrich's own phrase is
"the Tasmanian case"; "Tasmania effect" is not a fixed term of art, so renaming
costs no recognizability. The underlying demographic claim is also contested
(Vaesen et al. 2016, *PNAS*).

The naming problem that matters for the code: Henrich's claim concerns *loss of
existing* technology in an isolated population, whereas this metric fires on
high fresh-creation plus low reuse — a failure to accumulate in the first place.
The label described something the metric does not measure.

Separately, the metric was broken. `fresh_rate >= 0.8 and reuse_rate < 0.2` had
no guard on library size, so it fired on the first artifact ever created, when
reuse is impossible by construction. It fired on all four Gate B smoke runs on
2026-08-10, including the successful one. A warning that always fires is one
nobody reads on the run where it matters.

"Ratchet failure" is after Tomasello's cultural ratchet (Tomasello, Kruger &
Ratner 1993), the canonical term for cumulative culture. The README already used
"LEAN as Cultural Ratchet," so the vocabulary was in the project's framing.

---

#### Acceptance Criteria

- [x] `potential_tasmania_effect` → `potential_ratchet_failure` on
      `LibraryAnalysis`, in both `metadata.json` writers in `lms/run.py`
- [x] `tasmania_threshold` → `ratchet_threshold`
- [x] Warning text reads "Ratchet failure detected!"
- [x] Minimum-evidence guard: no flag below `MIN_ARTIFACTS_FOR_RATCHET` (5) or
      `MIN_GENERATIONS_FOR_RATCHET` (2)
- [x] A small library reports "too small to judge" rather than claiming
      accumulation is healthy — the same overclaim in the other direction
- [x] Regression tests for both guard conditions
- [x] `lms/metrics.py` module docstring records the rename and its reasons, and
      keeps the Henrich and Vaesen citations
- [x] `CLAUDE.md` / `README.md` keep the Henrich citation but state the naming
      decision

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/metrics.py` | MODIFY | Field, threshold, guard, warning text, rationale |
| `lms/run.py` | MODIFY | Both `metadata.json` writers |
| `lms/artifacts.py` | MODIFY | Docstring reference |
| `tests/test_metrics.py` | MODIFY | Rename + 2 guard regression tests |
| `CLAUDE.md`, `README.md`, `docs/planning/upcoming-sprints.md` | MODIFY | Prose |

---

#### Implementation Notes

- The Henrich citation is **kept**, in `metrics.py`, `CLAUDE.md` and `README.md`.
  The decision is to stop using it as a label, not to drop the source.
- Archived `metadata.json` files keep the old key. They are a historical record;
  nothing reads the field back, so no migration is needed.
- Untracked working notes still using the old name: `docs/research/`,
  `docs/interview_prep.md`, `specs/soft_prompt_genome.md`. Not in the repo, so
  not touched here.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `uv run pytest` clean — 444 passed
- [x] `uv run ruff check` / `uv run mypy` clean on the files this card touches
