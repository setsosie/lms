### 26Q4-EVO-01: Per-run prompt override hook

**User Story**: As the promptbreeder loop, I want to put a bred prompt in
front of a run's agents without editing source, so that each LMS run can
evaluate one genome — with the exact prompt that ran recorded in the run's
metadata.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | ✅ DONE (2026-08-28) |
| **Branch** | `claude/repo-improvements-experiments-2e96q8` |
| **Dependencies** | None |
| **PR Size Target** | <300 lines |

---

#### Context

The text-space evolution arm (`26Q4-EVO-02/03`) treats each LMS run as one
fitness evaluation: a strong breeder model mutates prompts, a run scores
them. That requires an injection point, and before this card there was none:

- `CURRENT_PROMPTS` (`lms/prompts.py`) was a hardcoded module global.
- Agents call `get_prompt("agent_system_goal")` directly (`lms/agent.py`);
  nothing between the registry and the agent takes a parameter.
- The nearest machinery, `VotingSystem.get_current_prompt`
  (`lms/voting.py`), is exported and wired to nothing — and is a different
  mechanism (population self-amendment, not external selection).

Provenance is the second half of the requirement. `metadata.json` already
records `prompt_versions` via `get_all_versions()`; an override that flows
through `CURRENT_PROMPTS` therefore shows up in the run record
automatically, provided its version string is distinguishable from the base.

---

#### Acceptance Criteria

- [x] `lms/prompts.py` gains `apply_prompt_overrides(overrides, source)`:
      replaces entries in `CURRENT_PROMPTS`, versioned as
      `<base>+override.<sha256[:8] of content>` so identical content always
      produces the identical version string
- [x] Unknown prompt names and empty/non-string content fail loudly at apply
      time (`ValueError` naming the offender), not as a `KeyError`
      mid-generation; nothing is half-applied on failure
- [x] `load_prompt_overrides(path)` reads a JSON object of name → content and
      rejects anything else with a readable error
- [x] `active_overrides()` reports `{source, versions}` provenance, `None`
      when no overrides are applied; `clear_prompt_overrides()` restores the
      base registry (tests, in-process breeders)
- [x] `lms/run.py` takes `--prompt-file` (fallback: `$LMS_PROMPT_FILE`),
      applies before the run, errors out before any tokens are spent on a
      missing/invalid file
- [x] `metadata.json` records `prompt_overrides` (provenance or null); a
      resume under overrides records `prompt_overrides_resume` so the
      original run's provenance is not overwritten
- [x] Tests: override reaches `get_prompt`; version flows into
      `get_all_versions()`; untouched prompts untouched; loud failures;
      file round-trip; CLI wiring

---

#### Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `lms/prompts.py` | MODIFY | Override apply/load/report/clear |
| `lms/run.py` | MODIFY | `--prompt-file` flag, metadata provenance |
| `tests/test_prompt_overrides.py` | CREATE | Behaviour tests |
| `scripts/verify/26Q4-01/verify_26Q4-EVO-01.sh` | CREATE | Verification script |

---

#### Decision Gates

- This card is the *only* injection point. Do not add per-agent or
  per-generation prompt parameters elsewhere — a second path would mean a
  prompt an agent saw that metadata cannot account for.
- Genome file format stays a flat JSON object of name → content. Lineage,
  fitness, and operator history belong to the population file
  (`26Q4-EVO-03`), not to the genome handed to a run.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `scripts/verify/26Q4-01/verify_26Q4-EVO-01.sh` exits 0
- [x] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
