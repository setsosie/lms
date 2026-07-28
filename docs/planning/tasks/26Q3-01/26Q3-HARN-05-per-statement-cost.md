### 26Q3-HARN-05: Per-statement cost accounting

**User Story**: As the calibration measurement, I want the full cost of every
statement — including the attempts that failed — attributed to that statement, so
that CVFN's numerator is real.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-05-per-statement-cost` |
| **Dependencies** | 26Q3-HARN-01 |
| **PR Size Target** | <300 lines |

---

#### Context

CVFN = (tokens + GPU wall-clock + human minutes) / (statements clearing all six
gates). The denominator comes from HARN-03/04. This task builds the numerator.

**Current State**:
- `lms/agent.py:473` — `tokens_per_artifact = usage.total_tokens // len(matches)`.
  Cost is split **evenly** across artifacts in a response, and a response that
  parses to **zero** artifacts contributes **zero** recorded per-artifact cost —
  its tokens vanish from attribution entirely. Failed attempts are the majority of
  spend at a 3–6% success rate, so the current per-artifact numbers understate
  true cost by roughly the inverse of the success rate.
- `lms/society.py:191-194, 457-458` — society-level totals *are* complete. So the
  aggregate is right and the attribution is wrong; both are needed.
- Wall-clock is not recorded anywhere.

---

#### Acceptance Criteria

- [ ] `AttemptRecord{statement_key, agent_id, generation, prompt_tokens,
      completion_tokens, wall_clock_s, outcome, gate_failed}` appended for **every**
      generation call, including ones that parse to zero artifacts
- [ ] `statement_key` is stable across retries of the same target statement (source
      anchor / tag where available, else a normalized name) so a retry chain is
      attributable
- [ ] Unattributable spend (agent produced nothing identifiable) accumulates in an
      explicit `overhead` bucket — never silently dropped
- [ ] Wall-clock recorded per attempt and per generation
- [ ] `cvfn_report(run_dir)` emits: total tokens, total wall-clock, verified-novel
      count, CVFN, and the gate-failure histogram
- [ ] Invariant test: `sum(attempt tokens) + overhead == society.total_tokens_used`
- [ ] Human review minutes ingestible from a separate file
      (`review_log.json`, produced in Phase D) and folded into CVFN

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/accounting.py` | CREATE | `AttemptRecord`, ledger, `cvfn_report` |
| `lms/agent.py` | MODIFY | Emit attempt records incl. zero-artifact responses |
| `lms/society.py` | MODIFY | Wire ledger; wall-clock timing |
| `lms/metrics.py` | MODIFY | CVFN alongside existing metrics |
| `tests/test_accounting.py` | CREATE | Conservation invariant + retry attribution |

---

#### Implementation Notes

- Keep `Artifact.tokens_used` for backward compatibility but stop treating it as
  the cost of record; the ledger is the source of truth.
- Report CVFN with an explicit denominator count. `CVFN = 2.1M tokens / 1
  statement` and `CVFN = 2.1M / 50` are wildly different confidence situations and
  the report must not hide that.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `cvfn_report` runs against an archived December run and produces a number
- [ ] `uv run pytest`, `uv run ruff check`, `uv run mypy` clean
