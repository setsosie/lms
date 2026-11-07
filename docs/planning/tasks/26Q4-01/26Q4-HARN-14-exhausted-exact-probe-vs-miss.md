### 26Q4-HARN-14: Tell an exhausted exact? from a miss

**User Story**: As the person reading a decisive N1, I want a probe that ran out of budget to stop counting as a search that found nothing, so that "not found" means the search finished.

> Context and draft criteria written by Claude on 2026-11-05 from the second pre-merge review of `26Q4-HARN-09`. The user asked for this card before Phase C but has not reviewed this text.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-14-exhausted-exact-probe-vs-miss` |
| **Dependencies** | 26Q4-HARN-09 |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`26Q4-HARN-09` at `aebe660`):

- `ExactProbeBackend.read_output` reads "could not close the goal" as a search that ran. Per the reviewer's reading of `Lean/Meta/Tactic/LibrarySearch.lean`, `tryOnEach` stops on `abortSpeculation` when `exact?` runs out of heartbeats, and `exact?` then prints that same message.
- At four stages, such a miss supports a decisive N1.
- Reproduced on core Lean, v4.32.1: `theorem t (a b c d : Nat) (h : a ≤ b) (k : c ≤ d) : a + c ≤ b + d := by exact?` reads as a miss at `-DmaxHeartbeats=640` and closes with `Nat.add_le_add h k` at 680.
- Not confirmed at the default budget (200000) on the real library, so how often it happens is unknown. core-03 did hit the default limit at R4, but with a timeout error rather than a miss.

**Investigation**: the output alone can't separate the two cases. Any fix has to change how the probe runs.

---

#### Acceptance Criteria

- [ ] The frequency is measured: the misses from a density run are re-probed at several times the default budget, on the box at the pin, and the number that then close is recorded in this card — accepted via Outcome Demo
- [ ] If any close, a probe miss no longer reads as a search that ran unless it held at the higher budget: `tests/test_novelty.py::TestExactProbeOutput` covers it
- [ ] If none close, this card records that, and the limit is stated in the runbook's R4 section
- [ ] `uv run pytest -q` passes, and `uv run ruff check .` and `uv run mypy lms/` are clean

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/novelty/mathlib_search.py` | MODIFY | How the probe runs, if the measurement calls for it |
| `tests/test_novelty.py` | MODIFY | Tests above |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | State the limit or the new behaviour |

---

#### Implementation Notes

- A raised budget costs walltime on every miss. The R4 job has a 4-hour limit and each probe has 120 s. Re-probing only misses keeps most of that cost away from closes.
- Bump the `exact_probe` cache key version if what a cached miss means changes.

---

#### Decision Gates

- If the box run shows a miss closing at a higher budget on more than a few statements → stop and show the list before choosing a budget.
- If a fix would add more than a few minutes to the R4 job → stop and show the cost.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- `26Q4-HARN-13` owns what a decisive N1 means at Gate 4.
- A heartbeat limit that Lean does report, which `26Q4-HARN-09` labels already.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-14.sh`.

---

#### Outcome Demo

**Where**: the GPU server, as a round folder Claude prepares; the user runs it.
**Run**: the round's `RUN_THIS.txt`, which re-probes the recorded misses at a raised budget at the pin.
**Expect**: a count of misses that close only at the higher budget, by statement. Report it whichever way it comes out.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-14.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator

---

#### Notes

A neighbouring gap from the independent review at the final gate of `26Q4-HARN-09` (2026-11-06, at `3185b0e`), reproduced with a probe, predating that PR and not blocking:

- **Some search failures raise instead of reading as unavailable.** A grep `subprocess.TimeoutExpired`, and a `ConnectionResetError` or `UnicodeDecodeError` from either HTTP backend. No verdict comes out wrong. In the density script, one reset ends the run, and Gate 4 catches it as an audit hole. It's the same gap as `26Q4-HARN-09`'s "a search that failed" criterion, in a form its tests don't cover. Decide here whether it belongs in this card or a new one.
