### 26Q4-HARN-14: Tell an exhausted exact? from a miss

**User Story**: As the person reading a decisive N1, I want a probe that ran out of budget to stop counting as a search that found nothing, so that "not found" means the search finished.

> Context and draft criteria written by Claude on 2026-11-05 from the second pre-merge review of `26Q4-HARN-09`. The user asked for this card before Phase C but has not reviewed this text.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | MEDIUM |
| **Status** | ✅ DONE |
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
- **Measured 2026-11-09: none of R4b's 18 misses was exhaustion.** See Notes.

**Investigation**: the output alone can't separate the two cases. Any fix has to change how the probe runs.

---

#### Acceptance Criteria

- [x] The frequency is measured: the misses from a density run are re-probed at several times the default budget, on the box at the pin, and the number that then close is recorded in this card — accepted via Outcome Demo. *Measured with no limit at all rather than a multiple, and on the workstation's build at the pin rather than the box: same toolchain and library, and heartbeats are deterministic. 0 of 18 close.*
- [ ] ~~If any close, a probe miss no longer reads as a search that ran unless it held at the higher budget~~ — not taken: none closed
- [x] If none close, this card records that, and the limit is stated in the runbook's R4 section
- [x] `uv run pytest -q` passes, and `uv run ruff check .` and `uv run mypy lms/` are clean

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/research/measure_exact_budget.py` | CREATE | The measurement, rerunnable for any list of statements |
| `tests/test_measure_exact_budget.py` | CREATE | It wraps the classifier's own probe and reads each outcome right |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | State the limit |

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

**Where**: any machine with `lean/.lake` built at the pin (`cd lean && lake exe cache get`), about 4.3 GB of memory free per probe.
**Run**:
```bash
uv run python scripts/research/measure_exact_budget.py /tmp/budget.jsonl core-17 ram-16
```
**Expect**: two `"verdict": "missed"` rows, with `heartbeats_thousands` near 193525 and 176650, in 3 to 5 minutes. The full 18 are listed in Notes.

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

**Measurement, 2026-11-09, at `fe3134f0`** (Lean v4.27.0-rc1), on the workstation's build of the pinned library. That build reproduced R4b's elaboration results exactly. Each of R4b's 18 `exact?` misses was re-run as the classifier's own probe in its own process, under `#count_heartbeats in`, which lifts the limit and reports what the command used, with async elaboration off. All 18 still missed: every search finished, and a search that misses with no limit also misses with one. Heartbeats used, in thousands, against the default limit of 200,000:

| Statement | Heartbeats used |
|---|---:|
| core-17 | 193,525 |
| ram-16 | 176,650 |
| core-13 | 124,650 |
| core-15 | 112,083 |
| ram-05 | 63,581 |
| ram-06 | 52,918 |
| yoneda_fully_faithful (control) | 50,032 |
| ram-20 | 49,347 |
| core-04 | 37,846 |
| core-20 | 26,189 |
| ram-17 | 25,376 |
| ram-21 | 23,409 |
| core-12 | 20,375 |
| ram-14 | 18,638 |
| ram-04 | 17,670 |
| core-07 | 17,214 |
| core-18 | 14,076 |
| yoneda_map_eq_postcomp (control) | 13,441 |

core-17 and ram-16 came within 12% of the limit, so the failure is plausible for other statements even though it didn't happen here. A probe that always ran under the count could report an exhausted search as unavailable. That changes what a probe costs, since the count lifts the limit, so it is not this card's.

The neighbouring gap found at the final gate of `26Q4-HARN-09` (some search failures raise instead of reading as unavailable) moved to its own card, `26Q4-HARN-20`, on 2026-11-09.
