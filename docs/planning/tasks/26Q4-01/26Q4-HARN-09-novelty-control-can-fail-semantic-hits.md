### 26Q4-HARN-09: The novelty control can fail, and semantic hits stop reading as N1

**User Story**: As the person making the slice decision, I want the Gate A control to be able to fail and a LeanSearch match to stop counting as "not found", so that the N1 density I read measures novelty rather than classifier misses.

> Context, AC, files and demo drafted by Claude on 2026-11-04 from the R4 run; the user approved the four-part scope but has not reviewed this text. Review them as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-09-novelty-control-can-fail-semantic-hits` |
| **Dependencies** | None |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `1a9172e`; the numbers are from runbook R4 at that commit, library pin `fe3134f0`):

| Report | N0 | N1 | INCONCLUSIVE | Upper density |
|---|---|---|---|---|
| Control (52) | 10 | 19 | 23 | 0.37 |
| Core (20) | 9 | 7 | 4 | 0.35 |
| Ramification (21) | 3 | 8 | 10 | 0.38 |

- **The control's decisive check cannot fire.** `DECISIVE_CONFIDENCE` is 0.8 and `_N1_CONFIDENCE_BY_STAGES` (`lms/novelty/__init__.py:68`) reaches it only at four stages. Stages available per control statement: 21 statements ran one, 31 ran three, none ran four.
- **`exact?` ran on 0 of the 52.** 24 are "statement not probeable", 18 "did not elaborate", and 10 stopped at the name stage. `probe_source` (`lms/novelty/mathlib_search.py:545`) keeps only the declaration header, so the `open`, `variable` and `universe` lines are lost: the failures read `Unknown identifier C` and `expected token` (scoped `⥤`, `⟶`).
- **The report overstates the ceiling.** `max_n1_confidence` (`lms/novelty/__init__.py:408`) counts configured backends, so the control reports `cap=0.90` when no statement could pass 0.75.
- **The semantic stage discards correct hits.** `_score` (`lms/novelty/__init__.py:261`) takes Jaccard overlap between a hit's tokens and every token in the statement, and below 0.5 the hit counts as not found. `rightAdjoint_preservesLimits` shares 9 of its 10 tokens with "A right adjoint functor preserves all limits" and scores 0.31.
- **Label tokens count against the match.** `query_tokens` (`lms/novelty/__init__.py:227`) includes the declaration name even on `names_are_labels` arcs: 6 of core-13's 16 tokens are label.
- **A heartbeat timeout is mislabelled.** `read_output` (`lms/novelty/mathlib_search.py:600`) calls any non-miss error "did not elaborate". core-03's error is at column 105, which is the `exact?` token. The verdict (INCONCLUSIVE) is right and the label is wrong.

**Investigation**: LeanSearch was re-queried on 2026-11-04 for all 34 N1 statements through the repo's own backend and scorer, and the declaration names were checked against source at `fe3134f0`. Hand reading of the discarded hits, which is not a D4 review:

| Report | N1s | In the library | Close | Not found |
|---|---|---|---|---|
| Control | 19 | 15 | 3 | 1 |
| Core | 7 | 6 (04, 13, 15, 17, 18, 20) | – | core-12 |
| Ramification | 8 | 4 (05, 06, 16, 17) | 14, 20 | 04, 21 |

---

#### Scope

Four changes to the classifier and its report, then one re-run of R4. The semantic stage still cannot return N0: its best score stays below `DECISIVE_CONFIDENCE`, so a match can only move a statement to INCONCLUSIVE with the hit as evidence.

---

#### Acceptance Criteria

- [ ] `probe_source` keeps the statement's `open`, `variable`, `universe` and `namespace` lines ahead of the declaration, and drops its `import` lines: `tests/test_novelty.py::TestExactProbe` covers a statement with all four
- [ ] A control statement whose stages all ran and came up empty reads decisive N1, so the check can fire: `tests/test_novelty.py` has a test for it at four stages
- [ ] Semantic scoring no longer divides by the whole statement's tokens: `tests/test_novelty.py` has a test where `rightAdjoint_preservesLimits` against the control's statement scores at least the 0.5 floor, so the verdict is INCONCLUSIVE, not N1
- [ ] A hit that shares only generic tokens stays below the floor: `tests/test_novelty.py` has a test for this, so the fix cannot turn every statement INCONCLUSIVE
- [ ] On `names_are_labels` arcs, label tokens are excluded from semantic scoring: `tests/test_novelty.py` has a test where the same statement and hit score the same under two different labels
- [ ] No semantic hit can produce N0, however strong: `tests/test_novelty.py` has a test at the highest semantic score
- [ ] `tests/test_novelty.py::TestRecordedFixtures` still passes, including `test_no_known_n1_is_ever_called_n0`
- [ ] The report adds the ceiling the statements actually reached and a count of statements per number of stages run, and the script's "Decisive N1 is unreachable" message uses the reached ceiling: covered in `tests/test_novelty.py`
- [ ] A `(deterministic) timeout` in probe output reports the heartbeat limit, not "did not elaborate", and the stage stays unavailable: `tests/test_novelty.py::TestExactProbeOutput` covers it
- [ ] The `exact_probe` cache key version is bumped, so cached probes built without context are not replayed
- [ ] `uv run pytest -q` passes, and `uv run ruff check .` and `uv run mypy lms/` are clean
- [ ] Runbook Checkpoint R4 and `data/novelty_control/README.md` describe what the control can now show
- [ ] R4 re-run at the merged commit — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/novelty/__init__.py` | MODIFY | Semantic scoring, label tokens, reached ceiling in the report |
| `lms/novelty/mathlib_search.py` | MODIFY | Probe context, timeout label, cache key version |
| `scripts/measure_n1_density.py` | MODIFY | Pass `names_are_labels` to the classifier; message and printout use the reached ceiling |
| `tests/test_novelty.py` | MODIFY | Tests above |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | Checkpoint R4 reading |
| `data/novelty_control/README.md` | MODIFY | What the control tests |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-09-novelty-control-can-fail-semantic-hits.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh` | CREATE | Verify script |

---

#### Implementation Notes

- Keep the scoring change small. The likely shape is dividing by the smaller token set rather than the union. Compare any candidate rule on the recorded hits for all 34 N1 statements before choosing it, and report how many move out of N1.
- `names_are_labels` currently only filters backends in `build_classifier`. Pass it to `NoveltyClassifier` as an argument defaulting to False, so `lms/gates/novelty.py` is unchanged.
- An unclosed `namespace` at the end of a probe file is valid Lean, so the probe needs no matching `end`.
- A statement whose first declaration is a `def` stays "not probeable". Such statements stop at three stages, and the new per-stage count makes that visible.
- Semantic answers are cached by query, not by score, so a re-run re-scores them without new LeanSearch calls. Only the probes run again.

---

#### Decision Gates

- If no simple rule moves most control N1s out of N1 while leaving a generic-token hit below the floor → stop and show the comparison. Do not stack thresholds until the control looks right.
- If a `TestRecordedFixtures` case would need its expected levels changed → stop and surface it. Those are live recordings, and the case may be right.
- If the re-run's control shows decisive N1s → they are the check working. Inspect each by hand and report them; do not adjust the classifier to remove them in this PR.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- Sprint DoD 2's wording ("N0 must account for essentially all"). A semantic match reads INCONCLUSIVE, so N0 will stay short of that after this fix. Rewording it is the user's decision.
- Writing the slice decision (DoD 5) and the D4 review of the arc queue.
- Repairing SCHEMATIC drafts, or changing any arc's statements.
- Probing a later `theorem` when a statement's first declaration is a `def`.
- `cvfn_report` counting gate-failed artifacts: that is a separate candidate card.
- `26Q3-HARN-28` owns the ANT arc → goal converter.
- `scripts/slurm/lms_run.sbatch`'s GRES default: a separate small change before R5b.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh`.

---

#### Outcome Demo

**Where**: `mahpiya`, as a round folder Claude prepares; the user runs it.
**Run**: the round's `RUN_THIS.txt`, which re-runs `scripts/slurm/n1_density.sbatch` pinned to the merged commit in `~/code/lms-box`.
**Expect**:
- The control's stage counts show statements at four stages, and `exact_probe` ran on some of them.
- The control's N1 count falls well below 19.
- Any decisive N1 on the control is listed by name for hand inspection.
- core-03 no longer reads "did not elaborate".
- The arcs' upper densities move toward the hand reading above (about 1 of 20 and 2–4 of 21). That is a prediction, so report the numbers either way.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
