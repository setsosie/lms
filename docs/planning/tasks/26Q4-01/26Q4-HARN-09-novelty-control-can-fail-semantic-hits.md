### 26Q4-HARN-09: The novelty control can fail, and semantic hits stop reading as N1

**User Story**: As the person making the slice decision, I want the Gate A control to be able to fail and a LeanSearch match to stop counting as "not found", so that the N1 density I read measures novelty rather than classifier misses.

> Context, AC, files and demo drafted by Claude on 2026-11-04 from the R4 run; the user approved the four-part scope but has not reviewed this text. Review them as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | ✅ DONE |
| **Branch** | `26Q4-HARN-09-novelty-control-can-fail-semantic-hits` |
| **Dependencies** | None |
| **PR Size Target** | <1000 lines (max 1500). Raised by the user on 2026-11-05, because tests are most of the diff |
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

Changes to the classifier, its probes and its report, then one re-run of R4. The semantic stage still cannot return N0: its best score stays below `DECISIVE_CONFIDENCE`, so a match can only move a statement to INCONCLUSIVE with the hit as evidence.

The first pre-merge review of this PR (2026-11-05) returned NO-GO, and the user chose the tightened rule below as "Option A". The criteria from "Single letters" onward come from that review and a second one on `aebe660`, which also returned NO-GO.

---

#### Acceptance Criteria

- [ ] `probe_source` keeps the statement's `open`, `variable`, `universe` and `namespace` lines ahead of the declaration, and drops its `import` lines: `tests/test_novelty.py::TestExactProbe` covers a statement with all four
- [ ] A control statement whose stages all ran and came up empty reads decisive N1, so the check can fire: `tests/test_novelty.py` has a test for it at four stages
- [ ] Semantic scoring no longer divides by the whole statement's tokens: `tests/test_novelty.py` has a test where `rightAdjoint_preservesLimits` against the control's statement scores at least the 0.5 floor, so the verdict is INCONCLUSIVE, not N1
- [ ] A hit with fewer than 3 distinct tokens cannot match: `tests/test_novelty.py` covers the bare `CategoryTheory.Category`. Longer bare names such as `Ideal.ramificationIdx` still match and go to review; see Implementation Notes
- [ ] On `names_are_labels` arcs, label tokens are excluded from semantic scoring: `tests/test_novelty.py` has a test where the same statement and hit score the same under two different labels
- [ ] No semantic hit can produce N0, however strong: `tests/test_novelty.py` has a test at the highest semantic score
- [ ] `tests/test_novelty.py::TestRecordedFixtures` still passes, including `test_no_known_n1_is_ever_called_n0`
- [ ] The report adds the ceiling the statements actually reached and a count of statements per number of stages run, and the script's "Decisive N1 is unreachable" message uses the reached ceiling: covered in `tests/test_novelty.py`
- [ ] Single letters take no part in scoring, so renaming a binder changes no verdict: `tests/test_novelty.py::TestSemanticRule` renames `K` and `L` on core-18 and ram-20
- [ ] A near hit (Jaccard at least 0.5, short of a match) stays N1 below `DECISIVE_CONFIDENCE`, and its evidence names the hit, so that statement cannot count without review at Gate 4 or in `cvfn_report`: `tests/test_novelty.py::TestNearHit`. This doesn't make every other N1 trustworthy: see Implementation Notes
- [ ] Tests fail on a wrong rule: `TestSemanticRule` replays real hits on both sides of each boundary, and the recorded run's moved statements and near hits are pinned by id
- [ ] The report states `n1_with_decisive_stages` and `n1_or_inconclusive_density`, and the script no longer calls the N1 density an upper bound
- [ ] `candidates` keeps every semantic hit, and Gate 4 stamps them on the artifact as `novelty_candidates`
- [ ] A probe is a `theorem`, so `exact?` cannot use a `variable` the statement leaves out, and it stops at `where`: `tests/test_lean_real.py::TestExactProbeContext` runs real Lean
- [ ] The probe keeps `section`, `end`, `include` and `omit`, and an earlier real error outranks a later heartbeat limit
- [ ] A close beside a miss reads unavailable, because the close may be another command's: `tests/test_lean_real.py::TestExactProbeContext`
- [ ] A search that failed leaves the stage unavailable and uncached, not an empty search: a malformed LeanSearch answer, a Loogle error, a grep failure, and a probe that gives no answer
- [ ] A close on a file with several declarations reads INCONCLUSIVE and names what closed, and only a single-declaration close reads N0: `tests/test_novelty.py::TestSeveralDeclarations`
- [ ] Findings not fixed here are carded before Phase C: `26Q4-HARN-13` and `26Q4-HARN-14`
- [ ] A `(deterministic) timeout` in probe output reports the heartbeat limit, not "did not elaborate", and the stage stays unavailable: `tests/test_novelty.py::TestExactProbeOutput` covers it
- [ ] The `exact_probe` cache key version is bumped, so cached probes built without context are not replayed
- [ ] `uv run pytest -q` passes, and `uv run ruff check .` and `uv run mypy lms/` are clean
- [ ] Runbook Checkpoint R4 and `data/novelty_control/README.md` describe what the control can and cannot show
- [ ] R4 re-run at the merged commit — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/novelty/__init__.py` | MODIFY | Semantic scoring, label tokens, reached ceiling in the report |
| `lms/novelty/mathlib_search.py` | MODIFY | Probe context, timeout label, cache key version, malformed answers |
| `lms/gates/novelty.py` | MODIFY | Stamp candidates on the artifact |
| `lms/artifacts.py` | MODIFY | `novelty_candidates` field |
| `scripts/measure_n1_density.py` | MODIFY | Pass `names_are_labels` to the classifier; message and printout use the reached ceiling |
| `tests/test_novelty.py` | MODIFY | Tests above |
| `data/ant_arcs/README.md` | MODIFY | How to read the arc densities |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-13-decisive-n1-meaning-gate-4.md` | CREATE | Card, before Phase C |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-14-exhausted-exact-probe-vs-miss.md` | CREATE | Card, before Phase C |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-13.sh` | CREATE | Stub for that card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-14.sh` | CREATE | Stub for that card |
| `docs/planning/upcoming-sprints.md` | MODIFY | The two cards in the Phase C pre-lock |
| `tests/test_measure_n1_density.py` | MODIFY | Labels reach the classifier; the script's printout |
| `tests/test_lean_real.py` | MODIFY | Probe context on real Lean |
| `tests/fixtures/novelty/semantic_hits_2026-11-04.json` | CREATE | The hits behind the rule table, so it can be re-derived |
| `docs/planning/current-sprint.md` | MODIFY | Task rows; report fields named in DoD 1 |
| `docs/runbooks/2026-10-08-box-restore.md` | MODIFY | Checkpoint R4 reading |
| `data/novelty_control/README.md` | MODIFY | What the control tests |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-09-novelty-control-can-fail-semantic-hits.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh` | CREATE | Verify script |

---

#### Implementation Notes

- **Rule chosen by the user on 2026-11-05 (Option A):** a hit matches when it has at least 3 distinct tokens and at least 0.8 of them (name and signature, without single letters) appear in the statement. A match only reaches INCONCLUSIVE. A near hit, Jaccard at least 0.5 without a match, stays N1 but must be reviewed. Every statement also keeps all of semantic search's hits as `candidates`, whether they matched or not, so D4 can read them. The density comes from D4, not from this threshold.
- Compared on the hits recorded in `tests/fixtures/novelty/semantic_hits_2026-11-04.json`, for the 34 N1 statements (of 7 core, 8 ramification, 19 control; 4 statements were hand-read as open). Each column counts statements leaving N1. Every row but the first ignores single letters:

| Rule | Core | Ramification | Control | Open statements left in N1 |
|---|---:|---:|---:|---:|
| Jaccard, as on `main` | 0 | 0 | 0 | 4 of 4 |
| Share of hit ≥ 0.6 | 6 | 4 | 17 | 1 of 4 |
| **Share of hit ≥ 0.8, at least 3 tokens** | **1** | **1** | **15** | **3 of 4** |
| Overlap over the smaller set ≥ 0.5 | 6 | 8 | 18 | 0 of 4 |

| Report | N1 at R4 | N1 replayed | N1 + INCONCLUSIVE replayed |
|---|---:|---:|---:|
| Control (52) | 19 (0.37) | 4 (0.08) | 42 (0.81) |
| Core (20) | 7 (0.35) | 6 (0.30) | 11 (0.55) |
| Ramification (21) | 8 (0.38) | 7 (0.33) | 18 (0.86) |

- **The 3-token minimum changed no recorded verdict.** It rules out `CategoryTheory.Category`. Three recorded names are that short (`minpoly`, `ZLattice.FG`, `differentIdeal`), but none would reach 0.8 anyway. Bare names with three tokens still match: `Ideal.ramificationIdx` alone is what moves ram-06, and `CategoryTheory.Epi` and `CategoryTheory.Mono` alone move one control statement. The cost is low, because a match only sends a statement to review.
- A near hit caps the N1's confidence rather than setting a flag, because `cvfn_report` and the arm reports read only `novelty_level` and `novelty_confidence`. A flag on the result would have left them counting it. The near hits in the recorded run are core-04, core-12, ram-04 and ram-14.
- The 12 statements that read INCONCLUSIVE from a semantic hit in that run all stay INCONCLUSIVE when their hits are re-fetched and re-scored, so dropping the old middle tier sends none of them to N1.
- This ladder is also the live Gate 4. Artifacts that carry a preamble can now reach four stages, so CVFN can move on later runs.
- **Rule chosen by the user on 2026-11-05:** the probe tests only a file's first declaration, so a close on a file with several reads INCONCLUSIVE and names what closed. All 41 arc statements are single declarations, so their N0s don't change. All 9 closes the ladder reaches in a local control run were on files with several, so `exact?` adds no N0 on the control: the forecast is N0 10, INCONCLUSIVE 38, N1 4, where it would otherwise be N0 19.
- **A decisive N1 still only means the search found nothing.** In Gate 4's configuration (four stages, names not labels) and with the other stages empty, core-13, -15, -17, -20 and ram-04, -05, -14, -17 read N1 0.90 with no review, and most are in the library by hand reading. Not a regression. `26Q4-HARN-13` owns it.
- **A miss doesn't prove `exact?` finished.** It can run out of budget and report a miss, which this PR can't detect. Reproduced only at low budgets. `26Q4-HARN-14` owns it.
- Hit names always contribute tokens, but `extract_identifiers` drops lowercase names from the statement, so core-04 gets no `minpoly` token. Counting them would also match core-04, ram-20 and ram-21, and ram-21 is hand-read as open. The rule table the user chose from didn't show this, and the rule is unchanged.
- A failed search cached with a message before this change no longer replays, on any stage but `exact_probe`, whose miss carries Lean's message as its error and is a real answer. A failure cached with no message can't be told from a real empty answer, so it still replays. Nothing shows any occurred.
- A local control run probes 18 of the 21 probeable statements, because 3 stop at the name stage. On a library build newer than the pin those 18 give 9 closed, 2 missed and 7 unavailable.
- `names_are_labels` currently only filters backends in `build_classifier`. Pass it to `NoveltyClassifier` as an argument defaulting to False, so `lms/gates/novelty.py` is unchanged.
- An unclosed `namespace` at the end of a probe file is valid Lean, so the probe needs no matching `end`.
- A `theorem` or `lemma` whose type is data, not a proposition, reads "did not elaborate", as it does when written that way. `exact?` on data only shows that some term exists, so `(n : Nat) : Nat` would close with `exact n` and read as a false N0. On a library build newer than the pin, four control statements are data. Two would have closed correctly (`initial_terminal_def_fixed`, `equivalence_of_categories`) and two would have missed, and all four lose the stage.
- A statement whose first declaration is a `def` stays "not probeable". Such statements stop at three stages, and the new per-stage count makes that visible.
- Semantic answers are cached by query, not by score, so a re-run with the same `.novelty_cache` and pin re-scores them without new LeanSearch calls, and only the probes run again. With another cache or pin, LeanSearch is queried again and every figure above may shift.

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
- `26Q4-HARN-13` owns what a decisive N1 means at Gate 4, and `26Q4-HARN-14` owns telling an exhausted `exact?` from a miss.
- Name search has no check against the statement, and can take its name from a comment. Both predate this card and can read N0.
- `scripts/slurm/lms_run.sbatch`'s GRES default: a separate small change before R5b.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh`.

---

#### Outcome Demo

**Where**: the GPU server, as a round folder Claude prepares; the user runs it.
**Run**: the round's `RUN_THIS.txt`, which copies R4's reports somewhere durable first (the re-run overwrites them), then re-runs `scripts/slurm/n1_density.sbatch` pinned to the merged commit with `REPO=~/code/lms-box`, since the job and the runbook's summary commands default to `~/code/lms`.
**Expect**:
- The control's stage counts show statements at four stages, and `exact_probe` ran on some of them. Report how many, and `n1_with_decisive_stages`. A local forecast, on a library build newer than the pin, has 11 statements at four stages and no N1 among them, so `n1_with_decisive_stages` of 0 and a decisive count of 0 are expected whatever else the run finds. That is weak evidence, not a pass.
- The control's N0 stays near R4's 10, because a close on a file with several declarations reads INCONCLUSIVE.
- The control's N1 count falls well below 19.
- Any decisive N1 on the control is listed by name for hand inspection.
- core-03 no longer reads "did not elaborate".
- The arcs' N1 densities fall only a little: replaying the recorded hits gives core 0.35 → 0.30 and ramification 0.38 → 0.33, against a hand reading of about 1 of 20 and 2–4 of 21 among the N1s. Those densities bound nothing: counting INCONCLUSIVE too gives 0.55 and 0.86, and all 10 of ramification's INCONCLUSIVE statements did not elaborate. The number for the slice decision comes from D4 review of the queue, reading each statement's `candidates`. Report the numbers either way.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <1000 lines changed (target) / <1500 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
