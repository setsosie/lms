### 26Q3-HARN-12: Make the committee architecture reachable, and add the review stage

**User Story**: As the calibration program, I want the tested committee
pipeline (planning panel → working groups → review → verify) reachable from a
supported CLI path, so that population-size runs study N collaborating agents
rather than N copies of one agent.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q3-HARN-12-committee-architecture-reachable` |
| **Dependencies** | None |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR — parts 1 and 2 delivered; part 3 (mode reconciliation) deferred, see Decision Gates |

---

#### Context

> Investigated 2026-08-10. All line numbers are from `main` @ `a93c262`.
> This card was originally scoped as "build task allocation and peer review."
> That was wrong: **most of it already exists and is tested.** It is simply
> unreachable. Rescoped 2026-08-10 to wiring + the one genuinely missing stage.

**The intended pipeline** — Planning committee → work committees → review
committee → finalized output — is three-quarters built:

| Stage | Code | Tests | State |
|---|---|---|---|
| Planning committee | `PlanningPanel` (`lms/planning.py:184`) — chair proposal → member votes → revised proposal → `WorkingGroupAssignment`s | `tests/test_planning.py` | exists |
| Work committees | `WorkingGroup` (`lms/working_group.py:227`) — chair opening → discussion rounds → chair summary → scribe finalize, with blackboard + consensus | `tests/test_working_group.py` | exists |
| **Review committee** | — | — | **was missing; added by this card** |
| Finalized output | verify → foundation (`lms/society.py:866-892`) | `tests/test_society.py` | exists |

`lms/dependency.py` (`DependencyGraph`, `TaskStatus`) already tracks which tags
are available, claimed, and done — real allocation machinery, not a stub.

**Fault 1 — the flags are dead.** `use_working_groups` is declared at
`lms/society.py:134` and `lms/config.py:41`, and is *only ever assigned in
tests* (`tests/test_society.py:726,797,826`). Nothing in `lms/` reads it.
`run_generation` (`lms/society.py:174`) dispatches on `iterative_mode` alone
(`lms/society.py:289`) and has no branch for groups. `run_generation_with_groups`
(`lms/society.py:734`) is called from **tests only** — `lms/run.py:254` and
`lms/run.py:524` both call `run_generation`. There is no `--groups` CLI
argument. Committee mode cannot be run today by any supported path.

**Fault 2 — no review committee.** `lms/planning.py:166` has panel members
review *the chair's plan*, not the produced Lean. `ReviewQueue` exists but is
used only in the non-iterative propose path. Nothing reviews a work committee's
output before it goes to the verifier. This is the one stage that must be
written rather than wired.

**Fault 3 — committees, retries, and review are mutually exclusive.** Group mode
verifies once (`lms/society.py:882`) with no retry loop. Iterative mode has
retries but no groups and hardcodes `reviews_total=0` (`lms/society.py:726`).
You can currently have exactly one of the three.

**Fault 4 — `library.add_reference` is never called in iterative mode.** It
exists only in the non-iterative path (`lms/society.py:524`).
`reused_artifact_count()` counts `a.referenced_by` (`lms/artifacts.py:388`),
which only `add_reference` populates (`lms/artifacts.py:335`), so
`calculate_reuse_rate` (`lms/metrics.py:41`) returns **0.0% by construction on
every `--iterative` run** — including `experiments/shakedown_3x3_d`, where reuse
demonstrably happened (`refs=['definition-Category-d3a579da']`).

**Fault 5 — a silent fallback.** `lms/society.py:759-761`: with no goal, group
mode returns `await self.run_generation(generation)`. A misconfigured committee
run degrades to flat mode with no error and no log line.

**Why it matters now.** In `experiments/shakedown_3x3_d` all three agents
formalized the same tag (0013). Goal progress read 11% on 2 verified artifacts
because the second bought no coverage. Population size cannot be studied while
N agents are N copies of one agent — see `experiments/26Q3-POP-01/`.

**Investigation**:
```bash
grep -rn "use_working_groups\|run_generation_with_groups\|reviews_total" lms/ tests/
# lms/society.py:134, lms/config.py:41   declared
# tests/test_society.py:726,797,826      assigned (tests only)
# lms/society.py:734                     defined; called from tests only
# lms/society.py:726                     reviews_total=0,  # No peer review in iterative mode
```

---

#### Acceptance Criteria

- [x] Committee mode has a supported CLI path:
      `uv run python -m lms.run --help | grep -q -- --groups`
- [x] `run_generation` dispatches on `use_working_groups` (the flag is read,
      not just declared): `uv run pytest tests/test_society.py -k dispatches_to_committee -q`
- [x] Committee mode without a goal is a loud error, not a silent flat-mode
      fallback: `uv run pytest tests/test_society.py -k without_goal_raises -q`
- [x] A review committee screens group output before the verifier — REJECT
      keeps code away from Lean, MODIFY replaces it, APPROVE passes it through:
      `uv run pytest tests/test_society.py -k review_committee -q`
- [x] Iterative mode links references, so reuse rate is measurable:
      `uv run pytest tests/test_society.py -k links_references -q`
- [x] `--groups` requires `--goal` and refuses `--iterative` at the CLI,
      before any tokens are spent (exercised by the verify script)

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/society.py` | MODIFY | `run_generation` dispatch; loud no-goal error; review-committee stage; review stats in group-mode `GenerationResult`; `add_reference` in iterative mode |
| `lms/run.py` | MODIFY | `--groups` / `--n-groups` flags, `build_parser()` extraction, fail-fast validation, committee mode threading into run/resume |
| `tests/test_society.py` | MODIFY | dispatch, loud-error, review-committee (approve/reject/modify/disabled), iterative-reuse tests |
| `docs/planning/tasks/26Q3-01/26Q3-HARN-12-committee-architecture-reachable.md` | CREATE | this card |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-12.sh` | CREATE | wiring checks |

---

#### Implementation Notes

Decisions taken (the card's open questions, settled):

- **Wire, don't rewrite** — held. `PlanningPanel`, `WorkingGroup`,
  `DependencyGraph` are untouched. The dispatch lives in
  `Society.run_generation`, which is the only entry point `lms/run.py` calls,
  so committee runs also get the HARN-09 foundation persistence for free
  (group mode previously never persisted the foundation at all).
- **Review stage placement**: between scribe-finalize and verify, as the card
  suggested. Reviewers are the society's `Agent`s — idle in committee mode —
  via the already-tested `Agent.review` / `PendingReview` path, not a new
  committee type. **Review can modify**: a MODIFY verdict replaces
  `artifact.lean_code` (with a `[Modified by …]` note), matching flat-mode
  semantics. REJECT adds the artifact to the library with a
  `Rejected by review committee (…)` error and releases the task tag back to
  AVAILABLE. `use_peer_review = False` skips the stage.
- **Mode reconciliation (committees + retries)**: NOT delivered — see Decision
  Gates. `--groups` and `--iterative` are mutually exclusive at the CLI with an
  error naming this card, which is the loud version of fault 3 rather than a
  fix for it.
- **`add_reference` in iterative mode**: the one-line fix is in. Note for
  metrics readers: the first run reporting a non-zero reuse rate after this is
  the first run where the number was **measurable**, not the first run where
  reuse happened (`shakedown_3x3_d` had real reuse and reported 0.0%).
- The silent no-goal fallback is now `ValueError("Committee mode requires a
  goal…")`.
- Group transcripts now reach the textbook for every group that produced code,
  including review-rejected ones — the transcript is the record of the
  session, not a reward for passing review.

---

#### Decision Gates

- Reconciling committee mode with the iterative retry loop **was stopped on**,
  as the card pre-authorized: retries would have to live inside
  `WorkingGroup.run_session` (re-entering the discussion with verifier
  feedback), which restructures it. That is its own card; until then the CLI
  refuses the combination loudly.
- If `PlanningPanel` turns out not to work against the live vLLM server (it has
  never been run end-to-end outside tests), stop and report before rewriting it
  — a mock-passing panel that fails on the box is a wiring bug, not a design flaw.
- If allocation needs `Goal` semantics to change, stop — `26Q3-HARN-03` and
  `-04` depend on them.

---

#### Out of Scope

- `26Q3-HARN-11` owns what agents are *told* about the foundation — including
  `_get_foundation_summary` (`lms/society.py:955`), which committee mode reads
  at `lms/society.py:771`. Not changed here.
- `26Q3-HARN-03` owns the T2/T4 gates; `26Q3-HARN-04` owns novelty N0/N1.
- The attempts-vs-library denominator split (`lms/society.py:645` counts
  attempts as `artifacts_created`; `total_artifacts` is `len(library)` at
  `lms/metrics.py:174`) is `26Q3-HARN-05`'s reporting defect. Not this card.
- `Society.load` pointing a resumed run's foundation at `output_dir` — separate
  found-work, not this card.
- Failure writeups stay in the textbook. Negative results are the feedback
  channel iterative retries depend on; the rule is channel separation
  (verified Lean is importable/citable, failure prose is readable and
  non-citable), not prohibition.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-12.sh`.

Behaviour goes in pytest, not in the bash script. The script must **fail at the
merge base** — measure discrimination against `origin/main` after committing.

---

#### Outcome Demo

**Where**: mahpiya (4×H100) — **pending human run** (the box is user-driven).
**Run**:
```bash
uv run python -m lms.run --groups --n-groups 3 --agents 3 --goal stacks-ch4-phase1 --generations 2 --verifier real --provider openai --output experiments/committee_smoke
```
(then inspect `experiments/committee_smoke/artifacts.json` for distinct
`stacks_tag`s per `group-*` creator and at least one
`Rejected by review committee` entry)

**Expect**: a single generation in which N agents across M committees attempt M
*different* Stacks tags rather than N copies of one; a review stage that rejects
at least one artifact before it reaches the verifier; and a non-zero reported
`Reuse Rate` on a run whose `artifacts.json` shows non-empty `references`.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-12.sh` exits 0
- [x] `uv run ruff format`, `uv run ruff check` clean on touched files; `uv run
      mypy` introduces no new errors (16 at base, 16 after — the 3 reported in
      `society.py` are the pre-existing flat-path `result` shadow, lines shifted)
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [x] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)
