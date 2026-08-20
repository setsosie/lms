### 26Q3-HARN-14: Committee groups need a Lean verify-feedback loop

**User Story**: As the calibration program, I want a committee group to see the
Lean error and get repair attempts, so that committee runs convert
single-retry-fixable failures into verified artifacts the way the iterative
path already does on the same box and model.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS — PR open |
| **Branch** | `26Q3-HARN-14-committee-group-lean-feedback-loop` |
| **Dependencies** | None (HARN-12 merged) |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

> Investigated 2026-08-19. Line numbers from `docs/step8-result` @ `30bff7e`.
> This is the card HARN-12's Decision Gates explicitly deferred to: "Reconciling
> committee mode with the iterative retry loop was stopped on … That is its own
> card."

**The evidence** — `experiments/committee_real_a` (2026-08-19, 4×H100,
Qwen3-Coder-30B, TP=4): 8 generations × 3 one-shot groups on tag 0013 →
22 artifacts, **0 verified**, 669,808 tokens, ~9 min wall-clock. Error
taxonomy:

| Class | Count | Retry-fixable? |
|---|---|---|
| Same guessed import (`Mathlib.CategoryTheory.Category`, missing olean) | ~7/22 | yes — error names the module |
| Syntax errors mid-file | ~8/22 | yes — error carries line + token |
| Universe / type errors | ~6/22 | yes — error carries the mismatch |

Every class is single-retry-fixable with the error message fed back. The
iterative agent path (`Agent.propose_iterative`, 5 attempts + error feedback)
produced `verified_lean` on the same box and model. Committee groups get
exactly one shot: scribe compiles, Lean judges, session ends.

**The one-shot path**: committee Phase 5 (`lms/society.py:1178-1247`) makes a
single `await self.verifier.verify(lean_code)` call (`lms/society.py:1199`).
On failure the error goes to the textbook (`lms/society.py:1237`) and the tag
flips back to `AVAILABLE` (`lms/society.py:1232`) — the error never reaches
any model this generation. The `WorkingGroup` object is still alive in the
`(pending, group)` tuple at that point; nothing ever speaks to it again. The
only "retry" today is a fresh ~30K-token group session on the same tag next
generation (669,808 / 22 ≈ 30K tokens per one-shot artifact), starting from
zero knowledge of the failure.

**The proven pattern**: `Agent.propose_iterative` (`lms/agent.py:265`) — loop
to `max_attempts` (`lms/agent.py:350`), on failure append the error as a user
message capped at 500 chars with "Analyze the error and try again"
(`lms/agent.py:460-469`), record each attempt in the ledger with
`gate_failed="lean_verify"` (`lms/agent.py:412`).

**Design taken — scribe repair turn, not discussion re-entry.** HARN-12's gate
sketched re-entering the group discussion with verifier feedback, which
restructures `run_session`. The taxonomy argues that is not needed: every
error class is fixable by one model reading the error text. So: on verify
failure, Phase 5 asks the group's scribe for a repair (one turn, failed code +
Lean error in context), re-cleans, re-verifies, up to `max_repair_attempts`.
A repair turn is one scribe call against a ~30K-token session — cheap relative
to the do-nothing alternative of re-running the whole group next generation.

**Investigation**:
```bash
grep -n "verifier.verify" lms/society.py
# 1199:  the only verify call on the committee path — no loop around it
grep -rn "max_attempts" lms/agent.py | head -2
# 271:   propose_iterative(max_attempts=5) — the pattern to mirror
grep -n "def repair" lms/working_group.py
# (no output — the seam does not exist yet)
```

---

#### Acceptance Criteria

- [x] `WorkingGroup` has a repair turn: given failed code + Lean error, the
      scribe returns a revised artifact dict through the existing
      `_parse_artifact` path:
      `uv run pytest tests/test_working_group.py -k repair -q`
- [x] Committee Phase 5 retries a failed verify up to `max_repair_attempts`,
      feeding the Lean error back, and a repair that verifies counts as
      verified (foundation, dependency graph, goal progress — same as a
      first-shot success):
      `uv run pytest tests/test_society.py -k repair_attempt_verifies -q`
- [x] Repair output is re-cleaned (`_clean_lean_code` — the HARN-02 leak
      applies to any scribe payload) and re-checked against the goal's import
      restrictions before Lean sees it:
      `uv run pytest tests/test_society.py -k "repair_output_is_recleaned or recheck_import" -q`
- [x] `max_repair_attempts = 0` reproduces today's one-shot behaviour exactly:
      `uv run pytest tests/test_society.py -k zero_repair -q`
- [x] Repair spend lands in the ledger keyed to the task tag with outcome
      `group_repair` (distinct from `group_session`, so cost analysis can
      separate them):
      `uv run pytest tests/test_working_group.py -k repair_spend_recorded -q`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/working_group.py` | MODIFY | `repair()` — one scribe turn: failed code + error → revised artifact; `_generate` takes an `outcome` arg (default `group_session`) |
| `lms/society.py` | MODIFY | Phase 5 (`:1178`): wrap verify in the repair loop; re-clean + re-run import check per attempt |
| `lms/config.py` | MODIFY | `max_repair_attempts: int = 2` next to `max_turns_per_group` (`:44`) |
| `lms/run.py` | MODIFY | thread the knob like `max_turns_per_group`; optional `--repair-attempts` flag |
| `tests/test_working_group.py` | MODIFY | repair-turn + ledger tests |
| `tests/test_society.py` | MODIFY | retry-loop, re-clean, zero-attempts tests |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-14.sh` | MODIFY | fill in checks |

---

#### Implementation Notes

- **Mirror `propose_iterative`'s feedback message** (`lms/agent.py:465-467`):
  error capped at 500 chars, "Analyze the error and try again", attempts-left
  count. Same cap, same tone — one pattern in the codebase, not two.
- Repair goes through the group's existing `_generate`
  (`lms/working_group.py:423-453`) so spend is auto-attributed to
  `config.task_tag`; pass `outcome="group_repair"`.
- **Repaired code does NOT go back through the review committee.** Review
  (Phase 4) screens before Lean; after a failure, Lean's error is a stricter
  judge than a reviewer. This mirrors MODIFY semantics: `lean_code` is
  replaced, artifact id stays, a note is appended
  (`[Repaired by scribe, attempt N]`).
- Do NOT re-enter discussion rounds, do NOT touch `run_session`'s structure,
  do NOT merge iterative-mode code paths — the scribe turn is the whole loop.
- Default `max_repair_attempts = 2`: taxonomy says one usually suffices; the
  second covers a fix that introduces a new error. Not 5 — a group session
  already spent ~30K tokens converging on the approach.

---

#### Decision Gates

- If scribe-only repair proves insufficient on the box (repair turns keep
  failing where `propose_iterative` succeeds), stop and report with the
  per-class fix rate — re-entering the discussion is a redesign, not a retry
  count bump.
- If the loop needs `Goal` or verifier semantics to change → stop
  (`26Q3-HARN-03`/`-04` depend on them).
- If the change exceeds the PR Size Target → stop and split, don't power through.
- An explicit "do NOT do X" in this card is binding even if X seems helpful.

---

#### Out of Scope

- **What the scribe is told about valid imports** — the ~7/22 guessed-import
  class could also be attacked statically (allowed-import list in the scribe
  prompt). That is prompt/foundation-exposure work: `26Q3-HARN-11`'s territory.
  This card only guarantees the error message reaches the scribe.
- `--groups --iterative` reconciliation (HARN-12 fault 3) stays refused at the
  CLI. This card gives committee mode its *own* loop; it does not merge modes.
- T2/T4 gates (`26Q3-HARN-03`), novelty N0/N1 (`26Q3-HARN-04`).
- Retry-vs-first-shot verified counts in `GenerationResult` — nice for the
  gate histogram, but reporting work; the ledger `group_repair` rows carry the
  raw signal.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-14.sh`.

The script checks that the work exists and that its tests pass. It does not
itself test — behavior is proven in `tests/`. It must **fail at the merge
base** — measure discrimination against `origin/main` after committing.

---

#### Outcome Demo

**Where**: mahpiya (4×H100) — human-run; the box is user-driven.
**Run**: re-run the `committee_real_a` configuration, e.g.
```bash
uv run python -m lms.run --groups --n-groups 3 --repair-attempts 2 --agents 3 --goal stacks-ch4-phase1 --generations 8 --verifier real --provider openai --output experiments/committee_real_b
```
(then check `experiments/committee_real_b/artifacts.json` for `verified_lean`
entries created by `group-*`, and the ledger for `group_repair` rows)

**Expect**: on the run where 0/22 verified one-shot, repair turns convert a
visible share of the single-retry-fixable classes (~21/22 of the taxonomy) to
`verified_lean`; every repair attempt appears in the ledger as `group_repair`
spend on its task tag.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `scripts/verify/26Q3-01/verify_26Q3-HARN-14.sh` exits 0 (and fails at the
      merge base — discrimination confirmed 2026-08-19)
- [x] `uv run ruff format`, `uv run ruff check` clean on touched files;
      `uv run mypy` introduces no new errors (all reported errors are on
      pre-existing lines: the flat-path result shadow, run.py verifier
      assignments, dotenv stub)
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [x] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)
