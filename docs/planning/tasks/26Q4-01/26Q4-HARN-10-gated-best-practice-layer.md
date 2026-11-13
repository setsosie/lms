### 26Q4-HARN-10: Gated best-practice layer — advice is kept only if it lowers cost on a fixed probe set

**User Story**: As the person testing whether cultural artifacts lower the cost of good output, I want a practice admitted to the culture only when fresh agents given it solve a fixed probe set more cheaply than fresh agents without it, so that accumulated advice is tested rather than assumed, and superstition cannot pass as best practice.

> **Sketch, not scheduled.** Drafted by Claude on 2026-11-04 at the user's request; the user has not reviewed this text. The Acceptance Criteria are a first draft to review as the author, and Points and Priority are placeholders. Not in the sprint table.

| Field | Value |
|-------|-------|
| **Story Points** | 5 (placeholder) |
| **Priority** | MEDIUM (placeholder) |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-10-gated-best-practice-layer` |
| **Dependencies** | `26Q3-HARN-28` (pinned statements, so a probe's success is well defined) |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | `0/2`: Part 1 the gate, Part 2 proposal and run wiring |

---

#### Context

Lean checks statements, and nothing checks advice. The working hypothesis is that output tokens per good token may fall as cultural artifacts and best practices accumulate. That cannot be tested while every note survives by default. Background: `docs/planning/2026-11-04-external-check-scale-definition-quality.md` §4 and §5.

**Current State** (read 2026-11-04 on the `26Q4-HARN-09` branch; `main` is `1a9172e`):
- `lms/prompts.py` tells agents to leave notes for successors ("Their notes are gold").
- `lms/society.py:687-710` adds a verified artifact's notes to `self.textbook` as a `success` entry, and a failed artifact's notes with its error. No step selects among entries.
- `lms/prompts.py` `apply_prompt_overrides` (with `--prompt-file` in `lms/run.py:833`) replaces prompts for a process and records a content hash in run metadata, so a practice can be injected with provenance.
- `lms/arms.py:278` has `_mcnemar_exact`, and `paired_comparison` pairs arms per statement.
- `lms/accounting.py:91-99` records prompt and completion tokens separately. The user has chosen output tokens as the budget unit.
- **Agent lifetime is not a setting.** A grep of `lms/agent.py` and `lms/artifacts.py` found no per-agent budget. `--max-tokens` in `lms/run.py` caps a run, and 20K per agent-generation is an observation from a shakedown, which counted prompts too.
- `scripts/slurm/lms_run.sbatch:58` defaults `MAX_MODEL_LEN` to 131,072, so a 200K context does not fit the served window today.

**Investigation**:
```bash
grep -nE "textbook.add|notes" lms/society.py            # notes reach the textbook unselected
grep -nE "def apply_prompt_overrides" lms/prompts.py    # the injection point
grep -nE "max_context|context_budget|budget" lms/agent.py lms/artifacts.py  # no lifetime cap
```

---

#### Scope

**The gate (Part 1).** For a candidate practice, on a fixed set of dev probes, run fresh agents with an empty textbook, paired by probe, with and without the practice, and return ADMIT, REJECT or INCONCLUSIVE.

**Proposal and wiring (Part 2).** At a generation boundary an agent proposes at most one practice from that generation's failures. Admitted practices reach later generations. Agent lifetime becomes a setting, 20K output tokens to start, 200K as the later target.

---

#### Acceptance Criteria

**Part 1: the gate**
- [ ] `lms/practices.py` exists with `Practice`, `ProbeSet` and `evaluate_practice`, which takes an injected attempt runner, so tests need no model
- [ ] `tests/test_practices.py::test_gate_admits_planted_improvement` passes: a stub runner with a planted effect returns ADMIT
- [ ] `tests/test_practices.py::test_gate_rejects_planted_harm` passes
- [ ] `tests/test_practices.py::test_gate_can_be_inconclusive` passes: no planted effect does not return ADMIT
- [ ] `tests/test_practices.py::test_probe_splits_are_disjoint` passes: dev and test probes share no statement, and the run refuses otherwise
- [ ] `tests/test_practices.py::test_leak_screen_refuses_probe_content` passes: a practice naming a probe's tag, declaration or statement is refused unrun
- [ ] `tests/test_practices.py::test_practice_length_cap` passes: admitted practices together stay under a fixed share of agent lifetime
- [ ] `tests/test_practices.py::test_gate_tokens_ledgered_separately` passes: gate spending is reported apart from the society's budget, in output tokens
- [ ] `tests/test_practices.py::test_censored_attempt_counts_as_miss` passes: an attempt reaching its lifetime unverified is a miss at that cost
- [ ] The gate's false-positive and detection rates are read on real agents, with a placebo practice and a known-useful one — accepted via Outcome Demo

**Part 2: proposal and wiring**
- [ ] `tests/test_practices.py::test_one_proposal_per_generation` passes
- [ ] `tests/test_practices.py::test_only_admitted_practices_reach_prompts` passes
- [ ] `tests/test_practices.py::test_lifetime_ends_agent_at_output_budget` passes: 20K and 200K both work in a stub run
- [ ] Run metadata records each admitted practice, its evidence and the prompt hash
- [ ] `uv run ruff check lms/practices.py` and `uv run mypy lms/practices.py` are clean

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/practices.py` | CREATE | Records, probe sets, gate (Part 1) |
| `tests/test_practices.py` | CREATE | Gate and wiring tests |
| `lms/society.py` | MODIFY | Part 2: proposal at the generation boundary; agent lifetime |
| `lms/prompts.py` | MODIFY | Part 2: proposal prompt; a slot for admitted practices |
| `lms/run.py` | MODIFY | Part 2: flags for practices, probe set and lifetime; run metadata |
| `goals/` probe-set file | CREATE | Dev and test probes, pinned statements |

---

#### Implementation Notes

**The default decision rule, for review.** Primary outcome: hit on the pinned statement, paired by probe and repeat. Secondary: output tokens to first verification. ADMIT needs exact McNemar p < 0.05 in favour, using `arms._mcnemar_exact`, and no increase in median output tokens. With no discordant pairs against, that takes at least 6 discordant pairs, so **only large effects can pass**. Treat that as the intended conservatism for now, not a measure of what small practices are worth.

**Cost, for sizing** (*estimate*): 10 probes × 3 repeats × 2 conditions = 60 lifetimes per candidate, so 1.2M output tokens at 20K and 12M at 200K. Reuse control attempts across candidates within a generation.

**Probes need calibrating first.** A probe all agents solve, or none do, carries no signal. Measure base rates with no practices and keep probes around 20–80% at the current lifetime, so the set depends on lifetime and must be recalibrated at 200K.

**Keep the final measure honest.** Gating on dev probes can overfit them. Report accumulation on test probes that never touch gating. Keep Yoneda and 2-Yoneda out of dev probes, so admission cannot tune to the main result.

**Controls.** Run a placebo practice (generic encouragement) and a known-useful one. The Sonnet simulation's two Yoneda failures on trailing mechanical steps suggest a candidate: finish with a tactic check such as `rfl` after rewriting.

**Mirror** `lms/arms.py` for pure functions over recorded results and refusal on unmeasurable input. Reuse `apply_prompt_overrides` provenance rather than adding a second mechanism. Do not remove the textbook: it is the comparison arm.

**The comparison this enables:** no notes; notes as today; every proposal kept; gated practices. If gated beats every-proposal-kept, selection matters. If neither beats no notes, advice is not doing the work.

---

#### Decision Gates

- If `26Q3-HARN-28` has not landed → stop. Do not build a second statement-pinning path here.
- If lifetime means cumulative output tokens but a context window fills first → stop and surface it. The user wants lifetime near max context, and the two differ under tool-heavy loops.
- If 200K needs `MAX_MODEL_LEN` above 131,072 and serving capacity drops → report the concurrency cost before changing the default.
- If fewer than about 10 calibrated dev probes exist at a lifetime → stop. The gate cannot resolve anything at that lifetime.
- If the placebo is admitted at the Outcome Demo → the gate is not working. Stop before admitting anything else.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- `26Q3-HARN-28` owns statement pinning; `26Q4-HARN-05` owns best-of-k; `26Q4-HARN-07` owns swarm mode.
- `26Q3-POP-01` owns the population sweep. This card supplies a mechanism, not the 1, 10, 100 result.
- No use of gate results as an agent reward or in training.
- No editing of admitted practices by agents, and no automatic eviction yet.
- No Axiom library in any path an agent can read.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-10.sh`.

---

#### Outcome Demo

Sketch: flag names are not final. Run by the user; assistants do not run on this host without the session's authorization steps.

**Where**: mahpiya
**Run**:
```bash
uv run python -m lms.practices calibrate --probes goals/probes_categories.json --lifetime 20000
uv run python -m lms.practices gate --practice placebo --probes goals/probes_categories.json --lifetime 20000
uv run python -m lms.practices gate --practice known-useful --probes goals/probes_categories.json --lifetime 20000
```
**Expect**: A calibrated probe list with base rates. The placebo is not admitted. The known-useful practice is admitted, or the report shows the gate lacks the power to tell at this size.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-10.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max), per part
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
