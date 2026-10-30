### 26Q4-HARN-02: Wire TraceStore so conversations.json records transcripts

**User Story**: As the analyst reading a finished run, I want every LLM
exchange the society paid for saved in `conversations.json`, so that a
failure (a lost thread, a scaffold that "verified", a reviewer that rubber-
stamped) can be traced to what was actually said, in every mode and not only
in committee transcripts buried in `textbook.json`.

> Context, AC, files and demo drafted by Claude on 2026-10-30 from the
> user's design choice (thread the store beside the cost ledger, not a
> provider wrapper); the Notes section is the 2026-10-09 evidence. Review
> them as the author.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | MEDIUM |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q4-HARN-02-wire-tracestore-conversation-logs` |
| **Dependencies** | None |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

> Summarize current state. Include investigation results (grep evidence).

**Current State** (`main` @ `80da9f5`):
- `Society.__init__` creates `self.trace_store` (`lms/society.py:185`),
  `Society.save` writes it (`:1682`) and `Society.load` restores it
  (`:1789`). Nothing in between writes to it, so `conversations.json` is
  `[]` in every mode, every run.
- There are exactly seven `provider.generate` calls outside `lms/providers/`:
  five in `lms/agent.py` (`propose`, three in `propose_iterative`, `review`),
  one in `PlanningPanel._generate`, one in `WorkingGroup._generate`.
- Every one of them already reaches the cost ledger, by two routes.
  `propose_iterative`, `PlanningPanel` and `WorkingGroup` take `ledger=` and
  record at the call. `propose` and `review` are recorded afterwards by
  `Society` (`_ledger_propose_response`, the review loops), which never sees
  the prompt.
- `tests/test_accounting.py` proves the ledger's conservation invariant in
  flat, iterative and committee modes. Its helpers are reusable here.

**Investigation**:
```bash
grep -rn "\.generate(" lms/ --include=*.py | grep -v "^lms/providers"   # 7 call sites
grep -n "trace_store\.\(add\|record\|log\|append\)" lms/*.py            # nothing
```

---

#### Acceptance Criteria

> Every criterion needs something that can **accept** it — a command, or a human step in the
> Outcome Demo.

- [ ] `Agent.propose`, `Agent.propose_iterative`, `Agent.review`,
      `PlanningPanel` and `WorkingGroup` take an optional `trace_store`
      beside the ledger, and `Society` passes `self.trace_store` wherever it
      passes `self.ledger` (and to `propose`/`review`). `None` is a no-op
- [ ] One `ConversationLog` per exchange the ledger records: one per
      `propose`, `review`, panel call and group call. A `propose_iterative`
      session is one log holding every turn, retries and writeup included
- [ ] Each log opens with the system prompt as a `system` turn. The user turn
      that sent a request carries that call's prompt tokens; the assistant
      reply carries its completion tokens. So the transcripts' tokens equal
      `ledger.total_tokens` in flat, iterative and committee modes
- [ ] Each log's `metadata["kind"]` names the exchange: `propose`, `review`,
      `iterative`, `planning`, or the group call's ledger outcome
- [ ] An iterative log records `success`, the produced artifact ids, and each
      failed attempt's verifier error
- [ ] `Society.save` writes the transcripts to `conversations.json`
- [ ] `uv run pytest tests/test_trace_store_wiring.py -v` passes; all of its
      tests fail at `main`

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/agent.py` | MODIFY | `trace_store` on `propose`, `propose_iterative`, `review` |
| `lms/planning.py` | MODIFY | `PlanningPanel(trace_store=...)`, record in `_generate` |
| `lms/working_group.py` | MODIFY | `WorkingGroup(trace_store=...)`, record in `_generate` |
| `lms/society.py` | MODIFY | Pass `self.trace_store` beside `self.ledger` |
| `lms/traces.py` | MODIFY | Only if a small helper for one-exchange logs earns its place |
| `tests/test_trace_store_wiring.py` | CREATE | Behaviour tests |

> This table is a fence the gate checks: `/pre-merge` treats a changed file not named here
> (tests and this task's own card/verify script excepted) as a blocking finding.

---

#### Implementation Notes

- Mirror the ledger's threading exactly: an optional constructor or keyword
  argument, stored or passed through, `None` meaning "record nothing". The
  tests reuse `tests/test_accounting.py`'s helpers, and the token
  invariant is that file's conservation check pointed at the transcripts.
- Record where the call happens, not where `Society` ledgers it afterwards:
  only the call site has the prompt.
- Do not wrap the provider. The user chose threading; a wrapper cannot know
  `agent_id` or `generation` without a new mechanism.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If making this work requires changing another task's code → stop; do not absorb the adjacent scope.
- If the change exceeds the PR Size Target → stop and split.
- An explicit "do NOT do X" in this card is binding even if X seems helpful.

---

#### Out of Scope

- `ReasoningTrace` / `detect_breakthrough`: `reasoning_traces.json` stays
  empty. Wiring breakthrough detection is its own card if it is wanted.
- Moving committee transcripts out of `textbook.json`. They stay where they
  are; agents read the textbook.
- Back-filling transcripts for runs made before this card.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-02.sh`.

The script checks that the work exists and that its tests pass. It does not itself test.

---

#### Outcome Demo

**Where**: the box, the next real run (runbook R5b or R6)
**Run**: `uv run python -c "import json,sys; c=json.load(open(sys.argv[1])); print(len(c), sum(t['tokens'] for x in c for t in x['turns']))" <run>/conversations.json`
**Expect**: a nonzero log count, and a token sum equal to the total in that
run's `attempts.json`.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-02.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)

---

#### Notes

> Found-work, verified against `main` @ `c06b79a` on 2026-10-09. Evidence only;
> the sections above are still the author's to write.

- `Society.__init__` creates `self.trace_store = TraceStore()` (`lms/society.py:180`),
  and `Society.load` restores it (`:1749`).
- Nothing ever writes to it: no caller of `TraceStore.add_conversation` or
  `add_trace` (`lms/traces.py:119`, `:123`) exists in `lms/`.
  `grep -n "trace_store\.\(add\|record\|log\|append\)" lms/*.py` returns nothing.
- So `conversations.json` is empty in every mode. Committee transcripts do
  exist, stored in `textbook.json` under `entry_type="transcript"`, where
  nothing that reads traces looks for them.
