### 26Q4-HARN-32: A verified artifact is read against its statement, part by part

**User Story**: As the person reading what a run produced, I want every verified artifact read against its statement's parts (the opening and each numbered condition), and a reviewer that knows `LMS.Foundation` is a valid import, so that a partial formalization is recorded as partial instead of passing as the whole statement, and reviews stop undoing work built on the foundation.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-32-artifact-read-against-statement-parts` |
| **Dependencies** | None. `26Q4-HARN-33` depends on this card |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State** (`main` @ `e9c7352`):
- **The review reads compilation only.** The review prompt (`lms/prompts.py:363` `REVIEW_SYSTEM_PROMPT_V1`, `:387` `REVIEW_USER_PROMPT_V1`) shows the reviewer the tag number, a description, the code and the authors' notes. It never shows the statement's text. It asks only "Will it compile in LEAN 4 with Mathlib?" (`:407`), imports, and type errors.
- **The reviewer is not told what the foundation holds.** The generators get it through `foundation.get_context_for_agent()` (`lms/foundation.py:1185`, used at `lms/agent.py:258` and `:352`). `Agent.review` (`lms/agent.py:771`) does not.
- **A MODIFY replaces the code outright** (`lms/society.py:1535`). Verification runs on the reviewer's version, and the group's scribe repairs it if Lean fails.
- **After verification, nothing reads the artifact against its statement.** The goal is marked formalized (`lms/society.py` after `_promote_to_foundation`).

Evidence: batch q4 (pin `a0c98b77`, track B, 4 runs × 10 generations, `docs/results/` write-up pending).
- **The 003T case.** `CategoryFibredInGroupoids` (run s3, generation 4) verified and closed 003T (Definition 4.35.1). It encodes 003T's opening and condition (2), but not condition (1), the existence of lifts. The empty category over C satisfies the artifact's structure.
  - The group's notes claimed (2) "encapsulates … existence and uniqueness".
  - D4 (2026-11-18) gave it partial credit: the opening and (2).
- **What the reviews did:** 106 of 120 reviews were MODIFY.
  - 14 MODIFYs removed `import LMS.Foundation`.
  - 8 added a Mathlib import that does not exist at the pinned Mathlib `fe3134f0`.
  - Of the 39 originals that carried a nonexistent import, 2 had it fixed by the review.
  - In the 003T review, the reviewer replaced the kept `IsCartesianOverC` with `Functor.IsCartesian`, which does not exist, and dropped the foundation import. The scribe's repair restored both.
- **Multi-part statements are common:** 19 of track B's 40 statements carry a top-level `enumerate` (8 with two items, 11 with three).

**Investigation**:
```bash
grep -n "Will it compile\|REVIEW_USER_PROMPT_V1 = " lms/prompts.py
grep -n "async def review" -A25 lms/agent.py | grep -n "format\|foundation"
grep -n 'review.decision == "MODIFY"' -A6 lms/society.py
grep -n "def _get_task_content\|def get_context_for_agent" lms/society.py lms/foundation.py
```

---

#### Scope

This card has three parts, all of them recording or review-side changes. Nothing here changes when a tag closes; `26Q4-HARN-33` does that.
1. **Split a statement into parts.** A deterministic function turns a goal statement's `content` into labelled parts: `opening` (the text before the first top-level `\begin{enumerate}`) and `(1)`, `(2)`, … (one per top-level `\item`). A statement without an `enumerate` has one part, `whole`.
2. **A coverage read after verification.** For each verified artifact, one reviewer call is shown the statement's labelled parts and the artifact's final Lean code. It answers one line per part, `covered` or `missing`. The result is stored on the artifact.
3. **The pre-verification review knows the foundation.** The review prompt gains the statement text, the foundation's entries (the same context the generators get), and the sentence that `import LMS.Foundation` / `open LMS.Foundation` are valid. A MODIFY that drops `import LMS.Foundation` from code that names a foundation entry is not adopted: the original code goes to Lean instead.

---

#### Acceptance Criteria

- [ ] `statement_parts(content) -> list[tuple[str, str]]` returns `[("opening", …), ("(1)", …), ("(2)", …)]` for 003T's goal text, and `[("whole", …)]` for a statement with no `enumerate`. Nested `enumerate` items stay inside their top-level item. Covered by `tests/test_statement_coverage.py::test_parts_*`.
- [ ] A new prompt version, `coverage_review` 1.0.0, is registered through the prompt registry, as the committee prompts are (`26Q4-HARN-12`). It asks for exactly one `<label>: covered|missing` line per part, plus one reason line per `missing`.
- [ ] Every artifact that verifies, in committee and in flat mode, gets one coverage call from an agent outside its group. The result is stored as `artifact.coverage = {"opening": True, "(1)": False, "(2)": True}` and saved in `artifacts.json`.
- [ ] An unparseable reply, a missing label or an unknown label stores `coverage = None` ("not read"), never a guess. Covered by `test_coverage_unparsed_is_none`.
- [ ] Fixture: the q4 003T artifact with a stub provider answering `opening: covered`, `(1): missing`, `(2): covered` stores exactly that. Covered by `test_coverage_003t_fixture`.
- [ ] `REVIEW_USER_PROMPT` gets a new version. It carries the statement text from `_get_task_content`, the output of `foundation.get_context_for_agent()`, and a line that `import LMS.Foundation` and `open LMS.Foundation` are valid and the listed names exist. The old version stays registered. Covered by `test_review_prompt_carries_statement_and_foundation`.
- [ ] Take an original that imports `LMS.Foundation` and names an entry the foundation holds. A MODIFY whose code drops that import is not adopted: the original code is verified, and the run log prints `[review] MODIFY by <agent> not adopted: drops import LMS.Foundation`. A MODIFY that keeps the import is adopted as today. Covered by `test_modify_dropping_foundation_import_not_adopted`.
- [ ] The run summary lists the coverage of each verified artifact (`003T CategoryFibredInGroupoids: opening ✓ (1) ✗ (2) ✓`) and a count of `not read`. Covered by `test_summary_prints_coverage`.
- [ ] Tag closure is unchanged by this card: a verified, promoted artifact closes its tag whatever its coverage reads. Covered by `test_coverage_does_not_change_closure`.
- [ ] All tests pass: `uv run pytest tests/test_statement_coverage.py -v`, and the full suite stays green.
- [ ] Whether the model's coverage read agrees with D4 on the 31 artifacts verified so far (q3's 27, q4's 4) — accepted via Outcome Demo.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/coverage.py` | CREATE | `statement_parts`, the coverage-reply parser, and the coverage call |
| `lms/prompts.py` | MODIFY | `coverage_review` 1.0.0 and a new version of the review user prompt |
| `lms/agent.py` | MODIFY | `Agent.review` takes the statement text and the foundation context; add `Agent.read_coverage` |
| `lms/society.py` | MODIFY | Pass the statement and foundation to reviews; the MODIFY guard; the coverage call after a verify succeeds (committee and flat paths) |
| `lms/artifacts.py` | MODIFY | `coverage` field, saved and loaded |
| `lms/run.py` | MODIFY | End-of-run summary: coverage lines and the `not read` count |
| `scripts/coverage_agreement.py` | CREATE | Compares a run's `coverage` fields against a D4 coverage file and prints agreement per part |
| `tests/test_statement_coverage.py` | CREATE | The criteria above |
| `tests/fixtures/q4_003t_artifact.json` | CREATE | The 003T artifact and its goal text |
| `docs/planning/tasks/26Q4-01/26Q4-HARN-32-artifact-read-against-statement-parts.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-32.sh` | CREATE | Verify script |

---

#### Implementation Notes

- **The coverage call reuses `Agent.review`'s plumbing:** a timed call, a ledger record with `outcome="coverage"`, and tokens charged to the statement. Reviewers are assigned as Phase 4 assigns them (`self.agents[i % len(self.agents)]`). It is one extra call per verified artifact, and q4 verified 4 of 120.
- **Read the final code.** The coverage call runs after verification, on the code Lean accepted, so a scribe repair after a MODIFY cannot leave the read stale.
- **Split the statement deterministically:** a small tokenizer over `\begin{enumerate}`, `\end{enumerate}` and `\item` that tracks depth. No model call, no regex over whole blocks.
- **Prompt versions go through the registry,** with the old version kept, so a run's `metadata.json` names the prompt version it used.
- **Do not introduce a coverage score** (a fraction or weight). Record per-part booleans; how to count them is `26Q4-HARN-33`'s and D4's business.

---

#### Decision Gates

- If the prompt registry cannot register a second version of the review prompt without changing the committee prompts' registration → stop and surface. Do not fork the registry.
- If flat mode has no agent outside the author's group to read coverage → stop and propose a rule (for example, the next agent in order). Do not let authors read their own artifact.
- If a statement's `content` has an `enumerate` whose items are not conditions (for example 0040's construction) and the split reads badly on a fixture → record it in the card's Notes and keep the mechanical split. Do not special-case tags.
- If the MODIFY guard needs more than "drops `import LMS.Foundation` while naming a foundation entry" → stop; a broader import policy is not this card.
- If the change exceeds the PR Size Target → stop and split, coverage read first.

---

#### Out of Scope

- `26Q4-HARN-33` owns what coverage *does*: a PARTIAL status, reopening the tag, and the brief to the next group. This card only records it.
- `26Q4-HARN-24` owns the name rule for tag closure; this card does not touch closure.
- `26Q4-HARN-15`, `-13` own novelty and N0 rules; coverage is not novelty.
- No normalizer for Mathlib import paths that do not exist (28 of q4's 112 failures). That is a separate card if wanted.
- No change to the generators' prompts or the working-group session.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-32.sh`.

---

#### Outcome Demo

**Where**: local, then the server (a round folder, per the runbook rule)
**Run**:
```bash
uv run pytest tests/test_statement_coverage.py -v
uv run python scripts/coverage_agreement.py --run experiments/<a run with coverage> --d4 references/d4_coverage.json
```
**Expect**:
- The first command passes.
- The second needs the 31 artifacts verified so far, read for coverage by the pinned 27B model (a short server round), and the user's D4 coverage grades for the same 31. It prints, per part, how often the model's `covered`/`missing` agrees with D4, and it lists each disagreement.
- On 003T it shows the model's read next to D4's "opening ✓ (1) ✗ (2) ✓".

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-32.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator

---

#### Notes

- **Added 2026-11-18 from batch q4,** after the user's D4 call on 003T: "We're going to have to figure out how to get the models to realize this is a good attempt, if not the exact answer."
- **Sprint 5 found-work,** attempted after the committed scope; if it does not land by 2026-11-27 it carries to Sprint 6.
- **Phase C** freezes at a named commit (Sprint 5 DoD 3). If this card lands after the freeze, Phase C runs without it unless the freeze moves. That choice is the user's.
