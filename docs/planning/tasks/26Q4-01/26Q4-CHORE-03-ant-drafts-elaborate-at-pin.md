### 26Q4-CHORE-03: The ANT arc drafts elaborate at the pin

**User Story**: As the slice decision, I want every ANT arc draft to elaborate against the pinned library, so that the N1-density run measures all 41 statements instead of silently skipping the 12 that don't.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔄 IN PROGRESS |
| **Branch** | `26Q4-CHORE-03-ant-drafts-elaborate-at-pin` |
| **Dependencies** | None |
| **Gate** | full |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

**Current State**:
- R4b (2026-11-07, at `7fab9b2a`) reported `statement did not elaborate` for 12 of the 41 drafts: `core-14`, `core-19`, `ram-02`, `-03`, `-07`, `-08`, `-10`, `-11`, `-12`, `-15`, `-18`, `-19`. Ten of the ramification arc's 21 were never measured, so its density says nothing about them.
- 11 of the 12 are marked SCHEMATIC, with placeholder names left in on purpose. `ram-02` only lacks a `DecidableEq` instance.
- No check elaborates the drafts at the pin. The 2026-10-08 pre-screen used a newer library, and the Lean CI job builds only `lean/`.
- `ram-11`'s draft is wrong, not just broken: its right side says the decomposition group acts trivially mod P (G_P = I_P, a statement about f), where Neukirch's says the inertia group is trivial.

**Investigation**:
```bash
uv run python scripts/check_arc_elaboration.py   # needs lean/.lake from `lake exe cache get`; 29 of 41 on main
```

---

#### Acceptance Criteria

- [ ] `scripts/check_arc_elaboration.py` elaborates every draft through `ExactProbeBackend.probe_source`, the probe's own framing, with auto-implicits off, and exits 1 naming each draft that fails
- [ ] The Lean CI job runs it, and runs on changes to `data/ant_arcs/`, the script, and `lms/novelty/mathlib_search.py`
- [ ] All 41 drafts elaborate at `fe3134f0`
- [ ] Each repaired draft states its informal statement in the pinned library's terms, keeping the book's hypotheses; its `notes` say what changed and why
- [ ] `ram-11` states e = 1 ⇔ I_P = 1
- [ ] Unchanged drafts are byte-identical, so their cached probe results still apply
- [ ] `data/ant_arcs/README.md` and `docs/review/ant_arc_statements_d4.md` record the repairs and the pin result

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/check_arc_elaboration.py` | CREATE | Elaborate every draft as the probe frames it |
| `tests/test_check_arc_elaboration.py` | CREATE | Framing and error attribution, without Lean |
| `.github/workflows/lean.yml` | MODIFY | Run the check after the build; widen the path filter |
| `data/ant_arcs/core_arc.json` | MODIFY | Repair `core-14`, `core-19` |
| `data/ant_arcs/ramification_arc.json` | MODIFY | Repair 10 drafts, correct `ram-11` |
| `data/ant_arcs/README.md` | MODIFY | Validation status at the pin |
| `docs/review/ant_arc_statements_d4.md` | MODIFY | Repaired entries |
| `docs/planning/tasks/26Q4-01/26Q4-CHORE-03-ant-drafts-elaborate-at-pin.md` | CREATE | This card |
| `scripts/verify/26Q4-01/verify_26Q4-CHORE-03.sh` | CREATE | Verify script |

---

#### Implementation Notes

- Every draft goes in one file, each in its own `section`, so the library loads once (about 4 minutes locally). Errors are mapped back to drafts by line.
- No `let` in a statement: the probe ends a header at the first top-level `:=`. `ram-08` writes its decomposition field out in full for that reason.
- Where the library has no definition (`ram-19`'s relative discriminant ideal), the statement writes out Neukirch's.
- Where the library needs a fact the book takes for granted (`Module.Finite A B`, separable residue fields), it is a hypothesis, named in the notes.

---

#### Decision Gates

- If the specified approach doesn't fit → stop, propose the alternative, wait for go-ahead.
- If a faithful statement can't be written at the pin → leave the draft, mark it, and say so in the PR.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- Proving any statement, or deciding N0/N1: the box run and D4 do that.
- Confirming `book_ref` numbers against the book: D4.
- The wild half of `ram-18`, which the draft never stated.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-CHORE-03.sh`.

---

#### Outcome Demo

**Where**: laptop with `lean/.lake` at the pin, or the Lean CI job
**Run**:
```bash
uv run python scripts/check_arc_elaboration.py
```
**Expect**: `41 of 41 drafts elaborate`. On `main`, `29 of 41`, with the 12 above named.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-CHORE-03.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)
- [ ] The R4 re-run of the 12 changed statements is a round folder, after merge

---

#### Notes

The repairs are statements, not proofs, and faithfulness is a D4 question: each repaired draft's `notes` start `REPAIRED 2026-11-09` so the reviewer can find them.
