### 26Q3-HARN-16: Committee mode cannot measure reuse

**User Story**: As the calibration program, I want committee artifacts to
carry and link references, so that reuse rate and the ratchet metric mean
something on committee runs.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-16-committee-reuse-invisible` |
| **Dependencies** | None |
| **PR Size Target** | <300 lines |
| **Parts** | single PR |

---

#### Context

> Investigated 2026-08-19 from `experiments/committee_real_b`. Same defect
> class HARN-12 fixed for iterative mode (`lms/society.py:858`).

`committee_real_b` printed `Reuse Rate: 0.0%` and **"WARNING: Ratchet
failure detected!"** on a run whose gen-5–7 groups demonstrably imported and
opened `LMS.Foundation` and built on the foundation's Category (the Functor
attempts). The warning is structural, not behavioural:

- `library.add_reference` is called on the flat path (`lms/society.py:605`)
  and the iterative path (`lms/society.py:858`) — **never on the committee
  path**.
- Committee artifacts are constructed without a `references` field
  (`lms/society.py:1063-1079`), and `_parse_artifact`
  (`lms/working_group.py`) has no references capture.
- The scribe prompt never asks for citations.

So `reused_artifact_count()` counts `referenced_by`, which nothing in
committee mode populates: reuse rate is 0.0% **by construction**, and the
ratchet metric (which keys on `fresh_creation_rate`) fires its warning on
every committee run regardless of actual behaviour.

**Investigation**:
```bash
grep -n "add_reference" lms/society.py
# 605 (flat), 858 (iterative) — nothing in the committee path (1030-1300)
grep -n "references" lms/working_group.py
# no capture in _parse_artifact
```

---

#### Acceptance Criteria

- [ ] The scribe artifact format includes a `references` field (foundation
      entry names / artifact ids used), and `_parse_artifact` captures it:
      `uv run pytest tests/test_working_group.py -k references -q`
- [ ] Committee Phase 3 populates `Artifact.references` and calls
      `library.add_reference` for resolvable ids, mirroring
      `lms/society.py:858`:
      `uv run pytest tests/test_society.py -k committee_links_references -q`
- [ ] A committee run whose artifact cites a library entry reports non-zero
      reuse (the metric is measurable, not necessarily large).

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/working_group.py` | MODIFY | `references` in scribe format + `_parse_artifact` capture |
| `lms/society.py` | MODIFY | populate `Artifact.references`, call `add_reference` in Phase 3 |
| `tests/test_working_group.py` | MODIFY | parse capture test |
| `tests/test_society.py` | MODIFY | linking test |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-16.sh` | CREATE | landing checks |

---

#### Implementation Notes

- Mirror the HARN-12 iterative fix exactly (`lms/society.py:858`) — resolve
  cited ids against the library, skip unresolvable ones silently (agents
  hallucinate ids; a warning note is fine, an exception is not).
- Foundation entries vs library ids: groups know foundation entries by
  *name*. Accept names too if cheap; otherwise ids only and say so in the
  scribe prompt.
- Note for metrics readers (HARN-12's lesson verbatim): the first committee
  run reporting non-zero reuse after this is the first run where the number
  was **measurable**, not the first where reuse happened —
  `committee_real_b` gens 5–7 built on the foundation and reported 0.0%.

---

#### Decision Gates

- If measuring reuse properly requires changing the ratchet metric itself,
  stop — metric semantics are CHORE-02/HARN-05 territory.
- Standard gates apply.

---

#### Out of Scope

- Making groups reuse *more* (prompting, API exposure — HARN-11).
- The ratchet metric's keying on `fresh_creation_rate`.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-16.sh`. Must fail at the merge
base.

---

#### Outcome Demo

**Where**: mahpiya (4×H100), human-run.
**Run**: committee run on a goal with ≥1 verified foundation entry;
**Expect**: `artifacts.json` shows non-empty `references` on citing
artifacts and the summary's Reuse Rate is non-zero when citation happened.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q3-01/verify_26Q3-HARN-16.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened within size target
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or `N/A` and why)
