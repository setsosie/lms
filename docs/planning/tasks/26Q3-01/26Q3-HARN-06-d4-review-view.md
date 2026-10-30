### 26Q3-HARN-06: D4 side-by-side review view

> **STRETCH** — pull in only if the CRITICAL cards land early. Needed before
> Phase D (Sep 14), not before Phase C.

**User Story**: As the human reviewer, I want each statement presented as source
quote beside Lean statement, so that D4 sign-off is fast enough to be measured as
a real throughput number rather than as a proxy for bad tooling.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | STRETCH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-HARN-06-d4-review-view` |
| **Dependencies** | 26Q3-HARN-03, 26Q3-HARN-04 |
| **PR Size Target** | <400 lines (was <300; raised 2026-10-30 for decoys and the correspondence record) |

---

#### Context

`specs/faithfulness_protocol.md` §3 D4 requires human sign-off that a definition
matches the source. `specs/ant_shakedown.md` §4.3 makes **minutes per definition**
an exit criterion, and §6 leaves the presentation format an open question.

Phase D measures D4 throughput. If the reviewer is squinting at raw JSON, the
measurement records the cost of bad tooling rather than the cost of review, and
the resulting per-statement human cost — a load-bearing term in whether a 12K
statement program is reachable at all — would be wrong.

**Amended 2026-10-30 — minutes measure cost, not accuracy.** Phase D reports
minutes per statement and a gibberish rate. Neither says how often the reviewer
signs off a statement that does not say what the book says, and a kernel check
cannot say it either: Lean confirms a proof of the statement as written, not
that the statement is the intended one. Seeding the queue with blinded,
known-unfaithful decoys turns the review into a measured instrument with a miss
rate. A per-statement correspondence record makes each sign-off auditable
later. Both are added below; the throughput measurement is unchanged.

---

#### Acceptance Criteria

- [ ] `scripts/build_review_page.py <run_dir>` emits a self-contained HTML page
- [ ] One card per statement clearing gates 1–5, showing:
      source quote (book, chapter, page) ‖ Lean statement ‖ novelty label with
      evidence ‖ gate results ‖ accumulated cost
- [ ] Keyboard-driven verdict: faithful / unfaithful / unsure, with a note field
- [ ] **Per-statement timer** — records seconds from card display to verdict; this
      is the throughput measurement
- [ ] Verdicts and timings written to `review_log.json` in the schema
      `26Q3-HARN-05` ingests
- [ ] Resumable: partial review sessions can be reopened without losing verdicts
- [ ] Statements whose novelty is `INCONCLUSIVE` are visually flagged as
      requiring a judgment call
- [ ] `--decoys <decoys.json>` mixes known-unfaithful statements into the queue
      in shuffled order (seed recorded). A decoy renders exactly like a real card;
      nothing on the page marks it. Each decoy records its mutation class
      (dropped hypothesis, swapped quantifier, weakened conclusion, changed
      constant or object)
- [ ] `review_log.json` keeps `is_decoy` and `mutation_class` per entry, and the
      session summary reports the decoy miss rate as a count ("signed off 1 of
      8 decoys"), by mutation class, kept apart from the real-statement verdicts
- [ ] Each real entry in `review_log.json` is a correspondence record: source
      anchor and quote (or citation fallback), the pinned formal statement and
      its hash, the foundation definitions the statement names, the axioms the
      T4 audit reports, who authored the statement and who authored the proof
      (agent or group ids, or "theorem card" when pinned), reviewer, verdict,
      discrepancy note
- [ ] `tests/test_review_schema.py` covers: decoy fields round-trip; a decoy
      entry never appears in the real-statement totals; the miss-rate count over
      a fixture log

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/build_review_page.py` | CREATE | Static HTML generator |
| `lms/review/schema.py` | CREATE | `review_log.json` schema |
| `tests/test_review_schema.py` | CREATE | Round-trip tests |
| `data/d4_decoys/README.md` | CREATE | How decoys are written and checked, one example per mutation class |

---

#### Implementation Notes

- Static HTML with inline CSS/JS, no server, no external assets. It must open from
  a file path on any machine.
- Source quotes require the ANT text as structured data. If that extraction isn't
  ready, fall back to a citation (chapter/section/page/statement number) and note
  the degradation — do not block on text ingestion.
- Decoys are an input file, written or checked by the human reviewer before
  the session; this card does not generate them automatically. Aim for 10–20%
  of the queue. A decoy must elaborate in Lean (`allow_sorry` is fine), or it is
  distinguishable by its gate results.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] Generated against an archived run and opened once to confirm it is usable
- [ ] `uv run pytest`, `uv run ruff check` clean
