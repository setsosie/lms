# ANT candidate-arc statement lists

Input data for the Sprint 3 N1-density measurement (`current-sprint.md` DoD
item 1), which resolves the open slice decision in
`docs/planning/calibration-program.md` §4. These files are what
`scripts/measure_n1_density.py <statements.json>` (26Q3-HARN-04, slice-selection
mode) consumes: the classifier labels each statement N0 / N1 / INCONCLUSIVE and
reports per-arc N1 density. The arc with the higher measured density becomes the
Phase C calibration slice.

## Files

| File | Arc | Statements |
|------|-----|-----------:|
| `core_arc.json` | Ch. I core: integrality → ideals → Minkowski → class number → units (§2–§7) | 20 |
| `ramification_arc.json` | Ch. I §8–§10 ramification + Ch. III §2 different/discriminant | 21 |

The ramification arc includes Ch. III §2 material because the program documents
("extensions of Dedekind domains, Hilbert ramification, different/discriminant",
`specs/ant_shakedown.md` §3) group it with Ch. I ramification; the JSON `source`
field and each `book_ref` record the true book location.

## Schema

```json
{
  "arc": "core | ramification",
  "source": "free-text provenance",
  "mathlib_rev": "optional pin",
  "statements": [
    {"id": "...", "book_ref": "Neukirch ANT Ch I §x.y", "name": "snake_case_name",
     "informal": "prose statement", "lean_statement": "theorem ... : ... := sorry",
     "notes": "..."}
  ]
}
```

`mathlib_rev` is pinned to `lean/lake-manifest.json`'s mathlib entry at drafting
time (`fe3134f0`). A novelty verdict is only meaningful relative to a Mathlib
revision — a stale N1 becomes N0 when upstream lands it (HARN-04 card,
Implementation Notes).

## Validation status: **all Lean drafts are UNVALIDATED**

None of the 41 `lean_statement` drafts has been elaborated against Mathlib.
Attempted locally 2026-08-19 via the lean-lsp MCP: every `import Mathlib.*`
snippet fails with "imports are out of date and must be rebuilt" — the local
olean cache is stale, and rebuilding it (`lake build` + Mathlib cache fetch) is
box-runbook work, not part of this data PR. Consequences:

- Drafts whose `notes` say **SCHEMATIC** contain deliberate placeholder names
  (marked `?` or named in the note) and will *not* elaborate as written. They
  carry the statement's shape for the classifier's name/semantic stages;
  the loogle/`exact?` stages need them repaired first.
- All other drafts are best-effort Mathlib-idiomatic Lean 4 and *may*
  elaborate, but none is confirmed. The first classifier run should report
  elaboration failures per statement so signatures can be repaired in one pass.
- **Proposition numbers in `book_ref` were drafted from memory and are
  unconfirmed.** The D4 review doc
  (`docs/review/ant_arc_statements_d4.md`) asks the reviewer to confirm or
  correct every reference; section-level references are high-confidence,
  x.y-level numbers are not.

## Expected shape of the result (prior, to be overwritten by measurement)

The program suspects the core arc is ~entirely N0 (Minkowski theory, class
number finiteness, Dirichlet units are all in Mathlib) and the ramification arc
is mixed — that suspicion is *why* the measurement exists. The per-statement
`notes` record a drafted-from-memory Mathlib-overlap guess so the classifier's
output can be sanity-checked against a human prior; where the two disagree,
trust neither — inspect the evidence field.
