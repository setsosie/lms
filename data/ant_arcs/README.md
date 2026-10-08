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
  "names_are_labels": true,
  "statements": [
    {"id": "...", "book_ref": "Neukirch ANT Ch I §x.y", "name": "snake_case_name",
     "informal": "prose statement", "lean_statement": "theorem ... : ... := sorry",
     "notes": "..."}
  ]
}
```

`names_are_labels: true` says the declaration names (`ant_c06_discr_ne_zero`)
are labels, not guesses at Mathlib names. The classifier's name-grep and loogle
stages search by declaration name, so on these files they can never match and
their silence is not evidence; `measure_n1_density.py` leaves them out and the
ladder is `exact_probe` + `semantic`. Two stages cap N1 confidence at 0.6, below
the 0.8 decisive line, so **every N1 here routes to D4** and the decisive
density is 0 by construction. Read the upper density and the review queue.

`mathlib_rev` is pinned to `lean/lake-manifest.json`'s mathlib entry at drafting
time (`fe3134f0`). A novelty verdict is only meaningful relative to a Mathlib
revision — a stale N1 becomes N0 when upstream lands it (HARN-04 card,
Implementation Notes).

## Validation status: pre-screened, not yet confirmed at the pin

**2026-08-19:** none of the 41 drafts had been elaborated; the local olean cache
was stale.

**2026-10-08 pre-screen.** All 41 elaborated in one file against a *newer*
Mathlib than the pin (rev `520045ab`, toolchain v4.32.1; the only local build
on the workstation), with two renames that happened between the revisions
translated for that run only: `Ideal.ramificationIdx f p P` →
`Ideal.ramificationIdx' p P`, and `NoZeroSMulDivisors` →
`Module.IsTorsionFree`. Both spellings in the files are the pinned rev's,
checked against its source.

| | Count |
|---|---:|
| Elaborate | **28 / 41** |
| Fail, all marked SCHEMATIC | 13 |
| Non-SCHEMATIC drafts that fail | **0** |

Getting there took five repairs, each recorded in the statement's `notes`:
`core-06` (`Basis` → `Module.Basis`), `core-17` (scoped `⁰` notation spelled
out), and `ram-16`/`-17`/`-20` (the instance arguments `differentIdeal`
requires at the pin). The 13 failures were left as drafted: each is already
marked SCHEMATIC.

What this does not establish: elaboration **at `fe3134f0`**. The box run of
`measure_n1_density.py` against the pinned build is the confirmation. A probe
whose statement does not elaborate now reports `statement did not elaborate`
and casts no vote either way (`exact_probe` stage), so a SCHEMATIC draft
surfaces as INCONCLUSIVE rather than as a confident N1.

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
