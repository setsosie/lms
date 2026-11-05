# Novelty-classifier control

`gate_a_control_arc.json` is the Gate A control for the N1-density
measurement (`current-sprint.md` DoD item 3; runbook Step 7b). It holds the 52
artifacts of the archived `stacks_ch4_phase1` run, which reimplement Mathlib's
category theory, so a working classifier should find almost none of them novel.

What it can show: a substantial **decisive** N1 count voids the arc densities.
A statement reads decisive N1 only if it ran all four stages and nothing matched,
so read `statements_by_stages_run` and `n1_with_decisive_stages` first. Few of
either means a decisive count of zero is weak evidence. `max_n1_confidence_reached`
at 0.9 is necessary for the check to fire and says no more than that. The expected
reading is some N0, a large INCONCLUSIVE share and few N1s. It is not "all N0": a
semantic match only ever reads INCONCLUSIVE, with the hit as evidence, so that a
search result can't label novel work a re-derivation.

What it cannot show: that the arc densities are right. Its statements are whole
files, with imports, comments and every later declaration, and the arcs' are one
declaration each, so a hit's tokens are far easier to find here. Replaying the
2026-11-04 hits, the match rule moved 15 of the control's 19 N1s, but none when
scoring only each statement's first declaration header.

Generated 2026-10-08 from the workstation's copy of the archive, which lives
under the gitignored `experiments/` and so never reaches a fresh checkout:

```bash
uv run python scripts/reextract_lean_code.py experiments/stacks_ch4_phase1
uv run python scripts/make_novelty_control.py experiments/stacks_ch4_phase1/artifacts.reextracted.json --out data/novelty_control/gate_a_control_arc.json
```

Unlike the ANT arcs, these declaration names are the agents' own (`opOpEquiv`,
`pullbackSymmetry`), so the file does not set `names_are_labels` and all four
search stages run. 11 of the 52 payloads have no parseable declaration name and
31 are not theorems; for those, the name, loogle, and `exact?` stages cannot form
a query and report themselves unavailable rather than "not found".
