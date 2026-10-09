# Novelty-classifier control

`gate_a_control_arc.json` is the Gate A control for the N1-density
measurement (`current-sprint.md` DoD item 3; runbook Step 7b). It holds the 52
artifacts of the archived `stacks_ch4_phase1` run, which reimplement Mathlib's
category theory, so a working classifier should read essentially all of them
N0. A substantial **decisive** N1 count here voids the arc densities.

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
