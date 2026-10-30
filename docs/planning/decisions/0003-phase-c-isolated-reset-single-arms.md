# 0003 — Phase C measures the collective against isolated, reset and single-agent arms

**Status**: Accepted
**Date**: 2026-10-30
**Deciders**: setsosie (solo)
**Tags**: experiment-design, calibration, scope
**Amends**: [[0001]] (arm set and analysis only; the model × harness control stands)

## Context

ADR 0001 runs one slice at "1 and 9 agents" on two models. Read against the
code, neither population arm isolates what the README's questions 1 and 2 ask:

- **"1 agent" is not one loop.** In committee mode `--agents 1` over the
  default three groups seats a chair, a scribe and one researcher per group
  (`Society._committee_members_per_role`): nine role sessions, a planning
  panel and a reviewer. `--agents 9` seats fifteen. The contrast is one
  researcher per group against three, inside one organization.
- **No arm removes the shared foundation.** A 9-agent win over a 1-agent run
  is consistent with "nine interacting agents beat one" and says nothing about
  whether nine agents *in isolation*, spending the same tokens, would have
  done as well. That comparison — the independent-loops null — is the one the
  Henrich claim needs.
- **Reuse rate does not show the ratchet pays**: it counts names, not whether
  carrying the foundation yields more verified statements.
- **The foundation is shared mutable state at a fixed path**
  (`Society.reset_foundation` docstring). Two runs in parallel on the box
  write the same `Foundation.lean`, so arms meant to be independent would
  share a medium.

## Decision

Phase C's local arm runs four configurations on the frozen harness, slice and
per-config budget, **one at a time**, and is analysed statement by statement:

| Arm | Command shape | Budget | Measures |
|---|---|---|---|
| **S1** single loop | flat `--iterative --agents 1` | 2M | the individual baseline (replaces committee `--agents 1`) |
| **C9** collective | committee `--groups --agents 9` | 2M | as in ADR 0001 |
| **I9** isolated null | 9 × S1, run sequentially at 2M/9 each, pooled | 2M total | C9 − I9 = collective gain over the same compute in isolation |
| **R9** reset | C9 with the foundation reset to seed every generation | 2M | C9 − R9 = what the carried foundation is worth (the ratchet) |

R9 resets `Foundation.lean` only. The textbook, the dependency graph's
statuses and the artifact library carry in both C9 and R9, so the contrast
prices importable verified code and nothing else. The reset flag defaults off,
so every other arm runs the frozen behaviour.

Every arm's denominator is **distinct statements** clearing the gates (keyed by
source tag), not verified artifacts: `cvfn_report` today counts a tag proved
twice as two.

The control arm's "1" is likewise S1 on the frontier model, not committee
`--agents 1`. The committee is called a *collective*, not a swarm: its planning
panel is a coordinator.

## Alternatives considered

- **ADR 0001 as written (committee 1 vs 9)** — rejected: neither arm removes
  the medium or holds organization fixed, so a difference is unattributable.
- **Add I9 and R9 but keep committee `--agents 1`** — rejected on budget; S1
  is the comparison the literature asks for (a single agent at matched
  compute), and committee-1 runs only if budget remains.
- **Parallel arms in separate `lean/` checkouts** — deferred: a Lake build and
  Mathlib cache per checkout is untested on the restored box.
- **A flat 9-agent shared-foundation arm (M9) as well** — optional, not
  committed: it splits C9 − I9 into medium and organization, at another 2M.

## Consequences

**What this enables:**
- Answers to questions 1 and 2 with a comparison each: C9 vs I9 (does the
  collective beat isolation at equal compute?) and C9 vs R9 (does the ratchet
  pay?), on top of ADR 0001's model × harness cell.
- Paired per-statement comparisons on the shared slice, at no extra run cost.

**What this costs:**
- Local tokens 4M → 8M, and sequential wall-clock for every arm.
- Two cards before Phase C: `26Q4-HARN-03` (arm-comparison report: I9
  pooling, paired tests) and `26Q4-HARN-04` (per-generation reset flag).

**What this commits us to:**
- **Pooling rule for I9**: numerator = all nine runs' ledgered cost; denominator
  = *distinct* statements clearing gates. A statement two isolated runs both
  prove counts once and is paid for twice. Duplication is reported.
- **No two Phase C runs execute concurrently** against the same `lean/` tree.
- **Error bars**: bootstrap over statements within a run; run-to-run noise
  estimated from `--repeats` on S1 once `26Q4-EVO-02` lands, reported as
  covering S1 only. Until then the report says between-run variance is
  unmeasured.
- **Question 3** (minimum viable culture) is not answered by two sizes; the
  verdict says so and defers it to a size sweep with an isolated null at
  each size.
- A Gate C zero on any arm is recorded as that arm's result, as before.

## Related

- [[0001]], [[0002]]; `calibration-program.md` §3; `26Q4-HARN-03`, `26Q4-HARN-04`
- `experiments/26Q3-POP-01/card.md` — arm A5 is the matched-budget precedent
- *Language Models as Agents* (Vol 6 draft): independent-loops null (Ch. 12),
  size/organization/retention axes (§12), carried-vs-reset library (Ch. 13, 19)
