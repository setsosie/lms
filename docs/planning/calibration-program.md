# Q3 2026 Calibration Program

**Window**: 2026-07-24 → 2026-09-30 (10 weeks)
**Single deliverable**: a defensible **go/no-go number** for the three-text program.
**Compute**: 4×H100 cluster, **user-driven** — Claude prepares runbooks and never executes on the cluster.
**Basis**: `docs/planning/2026-07-24-feasibility-assessment.md`

---

## 1. Why this program exists

The three-text roadmap's governing inputs (~50 statements per WC cycle, ~400–500K
tokens per cycle) come from a run verified by a regex, not by Lean. Every real-
verified run to date sits at 0–6% verification, 0.0 reuse, and produced only N0
content. We are not going to re-plan on top of that. We are going to replace the
number.

**Everything in this program serves one measurement.** If a task doesn't move us
toward that number, it isn't in scope this quarter.

## 2. The number

**CVFN — cost per verified-faithful-novel statement.**

A statement counts toward the denominator only if it clears all six:

| # | Gate | Enforced by | Source |
|---|---|---|---|
| 1 | Compiles under real Lean 4 + Mathlib | `RealLeanVerifier`, no mock fallback | — |
| 2 | No `sorry`, no new `axiom`, no `native_decide` | T4 axiom/sorry audit | `faithfulness_protocol.md` §4 |
| 3 | Non-vacuous — not a trivial `example`, hypotheses satisfiable | T2 non-vacuity witness | §4 |
| 4 | **Novelty ≥ N1** — not already in Mathlib | novelty classifier (Phase A) | §6.1 |
| 5 | Source-anchored to a specific ANT statement | T3 source anchor | §4 |
| 6 | Human D4 sign-off: the Lean says what the book says | you, reviewing | §3 D4 |

Numerator, recorded per statement:

- agent tokens (prompt + completion, all attempts including failures)
- GPU wall-clock seconds
- human review minutes

**Gates 1–5 are machine-checked and must be built before any run.** Gate 6 is you.
An artifact failing any gate is logged with the reason — the failure histogram is
as informative as the number itself.

## 3. Phases and gates

### Phase A — Make the harness incapable of lying (Jul 27 – Aug 7, local)

No cluster needed. This is the sprint that starts now.

- Record verifier provenance in `metadata.json`; **`verified: true` is unreachable
  under the mock verifier** — mock records `verified_heuristic` instead.
- Fix the `lean_code` extraction bug (leading `"|\n  "` YAML block-scalar leak).
- Implement T2 / T4 as merge gates.
- Implement the **novelty classifier**: does this statement already exist in
  Mathlib? (name search + `exact?`/`loogle` via the lean-lsp MCP). Emits N0/N1.
- Per-statement cost accounting (tokens, wall-clock) replacing per-generation.
- Carry over `26Q2-INFRA-01` (`base_url`) — needed for the cluster's vLLM `/v1`.

**Gate A** (blocks Phase C, not Phase B): re-run the archived
`stacks_ch4_phase1` artifacts through the new gates and confirm the harness now
reports them as **N0 / not-verified**. If the new pipeline still scores that run
at 92%, the gates don't work.

### Phase B — Stand up the oracle (Aug 10 – Aug 21, **you drive**)

Runbook: `docs/infrastructure/cluster-runbook-calibration.md`. Claude writes the
exact commands; you run them and paste back the output.

1. `elan` + `lake exe cache get` + `lake build` on the cluster → Mathlib compiles.
2. vLLM serving the chosen model, `/v1/chat/completions` responds.
3. **End-to-end smoke**: 1 agent, 1 statement, real verification, one artifact
   through all five machine gates.

**Gate B**: that smoke test passes. Until it does, no calibration run.

**Contingency (hard date: Aug 21).** Cluster time competes with FLAIME (P0). If
Gate B is not green by Aug 21, Phase C runs on **API models with a hard $300 cap**
instead, and the local-first policy is formally suspended for the calibration only.
Rationale: the go/no-go date is worth more than the execution-environment policy,
and Phase A's `base_url` work makes the swap a config change. This is a
pre-commitment, not a decision to make in the moment.

> **Resolved 2026-09-04 (`decisions/0001-phase-c-model-harness-control.md`).**
> The cluster was serving; the fallback did not fire as a swap. The $300 now
> funds the **control arm** of Phase C below.

### Phase C — The calibration run (Aug 24 – Sep 11)

**One fixed slice, three population sizes, identical budget.**

- **Slice**: a ~20-statement contiguous arc from Neukirch ANT Ch. I, selected in
  Phase A by running the novelty classifier over candidates and picking the one
  with the **highest N1 density**. See §4 — the currently-decided core arc is
  probably the wrong choice for this purpose.
- **Configs (amended 2026-09-04, ADR 0001)**: two arms on the same slice, the
  same harness (frozen at the 2026-09-07 state of `main`), the same
  accounting. *Local arm*: Qwen3.6-27B-FP8 at 1 and 9 agents. *Control arm*:
  one frontier API model at 9 agents, then 1 if budget remains. This yields
  CVFN, the population-size comparison, **and** the model-vs-harness
  attribution the Prove2Me paper (arXiv 2608.28433 §7) leaves to future work.
  ~~1 / 3 / 9 agents on one model~~ — a Qwen-only result cannot say whether a
  bad number is the model or the harness.
- **Budget**: 2M tokens per local config (4M total, hard-capped); the control
  arm is capped at $300 with tokens recorded, so the arms compare on tokens
  too. If a config exhausts its budget at zero verified statements, that is a
  result — record and stop.

**Gate C**: ≥1 statement clears all five machine gates in at least one config.
If zero across all three, the finding is "the pipeline cannot produce novel
verified Lean at current capability" — proceed straight to Phase E and write that
up. That is a legitimate, publishable go/no-go outcome, not a failure of the plan.

### Phase D — Human D4 review (Sep 14 – Sep 18, you)

You review every statement that cleared gates 1–5, against Neukirch. Measured:

- **minutes per statement** (this is the number that decides whether human review
  can ever keep up with a 12K-statement program)
- **gibberish rate**: fraction that compiles and passes machine gates but is
  unfaithful or useless

Presentation format: side-by-side book quote / Lean statement. Building that view
is a Phase A deliverable — if D4 review is slow because the format is bad, we
learn the wrong thing.

### Phase E — Verdict (Sep 21 – Sep 30)

Write `docs/planning/2026-09-30-calibration-verdict.md`:

- CVFN with error bars, and the gate-failure histogram
- population-size effect on CVFN across 1/3/9 agents
- re-forecast of the three-text program using real inputs — or a documented kill
- decision on the soft-prompt-genome arm, which is entirely downstream of whether
  fitness (proof success) has any dynamic range at all

## 4. Open decision — which ANT slice

`specs/ant_shakedown.md` decided (2026-06-09) on the **Ch. I core arc**:
integrality → Minkowski → class number → units.

**That choice conflicts with this program's goal.** It was chosen so the *math*
couldn't be the failure mode — correct for testing workflow mechanics. But
Minkowski theory, finiteness of the class number, and Dirichlet's unit theorem
are all in Mathlib. A slice that is ~entirely N0 cannot produce a CVFN number,
because CVFN counts only N1-and-above.

Proposed revision: target the parts of Ch. I with genuine N1 density — the
ramification material (extensions of Dedekind domains, Hilbert ramification
theory, different and discriminant) — which `ant_shakedown.md` §3 already lists as
"partial overlap, more novel statements."

**Do not decide this from memory of Mathlib's contents.** Phase A builds the
novelty classifier; run it over both candidate arcs and pick on measured N1
density. Recorded here as an explicit decision point, not a silent change.

## 5. What is explicitly out of scope this quarter

Deferred, not cancelled:

- Shared kernel Phase 1 (spectral sequences, fibred categories, core algebra)
- ART, CNF, Stacks formalization; OCR of `cnf_2nd.pdf`
- Soft-prompt genome arm: SPG-01, SPG-02, SPG-03 — all of it. The genome's fitness
  function *is* formalization success; evolving against a fitness signal that is
  ~0 for every genome is a flat landscape by construction. SPG resumes only if
  Phase C shows dynamic range in proof success.
- `specs/interactive_society.md`, the benchmark release, any paper

**Reframed 2026-09-04:** the shared kernel, ART, CNF and Stacks are no longer
LMS *production* targets — see `decisions/0002-lms-measures-collective-formalization.md`
(Proposed). They remain slices and test material.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Cluster time never materializes (FLAIME is P0) | ~~Aug 21 pre-commitment: fall back to API + $300 cap~~ — resolved 2026-09-04, the $300 funds the control arm (ADR 0001) |
| A bad CVFN cannot be attributed to model vs harness | Frontier control arm on the same slice and harness (ADR 0001, 2026-09-04) |
| Solo bandwidth — this is off `FOCUS_PLAN_MAR2026_JAN2027.md` entirely | Phase A is small and local; Phases B/D are the only ones needing you, ~2 days total |
| Chosen slice turns out ~all N0 | Novelty classifier picks the slice on measurement (§4) |
| Agents game the gates (as they did with trivial `example`s) | T2 non-vacuity + N1 requirement + D4 human review; gate-failure histogram makes gaming visible |
| The number comes back terrible | That is a *result*. Phase E writes it up as a kill and we stop honestly |

## 7. Bottom line

Ten weeks, one number, three ways it can end:

- **CVFN is decent** → re-forecast the three-text program on real inputs and restart it.
- **CVFN is bad but non-zero** → the pipeline works and is uneconomic; scope down to
  a single tractable target and publish the calibration.
- **CVFN is undefined (zero N1 statements)** → the collective-formalization thesis
  is not testable at current capability. Write it up, shelve the program, and the
  10 weeks bought a defensible answer rather than another quarter of estimates.

**Read against the control (added 2026-09-04, ADR 0001).** Each outcome above
is reported per arm. A zero on the local arm with a non-zero on the control arm
says the harness works and the model is the bottleneck. A zero on both says the
harness or the slice is. Non-zero on both gives the attribution the program now
exists to produce.
