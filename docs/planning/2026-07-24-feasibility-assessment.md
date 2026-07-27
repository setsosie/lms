# Feasibility assessment — LMS three-text program

**Date**: 2026-07-24
**Author**: repo audit (Claude Code session)
**Verdict**: the program as scoped in `upcoming-sprints.md` is **not feasible**, because
its governing numbers descend from a mock-verified run. The idea is not refuted;
the estimate is unsupported. One measurement decides it.

---

## 1. What exists and works

| Asset | State |
|---|---|
| Python harness (`lms/`) | ~8,300 lines: society/generation loop, provider abstraction (Anthropic/OpenAI/Google), artifact library, dependency graph, planning panel, working groups, metrics |
| Test suite | 304 passing, 11 skipped. (44 "failures" are `pytest-asyncio` missing from `.venv` — a packaging bug, not broken code) |
| Lean output (`lean/`) | 21 files, 2,394 lines, ~170 declarations, 3 `sorry` |
| Genuinely novel Lean | `Categories/Localization.lean` (4 theorems, Stacks tags 04VB/04VD/05Q2) + `Categories/Compat.lean` — the WC-3 output |
| Specs | ~7,500 lines. `faithfulness_protocol.md` and `benchmark.md` are the strongest assets in the repo (see §4) |

This is a real prototype with real planning behind it. The problems below are
about *measurement*, not about the code being absent.

## 2. The load-bearing defect: the roadmap is calibrated on a mock verifier

`lms/run.py:493` defaults `--verifier` to `mock`. `MockLeanVerifier`
(`lms/lean/mock.py`) accepts any code matching the regex
`^\s*(theorem|lemma|def|axiom|example|structure|inductive|class)\s+`. It does not
invoke Lean.

**No experiment records which verifier it used.** `metadata.json` has no verifier
field. That single omission is the root cause of everything below.

Reconstructing from the data:

| Run | Agents × gens | Tokens | Artifacts | Verified | Reuse rate | Reading |
|---|---|---|---|---|---|---|
| `stacks_ch4_phase1` | 3 × 5 | 163K | 52 | 48 (92%) | 0.038 | **mock** — see below |
| `stacks_ch4_real_opus` | 3 × 5 | 270K | 61 | 0 (0%) | 0.0 | real verifier, nothing compiled |
| `run_20251218_084146` | 3 × 2 | 455K | 6 | 0 (0%) | 0.0 | real |
| `run_20251218_101548` | 3 × 2 | 446K | 6 | ~0 (4%) | 0.0 | real |
| `run_20251218_105831` | 15 × 5 | **8.99M** | 75 | **~2 (3%)** | 0.0 | real |
| `run_20251218_130736` | 15 × 2 | 2.39M | 30 | ~2 (6%) | 0.0 | real |
| `run_20251218_154258` | 3 × 5 | 1.05M | 10 | ~1 (5%) | 0.0 | real |

Evidence that `stacks_ch4_phase1` was mock-verified:

1. Its `lean_code` fields begin with a literal `"|\n  "` — a YAML block-scalar
   marker leaked into the payload. That string cannot compile; the mock regex
   matches it anyway because it anchors on `^\s*`.
2. Its "verified" artifacts are largely `example` blocks that cite Mathlib
   (`example {C : Type*} [CategoryTheory.Category C] (X : C) : X ⟶ X := 𝟙 X`).
   These typecheck trivially and prove nothing about the target statement.
3. Every run known to use a real verifier lands at 0–6%.

**`upcoming-sprints.md` § "Fixed quantities (high confidence)" takes ~50
statements per WC cycle and ~400–500K tokens per cycle from this run.** Both
propagate into "~230–280 WC cycles", into the $2–3K cost figure, and into the
3–4 month headline.

Substituting the real-verification measurements: the 15-agent run spent **9.0M
tokens to produce roughly 2 verified artifacts**, and those were N0
(re-derivations of material Mathlib already contains). The corpus target is
~11,600–14,000 **novel** statements. The gap between the planning assumption and
the only real measurement is one to two orders of magnitude.

## 3. Secondary findings

**3.1 No verification oracle is installed.** `which lean lake elan` → nothing.
`~/.elan` does not exist. `lean/.lake` does not exist; Mathlib has never been
built in this checkout. The project's central claim is LEAN-as-perfect-oracle and
there is currently no oracle anywhere in the loop.

**3.2 Local-first has no local box.** This machine is an RTX 3070 Laptop (8 GB
VRAM), 7 GB RAM, WSL2, 43 GB free disk. `specs/model_quickref.md` assumes 4×H100
NVL / 376 GB — the org cluster, per `~/code/temp/COMPUTE_ALLOCATION_Q3Q4_2026.md`.
`26Q2-INFRA-01` (add `base_url` to `ProviderConfig`) is correctly sized at 3
points, but it unblocks nothing until a serving box is actually stood up.

**3.3 The collective hypothesis has never been tested.** Reuse rate is **0.0 in
every real-verified run**, including the 15-agent one. The single non-zero value
(0.038) is from the mock run. Population size cannot be shown to matter when the
per-agent success rate is ~3%: there is nothing to reuse. Any phase-transition
claim is currently unmeasurable, not unsupported-but-plausible.

**3.4 Sprint 1 lapsed.** Dated Jun 16–27, still marked 🔄 ACTIVE, 0 of 24 points
delivered. Last code commit was 2025-12-18; last activity of any kind 2026-06-18.

**3.5 No allocated time.** `~/code/temp/FOCUS_PLAN_MAR2026_JAN2027.md` lists
"Math formalization plans" under *What's NOT on This Plan — explicitly paused,
resume Q1 2027*, behind FLAIME, PhoNet, and AudEff, for a solo researcher.
Whatever plan we write competes with that.

## 4. The good news: the fix is already specified

`specs/faithfulness_protocol.md` was written to prevent exactly this failure. It
already defines:

- the **N0–N3 novelty ladder** — N0 = "re-proof of an existing Mathlib result,
  *calibration only, never claimed*". Every verified artifact in every run to
  date is N0.
- **T2 non-vacuity witness** — would reject the trivial `example` blocks.
- **T4 axiom/sorry audit** — would reject `sorry`-carrying and axiom-smuggling proofs.
- **D1–D4 definition checks**, with D4 as human sign-off by a reviewer who knows
  the source text.

`specs/benchmark.md` already defines Pass Rate, Compute Efficiency (tokens per
verified Lean line), Citation Density, and Definition Drift.

**Nothing in `lms/` implements any of it.** The protocol exists as prose; the
harness counts a regex match as a verified theorem. Closing that gap is a
tractable amount of work and it is the whole ballgame.

## 5. Verdict

Three-text program at ~11.6–14K novel statements in 3–4 months: **no.** That
forecast rests on a number produced by a regex.

What is feasible, and what the plan should be reduced to: **make the harness
incapable of lying about verification, then buy one honest measurement of
cost-per-verified-faithful-novel-statement.** That number either revives the
roadmap with real inputs or kills it defensibly. Everything else in the program
is downstream of it — including whether the collective hypothesis is testable at
all at current model capability.

See `docs/planning/calibration-program.md` for the Q3 plan.
