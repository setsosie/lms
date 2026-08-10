# Sprint 2: Make the harness incapable of lying about verification

**Dates**: Jul 27 – Aug 7, 2026
**Quarter**: Q3 2026 · folder `26Q3-01`
**Program**: `docs/planning/calibration-program.md` — Phase A
**Sprint Goal**: Every one of the five machine gates that a statement must clear
to count toward the go/no-go number is implemented and tested. At sprint close,
the harness cannot record a mock-verified or Mathlib-duplicate artifact as a
verified novel statement.

**Status**: 🔄 ACTIVE

> **Execution note**: every task below is local Python + Lean-LSP work that Claude
> executes. Your cost this sprint is **review, not implementation**. The cluster is
> not needed until Sprint 3.

---

## Sprint 1 close-out (26Q2-01) — ❌ NOT DELIVERED

| Metric | Value |
|--------|-------|
| Dates | Jun 16 – Jun 27, 2026 |
| Planned points | 24 |
| Delivered | **0** |
| Last commit in window | 2026-06-18 (planning docs only) |

Honest postmortem: the sprint was scaffolded and then never worked. No code
landed in the window or in the 4 weeks after it. Contributing cause on the
planning side — `26Q2-ANT-01` was committed as IN PROGRESS while its hard
dependency (a working Lean oracle, `26Q2-INFRA-01`, a serving box) was PENDING,
so the sprint had no executable first step.

Disposition of Sprint 1 scope:

| Task | Disposition |
|---|---|
| `26Q2-INFRA-01` local serving path | **Carried** → `26Q3-INFRA-01` this sprint |
| `26Q2-ANT-01` ANT shakedown | **Superseded** by the calibration program, Phases B–D. Rescoped: measures CVFN, not just workflow mechanics |
| `26Q2-SPG-01/02/03` genome arm | **Deferred to Q4** — see `calibration-program.md` §5. Fitness = proof success; that signal is ~0 today, so the landscape is flat by construction |

## Sprint 2 Summary

| Metric | Value |
|--------|-------|
| Planned points | 24 (+3 stretch) |
| Carryover | 3 (`26Q3-INFRA-01`) |
| Delivered to date | 3 (`26Q3-CHORE-01`, `26Q3-HARN-07`) |
| Track | Harness honesty (Phase A of the calibration program) |
| Execution | Local, Claude-executed; cluster not required |
| Status | 🔄 ACTIVE |

## Harness Track

Implements the machine gates from `specs/faithfulness_protocol.md`, which the
harness has never enforced. Gate numbering matches
`calibration-program.md` §2.

| Task | Points | Priority | Epic | Status | PR / Branch | Notes |
|------|--------|----------|------|--------|-------------|-------|
| 26Q3-CHORE-01: Fix test environment | 1 | HIGH | CHORE | ✅ DONE | #11 | `pytest-asyncio` missing from `.venv` → 44 spurious failures. Move to `[dependency-groups] dev`. Real baseline was **57** failed, not 44; suite now 392 passed |
| 26Q3-HARN-01: Verifier provenance + mock can't mark verified | 3 | CRITICAL | HARN | 🔲 PENDING | — | Record verifier in `metadata.json`; mock emits `verified_heuristic`, never `verified`. **Root cause of the bad roadmap numbers** |
| 26Q3-HARN-02: Fix `lean_code` extraction | 2 | CRITICAL | HARN | 🔲 PENDING | — | Leading `"\|\n  "` YAML block-scalar leak reaches the verifier as source |
| 26Q3-HARN-03: T2 / T4 machine gates | 5 | CRITICAL | HARN | 🔲 PENDING | — | Gate 2 (no `sorry`/new `axiom`/`native_decide`) + Gate 3 (non-vacuity; reject trivial `example`) |
| 26Q3-HARN-04: Novelty classifier (N0 / N1) | 5 | CRITICAL | HARN | 🔲 PENDING | — | Gate 4. Mathlib name search + `exact?`/`loogle` via lean-lsp MCP. Every artifact produced to date is N0 |
| 26Q3-HARN-05: Per-statement cost accounting | 3 | HIGH | HARN | 🔲 PENDING | — | Tokens + wall-clock attributed per statement incl. failed attempts; replaces per-generation totals. This is the CVFN numerator |
| 26Q3-INFRA-01: Local OpenAI-compatible serving path | 3 | HIGH | INFRA | 🔲 PENDING | — | Carryover. `base_url` on `ProviderConfig`. Also the API-fallback lever for the Aug 21 contingency |
| 26Q3-HARN-07: Verifier must invoke Lean with the project env | 2 | CRITICAL | HARN | ✅ DONE | #12 | Found on the box 2026-07-28. `real.py` runs bare `lean` with no `LEAN_PATH`, so it can only check **import-free** Lean. Every real agent proof imports Mathlib → CVFN numerator is structurally 0 until fixed. Use `lake env lean` |
| 26Q3-HARN-06: D4 side-by-side review view | 3 | STRETCH | HARN | 🔲 PENDING | — | Book quote ‖ Lean statement. If D4 is slow because the format is bad, Phase D measures the wrong thing |

## Gate A — sprint exit criterion

Re-run the archived `experiments/stacks_ch4_phase1/artifacts.json` through the
rebuilt pipeline. **It must now report ~0 verified novel statements** (it
currently reports 48/52 = 92%).

If the rebuilt gates still score that run highly, the gates do not work and
Sprint 3 does not start.

Secondary check: `experiments/run_20251218_105831` (15 agents, 8.99M tokens)
should report its ~2 verified artifacts as **N0**, with a populated gate-failure
histogram over the other 73.

## Risk register

| Risk | Mitigation | Task |
|------|------------|------|
| Novelty classifier is unreliable (Mathlib search is fuzzy) | Report N0/N1 with a confidence field; anything low-confidence routes to D4 human review rather than being counted | 26Q3-HARN-04 |
| Non-vacuity checking is undecidable in general | Implement the tractable subset: reject `example` with no new named declaration, reject statements whose hypotheses are unsatisfiable by a witness search. Log what it can't decide | 26Q3-HARN-03 |
| Gates get built but the archived-run check is skipped | Gate A is the sprint exit criterion, not a nice-to-have | — |
| Sprint 2 lapses like Sprint 1 | Tasks are Claude-executed and locally verifiable; no external dependency for any of them | — |

## Next sprint (Sprint 3, Aug 10 – Aug 21) — pre-lock

Phase B of the calibration program. Contents:

- `26Q3-CLUSTER-01`: stand up Lean + Mathlib + vLLM on the 4×H100 box (**you drive**,
  Claude writes `docs/infrastructure/cluster-runbook-calibration.md`)
- `26Q3-CAL-01`: select the ANT slice by measured N1 density — resolves the open
  decision in `calibration-program.md` §4 (the committed Ch. I core arc is
  probably ~all N0, which would make CVFN undefined)
- **Aug 21 hard checkpoint**: if Gate B isn't green, invoke the pre-committed
  API fallback ($300 cap) so the Sep 30 verdict date holds

## Sync Log

- **2026-07-28** — 2 tasks reconciled: `26Q3-CHORE-01` → DONE (#11),
  `26Q3-HARN-07` → DONE (#12). Delivered 0 → 3 pts. Also merged: #10 (runbook
  Step 2c/3c honesty, not a sprint row). No open PRs.
  Found during sync: `pytest` mutates the tracked WC-3 corpus on every run —
  `lms/society.py:122` defaults `FoundationFile` to the cwd-relative
  `lean/LMS/Foundation.lean`, so `tests/test_society.py` overwrites the real
  Yoneda corpus with mock output. Folded into `26Q3-HARN-01`.
