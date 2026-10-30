### 26Q4-HARN-03: Arm-comparison report — pooled isolated arm, distinct statements, paired tests

**User Story**: As the Phase C analyst, I want one command that compares saved run directories as arms — pooling the isolated arm's nine runs and pairing every arm statement by statement — so that "the collective beat isolation" and "the ratchet pays" are each a number with a test behind it, not two CVFNs side by side.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q4-HARN-03-arm-comparison-report` |
| **Dependencies** | None (reads the run-directory format `cvfn_report` already reads). Required by ADR 0003 (Accepted 2026-10-30) |
| **PR Size Target** | <400 lines (max 800) |
| **Parts** | single PR |

---

#### Context

ADR 0003 (Accepted, 2026-10-30) adds three arms to Phase C's local arm: S1
(one loop), I9 (nine S1 runs at 2M/9 tokens each, run sequentially), and R9
(C9 with the foundation reset every generation). Two of the four contrasts it
names need analysis that does not exist.

**Current State**:
- `lms/accounting.py:348` `cvfn_report(run_dir)` reads **one** run directory.
  Its denominator counts verified **artifacts**, not distinct statements:
  `verified = [a for a in artifacts if a.get("status") == "verified_lean"]`,
  then `denominator = sum(1 for a in verified if _counts_as_novel(a))`. A tag
  verified twice — which `dag-phase.md` records happening when the chair puts
  several groups on one tag — counts twice.
- `_counts_as_novel` (`lms/accounting.py:338`) is private; it is the Gate 4
  rule (decisive N1 or a D4 verdict) and must not be re-implemented.
- No code compares runs. Every comparison to date has been two printed CVFNs.

**Investigation**:
```bash
grep -rn "def cvfn_report\|def _counts_as_novel\|denominator = " lms/accounting.py
# 338 _counts_as_novel, 348 cvfn_report; denominator counts artifacts
grep -rln "compare\|pooled\|mcnemar" lms/ scripts/ tests/
# nothing relevant
grep -n "stacks_tag" lms/society.py | head
# goal-directed flat, iterative and committee artifacts all carry stacks_tag
```

---

#### Scope

- Statement-level outcomes for one run, for a pooled set of runs, and a paired
  comparison of two arms over the slice they share. Report-only: reads run
  directories, writes nothing into them.

---

#### Acceptance Criteria

- [ ] `lms/accounting.py` exposes `counts_as_novel` (public name); `_counts_as_novel` stays as an alias so existing callers are untouched
- [ ] `lms/arms.py` exists with:
  - `statement_outcomes(run_dir: Path) -> dict[str, bool]` — key = `tag:<stacks_tag>` via `accounting.statement_key`; value True iff at least one artifact for that statement is `verified_lean` and `counts_as_novel` (or, on runs without novelty labels, `verified_lean`, with the kind reported as in `cvfn_report`)
  - `ArmReport` dataclass: arm name, run dirs, total ledgered tokens, distinct statements counted, **artifact-count** denominator (what `cvfn_report` would say), duplicate statements (counted by ≥2 artifacts, within or across runs), CVFN on the distinct denominator, denominator kind
  - `arm_report(name: str, run_dirs: list[Path]) -> ArmReport` — one dir for S1/C9/R9, several for I9; tokens summed over all dirs, statements unioned
  - `paired_comparison(a: ArmReport, b: ArmReport, statements: list[str]) -> PairedResult` — the 2×2 table over the shared statement list, an exact two-sided McNemar p-value on the discordant pairs, and a statement-bootstrap 95% interval on the difference in counted statements (fixed seed, recorded)
- [ ] An artifact with no `stacks_tag` is **not** keyed by declaration name; it is counted in an `untagged_verified` field and the report prints it
- [ ] `scripts/compare_arms.py --arm NAME=DIR[,DIR...] ... [--pair A:B ...] [--json OUT]` prints one row per arm and one block per pair; `--json` writes the same content
- [ ] `tests/test_arms.py` covers: a tag verified twice in one run counts once; the same tag verified in two pooled runs counts once and is reported as a duplicate; tokens sum across pooled dirs including overhead; an untagged verified artifact is excluded and counted; McNemar p on a hand-computed table; bootstrap interval reproducible under its seed; a run with no novelty labels reports `verified_lean_unfiltered`
- [ ] All tests pass: `uv run pytest tests/test_arms.py tests/test_accounting.py -v`
- [ ] `uv run ruff check lms/arms.py scripts/compare_arms.py` and `uv run mypy lms/arms.py` clean
- [ ] Run against two archived run directories, the report is readable on one screen — accepted via Outcome Demo

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/arms.py` | CREATE | Statement outcomes, pooled arm report, paired comparison |
| `lms/accounting.py` | MODIFY | Public `counts_as_novel` alias only |
| `scripts/compare_arms.py` | CREATE | CLI over `lms.arms` |
| `tests/test_arms.py` | CREATE | Behaviour tests with tmp-path run directories |
| `scripts/verify/26Q4-01/verify_26Q4-HARN-03.sh` | CREATE | Verification script |

---

#### Implementation Notes

- Mirror `cvfn_report`'s reading of `results.json`, `artifacts.json` and `attempts.json` (ledger first, `results.json` totals as fallback). Build test fixtures the way `tests/test_accounting.py` builds run directories.
- `lms/accounting.py` must stay free of `lms` imports (see `lms/metrics.py`'s note); `lms/arms.py` imports from it, not the reverse.
- No new dependencies. The exact McNemar test is a two-sided binomial on the discordant count with p = 0.5 (`math.comb`); the bootstrap uses `random.Random(seed)`.
- The statement list for a pair is the slice's tag list from the goal recorded in the run (`goal.json` or metadata), intersected across both arms; refuse to pair two arms whose recorded goals differ.

---

#### Decision Gates

- If artifacts on a real run lack `stacks_tag` in a mode Phase C uses → stop and surface. Do not invent a declaration-name or text-similarity matching rule; that is a design decision.
- If the recorded goal is missing from older run directories, so pairs cannot be checked for a shared slice → stop and propose where the slice list comes from.
- If making `cvfn_report` itself use distinct statements looks necessary → do not change it here. Historical reports must stay comparable; surface it as a separate card.
- If the PR exceeds 800 lines → stop and split (report vs CLI).

---

#### Out of Scope

- `26Q4-HARN-04` owns the per-generation reset flag that produces R9 runs.
- `26Q4-EVO-02` owns `--repeats` and run-to-run variance; this card reports within-run (statement-bootstrap) uncertainty only, and says so in its output.
- No change to `cvfn_report`'s denominator, to gates, to `lms/metrics.py`, or to any run-producing code.
- No plotting.

---

#### Verification Script

See `scripts/verify/26Q4-01/verify_26Q4-HARN-03.sh`.

---

#### Outcome Demo

**Where**: local
**Run**:
```bash
uv run python scripts/compare_arms.py \
  --arm A=experiments/run_20251218_154258 \
  --arm B=experiments/run_20251218_130736,experiments/run_20251218_105831 \
  --pair A:B
```
**Expect**: two arm rows showing ledgered tokens, distinct statements, the artifact-count denominator beside it, duplicates and untagged counts; one pair block with the 2×2 table, a McNemar p and a bootstrap interval, or a refusal naming the goal mismatch if the archived runs used different goals. These archived runs are mock-verified, so the numbers are a format check, not a result.

---

#### Definition of Done

- [ ] All acceptance criteria checked off
- [ ] `scripts/verify/26Q4-01/verify_26Q4-HARN-03.sh` exits 0
- [ ] `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean
- [ ] PR opened with <400 lines changed (target) / <800 (max)
- [ ] Tests included with implementation
- [ ] Outcome Demo run by a human validator
