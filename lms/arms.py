"""Arm comparison over saved run directories.

An arm is one or more runs read as a unit: one directory for a single-society
arm, several for a pooled isolated arm. Its numerator is every run's ledgered
cost; its denominator is **distinct statements** clearing the gates, keyed by
source tag. A statement two pooled runs both prove counts once and is paid for
twice, and the duplication is reported rather than hidden. `cvfn_report`'s
artifact-count denominator is reported beside it so historical numbers stay
comparable.

Two arms are compared statement by statement over the slice their recorded
goal shares: a 2x2 table, an exact McNemar test on the discordant pairs, and a
statement-bootstrap interval on the difference in counted statements. That
interval covers within-run uncertainty only; run-to-run variance is not
measured here.

Report-only: reads run directories, writes nothing into them.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lms.accounting import calculate_cvfn, counts_as_novel, cvfn_report, statement_key

#: Resamples drawn for the statement bootstrap.
BOOTSTRAP_RESAMPLES = 2_000


def _load_artifacts(run_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """A run's artifacts, and whether they carry novelty labels."""
    data = json.loads((Path(run_dir) / "artifacts.json").read_text())
    artifacts: list[dict[str, Any]] = data.get("artifacts", [])
    return artifacts, any(a.get("novelty_level") for a in artifacts)


def _counted(artifact: dict[str, Any], has_novelty: bool) -> bool:
    """The rule `cvfn_report` applies to one artifact, without the tally."""
    if artifact.get("status") != "verified_lean":
        return False
    return counts_as_novel(artifact) if has_novelty else True


def _tag_key(tag: str | None) -> str | None:
    """Statement key for a source tag, normalized as `Goal.mark_formalized` does.

    Flat and iterative agents write the tag as free text and often copy the
    prompt example's quotes; `"0013"` and `0013` are one statement.
    """
    tag = (tag or "").strip("\"'")
    return statement_key(stacks_tag=tag) if tag else None


def statement_outcomes(run_dir: Path) -> dict[str, bool]:
    """Whether each tagged statement a run attempted cleared the gates.

    True iff at least one artifact for the statement is `verified_lean` and,
    on runs with novelty labels, counts as novel. Artifacts without a source
    tag are left out: keying them by declaration name would let two runs'
    unrelated `foo`s count as one statement.
    """
    artifacts, has_novelty = _load_artifacts(run_dir)
    outcomes: dict[str, bool] = {}
    for a in artifacts:
        key = _tag_key(a.get("stacks_tag"))
        if key is None:
            continue
        outcomes[key] = outcomes.get(key, False) or _counted(a, has_novelty)
    return outcomes


def _recorded_goal(run_dir: Path) -> tuple[str | None, list[str]]:
    """The goal name and slice statement keys a run recorded, if any."""
    path = Path(run_dir) / "goal.json"
    if not path.exists():
        return None, []
    goal = json.loads(path.read_text())
    keys = [_tag_key(d.get("tag")) for d in goal.get("definitions", [])]
    return goal.get("name"), [k for k in keys if k is not None]


@dataclass
class ArmReport:
    """One arm's cost and distinct-statement count, pooled over its runs."""

    name: str
    run_dirs: list[str]
    #: Ledgered tokens summed over every run, overhead included.
    total_tokens: int
    distinct_statements: int
    #: What `cvfn_report` would say: verified artifacts, summed over runs.
    artifact_denominator: int
    #: Statements counted by two or more artifacts, within or across runs.
    duplicate_statements: list[str]
    #: Artifacts that would have counted but carry no source tag.
    untagged_verified: int
    #: Statements counted in `distinct_statements` but outside the goal's tag
    #: slice, so no pairing can see them.
    off_slice_verified: int
    cvfn_tokens_per_statement: float | None
    denominator_kind: str
    unmeasurable_reason: str | None = None
    outcomes: dict[str, bool] = field(default_factory=dict)
    #: Distinct goal names the runs recorded, None for a run without one.
    goal_names: list[str | None] = field(default_factory=list)
    goal_statements: list[str] = field(default_factory=list)

    def format(self) -> str:
        if self.unmeasurable_reason is not None:
            cvfn = f"unmeasurable ({self.unmeasurable_reason})"
        elif self.cvfn_tokens_per_statement is None:
            cvfn = "undefined"
        else:
            cvfn = f"{self.cvfn_tokens_per_statement:,.0f}"
        goals = (
            f", {len(self.goal_names)} different goals"
            if len(self.goal_names) > 1
            else ""
        )
        return (
            f"{self.name}: {len(self.run_dirs)} run(s){goals}, "
            f"{self.total_tokens:,} tokens, "
            f"{self.distinct_statements} distinct statements "
            f"({self.artifact_denominator} by artifact count), "
            f"{len(self.duplicate_statements)} duplicated, "
            f"{self.untagged_verified} untagged, "
            f"{self.off_slice_verified} off-slice, "
            f"CVFN {cvfn} [{self.denominator_kind}]"
        )


def _pooled_kind(kinds: list[str]) -> str:
    """One denominator kind for a non-empty pool; the weakest run's kind wins."""
    for kind in ("unmeasurable", "verified_lean_unfiltered"):
        if kind in kinds:
            return kind
    return "verified_novel"


def arm_report(name: str, run_dirs: list[Path]) -> ArmReport:
    """Pool one or more run directories into an arm.

    Tokens are summed over all runs, the ledger read first and `results.json`
    totals as the fallback, exactly as `cvfn_report` reads each run. Statements
    are unioned. Runs recording different goals still pool, but the arm then
    refuses to pair.
    """
    if not run_dirs:
        raise ValueError(f"arm {name}: no run directories")
    reports = [cvfn_report(Path(d)) for d in run_dirs]
    goals = [_recorded_goal(Path(d)) for d in run_dirs]
    outcomes: dict[str, bool] = {}
    counted_by: Counter[str] = Counter()
    untagged = 0
    for d in run_dirs:
        artifacts, has_novelty = _load_artifacts(Path(d))
        for a in artifacts:
            if not _counted(a, has_novelty):
                continue
            key = _tag_key(a.get("stacks_tag"))
            if key is None:
                untagged += 1
            else:
                counted_by[key] += 1
        for key, ok in statement_outcomes(Path(d)).items():
            outcomes[key] = outcomes.get(key, False) or ok

    total_tokens = sum(r.total_tokens for r in reports)
    distinct = sum(outcomes.values())
    kind = _pooled_kind([r.denominator_kind for r in reports])
    reason = next(
        (r.unmeasurable_reason for r in reports if r.unmeasurable_reason), None
    )
    goal_statements = list(dict.fromkeys(k for g in goals for k in g[1]))
    in_slice = set(goal_statements)
    off_slice = sum(1 for k, ok in outcomes.items() if ok and k not in in_slice)
    return ArmReport(
        name=name,
        run_dirs=[str(d) for d in run_dirs],
        total_tokens=total_tokens,
        distinct_statements=distinct,
        artifact_denominator=sum(r.denominator for r in reports),
        duplicate_statements=sorted(k for k, n in counted_by.items() if n >= 2),
        untagged_verified=untagged,
        off_slice_verified=off_slice if in_slice else 0,
        cvfn_tokens_per_statement=(
            None if reason else calculate_cvfn(total_tokens, distinct)
        ),
        denominator_kind=kind,
        unmeasurable_reason=reason,
        outcomes=outcomes,
        goal_names=list(dict.fromkeys(g[0] for g in goals)),
        goal_statements=goal_statements,
    )


def shared_statements(a: ArmReport, b: ArmReport) -> list[str]:
    """The slice two arms can be paired on: their goals' statements in common.

    Refuses when either arm has no recorded goal or the goals differ; a
    comparison over two different slices is not paired.
    """
    if None in a.goal_names + b.goal_names:
        raise ValueError(f"cannot pair {a.name}:{b.name}: no recorded goal")
    if len(a.goal_names) != 1 or a.goal_names != b.goal_names:
        raise ValueError(
            f"cannot pair {a.name}:{b.name}: recorded goals differ "
            f"({a.name}: {a.goal_names}; {b.name}: {b.goal_names})"
        )
    theirs = set(b.goal_statements)
    return [s for s in a.goal_statements if s in theirs]


@dataclass
class PairedResult:
    """Two arms compared statement by statement over a shared slice."""

    a_name: str
    b_name: str
    statements: int
    both: int
    a_only: int
    b_only: int
    neither: int
    #: Exact two-sided McNemar p on the discordant pairs.
    mcnemar_p: float
    #: Counted statements, A minus B.
    difference: int
    #: 95% statement-bootstrap interval on `difference`.
    bootstrap_ci: tuple[int, int]
    seed: int
    a_kind: str
    b_kind: str
    resamples: int = BOOTSTRAP_RESAMPLES

    def format(self) -> str:
        lo, hi = self.bootstrap_ci
        mismatch = (
            [
                f"  !! denominator kind mismatch: {self.a_name} {self.a_kind}, "
                f"{self.b_name} {self.b_kind}"
            ]
            if self.a_kind != self.b_kind
            else []
        )
        return "\n".join(
            [
                f"{self.a_name} vs {self.b_name} over {self.statements} statements",
                *mismatch,
                f"  both {self.both}  {self.a_name} only {self.a_only}  "
                f"{self.b_name} only {self.b_only}  neither {self.neither}",
                f"  McNemar exact p = {self.mcnemar_p:.4g}",
                f"  difference {self.difference:+d}, 95% bootstrap [{lo:+d}, {hi:+d}] "
                f"(seed {self.seed}, {self.resamples} resamples)",
                "  within-run uncertainty only; between-run variance unmeasured",
            ]
        )


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_comparison(
    a: ArmReport, b: ArmReport, statements: list[str], seed: int = 0
) -> PairedResult:
    """2x2 table, exact McNemar and bootstrap interval over `statements`.

    Refuses an unmeasurable arm: its counts are inflated by construction, so a
    test on them would look like a result. Arms with two measurable but
    different denominator kinds are compared, and the result says so.
    """
    for arm in (a, b):
        if arm.denominator_kind == "unmeasurable":
            raise ValueError(
                f"cannot pair {a.name}:{b.name}: {arm.name} is unmeasurable "
                f"({arm.unmeasurable_reason})"
            )
    pairs = [(a.outcomes.get(s, False), b.outcomes.get(s, False)) for s in statements]
    both = sum(1 for x, y in pairs if x and y)
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)

    deltas = [int(x) - int(y) for x, y in pairs]
    rng = random.Random(seed)
    if deltas:
        draws = sorted(
            sum(rng.choices(deltas, k=len(deltas))) for _ in range(BOOTSTRAP_RESAMPLES)
        )
        ci = (
            draws[round(0.025 * (BOOTSTRAP_RESAMPLES - 1))],
            draws[round(0.975 * (BOOTSTRAP_RESAMPLES - 1))],
        )
    else:
        ci = (0, 0)

    return PairedResult(
        a_name=a.name,
        b_name=b.name,
        statements=len(statements),
        both=both,
        a_only=a_only,
        b_only=b_only,
        neither=len(pairs) - both - a_only - b_only,
        mcnemar_p=_mcnemar_exact(a_only, b_only),
        difference=a_only - b_only,
        bootstrap_ci=ci,
        seed=seed,
        a_kind=a.denominator_kind,
        b_kind=b.denominator_kind,
    )
