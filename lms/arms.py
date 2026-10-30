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
        tag = a.get("stacks_tag")
        if not tag:
            continue
        key = statement_key(stacks_tag=tag)
        outcomes[key] = outcomes.get(key, False) or _counted(a, has_novelty)
    return outcomes


def _recorded_goal(run_dir: Path) -> tuple[str | None, list[str]]:
    """The goal name and slice statement keys a run recorded, if any."""
    path = Path(run_dir) / "goal.json"
    if not path.exists():
        return None, []
    goal = json.loads(path.read_text())
    tags = [d["tag"] for d in goal.get("definitions", []) if d.get("tag")]
    return goal.get("name"), [statement_key(stacks_tag=t) for t in tags]


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
    cvfn_tokens_per_statement: float | None
    denominator_kind: str
    unmeasurable_reason: str | None = None
    outcomes: dict[str, bool] = field(default_factory=dict)
    goal_name: str | None = None
    goal_statements: list[str] = field(default_factory=list)

    def format(self) -> str:
        if self.unmeasurable_reason is not None:
            cvfn = f"unmeasurable ({self.unmeasurable_reason})"
        elif self.cvfn_tokens_per_statement is None:
            cvfn = "undefined"
        else:
            cvfn = f"{self.cvfn_tokens_per_statement:,.0f}"
        return (
            f"{self.name}: {len(self.run_dirs)} run(s), "
            f"{self.total_tokens:,} tokens, "
            f"{self.distinct_statements} distinct statements "
            f"({self.artifact_denominator} by artifact count), "
            f"{len(self.duplicate_statements)} duplicated, "
            f"{self.untagged_verified} untagged, "
            f"CVFN {cvfn} [{self.denominator_kind}]"
        )


def _pooled_kind(kinds: list[str]) -> str:
    """One denominator kind for a pool; the weakest run's kind wins."""
    for kind in ("unmeasurable", "verified_lean_unfiltered"):
        if kind in kinds:
            return kind
    return "verified_novel"


def arm_report(name: str, run_dirs: list[Path]) -> ArmReport:
    """Pool one or more run directories into an arm.

    Tokens are summed over all runs, the ledger read first and `results.json`
    totals as the fallback, exactly as `cvfn_report` reads each run. Statements
    are unioned. Pooled runs must record the same goal.
    """
    reports = [cvfn_report(Path(d)) for d in run_dirs]
    goals = [_recorded_goal(Path(d)) for d in run_dirs]
    goal_names = {g[0] for g in goals}
    if len(goal_names) > 1:
        raise ValueError(f"arm {name}: pooled runs record different goals")

    outcomes: dict[str, bool] = {}
    counted_by: Counter[str] = Counter()
    untagged = 0
    for d in run_dirs:
        artifacts, has_novelty = _load_artifacts(Path(d))
        for a in artifacts:
            if not _counted(a, has_novelty):
                continue
            if a.get("stacks_tag"):
                counted_by[statement_key(stacks_tag=a["stacks_tag"])] += 1
            else:
                untagged += 1
        for key, ok in statement_outcomes(Path(d)).items():
            outcomes[key] = outcomes.get(key, False) or ok

    total_tokens = sum(r.total_tokens for r in reports)
    distinct = sum(outcomes.values())
    kind = _pooled_kind([r.denominator_kind for r in reports])
    reason = next(
        (r.unmeasurable_reason for r in reports if r.unmeasurable_reason), None
    )
    goal_statements = list(dict.fromkeys(k for g in goals for k in g[1]))
    return ArmReport(
        name=name,
        run_dirs=[str(d) for d in run_dirs],
        total_tokens=total_tokens,
        distinct_statements=distinct,
        artifact_denominator=sum(r.denominator for r in reports),
        duplicate_statements=sorted(k for k, n in counted_by.items() if n >= 2),
        untagged_verified=untagged,
        cvfn_tokens_per_statement=(
            None if reason else calculate_cvfn(total_tokens, distinct)
        ),
        denominator_kind=kind,
        unmeasurable_reason=reason,
        outcomes=outcomes,
        goal_name=goals[0][0] if goals else None,
        goal_statements=goal_statements,
    )


def shared_statements(a: ArmReport, b: ArmReport) -> list[str]:
    """The slice two arms can be paired on: their goals' statements in common.

    Refuses when either arm has no recorded goal or the goals differ; a
    comparison over two different slices is not paired.
    """
    if a.goal_name is None or b.goal_name is None:
        raise ValueError(f"cannot pair {a.name}:{b.name}: no recorded goal")
    if a.goal_name != b.goal_name:
        raise ValueError(
            f"cannot pair {a.name}:{b.name}: recorded goals differ "
            f"({a.goal_name!r} vs {b.goal_name!r})"
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
    resamples: int = BOOTSTRAP_RESAMPLES

    def format(self) -> str:
        lo, hi = self.bootstrap_ci
        return "\n".join(
            [
                f"{self.a_name} vs {self.b_name} over {self.statements} statements",
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
    """2x2 table, exact McNemar and bootstrap interval over `statements`."""
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
    )
