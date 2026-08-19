"""Per-statement cost accounting — the CVFN numerator.

CVFN = (tokens + GPU wall-clock + human review minutes) / (statements clearing
all gates). The gate work (HARN-03/04) filters the denominator; this module
records the numerator: every generation call, including the ones that failed or
parsed to nothing, attributed to the statement that spent it.

Society-level totals were always complete; per-artifact attribution was not.
Cost was split evenly across the artifacts parsed from a response, and a
response parsing to zero artifacts contributed zero recorded cost. At a 3-6%
success rate the failed attempts are most of the spend, so per-artifact numbers
understated true cost by roughly the inverse of the success rate.

The ledger is the cost of record. `Artifact.tokens_used` remains for backward
compatibility but is no longer authoritative. Spend that cannot be attributed
to any statement accumulates in an explicit overhead bucket
(`statement_key == OVERHEAD_KEY`) — never silently dropped.

Scope: flat and iterative modes are ledgered. Working-group mode is not (its
wiring is 26Q3-HARN-12's card); the conservation invariant holds for non-group
runs.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Statement key for spend that cannot be attributed to any statement.
OVERHEAD_KEY = "__overhead__"

#: Novelty levels that count toward the CVFN denominator (populated by the
#: HARN-04 classifier; absent on runs predating it).
NOVEL_LEVELS = frozenset({"N1", "N2", "N3"})

_DECL_RE = re.compile(
    r"^\s*(?:noncomputable\s+)?(?:private\s+|protected\s+)?"
    r"(?:theorem|lemma|def|abbrev|structure|class|instance|inductive)\s+"
    r"([A-Za-z_][A-Za-z0-9_.']*)",
    re.MULTILINE,
)


def statement_key(
    stacks_tag: str | None = None,
    lean_code: str | None = None,
    natural_language: str | None = None,
) -> str:
    """Stable key for the statement an attempt was spent on.

    Source anchor (Stacks tag) where available, else the declared Lean name,
    else a normalized slug of the natural-language statement — stable across
    retries of the same target so a retry chain is attributable. Falls back to
    `OVERHEAD_KEY` when nothing identifiable was produced.
    """
    if stacks_tag:
        return f"tag:{stacks_tag}"
    if lean_code:
        m = _DECL_RE.search(lean_code)
        if m:
            return f"decl:{m.group(1)}"
    if natural_language:
        slug = re.sub(r"[^a-z0-9]+", "-", natural_language.lower()).strip("-")[:60]
        if slug:
            return f"nl:{slug}"
    return OVERHEAD_KEY


@dataclass
class AttemptRecord:
    """One generation call, attributed to a statement (or to overhead).

    Appended for every call — including calls that parsed to zero artifacts,
    which is where most of the spend goes at low success rates.
    """

    statement_key: str
    agent_id: str
    generation: int
    prompt_tokens: int
    completion_tokens: int
    wall_clock_s: float
    outcome: str
    gate_failed: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement_key": self.statement_key,
            "agent_id": self.agent_id,
            "generation": self.generation,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_clock_s": self.wall_clock_s,
            "outcome": self.outcome,
            "gate_failed": self.gate_failed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttemptRecord":
        return cls(
            statement_key=d["statement_key"],
            agent_id=d["agent_id"],
            generation=d["generation"],
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            wall_clock_s=d["wall_clock_s"],
            outcome=d["outcome"],
            gate_failed=d.get("gate_failed"),
        )


@dataclass
class StatementCost:
    """Rolled-up cost of one statement across its whole retry chain."""

    statement_key: str
    tokens: int = 0
    wall_clock_s: float = 0.0
    attempts: int = 0
    verified: bool = False


class CostLedger:
    """Append-only record of every generation call in a run."""

    def __init__(self) -> None:
        self.records: list[AttemptRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def record(self, rec: AttemptRecord) -> None:
        self.records.append(rec)

    def record_overhead(
        self,
        agent_id: str,
        generation: int,
        prompt_tokens: int,
        completion_tokens: int,
        wall_clock_s: float,
        outcome: str = "unattributable",
    ) -> None:
        """Spend with nothing identifiable to attribute it to."""
        self.record(
            AttemptRecord(
                statement_key=OVERHEAD_KEY,
                agent_id=agent_id,
                generation=generation,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                wall_clock_s=wall_clock_s,
                outcome=outcome,
            )
        )

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def overhead_tokens(self) -> int:
        return sum(
            r.total_tokens for r in self.records if r.statement_key == OVERHEAD_KEY
        )

    @property
    def total_wall_clock_s(self) -> float:
        return sum(r.wall_clock_s for r in self.records)

    def per_statement(self) -> dict[str, StatementCost]:
        """Cost per statement including every failed attempt in its chain."""
        costs: dict[str, StatementCost] = {}
        for r in self.records:
            c = costs.setdefault(r.statement_key, StatementCost(r.statement_key))
            c.tokens += r.total_tokens
            c.wall_clock_s += r.wall_clock_s
            c.attempts += 1
            if r.outcome == "verified_lean":
                c.verified = True
        return costs

    def gate_failure_histogram(self) -> Counter[str]:
        return Counter(r.gate_failed for r in self.records if r.gate_failed)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"records": [r.to_dict() for r in self.records]}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> "CostLedger":
        ledger = cls()
        data = json.loads(path.read_text())
        ledger.records = [AttemptRecord.from_dict(d) for d in data["records"]]
        return ledger


def calculate_cvfn(cost_tokens: int, verified_novel_count: int) -> float | None:
    """Tokens per verified-novel statement; None when the denominator is 0.

    Undefined, not infinite: a run with zero qualifying statements has no CVFN,
    and reporting one would hide exactly the situation the number exists to
    expose.
    """
    if verified_novel_count <= 0:
        return None
    return cost_tokens / verified_novel_count


@dataclass
class CVFNReport:
    """CVFN with its denominator stated explicitly.

    `CVFN = 2.1M tokens / 1 statement` and `CVFN = 2.1M / 50` are wildly
    different confidence situations; the denominator count is always reported
    alongside the ratio.
    """

    run_dir: str
    total_tokens: int
    total_wall_clock_s: float
    review_minutes: float
    verified_count: int
    #: Denominator actually used, and what it counts. "verified_novel" when
    #: novelty labels (HARN-04) are present; "verified_lean_unfiltered" when
    #: they are not — an upper bound on the true denominator, flagged as such.
    denominator: int
    denominator_kind: str
    cvfn_tokens_per_statement: float | None
    status_histogram: dict[str, int] = field(default_factory=dict)
    gate_failures: dict[str, int] = field(default_factory=dict)
    ledger_present: bool = False
    overhead_tokens: int = 0

    def format(self) -> str:
        lines = [
            f"CVFN report — {self.run_dir}",
            f"  total tokens:      {self.total_tokens:,}"
            + ("" if self.ledger_present else "  (society totals; no attempt ledger)"),
            f"  overhead tokens:   {self.overhead_tokens:,}"
            if self.ledger_present
            else "  overhead tokens:   n/a (run predates the ledger)",
            f"  total wall-clock:  {self.total_wall_clock_s:,.1f} s",
            f"  review minutes:    {self.review_minutes:,.1f}",
            f"  verified (lean):   {self.verified_count}",
            f"  denominator:       {self.denominator} ({self.denominator_kind})",
        ]
        if self.cvfn_tokens_per_statement is None:
            lines.append(
                f"  CVFN:              undefined — {self.total_tokens:,} tokens / "
                f"0 statements"
            )
        else:
            lines.append(
                f"  CVFN:              {self.cvfn_tokens_per_statement:,.0f} "
                f"tokens/statement ({self.total_tokens:,} / {self.denominator})"
            )
        if self.status_histogram:
            lines.append("  status histogram:")
            for status, n in sorted(self.status_histogram.items()):
                lines.append(f"    {status}: {n}")
        if self.gate_failures:
            lines.append("  gate failures (ledger):")
            for gate, n in sorted(self.gate_failures.items()):
                lines.append(f"    {gate}: {n}")
        return "\n".join(lines)


def _load_review_minutes(run_dir: Path) -> float:
    """Human review minutes from `review_log.json` (produced in Phase D).

    Accepts `[{"statement_key": ..., "minutes": ...}, ...]`, the same list
    under an `"entries"` key, or a bare `{"minutes": <total>}`.
    """
    path = run_dir / "review_log.json"
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "entries" in data:
            data = data["entries"]
        else:
            return float(data.get("minutes", 0.0))
    return float(sum(e.get("minutes", 0.0) for e in data))


def cvfn_report(run_dir: Path) -> CVFNReport:
    """CVFN over a saved run directory, archived or fresh.

    Reads `attempts.json` (the ledger) when present; older runs fall back to
    the society totals in `results.json`, which were always complete even when
    attribution was not. Novelty labels are honored when the artifacts carry
    them; without them the denominator is the unfiltered verified count and is
    labelled as such.
    """
    run_dir = Path(run_dir)

    results = json.loads((run_dir / "results.json").read_text())
    checkpoint = results.get("checkpoint", {})
    generations = results.get("generations", [])

    ledger: CostLedger | None = None
    attempts_path = run_dir / "attempts.json"
    if attempts_path.exists():
        ledger = CostLedger.load(attempts_path)

    if ledger is not None and ledger.records:
        total_tokens = ledger.total_tokens
        overhead = ledger.overhead_tokens
        gate_failures = dict(ledger.gate_failure_histogram())
    else:
        total_tokens = checkpoint.get("total_tokens_used", 0)
        overhead = 0
        gate_failures = {}

    # Per-generation wall-clock (0.0 on runs predating the field).
    total_wall_clock = sum(g.get("wall_clock_s", 0.0) for g in generations)

    artifacts_data = json.loads((run_dir / "artifacts.json").read_text())
    artifacts = artifacts_data.get("artifacts", [])
    status_histogram: Counter[str] = Counter(
        a.get("status", "unverified") for a in artifacts
    )
    verified = [a for a in artifacts if a.get("status") == "verified_lean"]

    has_novelty = any(a.get("novelty_level") for a in artifacts)
    if has_novelty:
        denominator = sum(1 for a in verified if a.get("novelty_level") in NOVEL_LEVELS)
        denominator_kind = "verified_novel"
    else:
        denominator = len(verified)
        denominator_kind = "verified_lean_unfiltered"

    return CVFNReport(
        run_dir=str(run_dir),
        total_tokens=total_tokens,
        total_wall_clock_s=total_wall_clock,
        review_minutes=_load_review_minutes(run_dir),
        verified_count=len(verified),
        denominator=denominator,
        denominator_kind=denominator_kind,
        cvfn_tokens_per_statement=calculate_cvfn(total_tokens, denominator),
        status_histogram=dict(status_histogram),
        gate_failures=gate_failures,
        ledger_present=ledger is not None and bool(ledger.records),
        overhead_tokens=overhead,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m lms.accounting <run_dir>", file=sys.stderr)
        raise SystemExit(2)
    print(cvfn_report(Path(sys.argv[1])).format())
