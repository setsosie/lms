#!/usr/bin/env python3
"""Compare saved run directories as arms.

Each `--arm` names one arm and the run directories it pools; a pooled arm sums
every run's tokens and counts each statement once. Each `--pair` compares two
named arms statement by statement over the slice their recorded goal shares,
or reports why it cannot.

Usage:
    uv run python scripts/compare_arms.py \
        --arm C9=runs/c9 --arm I9=runs/i1,runs/i2,runs/i3 \
        --pair C9:I9 --json comparison.json
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lms.arms import (  # noqa: E402
    ArmReport,
    arm_report,
    paired_comparison,
    shared_statements,
)


def _parse_arm(spec: str) -> tuple[str, list[Path]]:
    name, sep, dirs = spec.partition("=")
    if not sep or not name or not dirs:
        raise argparse.ArgumentTypeError(f"expected NAME=DIR[,DIR...], got {spec!r}")
    return name, [Path(d) for d in dirs.split(",")]


def _parse_pair(spec: str) -> tuple[str, str]:
    a, sep, b = spec.partition(":")
    if not sep or not a or not b:
        raise argparse.ArgumentTypeError(f"expected A:B, got {spec!r}")
    return a, b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        type=_parse_arm,
        action="append",
        required=True,
        metavar="NAME=DIR[,DIR...]",
        help="an arm and the run directories it pools",
    )
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        action="append",
        default=[],
        metavar="A:B",
        help="two named arms to compare statement by statement",
    )
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    arms: dict[str, ArmReport] = {}
    for name, dirs in args.arm:
        if name in arms:
            parser.error(f"--arm {name!r} given twice")
        arms[name] = arm_report(name, dirs)
    for a, b in args.pair:
        for missing in {a, b} - arms.keys():
            parser.error(f"--pair names unknown arm {missing!r}")

    for report in arms.values():
        print(report.format())

    pairs: list[dict[str, Any]] = []
    for a, b in args.pair:
        print()
        try:
            statements = shared_statements(arms[a], arms[b])
            result = paired_comparison(arms[a], arms[b], statements)
        except ValueError as exc:
            print(f"{a} vs {b}: refused, {exc}")
            pairs.append({"a": a, "b": b, "refused": str(exc)})
            continue
        print(result.format())
        pairs.append(asdict(result))

    if args.json is not None:
        payload = {"arms": [asdict(r) for r in arms.values()], "pairs": pairs}
        args.json.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
