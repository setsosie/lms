#!/usr/bin/env python3
"""Measure the N1 density of a candidate ANT arc — 26Q3-HARN-04 slice-selection mode.

Resolves the open decision in `docs/planning/calibration-program.md` §4: which
Neukirch ANT Ch. I slice to calibrate on. Run it over each candidate arc file
and pick on **measured** N1 density, not on recollection of Mathlib's contents.

Usage:
    uv run python scripts/measure_n1_density.py data/ant_arcs/core_arc.json \
        --json-out experiments/ant_slice_selection/core_arc_report.json

Input schema (one file per arc):
    {
      "arc": "core | ramification",
      "source": "free-text provenance",
      "mathlib_rev": "optional pin",
      "statements": [
        {"id": "...", "book_ref": "Neukirch ANT Ch I §x.y", "name": "snake_case_name",
         "informal": "prose statement", "lean_statement": "theorem ... := sorry",
         "notes": "..."}
      ]
    }

The report carries the full confidence distribution and the D4 review queue,
not just a density number: on a density measurement, systematic bias matters
more than per-item error.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lms.novelty import NoveltyClassifier, measure_density
from lms.novelty.mathlib_search import DiskCache, default_backends, detect_mathlib_rev

REQUIRED_STATEMENT_KEYS = ("id", "name", "lean_statement")


def load_arc(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if "statements" not in doc or not isinstance(doc["statements"], list):
        raise SystemExit(f"{path}: missing 'statements' list")
    for i, stmt in enumerate(doc["statements"]):
        missing = [k for k in REQUIRED_STATEMENT_KEYS if not stmt.get(k)]
        if missing:
            raise SystemExit(f"{path}: statements[{i}] missing {missing}")
    return doc


def build_classifier(args: argparse.Namespace) -> NoveltyClassifier:
    cache = DiskCache(args.cache_dir) if args.cache_dir else None
    backends = default_backends(args.lean_project)
    if args.offline:
        # Cache-only: every backend that would touch the network or the
        # toolchain is dropped; only cached answers and the local name grep
        # remain. Missing cache entries then surface as INCONCLUSIVE, loudly.
        backends = [b for b in backends if b.stage == "name"]
    return NoveltyClassifier(
        backends,
        cache=cache,
        mathlib_rev=detect_mathlib_rev(args.lean_project),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("statements_json", type=Path, help="arc statements file")
    parser.add_argument("--lean-project", type=Path, default=Path("lean"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".novelty_cache"),
        help="disk cache for search results (keyed by statement + Mathlib rev)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no network, no toolchain: cached answers + local name search only",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--max-statements", type=int, default=None, help="classify only the first N"
    )
    args = parser.parse_args(argv)

    doc = load_arc(args.statements_json)
    if args.max_statements is not None:
        doc = {**doc, "statements": doc["statements"][: args.max_statements]}

    classifier = build_classifier(args)
    report = measure_density(doc, classifier)

    print(f"Arc: {report['arc']}   (Mathlib {report['mathlib_rev'] or 'unknown'})")
    print(f"{'id':<28} {'level':<13} {'conf':>5}  evidence")
    for r in report["statements"]:
        top = r["evidence"][0] if r["evidence"] else ""
        print(f"{r['id']:<28} {r['level']:<13} {r['confidence']:>5.2f}  {top[:70]}")
    counts = report["counts"]
    print(
        f"\nTotal {report['total_statements']}: "
        f"N0={counts['N0']}  N1={counts['N1']}  INCONCLUSIVE={counts['INCONCLUSIVE']}"
    )
    print(
        f"N1 density: {report['n1_density']:.2f} (upper) / "
        f"{report['n1_density_decisive']:.2f} (decisive)"
    )
    print(f"Confidence distribution: {report['confidence_distribution']}")
    if report["needs_review"]:
        print(
            f"D4 review queue ({len(report['needs_review'])}): {report['needs_review']}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
