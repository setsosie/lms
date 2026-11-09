"""Re-run exact? probes with no heartbeat limit, and count the heartbeats each search used.

An exact? that runs out of heartbeats reports the same "could not close the goal" as a search that
finished and found nothing. This tells them apart for a given list of statements: it runs the
classifier's own probe under `#count_heartbeats in`, which lifts the limit and reports what the
command used. A probe that closes having used more than the default 200000 (thousands) was cut off
at the default; one that still misses with no limit is a genuine miss.

    uv run python scripts/research/measure_exact_budget.py OUT.jsonl ID [ID ...] [--lean-dir lean]

Needs `lean/.lake` at the pin. One process per probe, as on the box, because the first search in a
process pays for building the search index. Each takes about 1.5 to 3 minutes and about 4.3 GB.
Rows already in OUT.jsonl are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lms.novelty.mathlib_search import ExactProbeBackend  # noqa: E402

ARCS = (
    "data/ant_arcs/core_arc.json",
    "data/ant_arcs/ramification_arc.json",
    "data/novelty_control/gate_a_control_arc.json",
)
WALL_LIMIT_S = 1200
_USED = re.compile(r"Used (\d+) heartbeats")
_CLOSE = re.compile(r"Try this:\s*(?:\[apply\]\s*)?(exact [^\n]*)")
_MISS = "could not close the goal"


def wrap_probe(lean_statement: str) -> str | None:
    """The probe the classifier would run, with its limit lifted and its heartbeats counted."""
    source = ExactProbeBackend.probe_source(lean_statement)
    if source is None:
        return None
    i = source.index("theorem ")
    # Async elaboration would move the proof out of the counted command.
    return (
        source[:i]
        + "#count_heartbeats in\nset_option Elab.async false in\n"
        + source[i:]
    )


def read_run(output: str, timed_out: bool) -> dict:
    used = _USED.search(output)
    close = _CLOSE.search(output)
    missed = _MISS in output
    if timed_out:
        verdict = "timed out"
    elif close and not missed:
        verdict = "closed"
    elif missed and not close:
        verdict = "missed"
    else:
        verdict = "other"
    return {
        "verdict": verdict,
        "heartbeats_thousands": int(used.group(1)) if used else None,
        "close": close.group(1)[:200] if close else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-run exact? probes with no heartbeat limit."
    )
    parser.add_argument("out")
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--lean-dir", default="lean")
    args = parser.parse_args(argv)
    by_id = {
        s["id"]: s for p in ARCS for s in json.loads(Path(p).read_text())["statements"]
    }
    out, lean_dir = Path(args.out), Path(args.lean_dir)
    done = set()
    if out.exists():
        done = {
            json.loads(line)["id"]
            for line in out.read_text().splitlines()
            if line.strip()
        }
    probe = lean_dir / "MeasureExactBudget.lean"
    try:
        for sid in args.ids:
            if sid in done:
                continue
            source = wrap_probe(by_id[sid]["lean_statement"])
            if source is None:
                print(f"{sid}: not probeable")
                continue
            probe.write_text(source)
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    [
                        "lake",
                        "env",
                        "lean",
                        "-DautoImplicit=false",
                        "-DrelaxedAutoImplicit=false",
                        probe.name,
                    ],
                    cwd=lean_dir,
                    capture_output=True,
                    text=True,
                    timeout=WALL_LIMIT_S,
                )
                row = read_run(proc.stdout + proc.stderr, timed_out=False)
            except subprocess.TimeoutExpired:
                row = read_run("", timed_out=True)
            row = {"id": sid, **row, "wall_s": round(time.monotonic() - start, 1)}
            with out.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        probe.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
