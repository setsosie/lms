"""Elaborate every ANT arc draft at the pinned library, as the novelty classifier's `exact?` probe frames it.

A draft that does not elaborate is never measured: its probe reports "statement did not elaborate" and
casts no vote, so the arc's N1 density silently leaves it out. This checks all of them in one Lean run.

    uv run python scripts/check_arc_elaboration.py [--lean-dir lean] [--only id,id] [ARC_JSON ...]

Each draft goes through `ExactProbeBackend.probe_source`, the probe's own framing (its `open`/`variable`
lines kept, auto-implicits off), with `sorry` in place of the search. Every draft sits in its own
section of one file, so the library loads once. Exits 1 if any draft fails to elaborate.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lms.novelty.mathlib_search import ExactProbeBackend  # noqa: E402

DEFAULT_ARCS = ("data/ant_arcs/core_arc.json", "data/ant_arcs/ramification_arc.json")
_LOCATION = re.compile(r":(\d+):(\d+): (error|warning|info)")
_ERROR = re.compile(r":(\d+):(\d+): error")


def build_source(
    statements: list[dict],
) -> tuple[str, list[tuple[str, int, int]], list[str]]:
    """One Lean file holding every probeable draft, the line span of each, and the ids not probeable."""
    lines = ["import Mathlib", ""]
    spans: list[tuple[str, int, int]] = []
    skipped: list[str] = []
    for s in statements:
        probe = ExactProbeBackend.probe_source(s["lean_statement"])
        if probe is None:
            skipped.append(s["id"])
            continue
        tag = re.sub(r"\W", "_", s["id"])
        body = probe.split("\n", 2)[2].replace("by exact?", "by sorry")
        body = body.replace("lms_novelty_probe", f"lms_probe_{tag}")
        first = len(lines) + 1
        lines += [f"section S_{tag}", body.rstrip(), f"end S_{tag}", ""]
        spans.append((s["id"], first, len(lines)))
    return "\n".join(lines) + "\n", spans, skipped


def map_errors(output: str, spans: list[tuple[str, int, int]]) -> dict[str, str]:
    """The first error Lean reported inside each statement's span, with its continuation lines."""
    found: dict[str, str] = {}
    lines = output.splitlines()
    for n, line in enumerate(lines):
        m = _ERROR.search(line)
        if not m:
            continue
        row = int(m.group(1))
        sid = next((i for i, first, last in spans if first <= row <= last), None)
        if sid is None or sid in found:
            continue
        message = [line[m.end() :].lstrip(": ").strip()]
        for more in lines[n + 1 : n + 6]:
            if _LOCATION.search(more):
                break
            message.append(more.strip())
        found[sid] = " ".join(x for x in message if x)[:400]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Elaborate every ANT arc draft at the pinned library."
    )
    parser.add_argument("arcs", nargs="*", default=list(DEFAULT_ARCS))
    parser.add_argument("--lean-dir", default="lean")
    parser.add_argument("--only", help="comma-separated statement ids")
    args = parser.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    statements = [
        s
        for path in args.arcs
        for s in json.loads(Path(path).read_text())["statements"]
        if only is None or s["id"] in only
    ]
    lake = shutil.which("lake")
    lean_dir = Path(args.lean_dir)
    if lake is None or not (lean_dir / ".lake" / "packages" / "mathlib").is_dir():
        print(
            f"no lake, or no library build under {lean_dir}/.lake; run `lake exe cache get` there first"
        )
        return 2
    source, spans, skipped = build_source(statements)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".lean", dir=lean_dir, delete=False
    ) as f:
        f.write(source)
        path = Path(f.name)
    try:
        proc = subprocess.run(
            [
                lake,
                "env",
                "lean",
                "-DautoImplicit=false",
                "-DrelaxedAutoImplicit=false",
                str(path),
            ],
            cwd=lean_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    finally:
        path.unlink(missing_ok=True)
    errors = map_errors(proc.stdout + proc.stderr, spans)
    for sid, _, _ in spans:
        print(f"FAIL {sid}: {errors[sid]}" if sid in errors else f"ok   {sid}")
    for sid in skipped:
        print(f"skip {sid}: not probeable, so the classifier never runs exact? on it")
    print(f"{len(spans) - len(errors)} of {len(spans)} drafts elaborate")
    if not errors and proc.returncode != 0:
        print("Lean failed outside every draft:")
        print((proc.stdout + proc.stderr)[-2000:])
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
