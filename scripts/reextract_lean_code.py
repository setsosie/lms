#!/usr/bin/env python
"""Re-extract `lean_code` from archived `artifacts.json` runs (26Q3-HARN-02).

Runs recorded before HARN-02 stored the raw regex capture in `lean_code`, so
every payload carries its YAML block-scalar marker (`|\\n  `) or markdown fence
into the field the verifier reads. This rewrites those files through
`_clean_lean_code` so Gate A can re-score historical runs on clean source.

Output goes to a `*.reextracted.json` sibling. The input is never modified --
the archived record of what the harness actually did stays intact, because the
point of re-scoring is to compare against it.

Re-extraction does not verify anything. A cleaned payload has only been
un-packaged; whether Lean accepts it is a separate question that requires a
real verifier pass.

Usage:
    uv run python scripts/reextract_lean_code.py experiments/stacks_ch4_phase1
    uv run python scripts/reextract_lean_code.py experiments/*/artifacts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lms.agent import _clean_lean_code  # noqa: E402

SUFFIX = ".reextracted.json"


def resolve_inputs(paths: list[str]) -> list[Path]:
    """Expand directories to the `artifacts.json` they contain."""
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidate = path / "artifacts.json"
            if candidate.exists():
                resolved.append(candidate)
        elif path.name.endswith(SUFFIX):
            continue  # Never re-extract our own output.
        elif path.exists():
            resolved.append(path)
    return resolved


def reextract(path: Path) -> tuple[int, int]:
    """Rewrite one run's artifacts through the cleaner.

    Returns:
        Tuple of (artifacts with Lean code, artifacts whose code changed).
    """
    data = json.loads(path.read_text())
    artifacts = data.get("artifacts", [])

    with_code = 0
    changed = 0
    for artifact in artifacts:
        original = artifact.get("lean_code")
        if not original:
            continue
        with_code += 1

        cleaned = _clean_lean_code(original)
        if cleaned == original:
            continue
        changed += 1

        # Preserve what the run actually recorded, so the rewrite is auditable
        # from the output file alone.
        artifact.setdefault("lean_code_raw", original)
        artifact["lean_code"] = cleaned

        # The old verdict was reached against the packaged source, which is not
        # the source in this file any more. Carrying `verified: true` forward
        # onto code that was never checked in this form is exactly the failure
        # mode HARN-01 closed, so the verdict resets and the prior claim moves
        # to a field no scorer reads by accident.
        artifact["prior_status"] = artifact.get("status")
        artifact["prior_verified"] = artifact.get("verified", False)
        artifact["status"] = "unverified"
        artifact["verified"] = False

    out_path = path.with_name(path.name.removesuffix(".json") + SUFFIX)
    if out_path.resolve() == path.resolve():
        raise ValueError(f"refusing to write over the input: {path}")
    out_path.write_text(json.dumps(data, indent=2))

    print(f"{path}")
    print(f"  {with_code} with lean_code, {changed} rewritten -> {out_path.name}")
    return with_code, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="artifacts.json files, or run directories containing one",
    )
    args = parser.parse_args()

    inputs = resolve_inputs(args.paths)
    if not inputs:
        print("No artifacts.json found in the given paths.", file=sys.stderr)
        return 1

    total_code = 0
    total_changed = 0
    for path in inputs:
        with_code, changed = reextract(path)
        total_code += with_code
        total_changed += changed

    print(f"\n{len(inputs)} run(s): {total_changed}/{total_code} payloads rewritten.")
    print(
        "Re-extraction is not verification -- these still need a real "
        "verifier pass before any of them counts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
