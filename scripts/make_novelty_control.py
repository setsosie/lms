#!/usr/bin/env python3
"""Convert an archived `artifacts.json` into the arc schema — Gate A control.

The novelty classifier's calibration (`current-sprint.md` DoD item 3): the
archived `stacks_ch4_phase1` artifacts are known Mathlib reimplementations,
so a working classifier must read them ~all N0. An N1-heavy result here means
the search queries are broken — and voids any arc density measured with them,
because a query that never matches anything produces confident "absent from
Mathlib" verdicts everywhere.

Feed it the `.reextracted.json` sibling (see `reextract_lean_code.py`), not
the raw archive: pre-HARN-02 records carry the YAML block-scalar leak in
`lean_code`, and searching on `|`-prefixed source tests the leak, not Mathlib.

Usage:
    uv run python scripts/reextract_lean_code.py experiments/stacks_ch4_phase1
    uv run python scripts/make_novelty_control.py \
        experiments/stacks_ch4_phase1/artifacts.reextracted.json \
        --out experiments/n1_density/gate_a_control_arc.json
    uv run python scripts/measure_n1_density.py \
        experiments/n1_density/gate_a_control_arc.json \
        --json-out experiments/n1_density/gate_a_control.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# The first named declaration in the payload. Good enough for search queries;
# artifacts whose code never declares anything fall back to their id.
_DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|instance|inductive)\s+"
    r"([A-Za-z_][A-Za-z0-9_'.]*)",
    re.MULTILINE,
)


def derive_name(lean_code: str, artifact_id: str) -> str:
    m = _DECL_RE.search(lean_code)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_]", "_", artifact_id)


def convert(artifacts_doc: dict, source: str) -> dict:
    statements = []
    skipped = 0
    for a in artifacts_doc["artifacts"]:
        code = a.get("lean_code")
        if not code or not code.strip():
            skipped += 1
            continue
        statements.append(
            {
                "id": a["id"],
                "book_ref": a.get("stacks_tag") or "",
                "name": derive_name(code, a["id"]),
                "informal": a.get("natural_language") or "",
                "lean_statement": code,
                "notes": f"verified={a.get('verified', False)} "
                f"created_by={a.get('created_by', '?')} gen={a.get('generation', '?')}",
            }
        )
    return {
        "arc": "gate_a_control",
        "source": source,
        "expected": "~all N0 — these are known Mathlib reimplementations",
        "skipped_no_code": skipped,
        "statements": statements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("artifacts_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    doc = json.loads(args.artifacts_json.read_text())
    if "artifacts" not in doc:
        raise SystemExit(
            f"{args.artifacts_json}: not an artifacts.json (no 'artifacts')"
        )

    arc = convert(doc, source=str(args.artifacts_json))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(arc, indent=2))
    print(
        f"{len(arc['statements'])} statements -> {args.out} "
        f"({arc['skipped_no_code']} skipped for empty lean_code)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
