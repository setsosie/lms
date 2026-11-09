"""The arc elaboration check frames drafts as the probe does and blames each error on the right draft.

The Lean run itself happens in CI's Lean job, which has the library build.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_arc_elaboration", ROOT / "scripts" / "check_arc_elaboration.py"
)
assert _spec and _spec.loader
check = importlib.util.module_from_spec(_spec)
sys.modules["check_arc_elaboration"] = check
_spec.loader.exec_module(check)

A = {"id": "core-01", "lean_statement": "theorem a (n : ℕ) : n = n := sorry"}
B = {
    "id": "ram-07",
    "lean_statement": "open scoped Pointwise\n\ntheorem b (n : ℕ) : n + 0 = n := sorry",
}
DEF = {"id": "ram-99", "lean_statement": "def c : ℕ := 3"}


def test_each_draft_gets_its_own_section_and_a_sorry_in_place_of_the_search() -> None:
    source, spans, skipped = check.build_source([A, B])
    assert source.startswith("import Mathlib\n")
    assert "by exact?" not in source
    assert source.count(":= by sorry") == 2
    assert "section S_core_01" in source and "end S_ram_07" in source
    assert [s[0] for s in spans] == ["core-01", "ram-07"]
    assert skipped == []


def test_the_probe_preamble_comes_along_inside_the_draft_section() -> None:
    source, spans, _ = check.build_source([A, B])
    lines = source.splitlines()
    _, first, last = spans[1]
    inside = "\n".join(lines[first - 1 : last])
    assert "open scoped Pointwise" in inside
    assert "open scoped Pointwise" not in "\n".join(lines[: first - 1])


def test_a_draft_the_probe_cannot_frame_is_reported_not_dropped() -> None:
    _, spans, skipped = check.build_source([A, DEF])
    assert [s[0] for s in spans] == ["core-01"]
    assert skipped == ["ram-99"]


def test_an_error_is_blamed_on_the_draft_whose_span_holds_its_line() -> None:
    _, spans, _ = check.build_source([A, B])
    row = spans[1][1] + 2
    out = f"/x/tmp1.lean:{row}:40: error: unknown identifier 'foo'\n"
    assert check.map_errors(out, spans) == {"ram-07": "unknown identifier 'foo'"}


def test_continuation_lines_stop_at_the_next_located_message() -> None:
    _, spans, _ = check.build_source([A, B])
    r1, r2 = spans[0][1] + 1, spans[1][1] + 2
    out = (
        f"/x/t.lean:{r1}:3: error: failed to synthesize\n  DecidableEq (Ideal S)\n"
        f"/x/t.lean:{r2}:5: error: type mismatch\n"
    )
    errors = check.map_errors(out, spans)
    assert errors["core-01"] == "failed to synthesize DecidableEq (Ideal S)"
    assert errors["ram-07"] == "type mismatch"


def test_warnings_and_errors_outside_every_span_are_not_blamed_on_a_draft() -> None:
    _, spans, _ = check.build_source([A])
    out = f"/x/t.lean:{spans[0][1] + 1}:8: warning: declaration uses 'sorry'\n/x/t.lean:1:0: error: unknown package\n"
    assert check.map_errors(out, spans) == {}
