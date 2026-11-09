"""The budget measurement runs the classifier's own probe, counted, and reads each outcome right.

The Lean runs themselves need the pinned library and minutes per probe, so they stay out of the suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "measure_exact_budget", ROOT / "scripts" / "research" / "measure_exact_budget.py"
)
assert _spec and _spec.loader
m = importlib.util.module_from_spec(_spec)
sys.modules["measure_exact_budget"] = m
_spec.loader.exec_module(m)

STATEMENT = "open Nat\n\ntheorem t (a b : ℕ) (h : a ≤ b) : a ≤ b + 1 := sorry"


def test_the_probe_is_the_classifiers_own_with_the_count_wrapped_around_the_theorem() -> (
    None
):
    src = m.wrap_probe(STATEMENT)
    assert src is not None
    assert src.startswith("import Mathlib\n")
    assert "open Nat" in src.split("#count_heartbeats in")[0]
    assert "#count_heartbeats in\nset_option Elab.async false in\ntheorem " in src
    assert src.rstrip().endswith(":= by exact?")


def test_a_definition_is_not_probeable() -> None:
    assert m.wrap_probe("def c : ℕ := 3") is None


def test_a_close_reads_as_closed_with_its_term_and_count() -> None:
    out = "t.lean:5:70: info: Try this:\n  [apply] exact Nat.le_succ_of_le h\nt.lean:3:0: info: Used 13531 heartbeats, which is less than the current maximum of 200000.\n"
    assert m.read_run(out, timed_out=False) == {
        "verdict": "closed",
        "heartbeats_thousands": 13531,
        "close": "exact Nat.le_succ_of_le h",
    }


def test_a_miss_with_no_limit_reads_as_missed() -> None:
    out = "t.lean:5:70: error: `exact?` could not close the goal. Try `apply?` to see partial suggestions.\nt.lean:3:0: info: Used 193525 heartbeats, which is less than the current maximum of 200000.\n"
    row = m.read_run(out, timed_out=False)
    assert row["verdict"] == "missed" and row["heartbeats_thousands"] == 193525


def test_a_run_cut_off_by_the_wall_clock_reads_as_timed_out() -> None:
    assert m.read_run("", timed_out=True)["verdict"] == "timed out"
