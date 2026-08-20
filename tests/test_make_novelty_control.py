"""Gate A control converter (scripts/make_novelty_control.py)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "make_novelty_control",
    Path(__file__).resolve().parent.parent / "scripts" / "make_novelty_control.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mnc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mnc)


class TestDeriveName:
    def test_theorem_declaration(self):
        code = "import Mathlib\n\ntheorem foo_bar (x : Nat) : x = x := rfl\n"
        assert mnc.derive_name(code, "artifact-1") == "foo_bar"

    def test_modifier_prefixes(self):
        code = "noncomputable def Cat.hom_comp : Nat := 0\n"
        assert mnc.derive_name(code, "artifact-1") == "Cat.hom_comp"

    def test_no_declaration_falls_back_to_id(self):
        assert mnc.derive_name("-- just a comment\n", "gen1/agent 2") == "gen1_agent_2"


class TestConvert:
    def test_produces_arc_schema_and_skips_empty(self):
        doc = {
            "artifacts": [
                {
                    "id": "a1",
                    "lean_code": "theorem t1 : True := trivial",
                    "natural_language": "truth",
                    "stacks_tag": "0013",
                    "verified": True,
                    "created_by": "agent-0",
                    "generation": 1,
                },
                {"id": "a2", "lean_code": "   ", "natural_language": "empty"},
            ]
        }
        arc = mnc.convert(doc, source="test")
        assert arc["arc"] == "gate_a_control"
        assert arc["skipped_no_code"] == 1
        (stmt,) = arc["statements"]
        # Every key measure_n1_density.load_arc requires, non-empty.
        assert stmt["id"] == "a1"
        assert stmt["name"] == "t1"
        assert stmt["lean_statement"] == "theorem t1 : True := trivial"
        assert stmt["book_ref"] == "0013"
        assert "verified=True" in stmt["notes"]
