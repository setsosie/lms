"""Slice-selection script (scripts/measure_n1_density.py): which stages vote."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from lms.novelty.mathlib_search import parse_declaration

ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "measure_n1_density", ROOT / "scripts" / "measure_n1_density.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mnd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mnd)

ARCS = [
    ROOT / "data" / "ant_arcs" / f for f in ("core_arc.json", "ramification_arc.json")
]


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(cache_dir=None, lean_project=tmp_path, offline=False)


class TestBuildClassifier:
    def test_label_names_leave_name_keyed_stages_out(self, tmp_path):
        classifier = mnd.build_classifier(args(tmp_path), names_are_labels=True)
        assert [b.stage for b in classifier.backends] == ["exact_probe", "semantic"]

    def test_real_names_keep_the_full_ladder(self, tmp_path):
        classifier = mnd.build_classifier(args(tmp_path))
        assert [b.stage for b in classifier.backends] == [
            "name",
            "loogle",
            "exact_probe",
            "semantic",
        ]


@pytest.mark.parametrize("path", ARCS, ids=lambda p: p.stem)
class TestArcFiles:
    def test_declares_its_names_are_labels(self, path):
        assert json.loads(path.read_text())["names_are_labels"] is True

    def test_every_declaration_name_is_a_label(self, path):
        # The flag is only true while the drafts keep the `ant_` label scheme;
        # a draft named after a Mathlib guess would make name search meaningful.
        statements = json.loads(path.read_text())["statements"]
        names = [parse_declaration(s["lean_statement"])[1] for s in statements]
        assert all(n and n.startswith("ant_") for n in names), names
