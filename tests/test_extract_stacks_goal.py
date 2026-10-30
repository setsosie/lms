"""Stacks goal extractor (scripts/goals/extract_stacks_goal.py): `\\ref` edges."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "extract_stacks_goal",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "goals"
    / "extract_stacks_goal.py",
)
assert _SPEC is not None and _SPEC.loader is not None
ext = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ext)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "stacks_project"


@pytest.fixture
def stacks(monkeypatch):
    monkeypatch.setattr(ext, "STACKS", FIXTURE)


def test_resolve_refs_label_cases():
    l2t = {"alpha-definition-thing": "0A02", "alpha-section-intro": "0A01"}
    # Same-file labels are written without the file stem.
    text = r"See \ref{definition-thing} and \ref{section-intro}."
    assert ext.resolve_refs(text, "alpha", l2t) == {"0A02", "0A01"}
    # Cross-file labels carry their stem and are used as written.
    assert ext.resolve_refs(r"\ref{alpha-definition-thing}", "beta", l2t) == {"0A02"}
    # A label the tags file does not know is dropped.
    assert ext.resolve_refs(r"\ref{nowhere-label}", "alpha", l2t) == set()


def test_extract_file_reads_refs_from_statement_and_proof(stacks):
    out = ext.extract_file("alpha.tex", 1)
    # Statement ref plus proof ref; the unknown proof ref is dropped.
    assert set(out["0A03"]["refs"]) == {"0A02", "0A01"}
    # The proof is read for references only; its text never enters content.
    assert "PROOF-ONLY-TEXT" not in out["0A03"]["content"]
    # A statement with no proof of its own does not borrow the next one's.
    assert set(out["0A04"]["refs"]) == set()
    assert set(out["0A05"]["refs"]) == {"0A06", "0A03"}


def test_build_goal_keeps_only_earlier_in_track_edges(stacks, capsys):
    alpha = ext.extract_file("alpha.tex", 1)
    beta = ext.extract_file("beta.tex", 2)
    track_alpha = [(t, False) for t in ("0A02", "0A03", "0A04", "0A05", "0A06")]
    goal = ext.build_goal(
        "n", "d", "s", [(track_alpha, alpha), ([("0B01", True)], beta)], "pre"
    )
    requires = {d["tag"]: d["requires"] for d in goal["definitions"]}
    assert requires == {
        "0A02": [],
        "0A03": ["0A02"],  # 0A01 is a section, not in the track
        "0A04": [],
        "0A05": ["0A03"],  # 0A06 comes later: a forward reference
        "0A06": [],
        "0B01": ["0A02", "0A04"],  # cross-file edges within one track
    }
    dropped = [line for line in capsys.readouterr().out.splitlines() if "0A06" in line]
    assert dropped and "0A05" in dropped[0]
