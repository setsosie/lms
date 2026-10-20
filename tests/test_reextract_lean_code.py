"""Re-extracting archived runs (scripts/reextract_lean_code.py).

The script rewrites a pre-cleaner `artifacts.json` through `_clean_lean_code`
so historical runs can be re-scored. Two properties make that safe, and both
were checked only by inline Python in the card's verify script, against an
archive that is not checked in: the input is never touched, and a verdict
reached against packaged source does not carry over to the cleaned source.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

from lms.agent import Agent
from lms.artifacts import Artifact

ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "reextract_lean_code", ROOT / "scripts" / "reextract_lean_code.py"
)
assert _SPEC is not None and _SPEC.loader is not None
rx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rx)

PACKAGED = "|\n  theorem t : True := trivial"
CLEAN = "theorem t : True := trivial"


def _archive(tmp_path: Path) -> Path:
    """A pre-cleaner run: one packaged verified payload, one already clean."""
    path = tmp_path / "artifacts.json"
    path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "id": "packaged",
                        "lean_code": PACKAGED,
                        "status": "verified_mock",
                        "verified": True,
                    },
                    {
                        "id": "clean",
                        "lean_code": CLEAN,
                        "status": "verified_mock",
                        "verified": True,
                    },
                ]
            }
        )
    )
    return path


def _by_id(path: Path) -> dict[str, dict]:
    return {a["id"]: a for a in json.loads(path.read_text())["artifacts"]}


def test_writes_a_sibling_and_leaves_the_input_byte_identical(tmp_path: Path) -> None:
    src = _archive(tmp_path)
    before = hashlib.sha256(src.read_bytes()).hexdigest()

    assert rx.reextract(src) == (2, 1)

    assert hashlib.sha256(src.read_bytes()).hexdigest() == before
    assert (tmp_path / "artifacts.reextracted.json").exists()


def test_a_rewritten_payload_loses_its_verdict(tmp_path: Path) -> None:
    rx.reextract(_archive(tmp_path))

    packaged = _by_id(tmp_path / "artifacts.reextracted.json")["packaged"]

    assert packaged["lean_code"] == CLEAN
    assert packaged["lean_code_raw"] == PACKAGED
    assert packaged["verified"] is False
    assert packaged["status"] == "unverified"
    assert packaged["prior_verified"] is True
    assert packaged["prior_status"] == "verified_mock"


def test_an_unchanged_payload_keeps_its_verdict(tmp_path: Path) -> None:
    rx.reextract(_archive(tmp_path))

    clean = _by_id(tmp_path / "artifacts.reextracted.json")["clean"]

    assert clean["verified"] is True
    assert "prior_verified" not in clean


def test_its_own_output_is_never_an_input(tmp_path: Path) -> None:
    rx.reextract(_archive(tmp_path))

    inputs = rx.resolve_inputs(
        [str(tmp_path), str(tmp_path / "artifacts.reextracted.json")]
    )

    assert inputs == [tmp_path / "artifacts.json"]


def test_raw_capture_survives_a_serialization_round_trip() -> None:
    block = (
        "<artifact>\ntype: theorem\nname: t\ndescription: d\n"
        f"lean: {PACKAGED}\nreferences: []\n</artifact>"
    )
    proposed, _ = Agent(id="v", provider=None)._parse_artifacts(block)  # type: ignore[arg-type]

    restored = Artifact.from_dict(proposed[0].to_dict())

    assert restored.lean_code == CLEAN
    assert restored.lean_code_raw == PACKAGED
