"""An artifact's own universe names survive the merge into the foundation.

The verifier keeps a candidate's `universe` lines, and the merge dropped them,
so `universe u₁ u₂` verified alone and then failed the merged module with
`unknown universe level`. A failed build rolls back the whole generation, so one
agent's choice of level names cost every other agent's work.

Each artifact's names are bound in a `section` of its own rather than in the
header: a header binding would break a later artifact that names the level
explicitly (`def foo.{u₁}`), and two artifacts may pick the same name.
"""

from __future__ import annotations

from pathlib import Path

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.foundation import FoundationEntry, FoundationFile


def artifact(artifact_id: str, code: str, generation: int = 0) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=ArtifactType.DEFINITION,
        natural_language="test",
        lean_code=code,
        created_by="agent-0",
        generation=generation,
        status=VerificationStatus.VERIFIED_LEAN,
    )


EXOTIC = "universe u₁ u₂\n\ndef pairUp (A : Type u₁) (B : Type u₂) : Type (max u₁ u₂) := A × B"
SAME_NAME = "universe u₁\n\ndef idLevel (A : Type u₁) : A → A := fun a => a"
EXPLICIT = "def explicitLevel.{u₁} (A : Type u₁) : A → A := fun a => a"
PLAIN = "def plainDef : Nat := 1"


def saved(foundation: FoundationFile) -> str:
    foundation.save()
    return foundation.path.read_text()


class TestBinding:
    def test_the_names_are_bound_above_their_entries(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", EXOTIC))

        text = saved(foundation)

        assert text.index("universe u₁ u₂") < text.index("def pairUp")

    def test_the_binding_is_scoped_to_its_artifact(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", EXOTIC))
        foundation.add_artifact(artifact("b", SAME_NAME))
        foundation.add_artifact(artifact("c", EXPLICIT))

        lines = saved(foundation).splitlines()

        assert lines.count("section") == 2
        assert lines.count("end") == 2
        assert lines.index("end") < lines.index("universe u₁")
        # The section closes before the artifact that names the level itself.
        assert max(i for i, x in enumerate(lines) if x == "end") < next(
            i for i, x in enumerate(lines) if x.startswith("def explicitLevel")
        )

    def test_a_header_name_is_not_bound_again(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(
            artifact("a", "universe u u₁\n\ndef f (A : Type u) (B : Type u₁) := A × B")
        )

        lines = saved(foundation).splitlines()

        assert "universe u₁" in lines
        assert "universe u u₁" not in lines

    def test_an_artifact_with_no_names_of_its_own_gets_no_section(
        self, tmp_path: Path
    ) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", "universe u\n\ndef f : Nat := 1"))
        foundation.add_artifact(artifact("b", PLAIN))

        assert "section" not in saved(foundation).splitlines()

    def test_a_name_the_seed_binds_is_not_bound_again(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.set_seed("universe u₁\n\ndef seeded (A : Type u₁) := A")
        foundation.add_artifact(artifact("a", SAME_NAME))

        lines = saved(foundation).splitlines()

        assert lines.count("universe u₁") == 1
        assert "section" not in lines


class TestComments:
    def test_a_trailing_comment_is_not_a_name(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(
            artifact("a", "universe u₁ -- two levels\n\ndef f (A : Type u₁) := A")
        )

        assert "universe u₁" in saved(foundation).splitlines()

    def test_a_docstring_line_is_not_a_binding(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(
            artifact("a", "/-- A note.\nuniverse of discourse\n-/\ndef f : Nat := 1")
        )

        text = saved(foundation)

        assert "section" not in text.splitlines()
        assert foundation.entries[0].universes == ()


class TestRoundTrips:
    def test_a_rollback_drops_the_binding(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", PLAIN))
        good = foundation.snapshot()
        foundation.add_artifact(artifact("b", EXOTIC))

        foundation.restore(good)

        assert "universe u₁ u₂" not in saved(foundation)

    def test_a_generation_boundary_drops_the_binding(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", PLAIN, generation=0))
        foundation.add_artifact(artifact("b", EXOTIC, generation=1))

        foundation.restore(foundation.snapshot_before(1))

        assert "universe u₁ u₂" not in saved(foundation)

    def test_the_binding_survives_a_checkpoint(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", EXOTIC))
        saved(foundation)

        loaded = FoundationFile.load(foundation.path)

        assert "universe u₁ u₂" in saved(loaded)

    def test_a_checkpoint_from_before_the_field_still_loads(self) -> None:
        old = {
            "artifact_id": "a",
            "name": "f",
            "entry_type": "def",
            "signature": "def f : Nat := 1",
            "lean_code": "def f : Nat := 1",
            "generation": 0,
            "author": "agent-0",
        }

        assert FoundationEntry.from_dict(old).universes == ()
