"""A name collision costs the artifact, and nothing lands on the wrong declaration.

The merge used to skip any entry whose name already existed and keep the
artifact's other entries. Those then bound to the other agent's declaration,
which is not what Lean verified them against. When the merged file still
compiled, nothing caught it. When it didn't, the whole generation rolled back:
in the team simulation, two agents' different `rep` cost all three Yoneda proofs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.foundation import FoundationFile
from lms.lean.mock import MockLeanVerifier
from tests.test_foundation_rollback import SilentProvider
from lms.config import ProviderConfig
from lms.society import Society


def artifact(artifact_id: str, code: str) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=ArtifactType.DEFINITION,
        natural_language="test",
        lean_code=code,
        created_by="agent-0",
        generation=0,
        status=VerificationStatus.VERIFIED_LEAN,
    )


@pytest.fixture
def foundation(tmp_path: Path) -> FoundationFile:
    return FoundationFile(tmp_path / "Foundation.lean")


class TestADifferingNameIsRejected:
    def test_nothing_from_the_artifact_lands(self, foundation: FoundationFile) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        added = foundation.add_artifact(
            artifact("b", "def rep : Nat := 2\n\ntheorem usesRep : rep = rep := rfl")
        )

        assert added is False
        assert [e.name for e in foundation.entries] == ["rep"]
        assert "usesRep" not in foundation._definition_names

    def test_the_first_declaration_stays(self, foundation: FoundationFile) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("b", "def rep : Nat := 2"))

        assert foundation.entries[0].lean_code == "def rep : Nat := 1"

    def test_against_the_seed(self, foundation: FoundationFile) -> None:
        foundation.set_seed("def seeded : Nat := 1")

        added = foundation.add_artifact(
            artifact("a", "def seeded : Nat := 2\n\ndef other : Nat := 3")
        )

        assert added is False
        assert foundation.entries == []

    def test_a_name_repeated_inside_the_artifact(
        self, foundation: FoundationFile
    ) -> None:
        """Lean accepts this in two namespaces, and the merge strips namespaces."""
        code = (
            "namespace A\ndef f : Nat := 1\nend A\nnamespace B\ndef f : Nat := 2\nend B"
        )

        assert foundation.add_artifact(artifact("a", code)) is False
        assert foundation.entries == []

    def test_a_definition_that_landed_is_not_marked_by_the_rejection(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("b", "def rep : Nat := 2"))

        assert foundation.rejection_reason("a") is None

    def test_the_reason_names_the_declaration(self, foundation: FoundationFile) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("b", "def rep : Nat := 2"))

        reason = foundation.rejection_reason("b")

        assert reason is not None
        assert "`rep`" in reason


class TestAnIdenticalDeclarationIsHarmless:
    @pytest.mark.parametrize(
        "copy",
        [
            "def rep : Nat := 1",
            "def  rep :  Nat :=\n  1",
            "def rep : Nat := 1 -- the same thing",
        ],
    )
    def test_the_rest_of_the_artifact_is_kept(
        self, foundation: FoundationFile, copy: str
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        added = foundation.add_artifact(
            artifact("b", f"{copy}\n\ndef other : Nat := 3")
        )

        assert added is True
        assert [e.name for e in foundation.entries] == ["rep", "other"]
        assert foundation.rejection_reason("b") is None

    def test_a_complete_copy_adds_nothing_and_is_not_rejected(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        assert foundation.add_artifact(artifact("b", "def rep : Nat := 1")) is False
        assert foundation.rejection_reason("b") is None


class TestSocietyReportsARejection:
    @pytest.fixture
    def society(self, tmp_path: Path) -> Society:
        return Society(
            n_agents=1,
            provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
            verifier=MockLeanVerifier(),
            foundation_path=tmp_path / "Foundation.lean",
        )

    def test_the_reason_reaches_the_notes_and_the_log(
        self, society: Society, capsys: pytest.CaptureFixture[str]
    ) -> None:
        society._promote_to_foundation(artifact("a", "def rep : Nat := 1"))
        rejected = artifact("b", "def rep : Nat := 2")

        society._promote_to_foundation(rejected)

        assert "`rep`" in (rejected.notes or "")
        out = capsys.readouterr().out
        assert "b" in out and "`rep`" in out

    def test_an_accepted_artifact_gets_no_note(self, society: Society) -> None:
        accepted = artifact("a", "def rep : Nat := 1")

        society._promote_to_foundation(accepted)

        assert accepted.notes is None
