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


class TestDeclarationsAreReadAsLeanReadsThem:
    @pytest.mark.parametrize(
        "modifier",
        ["@[reducible] ", "@[reducible]\n", "private ", "protected ", "noncomputable "],
    )
    def test_a_modifier_cannot_hide_a_collision(
        self, foundation: FoundationFile, modifier: str
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        added = foundation.add_artifact(
            artifact(
                "b",
                f"{modifier}def rep : Nat := 2\n\ntheorem pos : 0 < rep := by decide",
            )
        )

        assert added is False
        assert "pos" not in foundation._definition_names

    def test_a_modified_declaration_is_its_own_entry(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(
            artifact(
                "a",
                "@[simp] theorem first : 1 = 1 := rfl\n\n@[simp] theorem next : 2 = 2 := rfl",
            )
        )

        assert [e.name for e in foundation.entries] == ["first", "next"]
        assert "first" in foundation.entries[0].lean_code
        assert "@[simp]" in foundation.entries[0].lean_code

    @pytest.mark.parametrize(
        "code,names",
        [
            ("def f : Nat := 1\n\ndef f' : Nat := 2", ["f", "f'"]),
            ("def f₁ : Nat := 1\n\ndef f₂ : Nat := 2", ["f₁", "f₂"]),
            ("def rep_α : Nat := 1\n\ndef rep_β : Nat := 2", ["rep_α", "rep_β"]),
            ("def φ : Nat := 1\n\ndef ψ : Nat := 2", ["φ", "ψ"]),
        ],
    )
    def test_a_whole_name_is_one_name(
        self, foundation: FoundationFile, code: str, names: list[str]
    ) -> None:
        assert foundation.add_artifact(artifact("a", code)) is True
        assert [e.name for e in foundation.entries] == names

    def test_a_primed_name_is_not_its_stem(self, foundation: FoundationFile) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        assert foundation.add_artifact(artifact("b", "def rep' : Nat := 2")) is True

    def test_a_universe_list_is_not_part_of_the_name(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(artifact("a", "def f.{r} (A : Sort r) : Sort r := A"))

        assert [e.name for e in foundation.entries] == ["f"]

    def test_a_name_repeated_inside_says_so(self, foundation: FoundationFile) -> None:
        code = (
            "namespace A\ndef f : Nat := 1\nend A\nnamespace B\ndef f : Nat := 2\nend B"
        )
        foundation.add_artifact(artifact("a", code))

        reason = foundation.rejection_reason("a")

        assert reason is not None
        assert "twice" in reason


class TestACommandAfterACopy:
    """A declaration's slice runs to the next one, so commands travel with it."""

    @pytest.mark.parametrize(
        "command",
        [
            "variable (n : Nat)",
            "section",
            "open Nat",
            "#check rep",
            "example : rep = 1 := rfl",
            "attribute [simp] rep",
        ],
    )
    def test_it_is_not_compared_as_the_body(
        self, foundation: FoundationFile, command: str
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        added = foundation.add_artifact(
            artifact("b", f"def rep : Nat := 1\n\n{command}\n\ndef other : Nat := 3")
        )

        assert added is True
        assert foundation.rejection_reason("b") is None
        assert "other" in foundation._definition_names

    def test_it_stays_in_the_file(self, foundation: FoundationFile) -> None:
        """Later entries were verified with it in scope."""
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))
        foundation.add_artifact(
            artifact(
                "b",
                "def rep : Nat := 1\n\nvariable (n : Nat)\n\ndef plus : Nat := n + rep",
            )
        )

        foundation.save()
        text = foundation.path.read_text()

        assert text.index("variable (n : Nat)") < text.index("def plus")

    def test_a_copy_of_a_seed_declaration_ignores_the_seeds_commands(
        self, foundation: FoundationFile
    ) -> None:
        foundation.set_seed(
            'structure S where\n  x : Nat\n\nscoped notation "Q" => S\n\ndef seeded : Nat := 1'
        )

        added = foundation.add_artifact(
            artifact("a", "structure S where\n  x : Nat\n\ndef other : S := ⟨1⟩")
        )

        assert added is True
        assert foundation.rejection_reason("a") is None


class TestRejectionsDontGoStale:
    def test_a_rollback_forgets_the_reason(self, foundation: FoundationFile) -> None:
        empty = foundation.snapshot()
        foundation.add_artifact(artifact("x", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("r", "def rep : Nat := 2"))
        assert foundation.rejection_reason("r") is not None

        foundation.restore(empty)

        assert foundation.rejection_reason("r") is None

    def test_the_same_artifact_can_land_later(self, foundation: FoundationFile) -> None:
        empty = foundation.snapshot()
        foundation.add_artifact(artifact("x", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("r", "def rep : Nat := 2"))
        foundation.restore(empty)

        assert foundation.add_artifact(artifact("r", "def rep : Nat := 2")) is True

        assert foundation.rejection_reason("r") is None
        assert foundation.entries[0].lean_code == "def rep : Nat := 2"


class TestAgentContext:
    def test_an_attribute_on_its_own_line_is_not_the_header(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(artifact("a", "@[reducible]\ndef rep : Nat := 1"))

        assert foundation.entries[0].declaration_header().startswith("def rep")

    def test_a_kept_command_is_not_offered_as_a_definition(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))
        foundation.add_artifact(
            artifact(
                "b", "def rep : Nat := 1\n\nvariable (n : Nat)\n\ndef plus : Nat := n"
            )
        )

        context = foundation.get_context_for_agent()

        assert "plus" in context
        assert "variable" not in context
        assert "tail" not in context


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
