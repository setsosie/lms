"""An artifact declaring a name the foundation already has is refused whole.

The merge used to skip any entry whose name already existed and keep the
artifact's other entries. Those then bound to the other agent's declaration,
which is not what Lean verified them against. When the merged file still
compiled, nothing caught it. When it didn't, the whole generation rolled back:
in the team simulation, two agents' different `rep` cost all three Yoneda proofs.

There's no exception for a copy. Deciding from text whether two declarations mean
the same needs the universes, the commands after them and what's in scope, and
each attempt to do it let a statement land meaning something else. An artifact
that imports the foundation can't redeclare a name anyway, so a copy can only
come from agents in one generation. The next generation can import it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.config import ProviderConfig
from lms.foundation import FoundationFile
from lms.lean.mock import MockLeanVerifier
from lms.society import Society
from tests.test_foundation_rollback import SilentProvider


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


class TestAnExistingNameRefusesTheArtifact:
    @pytest.mark.parametrize(
        "copy",
        ["def rep : Nat := 2", "def rep : Nat := 1"],
        ids=["differs", "identical"],
    )
    def test_nothing_from_the_artifact_lands(
        self, foundation: FoundationFile, copy: str
    ) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))

        added = foundation.add_artifact(
            artifact("b", f"{copy}\n\ntheorem usesRep : rep = rep := rfl")
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
            artifact("a", "def seeded : Nat := 1\n\ndef other : Nat := 3")
        )

        assert added is False
        assert foundation.entries == []

    @pytest.mark.parametrize("second", ["1", "2"], ids=["identical", "differs"])
    def test_a_name_repeated_inside_the_artifact(
        self, foundation: FoundationFile, second: str
    ) -> None:
        """Lean accepts this in two namespaces, and the merge strips namespaces."""
        code = f"namespace A\ndef f : Nat := 1\nend A\nnamespace B\ndef f : Nat := {second}\nend B"

        assert foundation.add_artifact(artifact("a", code)) is False
        assert foundation.entries == []
        assert "twice" in (foundation.rejection_reason("a") or "")

    def test_the_reason_names_the_declaration(self, foundation: FoundationFile) -> None:
        foundation.add_artifact(artifact("a", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("b", "def rep : Nat := 2"))

        assert foundation.rejection_reason("a") is None
        assert "`rep`" in (foundation.rejection_reason("b") or "")

    def test_a_copy_cant_land_a_different_instance(
        self, foundation: FoundationFile
    ) -> None:
        """`b.n` is 1 alone, and would be 0 bound to the first instance."""
        foundation.add_artifact(
            artifact(
                "a",
                "structure Foo where\n  n : Nat\n\ninstance : Inhabited Foo := ⟨⟨0⟩⟩",
            )
        )

        foundation.add_artifact(
            artifact(
                "b",
                "structure Foo where\n  n : Nat\n\ninstance : Inhabited Foo := ⟨⟨1⟩⟩\n\n"
                "def b : Foo := default",
            )
        )

        assert "b" not in foundation._definition_names


class TestARefusalKeepsNoState:
    """Committee ids hash the code, so identical work can come back under one id."""

    def test_it_can_land_once_the_name_is_gone(
        self, foundation: FoundationFile
    ) -> None:
        empty = foundation.snapshot()
        foundation.add_artifact(artifact("x", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("r", "def rep : Nat := 2"))
        foundation.restore(empty)

        assert foundation.add_artifact(artifact("r", "def rep : Nat := 2")) is True

        assert foundation.rejection_reason("r") is None
        assert foundation.entries[0].lean_code == "def rep : Nat := 2"

    def test_it_can_land_after_a_rollback_on_resume(
        self, foundation: FoundationFile
    ) -> None:
        later = artifact("x", "def rep : Nat := 1")
        later.generation = 1
        foundation.add_artifact(later)
        foundation.add_artifact(artifact("r", "def rep : Nat := 2"))
        foundation.save()

        loaded = FoundationFile.load(foundation.path)
        loaded.restore(loaded.snapshot_before(1))

        assert loaded.add_artifact(artifact("r", "def rep : Nat := 2")) is True

    def test_it_is_refused_again_while_the_name_stays(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(artifact("x", "def rep : Nat := 1"))
        foundation.add_artifact(artifact("r", "def rep : Nat := 2"))

        assert foundation.add_artifact(artifact("r", "def rep : Nat := 2")) is False
        assert foundation.rejection_reason("r") is not None


class TestDeclarationsAreReadAsLeanReadsThem:
    @pytest.mark.parametrize(
        "modifier",
        [
            "@[reducible] ",
            "@[reducible]\n",
            "@[simp,\n  reducible] ",
            '@[deprecated "x [y]" (since := "2024-01-01")] ',
            "private ",
            "protected ",
            "noncomputable ",
        ],
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

    @pytest.mark.parametrize(
        "second",
        [
            "scoped instance foo : Inhabited Nat := ⟨2⟩",
            "local instance foo : Inhabited Nat := ⟨2⟩",
            "instance (priority := low) foo : Inhabited Nat := ⟨2⟩",
        ],
    )
    def test_an_instance_form_cannot_hide_a_collision(
        self, foundation: FoundationFile, second: str
    ) -> None:
        foundation.add_artifact(artifact("a", "instance foo : Inhabited Nat := ⟨1⟩"))

        assert foundation.add_artifact(artifact("b", second)) is False
        assert "`foo`" in (foundation.rejection_reason("b") or "")

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

    @pytest.mark.parametrize(
        "first,second",
        [
            ("class inductive P where\n  | mk", "class inductive Q where\n  | mk"),
            ("class abbrev P := Inhabited Nat", "class abbrev Q := Inhabited Nat"),
        ],
    )
    def test_class_keywords_name_what_follows(
        self, foundation: FoundationFile, first: str, second: str
    ) -> None:
        foundation.add_artifact(artifact("a", first))

        assert foundation.add_artifact(artifact("b", second)) is True
        assert [e.name for e in foundation.entries] == ["P", "Q"]


class TestAScopedUniverseLine:
    def test_a_declaration_on_the_same_line_is_kept(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(
            artifact("a", "universe u₁ in def foo : Type u₁ := PUnit")
        )

        foundation.save()

        assert [e.name for e in foundation.entries] == ["foo"]
        assert "def foo" in foundation.path.read_text()

    def test_a_comment_is_not_a_declaration(self, foundation: FoundationFile) -> None:
        foundation.add_artifact(
            artifact(
                "a",
                "def g : Nat := 1\n\nuniverse u -- objects live in Type u\n\n"
                "def h : Type u := PUnit",
            )
        )

        foundation.save()

        assert "\nType u" not in foundation.path.read_text()


class TestAgentContext:
    @pytest.mark.parametrize(
        "attribute", ["@[reducible]", "@[to_additive [x]]"], ids=["plain", "nested"]
    )
    def test_an_attribute_on_its_own_line_is_not_the_header(
        self, foundation: FoundationFile, attribute: str
    ) -> None:
        foundation.add_artifact(artifact("a", f"{attribute}\ndef rep : Nat := 1"))

        assert foundation.entries[0].declaration_header().startswith("def rep")

    def test_an_attribute_and_binders_on_one_line_keep_the_line(
        self, foundation: FoundationFile
    ) -> None:
        foundation.add_artifact(
            artifact(
                "a",
                "@[reducible] def pick {M : Type} [Inhabited M]\n    (a : M) : M := a",
            )
        )

        assert "def pick" in foundation.entries[0].declaration_header()


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
