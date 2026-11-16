"""The foundation keeps the imports its entries verified with.

An artifact verifies alone with its own `import` lines, and the merge into
`LMS.Foundation` dropped them on the assumption that the header covers
everything. A category artifact importing `Mathlib.CategoryTheory.Category.Basic`
then failed the merged build, and the rollback took its generation's work. A
failed build also logged only Lake's `error: build failed`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import lms.society
from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.config import ProviderConfig
from lms.foundation import FoundationEntry, FoundationFile
from lms.lean.mock import MockLeanVerifier
from lms.lean.project import LeanProject, lean_error_lines
from lms.seed import load_seed
from lms.society import Society
from tests.test_foundation_rollback import FakeProject, ProjectVerifier
from tests.test_society import MockProvider

CAT = "Mathlib.CategoryTheory.Category.Basic"
SUB = "Mathlib.Algebra.Group.Subgroup.Basic"
HEADER_IMPORTS = FoundationFile.FOUNDATION_IMPORTS
#: The batch's shape: indented, importing Mathlib's category.
ON_MATHLIB = f"  import {CAT}\n\n  def endo (C : Type u) [CategoryTheory.Category.{{v}} C] : Nat := 1"
SEED = f"import {SUB}\n\nuniverse u\n\ndef seeded : Nat := 1"
WARNING = "Mathlib's own category API does NOT exist here"


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


def importing(name: str, *modules: str) -> str:
    return "".join(f"import {m}\n" for m in modules) + f"def {name} : Nat := 1"


def extra_imports_in(text: str) -> list[str]:
    """The saved file's imports beyond the header's, in file order."""
    imports = [x.split()[1] for x in text.splitlines() if x.startswith("import ")]
    assert imports[: len(HEADER_IMPORTS)] == list(HEADER_IMPORTS)
    return imports[len(HEADER_IMPORTS) :]


def saved(foundation: FoundationFile) -> str:
    foundation.save()
    return foundation.path.read_text()


class TestEntries:
    def test_an_indented_import_goes_to_the_top(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        assert foundation.add_artifact(artifact("a", ON_MATHLIB))

        lines = saved(foundation).splitlines()

        assert foundation.entries[0].imports == (CAT,)
        assert lines.index(f"import {CAT}") < lines.index("universe u v w")
        assert not any(
            x.strip().startswith("import")
            for x in lines[lines.index("universe u v w") :]
        )

    def test_the_union_is_deduplicated_and_sorted(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.add_artifact(artifact("a", importing("f", CAT, SUB)))
        foundation.add_artifact(
            artifact("b", importing("g", SUB, "Mathlib.Order.Basic"))
        )

        assert extra_imports_in(saved(foundation)) == sorted(
            {CAT, SUB, "Mathlib.Order.Basic"}
        )

    def test_own_module_header_comments_and_trailing_text_are_dropped(
        self, tmp_path: Path
    ) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        code = (
            f"import LMS.Foundation\nimport {HEADER_IMPORTS[0]}\nimport {SUB} -- groups\n"
            "/-- A note.\nimport Not.A.Module\n-/\ndef f : Nat := 1\ndef g : Nat := 2"
        )
        foundation.add_artifact(artifact("a", code))

        assert [e.imports for e in foundation.entries] == [(SUB,), (SUB,)]
        assert extra_imports_in(saved(foundation)) == [SUB]

    def test_an_artifact_without_imports_changes_nothing(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.add_artifact(artifact("a", "def f : Nat := 1"))

        assert saved(foundation).startswith(FoundationFile.FOUNDATION_HEADER)


class TestRoundTrips:
    def test_save_and_load_keep_them(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.set_seed(SEED)
        foundation.add_artifact(artifact("a", ON_MATHLIB))
        before = saved(foundation)

        loaded = FoundationFile.load(foundation.path)

        assert (loaded.seed_imports, loaded.entries[0].imports) == ((SUB,), (CAT,))
        assert "import" not in loaded.seed_source
        assert saved(loaded) == before

    def test_a_seed_of_imports_alone_round_trips(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.set_seed(f"import {CAT}\n")
        before = saved(foundation)

        loaded = FoundationFile.load(foundation.path)

        assert (loaded.seed_entries, loaded.seed_imports) == ([], (CAT,))
        assert saved(loaded) == before

    def test_an_entry_from_before_the_field_loads(self) -> None:
        old = FoundationEntry(
            "a", "f", "def", "def f", "def f : Nat := 1", 0, "x"
        ).to_dict()
        del old["imports"]

        assert FoundationEntry.from_dict(old).imports == ()

    def test_a_rollback_keeps_only_what_is_left(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.set_seed(SEED)
        foundation.add_artifact(artifact("a", importing("f", "Mathlib.Order.Basic")))
        foundation.add_artifact(artifact("b", ON_MATHLIB, generation=1))

        foundation.restore(foundation.snapshot_before(1))

        assert extra_imports_in(saved(foundation)) == [SUB, "Mathlib.Order.Basic"]


class TestSociety:
    """The paths that rebuild the foundation write what is left's imports."""

    def _society(self, tmp_path: Path, outcomes: list[bool]) -> Society:
        return Society(
            n_agents=1,
            provider=MockProvider(ProviderConfig(api_key="k", model="m")),
            verifier=ProjectVerifier(FakeProject(outcomes)),
            foundation_path=tmp_path / "F.lean",
            seed_source=SEED,
        )

    @pytest.mark.asyncio
    async def test_a_build_writes_them_and_a_failed_one_drops_them(
        self, tmp_path: Path
    ) -> None:
        society = self._society(tmp_path, [True, True, False])
        await society.reset_foundation()
        society.foundation.add_artifact(artifact("a", ON_MATHLIB))
        assert await society._write_and_build_foundation() is True
        assert extra_imports_in(society.foundation.path.read_text()) == [SUB, CAT]

        society.foundation.add_artifact(
            artifact("b", importing("g", "Mathlib.Order.Basic"))
        )
        assert await society._write_and_build_foundation() is False

        assert extra_imports_in(society.foundation.path.read_text()) == [SUB, CAT]

    @pytest.mark.asyncio
    async def test_a_reset_to_the_seed_keeps_only_the_seeds(
        self, tmp_path: Path
    ) -> None:
        society = self._society(tmp_path, [])
        await society.reset_foundation()
        society.foundation.add_artifact(artifact("a", ON_MATHLIB))
        await society._write_and_build_foundation()

        await society._reset_foundation_to_seed(0)

        assert extra_imports_in(society.foundation.path.read_text()) == [SUB]

    @pytest.mark.asyncio
    async def test_a_resumed_run_writes_them(self, tmp_path: Path) -> None:
        config = ProviderConfig(api_key="k", model="m")
        society = Society(
            n_agents=1,
            provider=MockProvider(config),
            verifier=MockLeanVerifier(),
            seed_source=SEED,
        )
        await society.reset_foundation()
        society.foundation.add_artifact(artifact("a", ON_MATHLIB))
        await society.persist_foundation()
        society.current_generation = 1
        society.save(tmp_path / "run")
        lms.society.DEFAULT_FOUNDATION_PATH.write_text("")
        lms.society.DEFAULT_FOUNDATION_PATH.with_suffix(".json").unlink()

        resumed = Society.load(
            tmp_path / "run", MockProvider(config), MockLeanVerifier()
        )
        assert await resumed.restore_foundation() is True

        assert extra_imports_in(resumed.foundation.path.read_text()) == [SUB, CAT]


class FakeProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self._output = (stdout.encode(), stderr.encode())

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._output


class TestBuildFailureLog:
    #: Lake's shape: Lean's errors on stdout, its summary alone on stderr.
    STDOUT = (
        "✖ [2/3] Building LMS.Foundation\n"
        "trace: .> lean ./LMS/Foundation.lean\n"
        "error: ./LMS/Foundation.lean:20:8: unknown namespace 'CategoryTheory'\n"
        "error: ./LMS/Foundation.lean:23:4: type mismatch\n  h\nhas type\n  Nat : Type\n"
        "warning: ./LMS/Foundation.lean:30:0: declaration uses 'sorry'\n"
        "error: Lean exited with code 1\nSome required builds logged failures:\n"
    )
    STDERR = "error: build failed\n"

    def test_the_positioned_errors_and_their_messages_are_picked(self) -> None:
        assert lean_error_lines(self.STDOUT + self.STDERR) == [
            "error: ./LMS/Foundation.lean:20:8: unknown namespace 'CategoryTheory'",
            "error: ./LMS/Foundation.lean:23:4: type mismatch",
            "h",
            "has type",
            "Nat : Type",
        ]

    def test_leans_own_format_counts(self) -> None:
        line = "LMS/F.lean:3:4: error: unknown identifier 'x'"

        assert lean_error_lines(line) == [line]

    def test_without_positions_other_error_lines_stand_in(self) -> None:
        assert lean_error_lines("error: unknown package 'Foo'\n" + self.STDERR) == [
            "error: unknown package 'Foo'"
        ]

    def test_the_output_is_bounded(self) -> None:
        lines = lean_error_lines(
            "".join(f"error: F.lean:{i}:0: {'x' * 999}\n" for i in range(99))
        )

        assert len(lines) == 21 and lines[-1] == "... (79 more lines)"
        assert max(len(line) for line in lines) <= 300

    @pytest.mark.parametrize("returncode", [0, 1])
    def test_a_failed_build_prints_them(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        returncode: int,
    ) -> None:
        async def fake_exec(*args: object, **kwargs: object) -> FakeProcess:
            return FakeProcess(returncode, self.STDOUT, self.STDERR)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        assert asyncio.run(LeanProject(tmp_path).build()) is (returncode == 0)

        out = capsys.readouterr().out
        assert ("unknown namespace 'CategoryTheory'" in out) is (returncode == 1)


class TestAgentContext:
    def test_the_warning_shows_for_a_seed_category(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.set_seed(load_seed())

        assert WARNING in foundation.get_context_for_agent()

    def test_the_warning_goes_once_mathlibs_category_is_imported(
        self, tmp_path: Path
    ) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.set_seed(load_seed())
        foundation.add_artifact(artifact("a", importing("f", CAT)))

        assert WARNING not in foundation.get_context_for_agent()

    def test_a_foundation_on_mathlib_lists_its_imports(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.add_artifact(artifact("a", ON_MATHLIB))

        context = foundation.get_context_for_agent()

        assert WARNING not in context
        assert "`import LMS.Foundation` brings these imports with it" in context
        assert all(f"    import {m}" in context for m in (*HEADER_IMPORTS, CAT))

    def test_a_seed_of_imports_alone_is_not_empty(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "F.lean")
        foundation.set_seed(f"import {CAT}\n")

        context = foundation.get_context_for_agent()

        assert "FOUNDATION: EMPTY" not in context
        assert f"    import {CAT}" in context
