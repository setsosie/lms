"""An artifact's `open` and `variable` lines reach the foundation with its entries.

The merge cut each artifact at its declarations, so whatever came before the
first one never reached `Foundation.lean`. Batch q3's track B verified 25
artifacts and kept none of them: every one opened with `open CategoryTheory`,
12 also declared the `variable` binders their definitions use, and the merged
module failed on `Unknown identifier S`, and on `⥤` and `≫` outside the
namespace that scopes that notation.

Each artifact's lines before its first declaration are written in a `section`
of its own, so one artifact's variables can't reach another's entries. The
core-Lean compile checks are in `tests/test_lean_real.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.foundation import FoundationEntry, FoundationFile

FIXTURES = (
    Path(__file__).parent / "fixtures" / "foundation_preamble" / "q3_verified.json"
)
Q3 = {row["id"]: row for row in json.loads(FIXTURES.read_text())["artifacts"]}


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


def saved(foundation: FoundationFile) -> str:
    foundation.save()
    return foundation.path.read_text()


def block(text: str, artifact_id: str) -> list[str]:
    """The lines written for one artifact: from its `-- From` line to the next."""
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f"-- From {artifact_id} ")
    )
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith(("-- From ", "end LMS.Foundation"))
        ),
        len(lines),
    )
    return [line.strip() for line in lines[start + 1 : end] if line.strip()]


#: The q3 artifact whose rollback opened the run: three `variable` lines and a
#: scoped `open`, all above its only declaration.
FIRST = "IsCartesian-bf42ea63"


class TestThePreambleIsWritten:
    def test_open_and_variable_lines_come_before_the_entry(
        self, tmp_path: Path
    ) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        assert foundation.add_artifact(artifact(FIRST, Q3[FIRST]["lean_code"]))

        lines = block(saved(foundation), FIRST)

        assert lines[0] == "section"
        assert lines[-1] == "end"
        order = [
            "open CategoryTheory",
            "open scoped CategoryTheory",
            "variable {C : Type u} [Category C]",
            "variable {S : Type v} [Category S]",
            "variable (p : S ⥤ C)",
        ]
        positions = [lines.index(line) for line in order]
        assert positions == sorted(positions)
        assert positions[-1] < next(
            i for i, line in enumerate(lines) if line.startswith("def IsCartesian")
        )

    def test_each_artifact_scopes_its_own_variables(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(
            artifact("a", "variable (n : Nat)\n\ndef addN (m : Nat) : Nat := n + m")
        )
        foundation.add_artifact(
            artifact("b", "variable (n : String)\n\ndef lenN : Nat := n.length")
        )

        text = saved(foundation)

        assert block(text, "a") == [
            "section",
            "variable (n : Nat)",
            "def addN (m : Nat) : Nat := n + m",
            "end",
        ]
        assert block(text, "b") == [
            "section",
            "variable (n : String)",
            "def lenN : Nat := n.length",
            "end",
        ]

    def test_an_artifact_with_nothing_above_its_entries_is_written_as_before(
        self, tmp_path: Path
    ) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(
            artifact("plain", "-- a note\n\ndef plainDef : Nat := 1")
        )

        assert block(saved(foundation), "plain") == ["def plainDef : Nat := 1"]

    def test_lines_between_entries_stay_with_them(self, tmp_path: Path) -> None:
        code = "open Nat\n\ndef one : Nat := succ zero\n\nvariable (k : Nat)\n\ndef addK : Nat := k + one"
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", code))

        assert block(saved(foundation), "a") == [
            "section",
            "open Nat",
            "def one : Nat := succ zero",
            "variable (k : Nat)",
            "def addK : Nat := k + one",
            "end",
        ]

    def test_a_block_comment_above_the_entries_is_not_copied(
        self, tmp_path: Path
    ) -> None:
        code = "/-! Module notes.\nnamespace is a word here -/\nopen Nat /- why\nthis -/\n\ndef one : Nat := succ zero"
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", code))

        assert block(saved(foundation), "a") == [
            "section",
            "open Nat",
            "def one : Nat := succ zero",
            "end",
        ]


class TestScopes:
    def test_a_namespace_and_its_end_go_and_a_section_and_its_end_stay(
        self, tmp_path: Path
    ) -> None:
        code = (
            "namespace Outer\nsection Inner\nvariable (k : Nat)\n\n"
            "def addK (m : Nat) : Nat := k + m\n\nend Inner\nend Outer"
        )
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", code))

        assert block(saved(foundation), "a") == [
            "section",
            "section Inner",
            "variable (k : Nat)",
            "def addK (m : Nat) : Nat := k + m",
            "end Inner",
            "end",
        ]

    def test_a_section_the_artifact_leaves_open_is_closed(self, tmp_path: Path) -> None:
        code = "noncomputable section\n\nsection Named\n\ndef three : Nat := 3"
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", code))

        assert block(saved(foundation), "a") == [
            "section",
            "noncomputable section",
            "section Named",
            "def three : Nat := 3",
            "end Named",
            "end",
            "end",
        ]

    def test_a_comment_line_that_reads_like_a_command_is_kept(
        self, tmp_path: Path
    ) -> None:
        code = (
            "namespace N\n\n/-- Counts up.\nend of the story\nnamespace trick -/\n"
            "def counter : Nat := 1\n\nend N"
        )
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("a", code))

        lines = block(saved(foundation), "a")

        assert "end of the story" in lines
        assert "namespace trick -/" in lines
        assert "end N" not in lines


def checked_lines(code: str) -> list[str]:
    """The artifact's lines that must reach the foundation verbatim.

    Leaves out what the merge drops on purpose: imports (hoisted to the
    header), `universe` lines (bound by the header or the artifact's section)
    and namespaces with the `end` that closes them.
    """
    masked = FoundationFile.BLOCK_COMMENT_PATTERN.sub(
        lambda m: re.sub(r"[^\n]", " ", m.group()), code
    )
    namespaces: list[bool] = []
    kept = []
    for line, bare in zip(code.splitlines(), masked.splitlines(), strict=True):
        s = bare.strip()
        if not s or s.startswith(("--", "import ", "universe ")):
            continue
        if s.startswith("namespace "):
            namespaces.append(True)
            continue
        if re.match(r"(noncomputable\s+)?section\b", s):
            namespaces.append(False)
        elif re.match(r"end\b", s):
            if namespaces.pop():
                continue
        kept.append(line.strip())
    return kept


class TestQ3Artifacts:
    @pytest.mark.parametrize("artifact_id", sorted(Q3))
    def test_no_line_of_a_verified_artifact_is_lost(
        self, tmp_path: Path, artifact_id: str
    ) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        assert foundation.add_artifact(
            artifact(artifact_id, Q3[artifact_id]["lean_code"])
        )

        have = set(block(saved(foundation), artifact_id))

        assert [
            line
            for line in checked_lines(Q3[artifact_id]["lean_code"])
            if line not in have
        ] == []

    def test_every_track_b_artifact_keeps_its_open(self, tmp_path: Path) -> None:
        track_b = [aid for aid, row in Q3.items() if row["run"] != "tracka"]
        assert len(track_b) == 25
        for artifact_id in track_b:
            foundation = FoundationFile(tmp_path / f"{artifact_id}.lean")
            foundation.add_artifact(artifact(artifact_id, Q3[artifact_id]["lean_code"]))
            assert "open CategoryTheory" in block(saved(foundation), artifact_id), (
                artifact_id
            )


class TestTheContextTravelsWithTheEntries:
    def test_it_round_trips_through_foundation_json(self, tmp_path: Path) -> None:
        path = tmp_path / "Foundation.lean"
        foundation = FoundationFile(path)
        foundation.add_artifact(artifact(FIRST, Q3[FIRST]["lean_code"]))
        foundation.save()

        loaded = FoundationFile.load(path)

        assert loaded.entries[0].context == foundation.entries[0].context
        assert "variable (p : S ⥤ C)" in [
            line.strip() for line in loaded.entries[0].context
        ]
        assert saved(loaded) == saved(foundation)

    def test_an_entry_saved_before_this_change_loads_without_one(self) -> None:
        old = FoundationEntry(
            artifact_id="a",
            name="f",
            entry_type="def",
            signature="def f : Nat",
            lean_code="def f : Nat := 1",
            generation=0,
            author="agent-0",
        ).to_dict()
        del old["context"]

        assert FoundationEntry.from_dict(old).context == ()

    def test_a_rollback_takes_it_away(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(
            artifact("a", "variable (n : Nat)\n\ndef addN (m : Nat) : Nat := n + m")
        )
        good = foundation.snapshot()
        foundation.add_artifact(
            artifact("b", "variable (s : String)\n\ndef lenS : Nat := s.length")
        )

        foundation.restore(good)
        text = saved(foundation)

        assert "variable (n : Nat)" in text
        assert "variable (s : String)" not in text


class TestAgentsSeeTheBinders:
    def test_the_variable_lines_are_shown_above_the_declaration(
        self, tmp_path: Path
    ) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact(FIRST, Q3[FIRST]["lean_code"], generation=3))

        context = foundation.get_context_for_agent()

        assert context.index("variable (p : S ⥤ C)") < context.index("def IsCartesian")
        assert "leading arguments" in context

    def test_an_entry_without_them_gets_no_note(self, tmp_path: Path) -> None:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        foundation.add_artifact(artifact("plain", "def plainDef : Nat := 1"))

        assert "leading arguments" not in foundation.get_context_for_agent()
