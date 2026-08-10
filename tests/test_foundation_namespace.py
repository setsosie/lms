"""Foundation names must be brought into scope, not just imported (26Q3-HARN-10).

Found on the box 2026-08-10 in `experiments/shakedown_3x3_c`, the first run
where the foundation actually reached the next generation (26Q3-HARN-09). All
five generation-1 and generation-2 artifacts wrote `import LMS.Foundation` and
all five failed with:

    error(lean.unknownIdentifier): Unknown identifier `Category`

Entries are written inside `namespace LMS.Foundation`, so the verified
definition is `LMS.Foundation.Category`. The module resolved; the name never
did. This had been true in every earlier run too -- `autoImplicit` silently
auto-bound the unresolved name and the error surfaced somewhere else as a type
mismatch, so it read as bad mathematics.
"""

from __future__ import annotations

from pathlib import Path

from lms.artifacts import Artifact, ArtifactType
from lms.foundation import FoundationFile
from lms.lean.interface import VerificationStatus
from lms.prompts import get_prompt


def _foundation_with_category(tmp_path: Path) -> FoundationFile:
    foundation = FoundationFile(tmp_path / "Foundation.lean")
    foundation.add_artifact(
        Artifact(
            id="definition-Category-2cbb71fd",
            type=ArtifactType.DEFINITION,
            natural_language="the Category structure",
            lean_code=(
                "structure Category (obj : Type u) where\n"
                "  Hom : obj → obj → Type v\n"
                "  id : (x : obj) → Hom x x"
            ),
            status=VerificationStatus.VERIFIED_LEAN,
            created_by="agent-0",
            generation=0,
        )
    )
    return foundation


# --- The preamble agents are handed ----------------------------------------


def test_preamble_opens_the_namespace_as_well_as_importing(tmp_path: Path) -> None:
    foundation = _foundation_with_category(tmp_path)

    preamble = foundation.get_preamble()

    assert "import LMS.Foundation" in preamble
    assert "open LMS.Foundation" in preamble


def test_agent_context_tells_agents_to_open_it(tmp_path: Path) -> None:
    """The regression. Context said only "import", and every artifact failed."""
    foundation = _foundation_with_category(tmp_path)

    context = foundation.get_context_for_agent()

    assert "open LMS.Foundation" in context, (
        "importing alone leaves every entry unreachable by its bare name"
    )


def test_agent_context_explains_why_the_open_is_needed(tmp_path: Path) -> None:
    """An instruction without its reason gets dropped under pressure."""
    foundation = _foundation_with_category(tmp_path)

    context = foundation.get_context_for_agent()

    assert "namespace" in context
    assert "unknown identifier" in context.lower()


def test_namespace_constant_matches_the_written_file(tmp_path: Path) -> None:
    """`NAMESPACE` and FOUNDATION_HEADER must not drift apart."""
    foundation = _foundation_with_category(tmp_path)
    foundation.save()

    written = foundation.path.read_text()

    assert f"namespace {FoundationFile.NAMESPACE}" in written


# --- The goal prompt, which contradicted the context ------------------------


def test_goal_prompt_no_longer_claims_the_foundation_is_auto_imported() -> None:
    """v2.5 said "(imported automatically)". Nothing imports it for the agent."""
    content = get_prompt("agent_system_goal").content

    assert "imported automatically" not in content
    assert "open LMS.Foundation" in content


def test_goal_prompt_example_is_valid_lean() -> None:
    """The "RIGHT ... WILL SUCCEED" example used `def X ... where`.

    That is not valid Lean for a declaration with fields -- `structure` is. An
    example that cannot compile is worse than no example.
    """
    content = get_prompt("agent_system_goal").content

    assert "def Cone" not in content
    assert "structure Cone" in content


def test_goal_prompt_version_was_bumped() -> None:
    """Prompt versions are recorded per run in metadata.json.

    Editing content without bumping the version makes two runs that record the
    same version incomparable.
    """
    assert get_prompt("agent_system_goal").version == "2.6.0"
