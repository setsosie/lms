"""Agents must be shown the *shape* of a foundation entry (26Q3-HARN-11).

Found on the box 2026-08-10 in `experiments/shakedown_3x3_d`, the first run to
produce cross-generational reuse. Gen 2 cited `definition-Category-d3a579da`,
the import elaborated, and the attempt then died on API shape:

    error: type expected, got (Category : Type v → Type (max v (u + 1)))

The agent used `Category` as a bare type. The foundation's `Category` is
indexed by an object type. Not Lean 3 syntax, not a missing Mathlib -- the
agent could not see the shape of the thing it was importing.
"""

from __future__ import annotations

from pathlib import Path

from lms.artifacts import Artifact, ArtifactType
from lms.foundation import FoundationFile
from lms.lean.interface import VerificationStatus
from lms.providers.base import ProviderConfig
from lms.society import Society
from tests.test_society import MockProvider

CATEGORY_SRC = (
    "structure Category (obj : Type u) where\n"
    "  Hom : obj → obj → Type v\n"
    "  id : (X : obj) → Hom X X\n"
    "  comp : {X Y Z : obj} → Hom X Y → Hom Y Z → Hom X Z\n"
    "  id_l : True\n"
    "  id_r : True\n"
    "  assoc : True"
)


def _foundation_with_category(path: Path) -> FoundationFile:
    foundation = FoundationFile(path)
    foundation.add_artifact(
        Artifact(
            id="definition-Category-d3a579da",
            type=ArtifactType.DEFINITION,
            natural_language="the Category structure",
            lean_code=CATEGORY_SRC,
            status=VerificationStatus.VERIFIED_LEAN,
            created_by="agent-0",
            generation=0,
        )
    )
    return foundation


# --- What the per-agent context renders -------------------------------------


def test_declaration_header_is_rendered_exactly_once(tmp_path: Path) -> None:
    """`signature` already carries `<type> <name>`; prepending them repeats it."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    assert "structure Categorystructure Category" not in context
    assert context.count("structure Category") == 1


def test_declaration_header_reproduces_the_source_line(tmp_path: Path) -> None:
    """The parameter is why the agent failed -- it must survive verbatim."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    assert "structure Category (obj : Type u) where" in context


def test_structure_fields_carry_their_types(tmp_path: Path) -> None:
    """A bare `Hom` does not tell an agent how to apply it."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    assert "Hom : obj → obj → Type v" in context
    assert "id : (X : obj) → Hom X X" in context


def test_no_structure_field_is_silently_dropped(tmp_path: Path) -> None:
    """`field_match[:5]` amputates the 6th field with no marker."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    for field in ("Hom", "id", "comp", "id_l", "id_r", "assoc"):
        assert field in context, f"field {field!r} never reached the agent"


# --- What committee mode renders (strictly weaker today) --------------------


def _society_with_category(tmp_path: Path) -> Society:
    society = Society(
        n_agents=1,
        provider=MockProvider(ProviderConfig(api_key="test", model="test")),
        foundation_path=tmp_path / "LMS" / "Foundation.lean",
    )
    society.foundation = _foundation_with_category(tmp_path / "LMS" / "Foundation.lean")
    return society


def test_committee_summary_survives_a_nonempty_foundation(tmp_path: Path) -> None:
    """`entries.values()` on a list, `entry.tag` on a tag-less dataclass."""
    summary = _society_with_category(tmp_path)._get_foundation_summary()

    assert "Category" in summary


def test_committee_summary_shows_shape_not_just_names(tmp_path: Path) -> None:
    """A work committee is told less than the agent that already failed."""
    summary = _society_with_category(tmp_path)._get_foundation_summary()

    assert "(obj : Type u)" in summary
