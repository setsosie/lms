"""Verified work must reach the next generation (26Q3-HARN-09).

Found on the box 2026-08-10. A 3-generation run verified `Category` in
generation 1; generations 2 and 3 cited it correctly, wrote
`import LMS.Foundation`, and got a module that predated the whole run --
because `foundation.save()` only ran inside `Society.save()`, which fires
every `checkpoint_interval` generations (default 10).

The failure was invisible for two compounding reasons: an import that resolves
against a stale module looks exactly like one that resolves against a current
one, and with `autoImplicit` on the missing name became an auto-bound variable
rather than an error. What the run actually reported was

    error: Function expected at Category, but this term has type ?m.1

which reads as broken mathematics rather than a harness bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType
from lms.config import ProviderConfig
from lms.foundation import FoundationFile
from lms.lean.mock import MockLeanVerifier
from lms.lean.real import RealLeanVerifier
from lms.society import Society
from tests.test_society import MockProvider, StubLeanVerifier


class RecordingProject:
    """Stands in for `LeanProject`, recording rebuild calls."""

    def __init__(self) -> None:
        self.rebuild_calls = 0

    async def rebuild_changed_sources(self) -> bool:
        self.rebuild_calls += 1
        return True


def _verified_artifact(name: str = "Category", generation: int = 0) -> Artifact:
    from lms.lean.interface import VerificationStatus

    return Artifact(
        id=f"definition-{name}-abcd1234",
        type=ArtifactType.DEFINITION,
        natural_language=f"the {name} structure",
        lean_code=f"structure {name} (obj : Type u) where\n  Hom : obj → obj → Type v",
        status=VerificationStatus.VERIFIED_LEAN,
        created_by="agent-0",
        generation=generation,
    )


ARTIFACT_RESPONSE = """<artifact>
type: definition
name: Category
description: the Category structure
lean: |
  structure Category (obj : Type u) where
    Hom : obj → obj → Type v
references: []
</artifact>"""


def _society(tmp_path: Path, verifier, responses: list[str] | None = None) -> Society:
    config = ProviderConfig(api_key="test", model="test")
    return Society(
        n_agents=1,
        provider=MockProvider(config, responses=responses),
        verifier=verifier,
        foundation_path=tmp_path / "Foundation.lean",
    )


# --- Fix 1: the foundation reaches disk, and gets recompiled ---------------


@pytest.mark.asyncio
async def test_generation_writes_foundation_to_disk(tmp_path: Path) -> None:
    """The regression. One generation, no checkpoint, file must exist."""
    society = _society(tmp_path, StubLeanVerifier())
    society.foundation.add_artifact(_verified_artifact())

    assert not society.foundation.path.exists(), "precondition: nothing on disk"

    await society.persist_foundation()

    assert society.foundation.path.exists()
    assert "Category" in society.foundation.path.read_text()


@pytest.mark.asyncio
async def test_persist_rebuilds_so_the_olean_is_not_stale(tmp_path: Path) -> None:
    """Saving without rebuilding leaves the old module importable.

    `ensure_built` only rebuilds when a *new* import name appears, and
    `LMS.Foundation`'s name never changes -- so nothing else in the system
    would ever recompile it.
    """
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]

    society = _society(tmp_path, verifier)
    society.foundation.add_artifact(_verified_artifact())

    await society.persist_foundation()

    assert project.rebuild_calls == 1


@pytest.mark.asyncio
async def test_persist_is_a_noop_with_an_empty_foundation(tmp_path: Path) -> None:
    """Nothing verified yet means no file and no build."""
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]

    society = _society(tmp_path, verifier)

    assert await society.persist_foundation() is True
    assert project.rebuild_calls == 0
    assert not society.foundation.path.exists()


@pytest.mark.asyncio
async def test_persist_survives_a_verifier_with_no_project(tmp_path: Path) -> None:
    """`MockLeanVerifier` has no Lean project to build against."""
    society = _society(tmp_path, MockLeanVerifier())
    society.foundation.add_artifact(_verified_artifact())

    assert await society.persist_foundation() is True
    assert society.foundation.path.exists()


@pytest.mark.asyncio
async def test_run_generation_makes_verified_work_importable(tmp_path: Path) -> None:
    """The actual regression, end to end through `run_generation`.

    Generation 0 verifies an artifact; before the next generation reads it, the
    foundation must be on disk and recompiled. This is the test that fails if
    the `persist_foundation` call is dropped from `run_generation` -- the
    negative test below would still pass vacuously.
    """
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]

    society = _society(tmp_path, verifier, responses=[ARTIFACT_RESPONSE])

    result = await society.run_generation(0)

    assert result.artifacts_verified == 1, "precondition: the artifact verified"
    assert society.foundation.path.exists(), "foundation never reached disk"
    assert "Category" in society.foundation.path.read_text()
    assert project.rebuild_calls == 1, "stale .olean would still be imported"


@pytest.mark.asyncio
async def test_run_generation_persists_only_when_something_verified(
    tmp_path: Path,
) -> None:
    """A generation that verifies nothing should not trigger a rebuild."""
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]

    society = _society(tmp_path, verifier)

    # MockProvider proposes nothing parseable, so nothing verifies.
    result = await society.run_generation(0)

    assert result.artifacts_verified == 0
    assert project.rebuild_calls == 0


# --- Fix 2: autoImplicit off, consistently ---------------------------------


def test_verifier_passes_strictness_flags_to_lean() -> None:
    """Without these a missing import becomes a metavariable, not an error."""
    assert "-DautoImplicit=false" in RealLeanVerifier.STRICTNESS_FLAGS
    assert "-DrelaxedAutoImplicit=false" in RealLeanVerifier.STRICTNESS_FLAGS


def test_foundation_header_matches_the_verifier_strictness() -> None:
    """An entry is verified standalone, then recompiled inside the library.

    If the two disagree on autoImplicit, code can pass verification and then
    fail the foundation build.
    """
    header = FoundationFile.FOUNDATION_HEADER
    assert "set_option autoImplicit false" in header
    assert "set_option relaxedAutoImplicit false" in header


# --- Fix 3: agents can see the shape of what they're told to reuse ---------


def test_agent_context_shows_the_declaration_signature(tmp_path: Path) -> None:
    """Regression: an agent wrote `C.Ob` against `structure Category (obj : ...)`.

    The context listed the name and a field list, so nothing revealed that the
    objects are a parameter rather than a field.
    """
    foundation = FoundationFile(tmp_path / "Foundation.lean")
    foundation.add_artifact(_verified_artifact())

    context = foundation.get_context_for_agent()

    assert "structure Category" in context
    assert "(obj : Type u)" in context, (
        "the parameter list must be visible or agents guess at the interface"
    )
