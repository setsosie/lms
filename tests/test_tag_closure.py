"""A verified artifact closes its tag only when it declares that tag's statement.

The match is on words: the tag name's words (kind word and section title
dropped) against the words of the names the artifact declares.
"""

from pathlib import Path
from unittest import mock

import pytest

from lms.agent import IterativeResponse
from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.dependency import TaskStatus
from lms.goals import Goal, StacksDefinition
from lms.society import Society, tag_name_matches
from tests.test_society import (
    GROUP_SESSION_RESPONSES,
    MockProvider,
    ProviderConfig,
    StubLeanVerifier,
    _committee_society,
)

FIBRED_GROUPOIDS = (
    "Definition: definition fibred groupoids (Categories fibred in groupoids)"
)

CASES = [
    # Batch q1, r6: both closures were made by the wrong statement.
    (
        "Definition: Functor",
        "structure CategoryDef (obj : Type u) where\n  hom : obj → obj → Type v",
        False,
    ),
    (
        "Definition: Natural Transformation",
        "structure Functor (C D : Type) where\n  obj : C → D",
        False,
    ),
    # The same tags, closed by their own statements.
    (
        "Definition: Functor",
        "structure Functor (C D : Type) where\n  obj : C → D",
        True,
    ),
    (
        "Definition: Natural Transformation",
        "structure NatTrans (F G : Functor C D) where\n  app : ∀ X, F.obj X ⟶ G.obj X",
        True,
    ),
    # Kernel tracks name tags by Stacks label, with a section title in parentheses.
    (
        "Definition: definition categories over C (Categories over categories)",
        "structure CatOverC (C : Type u) where\n  S : Type v",
        True,
    ),
    (
        "Lemma: lemma composition cartesian (Fibred categories)",
        "theorem comp_cartesian {X Y Z : S} : True := trivial",
        True,
    ),
    # Batch q4's 003T artifact names its statement.
    (
        FIBRED_GROUPOIDS,
        "structure CategoryFibredInGroupoids (C : Type u) [Category C] where\n  S : Type v",
        True,
    ),
    # The section title is not part of the statement's name: "Categories" there must not match.
    (
        FIBRED_GROUPOIDS,
        "structure CategoryOver (C : Type u) where\n  S : Type v",
        False,
    ),
]


@pytest.mark.parametrize(("tag_name", "lean_code", "closes"), CASES)
def test_rule_a_name_match(tag_name: str, lean_code: str, closes: bool) -> None:
    assert tag_name_matches(tag_name, lean_code) is closes


def test_short_words_never_match() -> None:
    # `is` is a prefix of `isomorphism`, but a match needs at least 3 letters.
    code = "def IsZero (x : Nat) : Prop := x = 0"
    assert tag_name_matches("Definition: Isomorphism", code) is False


def test_milestone_prefix_is_dropped() -> None:
    code = "theorem yoneda_embedding_faithful : True := trivial"
    assert tag_name_matches("MILESTONE: Yoneda lemma", code) is True


# --- Where a verified artifact closes its tag --------------------------------


def _committee(tmp_path: Path, tag_name: str) -> Society:
    """The group writes `structure Category`; the tag is named `tag_name`."""
    society = _committee_society(tmp_path, GROUP_SESSION_RESPONSES, StubLeanVerifier())
    society.use_peer_review = False
    assert society.goal is not None
    society.goal.definitions[0].name = tag_name
    return society


@pytest.mark.asyncio
async def test_committee_artifact_closes_its_own_tag(tmp_path: Path) -> None:
    society = _committee(tmp_path, "Definition: Category")
    await society.run_generation(0)
    assert society.dependency_graph is not None
    assert society.dependency_graph.nodes["T1"].status == TaskStatus.DONE
    assert society.goal is not None and society.goal.definitions[0].formalized


@pytest.mark.asyncio
async def test_committee_artifact_on_another_statement_reopens_the_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    society = _committee(tmp_path, "Definition: Functor")
    result = await society.run_generation(0)
    assert result.artifacts_verified == 1
    assert society.dependency_graph is not None
    assert society.dependency_graph.nodes["T1"].status == TaskStatus.AVAILABLE
    assert society.goal is not None and not society.goal.definitions[0].formalized
    # It compiled, so it is still promoted.
    (artifact,) = society.library.all()
    assert society.foundation.holds(artifact.id)
    out = capsys.readouterr().out
    assert (
        f'[goal] T1 not closed by {artifact.id}: declares Category, none matches "Definition: Functor"'
        in out
    )


def _one_tag_society(
    tmp_path: Path, tag_name: str, responses: list[str] | None = None
) -> Society:
    goal = Goal(
        name="Test Goal",
        description="Test",
        source="Test",
        definitions=[
            StacksDefinition(tag="T1", section="1.1", name=tag_name, content="...")
        ],
    )
    society = Society(
        n_agents=1,
        provider=MockProvider(
            ProviderConfig(api_key="test", model="test"), responses=responses or []
        ),
        verifier=StubLeanVerifier(),
        goal=goal,
        foundation_path=tmp_path / "LMS" / "Foundation.lean",
    )
    return society


@pytest.mark.parametrize(
    ("tag_name", "closes"),
    [("Definition: Category", True), ("Definition: Functor", False)],
)
@pytest.mark.asyncio
async def test_iterative_artifact_closes_only_its_own_tag(
    tmp_path: Path, tag_name: str, closes: bool
) -> None:
    society = _one_tag_society(tmp_path, tag_name)
    society.iterative_mode = True
    made = Artifact(
        id="cat-1",
        type=ArtifactType.DEFINITION,
        natural_language="Category",
        created_by="agent-0-mock",
        generation=0,
        lean_code="structure Category where\n  Obj : Type",
        stacks_tag="T1",
        status=VerificationStatus.VERIFIED_LEAN,
    )
    response = IterativeResponse(
        attempts=[], final_artifact=made, success=True, writeup=""
    )
    with mock.patch.object(
        society.agents[0], "propose_iterative", mock.AsyncMock(return_value=response)
    ):
        await society.run_generation(0)
    assert society.goal is not None
    assert society.goal.definitions[0].formalized is closes


FLAT_RESPONSE = """<artifact>
type: definition
name: Category
stacks_tag: T1
description: Category structure
lean: |
  structure Category where
    Obj : Type
references: []
</artifact>"""


@pytest.mark.parametrize(
    ("tag_name", "closes"),
    [("Definition: Category", True), ("Definition: Functor", False)],
)
@pytest.mark.asyncio
async def test_flat_artifact_closes_only_its_own_tag(
    tmp_path: Path, tag_name: str, closes: bool
) -> None:
    society = _one_tag_society(tmp_path, tag_name, responses=[FLAT_RESPONSE])
    society.use_peer_review = False
    result = await society.run_generation(0)
    assert result.artifacts_verified == 1
    assert society.goal is not None
    assert society.goal.definitions[0].formalized is closes
