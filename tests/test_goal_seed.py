"""A goal file names the foundation it starts from.

`Goal.seed` picks the generation-0 axiom layer, and a goal setting `seed=""`
starts bare. Only in-code goals could say so: `Goal.load` dropped the key, so
every `goals/*.json` goal started from the shipped from-scratch `Category`
class. The kernel tracks also tell agents to build on Mathlib, and in batch q1
52 of 73 artifacts on track B imported both. The two `Category` classes then
collided, and 24 artifacts failed with `failed to synthesize
CategoryTheory.Category`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lms.config import ProviderConfig
from lms.goals import Goal, StacksDefinition, get_goal
from lms.lean.mock import MockLeanVerifier
from lms.seed import load_seed
from lms.society import Society
from tests.test_society import MockProvider

GOALS = Path(__file__).resolve().parent.parent / "goals"
KERNEL_GOALS = ("stacks-kernel-track-a", "stacks-kernel-track-b")


def _goal(**fields: object) -> Goal:
    return Goal(
        name="g",
        description="d",
        source="s",
        definitions=[StacksDefinition(tag="0001", section="1", name="n", content="c")],
        **fields,  # type: ignore[arg-type]
    )


def _society(goal: Goal) -> Society:
    provider = MockProvider(ProviderConfig(api_key="k", model="m"))
    return Society(
        n_agents=1, provider=provider, verifier=MockLeanVerifier(), goal=goal
    )


@pytest.mark.parametrize("seed", ["", "category"])
def test_a_seed_survives_save_and_load(tmp_path: Path, seed: str) -> None:
    path = tmp_path / "goal.json"
    _goal(seed=seed).save(path)
    assert Goal.load(path).seed == seed


def test_seeded_tags_survive_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "goal.json"
    _goal(seed="category", seeded_tags=["0001"]).save(path)
    assert Goal.load(path).seeded_tags == ["0001"]


def test_a_goal_file_without_a_seed_still_gets_the_default(tmp_path: Path) -> None:
    """Older goal files, and checkpoints written before the key existed."""
    path = tmp_path / "goal.json"
    _goal().save(path)
    assert "seed" not in json.loads(path.read_text())
    loaded = Goal.load(path)
    assert loaded.seed is None
    assert loaded.seeded_tags == []
    assert _society(loaded).seed_source == load_seed()


@pytest.mark.parametrize("name", KERNEL_GOALS)
def test_the_kernel_goals_start_bare(name: str) -> None:
    goal = get_goal(name)
    assert goal.seed == ""
    assert _society(goal).seed_source == ""


def test_the_category_chapter_keeps_its_from_scratch_seed() -> None:
    """It forbids Mathlib.CategoryTheory, so the seed is the only Category."""
    goal = get_goal("stacks-ch4-phase1")
    assert goal.seed == "category"
    assert "class Category" in _society(goal).seed_source


@pytest.mark.parametrize("name", KERNEL_GOALS)
def test_the_extractor_writes_the_bare_seed(name: str) -> None:
    """Regenerating the files from the Stacks sources must not drop the key."""
    from scripts.goals.extract_stacks_goal import build_goal

    built = build_goal(name="n", description="d", source="s", picks=[], preamble="p")
    assert built["seed"] == ""
    on_disk = json.loads((GOALS / f"{name.replace('-', '_')}.json").read_text())
    assert on_disk["seed"] == built["seed"]
