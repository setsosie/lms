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

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.config import Config, ProviderConfig
from lms.dependency import DependencyGraph, TaskStatus
from lms.foundation import FoundationFile, FoundationSnapshot
from lms.goals import Goal, StacksDefinition
from lms.lean.mock import MockLeanVerifier
from lms.lean.real import RealLeanVerifier
from lms.run import run_experiment
from lms.society import GenerationResult, Society
from tests.test_foundation_rollback import FakeProject, ProjectVerifier
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


@pytest.mark.asyncio
async def test_reset_writes_an_empty_but_importable_foundation(
    tmp_path: Path,
) -> None:
    """A run must start from an empty foundation that still imports.

    The foundation cannot live under the experiment directory -- `import
    LMS.Foundation` resolves through Lake's LEAN_PATH -- so it is shared
    mutable state across runs, and nothing used to clear it. Writing the empty
    module rather than deleting it keeps the import resolving from generation
    1; an empty module is valid, a missing one is not.
    """
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]

    society = _society(tmp_path, verifier)

    assert await society.reset_foundation() is True

    assert society.foundation.path.exists()
    text = society.foundation.path.read_text()
    assert "namespace LMS.Foundation" in text, "must still be a valid module"
    assert "-- From " not in text, "must carry no entries from a previous run"
    assert project.rebuild_calls == 1, "an unbuilt module is not importable"


@pytest.mark.asyncio
async def test_reset_discards_a_previous_runs_definitions(tmp_path: Path) -> None:
    """Regression: runs were not independent.

    The copy checked into the repo is the output of a December mock-verified
    run, so on a fresh clone agents imported a `Category` that Lean had never
    checked. On a used box they imported whatever the last run left behind.
    """
    stale = tmp_path / "Foundation.lean"
    stale.write_text(
        "-- From definition-Stale-00000000 (gen 0, ghost)\nstructure Stale where\n"
    )

    society = _society(tmp_path, StubLeanVerifier())

    await society.reset_foundation()

    assert "Stale" not in society.foundation.path.read_text()


@pytest.mark.asyncio
async def test_run_experiment_resets_before_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reset has to be wired into the actual entry point.

    `run_experiment` has no other test coverage, so without this the only
    guard on the call was grepping the function's source from a shell script.

    `DEFAULT_FOUNDATION_PATH` is redirected because `run_experiment` does not
    take a foundation path — left alone this test would rewrite the tracked
    corpus, which is the very problem the card is about (issue #19).
    """
    monkeypatch.setattr(
        "lms.society.DEFAULT_FOUNDATION_PATH", tmp_path / "Foundation.lean"
    )
    monkeypatch.setattr("lms.run.create_provider", lambda name, cfg: MockProvider(cfg))

    calls: list[str] = []

    async def _spy(self: Society) -> bool:
        calls.append("reset")
        return True

    monkeypatch.setattr(Society, "reset_foundation", _spy)

    await run_experiment(
        n_agents=1,
        n_generations=0,
        provider_name="anthropic",
        output_dir=tmp_path / "out",
        config=Config(anthropic=ProviderConfig(api_key="k", model="m")),
        verifier_type="mock",
    )

    assert calls == ["reset"], "a run must not inherit the previous run's foundation"


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


# --- Reset to the seed every generation: the R9 arm (ADR 0003) --------------
#
# C9 against R9 prices what the carried foundation is worth. R9 resets only
# `Foundation.lean`; the textbook, the goal's and graph's statuses and the
# artifact library carry in both arms, so those must survive the reset.

SEED = "class Category (obj : Type u) where\n  Hom : obj → obj → Type v\n"


def _goal() -> Goal:
    return Goal(
        name="t",
        description="d",
        source="s",
        seed="",
        definitions=[
            StacksDefinition(tag="T0", section="0", name="seeded", content="c"),
            StacksDefinition(tag="T1", section="1", name="one", content="c"),
            StacksDefinition(
                tag="T2", section="2", name="two", content="c", requires=["T1"]
            ),
            StacksDefinition(tag="T3", section="3", name="three", content="c"),
        ],
        seeded_tags=["T0"],
    )


def _r9_society(
    tmp_path: Path, verifier, *, reset: bool, goal: Goal | None = None
) -> Society:
    society = Society(
        n_agents=1,
        provider=MockProvider(ProviderConfig(api_key="test", model="test")),
        verifier=verifier,
        goal=goal,
        foundation_path=tmp_path / "Foundation.lean",
        seed_source=SEED,
    )
    society.reset_foundation_each_generation = reset
    if goal is not None:
        society.dependency_graph = DependencyGraph.from_goal(goal)
    return society


class _Generations:
    """A generation's work, reduced to one verified and promoted artifact.

    Generation `g` declares `names[g]` (default `Pointed<g>`) for tag `T<g+1>`.
    What the society looked like when each generation was dispatched is kept,
    because that is the moment the reset has to have happened by.
    """

    def __init__(self, society: Society, names: dict[int, str] | None = None) -> None:
        self.society = society
        self.names = names or {}
        self.state: dict[int, FoundationSnapshot] = {}
        self.written: dict[int, str] = {}
        self.refs: dict[int, set[str]] = {}
        self.builds: dict[int, int] = {}
        self.last_good: dict[int, FoundationSnapshot | None] = {}
        self.added: dict[int, bool] = {}
        society._run_generation_impl = self  # type: ignore[method-assign]

    async def __call__(self, generation: int) -> GenerationResult:
        society = self.society
        self.state[generation] = society.foundation.snapshot()
        self.written[generation] = society.foundation.path.read_text()
        self.last_good[generation] = society._last_good_foundation
        self.refs[generation] = society._derive_references(
            "def uses : Type := Pointed0"
        )
        project = getattr(society.verifier, "project", None)
        self.builds[generation] = getattr(project, "rebuild_calls", -1)

        name = self.names.get(generation, f"Pointed{generation}")
        made = Artifact(
            id=f"gen{generation}-{name}",
            type=ArtifactType.DEFINITION,
            natural_language=name,
            lean_code=f"structure {name} (obj : Type u) where\n  pt : obj",
            status=VerificationStatus.VERIFIED_LEAN,
            created_by="agent-0",
            generation=generation,
            stacks_tag=f"T{generation + 1}",
        )
        society.library.add(made)
        self.added[generation] = society.foundation.add_artifact(made)
        if society.goal is not None:
            society.goal.mark_formalized(f"T{generation + 1}", made.id)
        if society.dependency_graph is not None:
            society.dependency_graph.update_status(
                f"T{generation + 1}", TaskStatus.DONE, made.id
            )
        society.textbook.add(f"note {generation}", "agent-0", generation)
        society.current_generation = generation + 1
        return GenerationResult(
            generation=generation,
            artifacts_created=1,
            artifacts_verified=1,
            artifacts_referenced=0,
            fresh_creations=1,
        )


async def _run(society: Society, n: int, names: dict[int, str] | None = None):
    generations = _Generations(society, names)
    await society.reset_foundation()
    for generation in range(n):
        await society.run_generation(generation)
    return generations


def test_reset_flag_defaults_off(tmp_path: Path) -> None:
    society = _society(tmp_path, StubLeanVerifier())

    assert society.reset_foundation_each_generation is False


@pytest.mark.asyncio
async def test_with_reset_generation_1_starts_from_exactly_the_seed(
    tmp_path: Path,
) -> None:
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=True)

    seen = await _run(society, 2)

    assert seen.state[1] == seen.state[0], "generation 1 must start where 0 did"
    assert seen.state[1].entries == ()
    assert seen.state[1].definition_names == frozenset({"Category"})
    assert seen.state[1].artifact_ids == frozenset()
    assert "class Category" in seen.written[1]
    assert "Pointed0" not in seen.written[1], "Lean would still import it"


@pytest.mark.asyncio
async def test_with_reset_code_naming_generation_0_work_cites_nothing(
    tmp_path: Path,
) -> None:
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=True)

    seen = await _run(society, 2)

    assert seen.refs[1] == set()


@pytest.mark.asyncio
async def test_without_reset_generation_0_work_carries_into_generation_1(
    tmp_path: Path,
) -> None:
    """The regression: C9 and every other arm keep the foundation."""
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=False)

    seen = await _run(society, 2)

    assert [e.name for e in seen.state[1].entries] == ["Pointed0"]
    assert seen.refs[1] == {"gen0-Pointed0"}
    assert "structure Pointed0" in seen.written[1]


@pytest.mark.parametrize(("reset", "accepted"), [(True, True), (False, False)])
@pytest.mark.asyncio
async def test_a_name_from_generation_0_can_be_declared_again_only_after_a_reset(
    tmp_path: Path, reset: bool, accepted: bool
) -> None:
    """The reset must forget the names it removes, or the redefinition R9
    forces on every generation would be refused as a duplicate."""
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=reset)

    seen = await _run(society, 2, names={0: "Pointed", 1: "Pointed"})

    assert seen.added[0] is True
    assert seen.added[1] is accepted


@pytest.mark.asyncio
async def test_the_reset_rebuilds_before_the_generation_is_dispatched(
    tmp_path: Path,
) -> None:
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]
    society = _r9_society(tmp_path, verifier, reset=True)

    seen = await _run(society, 2)

    # run start, generation 0's promotion, then the reset
    assert seen.builds[1] == 3
    assert project.rebuild_calls == 4


@pytest.mark.parametrize("with_project", [True, False])
@pytest.mark.asyncio
async def test_the_seed_becomes_the_last_good_state(
    tmp_path: Path, with_project: bool
) -> None:
    """What a failed build in the generation rolls back to."""
    verifier = ProjectVerifier(FakeProject([])) if with_project else StubLeanVerifier()
    society = _r9_society(tmp_path, verifier, reset=True)

    seen = await _run(society, 2)

    assert seen.last_good[1] == seen.state[0]


@pytest.mark.asyncio
async def test_without_reset_there_is_no_extra_build(tmp_path: Path) -> None:
    verifier = StubLeanVerifier()
    project = RecordingProject()
    verifier.project = project  # type: ignore[attr-defined]
    society = _r9_society(tmp_path, verifier, reset=False)

    seen = await _run(society, 2)

    assert seen.builds[1] == 2
    assert project.rebuild_calls == 3


@pytest.mark.asyncio
async def test_a_seed_that_no_longer_builds_stops_the_run(tmp_path: Path) -> None:
    """Going on would let generation 1 import generation 0's compiled module."""
    project = FakeProject([True, True, False, False])
    society = _r9_society(tmp_path, ProjectVerifier(project), reset=True)

    with pytest.raises(RuntimeError, match="seed"):
        await _run(society, 2)

    assert society.foundation.entries == [], "the rollback went back to generation 0"
    assert "Pointed0" not in society.foundation.path.read_text()


@pytest.mark.asyncio
async def test_the_seed_banner_and_seeded_tags_happen_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    goal = _goal()
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=True, goal=goal)
    marked: list[list[str]] = []
    mark_seeded = goal.mark_seeded

    def spy() -> list[str]:
        marked.append(mark_seeded())
        return marked[-1]

    goal.mark_seeded = spy  # type: ignore[method-assign]

    await _run(society, 3)

    out = capsys.readouterr().out
    assert out.count("Seeded foundation") == 1
    assert out.count("Tags given by the seed") == 1
    assert marked == [["T0"]]
    assert out.count("[foundation] reset to seed") == 2


@pytest.mark.asyncio
async def test_the_reset_leaves_the_textbook_and_statuses_alone(
    tmp_path: Path,
) -> None:
    """ADR 0003: R9 differs from C9 in the foundation and nothing else.

    Checked after generation 1 has verified and promoted work too, which is
    when tags whose artifacts left the foundation are reconciled.
    """
    goal = _goal()
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=True, goal=goal)
    graph = society.dependency_graph
    assert graph is not None

    await _run(society, 2)

    t1 = next(d for d in goal.definitions if d.tag == "T1")
    assert t1.formalized is True
    assert t1.artifact_ids == ["gen0-Pointed0"]
    assert graph.nodes["T1"].status is TaskStatus.DONE
    assert graph.nodes["T2"].status is TaskStatus.DONE
    assert [e.content for e in society.textbook.entries] == ["note 0", "note 1"]
    assert "gen0-Pointed0" in society.library


@pytest.mark.asyncio
async def test_a_rolled_back_artifact_still_reopens_its_tag_with_reset(
    tmp_path: Path,
) -> None:
    """Only the reset's removals are exempt: a build that drops this
    generation's work reopens its tag exactly as in C9."""
    goal = _goal()
    # run start, generation 0 promotion, generation 1 reset, generation 1 fails
    project = FakeProject([True, True, True, False, True])
    society = _r9_society(tmp_path, ProjectVerifier(project), reset=True, goal=goal)

    await _run(society, 2)

    t2 = next(d for d in goal.definitions if d.tag == "T2")
    t1 = next(d for d in goal.definitions if d.tag == "T1")
    assert t2.formalized is False
    assert t1.formalized is True
    assert society.foundation.entries == [], "rolled back to the seed, not generation 0"
    assert "Pointed0" not in society.foundation.path.read_text()


@pytest.mark.asyncio
async def test_a_resumed_run_keeps_resetting_and_keeps_earlier_tags(
    tmp_path: Path,
) -> None:
    """The flag comes from the command line, not the checkpoint, so the
    seed state has to be recoverable from what the checkpoint does hold."""
    run_dir = tmp_path / "run"
    goal = _goal()
    society = _r9_society(tmp_path, StubLeanVerifier(), reset=True, goal=goal)
    await _run(society, 2)
    society.save(run_dir)
    goal.save(run_dir / "goal.json")

    resumed = Society.load(
        run_dir,
        MockProvider(ProviderConfig(api_key="test", model="test")),
        StubLeanVerifier(),
    )
    resumed.reset_foundation_each_generation = True
    assert await resumed.restore_foundation() is True

    assert resumed.goal is not None
    t1 = next(d for d in resumed.goal.definitions if d.tag == "T1")
    assert t1.formalized is True, "the resume must not reopen generation 0's tag"

    seen = _Generations(resumed)
    await resumed.run_generation(2)

    assert seen.state[2].entries == ()
    assert seen.state[2].definition_names == frozenset({"Category"})
