"""A resumed run must import the foundation it was building (26Q4-HARN-01).

`import LMS.Foundation` resolves through Lake, so the foundation file lives in
the Lean project, never under the run directory. `Society.save` only wrote it
there, and `Society.load` only looked for it at
`<run>/LMS/Foundation.json`, a path nothing ever writes. So the restore
branch never ran, and every resumed society started from an empty foundation:
no seed `Category`, and none of the verified work the checkpoint's own
`artifacts.json` records.

The project's copy cannot stand in for the run's either. The batch script
restores the tracked `lean/LMS/Foundation.*` when each job ends, and the next
run's `reset_foundation` overwrites whatever is left. The run directory is the
only place a checkpoint's foundation can survive between jobs.

Two more gaps stood between a walltime kill and any resume at all:
`metadata.json`, which `resume_experiment` reads first, was written only when
a run finished, and periodic checkpoints never wrote `goal.json`, without
which a committee run cannot resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lms.society
from lms.artifacts import Artifact, ArtifactType
from lms.config import Config, ProviderConfig
from lms.lean.interface import VerificationStatus
from lms.lean.mock import MockLeanVerifier
from lms.goals import Goal
from lms.run import resume_experiment, run_experiment
from lms.society import FOUNDATION_SNAPSHOT, GenerationResult, Society
from tests.test_foundation_rollback import FakeProject, ProjectVerifier
from tests.test_society import MockProvider

SEED = "class Category (obj : Type u) where\n  Hom : obj → obj → Type v\n"


def _verified_artifact() -> Artifact:
    return Artifact(
        id="definition-Pointed-abcd1234",
        type=ArtifactType.DEFINITION,
        natural_language="a type with a chosen point",
        lean_code="structure Pointed (obj : Type u) where\n  pt : obj",
        status=VerificationStatus.VERIFIED_LEAN,
        created_by="agent-0",
        generation=0,
    )


async def _checkpoint(run_dir: Path) -> Society:
    """A society one generation in: seeded, one verified entry, saved.

    No `foundation_path`, so the foundation lives at `DEFAULT_FOUNDATION_PATH`
    exactly as in a real run (conftest points that at a temp dir).
    """
    config = ProviderConfig(api_key="test", model="test")
    society = Society(
        n_agents=1,
        provider=MockProvider(config),
        verifier=MockLeanVerifier(),
        seed_source=SEED,
    )
    await society.reset_foundation()
    assert society.foundation.add_artifact(_verified_artifact())
    await society.persist_foundation()
    society.current_generation = 1
    society.save(run_dir)
    return society


def _end_of_job() -> None:
    """What the batch script does to the project copy when a job exits."""
    project = lms.society.DEFAULT_FOUNDATION_PATH
    project.write_text("")
    project.with_suffix(".json").unlink()


def _load(run_dir: Path) -> Society:
    config = ProviderConfig(api_key="test", model="test")
    return Society.load(run_dir, MockProvider(config), MockLeanVerifier())


@pytest.mark.asyncio
async def test_save_keeps_the_foundation_in_the_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    await _checkpoint(run_dir)

    snapshot = json.loads((run_dir / FOUNDATION_SNAPSHOT).read_text())

    assert [e["name"] for e in snapshot["entries"]] == ["Pointed"]
    assert "class Category" in snapshot["seed_source"]


@pytest.mark.asyncio
async def test_save_keeps_an_empty_foundation_too(tmp_path: Path) -> None:
    """A run that has verified nothing yet still has a seed to restore."""
    config = ProviderConfig(api_key="test", model="test")
    society = Society(
        n_agents=1,
        provider=MockProvider(config),
        verifier=MockLeanVerifier(),
        seed_source=SEED,
    )
    await society.reset_foundation()
    society.save(tmp_path / "run")

    snapshot = json.loads((tmp_path / "run" / FOUNDATION_SNAPSHOT).read_text())

    assert snapshot["entries"] == []
    assert "class Category" in snapshot["seed_source"]


@pytest.mark.asyncio
async def test_load_restores_entries_and_seed_after_the_project_copy_is_gone(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    original = await _checkpoint(run_dir)
    _end_of_job()

    resumed = _load(run_dir)

    assert [e.name for e in resumed.foundation.entries] == ["Pointed"]
    assert resumed.foundation.seed_source == original.foundation.seed_source
    assert "Category" in resumed.foundation._definition_names


@pytest.mark.asyncio
async def test_load_keeps_the_foundation_where_lean_imports_it(tmp_path: Path) -> None:
    """Not under the run directory: Lake never looks there."""
    run_dir = tmp_path / "run"
    await _checkpoint(run_dir)

    resumed = _load(run_dir)

    assert resumed.foundation.path == lms.society.DEFAULT_FOUNDATION_PATH
    assert run_dir not in resumed.foundation.path.parents


@pytest.mark.asyncio
async def test_load_refuses_a_checkpoint_without_a_foundation(tmp_path: Path) -> None:
    """Checkpoints written before the fix cannot be resumed faithfully.

    Resuming one would start from an empty foundation while `artifacts.json`
    still lists the verified work, which is the failure this card removes.
    """
    run_dir = tmp_path / "run"
    await _checkpoint(run_dir)
    (run_dir / FOUNDATION_SNAPSHOT).unlink()

    with pytest.raises(ValueError, match=FOUNDATION_SNAPSHOT):
        _load(run_dir)


@pytest.mark.asyncio
async def test_resume_writes_the_foundation_before_the_first_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    await _checkpoint(run_dir)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "n_agents": 1,
                "generations_completed": 1,
                "total_tokens_used": 0,
            }
        )
    )
    _end_of_job()
    monkeypatch.setattr("lms.run.create_provider", lambda name, cfg: MockProvider(cfg))

    imported: list[str] = []

    async def _generation(self: Society, generation: int) -> GenerationResult:
        imported.append(self.foundation.path.read_text())
        self.current_generation = generation + 1
        return GenerationResult(
            generation=generation,
            artifacts_created=0,
            artifacts_verified=0,
            artifacts_referenced=0,
            fresh_creations=0,
        )

    monkeypatch.setattr(Society, "run_generation", _generation)

    await resume_experiment(
        checkpoint_dir=run_dir,
        target_generations=2,
        config=Config(anthropic=ProviderConfig(api_key="k", model="m")),
        verifier_type="mock",
    )

    assert len(imported) == 1
    assert "class Category" in imported[0]
    assert "structure Pointed" in imported[0]


@pytest.mark.asyncio
async def test_a_run_killed_mid_way_leaves_what_resume_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata before the first generation, the goal at every checkpoint.

    Checked from inside the second generation, which is where a walltime kill
    finds the run: the end-of-run writes have not happened yet.
    """
    monkeypatch.setattr("lms.run.create_provider", lambda name, cfg: MockProvider(cfg))
    run_dir = tmp_path / "run"
    seen: list[tuple[int, dict, bool]] = []

    async def _generation(self: Society, generation: int) -> GenerationResult:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        seen.append((generation, metadata, (run_dir / "goal.json").exists()))
        self.current_generation = generation + 1
        return GenerationResult(
            generation=generation,
            artifacts_created=0,
            artifacts_verified=0,
            artifacts_referenced=0,
            fresh_creations=0,
        )

    monkeypatch.setattr(Society, "run_generation", _generation)

    await run_experiment(
        n_agents=1,
        n_generations=2,
        provider_name="anthropic",
        output_dir=run_dir,
        config=Config(anthropic=ProviderConfig(api_key="k", model="m")),
        goal=Goal(name="t", description="d", source="s", seed=""),
        verifier_type="mock",
        checkpoint_interval=1,
    )

    (_, first, _), (_, second, goal_saved) = seen
    assert first["provider"] == "anthropic"
    assert first["n_agents"] == 1
    assert second["generations_completed"] == 1
    assert goal_saved, "a committee resume needs the checkpoint's goal"


def _interrupted_checkpoint(run_dir: Path) -> None:
    """Killed during generation 1: its entry is in, `current_generation` is not bumped."""
    config = ProviderConfig(api_key="test", model="test")
    society = Society(
        n_agents=1,
        provider=MockProvider(config),
        verifier=MockLeanVerifier(),
        seed_source=SEED,
    )
    assert society.foundation.add_artifact(_verified_artifact())
    in_flight = Artifact(
        id="definition-Broken-ef567890",
        type=ArtifactType.DEFINITION,
        natural_language="an entry that breaks the merged module",
        lean_code="def Broken : Nat := 1",
        status=VerificationStatus.VERIFIED_LEAN,
        created_by="agent-0",
        generation=1,
    )
    assert society.foundation.add_artifact(in_flight)
    society.current_generation = 1
    society.save(run_dir)


def _load_onto(run_dir: Path, project: FakeProject) -> Society:
    config = ProviderConfig(api_key="test", model="test")
    return Society.load(run_dir, MockProvider(config), ProjectVerifier(project))


@pytest.mark.asyncio
async def test_an_interrupted_generation_that_breaks_the_build_is_dropped(
    tmp_path: Path,
) -> None:
    """Its entries never built together, so a failure there must not strand the run."""
    run_dir = tmp_path / "run"
    _interrupted_checkpoint(run_dir)
    resumed = _load_onto(run_dir, FakeProject([False, True, True]))

    assert await resumed.restore_foundation() is True

    assert [e.name for e in resumed.foundation.entries] == ["Pointed"]
    written = resumed.foundation.path.read_text()
    assert "def Broken" not in written
    assert "structure Pointed" in written


@pytest.mark.asyncio
async def test_resume_refuses_when_the_boundary_state_does_not_build_either(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _interrupted_checkpoint(run_dir)
    resumed = _load_onto(run_dir, FakeProject([False, False, False]))

    assert await resumed.restore_foundation() is False


@pytest.mark.asyncio
async def test_a_clean_checkpoint_builds_once_and_keeps_everything(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _interrupted_checkpoint(run_dir)
    project = FakeProject([True])
    resumed = _load_onto(run_dir, project)

    assert await resumed.restore_foundation() is True

    assert project.builds == 1
    assert [e.name for e in resumed.foundation.entries] == ["Pointed", "Broken"]
