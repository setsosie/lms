"""A goal tag reads DONE only while the foundation holds work behind it.

Every promotion path marks a tag as it processes results, and the foundation
build runs afterwards. A build that rolled the artifact back left the tag DONE,
so the next generation's prompt said a declaration existed that didn't, and told
agents not to redefine it. On a goal with real edges, the graph then released
everything downstream of nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.config import ProviderConfig
from lms.dependency import DependencyGraph, TaskStatus
from lms.foundation import FoundationFile
from lms.goals import Goal, StacksDefinition
from lms.society import Society
from tests.test_foundation_rollback import FakeProject, ProjectVerifier, SilentProvider


def artifact(artifact_id: str, code: str, tag: str = "T1") -> Artifact:
    return Artifact(
        id=artifact_id,
        type=ArtifactType.DEFINITION,
        natural_language="test",
        created_by="agent-0",
        generation=1,
        lean_code=code,
        status=VerificationStatus.VERIFIED_LEAN,
        stacks_tag=tag,
    )


def make_goal() -> Goal:
    return Goal(
        name="test",
        description="test",
        source="test",
        seed="",
        definitions=[
            StacksDefinition(
                tag="T1", section="1.1", name="one", content="one", requires=[]
            ),
            StacksDefinition(
                tag="T2", section="1.2", name="two", content="two", requires=["T1"]
            ),
            StacksDefinition(
                tag="T3", section="1.3", name="three", content="three", requires=[]
            ),
        ],
    )


def make_society(tmp_path: Path, outcomes: list[bool]) -> Society:
    society = Society(
        n_agents=1,
        provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
        verifier=ProjectVerifier(FakeProject(outcomes)),
        goal=make_goal(),
        foundation_path=tmp_path / "Foundation.lean",
    )
    return society


def promote(society: Society, made: Artifact) -> None:
    """What every promotion path does: foundation, goal, library."""
    society._promote_to_foundation(made)
    society.goal.mark_formalized(made.stacks_tag, made.id)
    society.library.add(made)


def definition(society: Society, tag: str = "T1") -> StacksDefinition:
    return next(d for d in society.goal.definitions if d.tag == tag)


class TestReopening:
    async def test_a_rolled_back_artifact_reopens_its_tag(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [True, False, True])
        await society._write_and_build_foundation()
        promote(society, artifact("a", "def bad : Nat := 1"))

        await society._write_and_build_foundation()
        society._reopen_tags_without_foundation_work()

        assert definition(society).formalized is False
        assert definition(society).artifact_ids == []
        assert "[DONE]" not in society.goal.to_prompt_context()

    async def test_a_refused_artifact_reopens_its_tag(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [True])
        society.foundation.add_artifact(artifact("first", "def rep : Nat := 1"))
        await society._write_and_build_foundation()
        promote(society, artifact("second", "def rep : Nat := 2"))

        society._reopen_tags_without_foundation_work()

        assert definition(society).formalized is False

    def test_a_gate_blocked_artifact_reopens_its_tag(self, tmp_path: Path) -> None:
        """Two paths mark the tag even when a gate kept the artifact out."""
        society = make_society(tmp_path, [])
        blocked = artifact("blocked", "def blocked : Nat := 1")
        society.goal.mark_formalized("T1", blocked.id)
        society.library.add(blocked)

        society._reopen_tags_without_foundation_work()

        assert definition(society).formalized is False

    def test_the_log_names_the_tag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        society = make_society(tmp_path, [])
        blocked = artifact("blocked", "def blocked : Nat := 1")
        society.goal.mark_formalized("T1", blocked.id)
        society.library.add(blocked)

        society._reopen_tags_without_foundation_work()

        out = capsys.readouterr().out
        assert "[goal] reopened" in out
        assert "T1" in out

    async def test_a_generation_reconciles_after_its_build(
        self, tmp_path: Path
    ) -> None:
        society = make_society(tmp_path, [True, False, True])
        await society._write_and_build_foundation()

        async def one_generation(generation: int) -> object:
            promote(society, artifact("a", "def bad : Nat := 1"))

            class Result:
                artifacts_verified = 1
                wall_clock_s = 0.0

            return Result()

        society._run_generation_impl = one_generation  # type: ignore[method-assign]

        await society.run_generation(1)

        assert definition(society).formalized is False


class TestWhatStaysDone:
    def test_a_surviving_artifact_keeps_the_tag(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        promote(society, artifact("kept", "def kept : Nat := 1"))
        blocked = artifact("blocked", "def blocked : Nat := 1")
        society.goal.mark_formalized("T1", blocked.id)
        society.library.add(blocked)

        society._reopen_tags_without_foundation_work()

        assert definition(society).formalized is True
        assert definition(society).artifact_ids == ["kept"]

    def test_a_copy_of_existing_work_still_backs_its_tag(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        promote(society, artifact("first", "def same : Nat := 1", tag="T2"))
        promote(society, artifact("copy", "def same : Nat := 1", tag="T1"))

        society._reopen_tags_without_foundation_work()

        assert definition(society, "T1").formalized is True

    def test_a_seeded_tag_is_left_alone(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society.goal.mark_formalized("T1", "seed")

        society._reopen_tags_without_foundation_work()

        assert definition(society).formalized is True

    def test_a_pre_marked_tag_is_left_alone(self, tmp_path: Path) -> None:
        """`stacks-full-path` starts with tags whose artifact this run never saw."""
        society = make_society(tmp_path, [])
        society.goal.mark_formalized("T1", "foundation-yoneda-complete")

        society._reopen_tags_without_foundation_work()

        assert definition(society).formalized is True

    def test_nothing_to_do_without_a_goal(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society.goal = None

        society._reopen_tags_without_foundation_work()


class TestCopiesAcrossAFailedBuild:
    """A copy has an id but no entries, so a rollback can't see it by its entries."""

    async def _built(self, tmp_path: Path, outcomes: list[bool]) -> Society:
        society = make_society(tmp_path, [True, *outcomes])
        promote(society, artifact("first", "def same : Nat := 1", tag="T2"))
        await society._write_and_build_foundation()
        return society

    async def test_a_copy_keeps_its_tag_when_another_artifact_breaks(
        self, tmp_path: Path
    ) -> None:
        society = await self._built(tmp_path, [False, False, True])
        promote(society, artifact("copy", "def same : Nat := 1", tag="T1"))
        promote(society, artifact("bad", "def bad : Nat := 1", tag="T3"))

        await society._write_and_build_foundation()
        society._reopen_tags_without_foundation_work()

        assert definition(society, "T1").formalized is True
        assert definition(society, "T3").formalized is False

    async def test_a_copy_keeps_its_tag_when_nothing_is_dropped(
        self, tmp_path: Path
    ) -> None:
        society = await self._built(tmp_path, [False, True])
        promote(society, artifact("copy", "def same : Nat := 1", tag="T1"))
        promote(society, artifact("new", "def fresh : Nat := 1", tag="T3"))

        assert await society._write_and_build_foundation() is True
        society._reopen_tags_without_foundation_work()

        assert definition(society, "T1").formalized is True

    async def test_a_copy_and_its_original_share_a_fate(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [True, False, True, False, True])
        await society._write_and_build_foundation()
        promote(society, artifact("first", "def same : Nat := 1", tag="T2"))
        promote(society, artifact("copy", "def same : Nat := 1", tag="T1"))
        promote(society, artifact("bad", "def bad : Nat := 1", tag="T3"))

        await society._write_and_build_foundation()
        society._reopen_tags_without_foundation_work()

        assert definition(society, "T1").formalized is True
        assert definition(society, "T2").formalized is True
        assert definition(society, "T3").formalized is False


class TestGateBlockedCommitteeTag:
    """Without this, the tag stays in progress: never offered again, dependents locked."""

    def test_it_goes_back_to_available(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society.dependency_graph = DependencyGraph.from_goal(society.goal)
        graph = society.dependency_graph
        graph.update_status("T1", TaskStatus.IN_PROGRESS)
        blocked = artifact("blocked", "def blocked : Nat := 1")

        society._release_gate_blocked_task("T1", blocked)

        assert graph.nodes["T1"].status is TaskStatus.AVAILABLE
        assert [n.tag for n in graph.available_tasks()][0] == "T1"
        assert "gate failure" in (blocked.notes or "")

    def test_it_leaves_a_task_another_group_finished(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society.dependency_graph = DependencyGraph.from_goal(society.goal)
        graph = society.dependency_graph
        graph.update_status("T1", TaskStatus.DONE, "other")

        society._release_gate_blocked_task(
            "T1", artifact("blocked", "def blocked : Nat := 1")
        )

        assert graph.nodes["T1"].status is TaskStatus.DONE


class TestGraph:
    def test_the_graph_reopens_and_dependents_lock(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society.dependency_graph = DependencyGraph.from_goal(society.goal)
        graph = society.dependency_graph
        blocked = artifact("blocked", "def blocked : Nat := 1")
        graph.update_status("T1", TaskStatus.DONE, blocked.id)
        society.goal.mark_formalized("T1", blocked.id)
        society.library.add(blocked)
        assert graph.nodes["T2"].status is TaskStatus.AVAILABLE

        society._reopen_tags_without_foundation_work()

        assert graph.nodes["T1"].status is TaskStatus.AVAILABLE
        assert graph.nodes["T2"].status is TaskStatus.BLOCKED


class TestResume:
    async def test_a_resume_rollback_reopens_the_tag(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [True, False, True])
        await society._write_and_build_foundation()
        promote(society, artifact("midway", "def bad : Nat := 1"))
        society._last_good_foundation = society.foundation.snapshot_before(1)

        await society.restore_foundation()

        assert definition(society).formalized is False

    def test_a_refusal_survives_a_checkpoint(self, tmp_path: Path) -> None:
        """A walltime kill saves mid-generation, before the tags are reconciled."""
        society = make_society(tmp_path, [])
        society._promote_to_foundation(artifact("first", "def rep : Nat := 1"))
        society._promote_to_foundation(artifact("second", "def rep : Nat := 2"))
        society.foundation.save()

        loaded = FoundationFile.load(society.foundation.path)

        assert loaded.holds("first") is True
        assert loaded.holds("second") is False

    async def test_a_real_load_reopens_a_copys_tag_when_its_original_goes(
        self, tmp_path: Path
    ) -> None:
        """A walltime kill saves mid-generation. If the resume build then fails, the
        copy's id must not outlive the declaration it copied."""
        society = make_society(tmp_path, [False, False, False, True])
        promote(society, artifact("first", "def same : Nat := 1", tag="T2"))
        promote(society, artifact("copy", "def same : Nat := 1", tag="T1"))
        society.foundation.save()

        society.foundation = FoundationFile.load(society.foundation.path)
        society._last_good_foundation = society.foundation.snapshot_before(1)
        await society.restore_foundation()

        assert definition(society, "T2").formalized is False
        assert definition(society, "T1").formalized is False
        assert "[DONE]" not in society.goal.to_prompt_context()

    async def test_a_refused_artifact_can_land_and_back_its_tag(
        self, tmp_path: Path
    ) -> None:
        """Committee ids hash the code, so identical work returns with the same id."""
        society = make_society(tmp_path, [True, False, True, True])
        await society._write_and_build_foundation()
        promote(society, artifact("x", "def rep : Nat := 1", tag="T1"))
        promote(society, artifact("r", "def rep : Nat := 2", tag="T2"))
        await society._write_and_build_foundation()
        society._reopen_tags_without_foundation_work()
        assert definition(society, "T2").formalized is False

        promote(society, artifact("r", "def rep : Nat := 2", tag="T2"))
        await society._write_and_build_foundation()
        society._reopen_tags_without_foundation_work()

        assert definition(society, "T2").formalized is True
        assert society.foundation.holds("r") is True


class TestCoreConceptRefusal:
    def test_it_says_why(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society._promote_to_foundation(
            artifact("a", "structure Category where\n  x : Nat")
        )
        other = artifact("b", "structure Cat where\n  y : Nat")

        society._promote_to_foundation(other)

        assert "`Cat`" in (other.notes or "")
        assert society.foundation.holds("b") is False


class TestHolds:
    def test_only_what_landed(self, tmp_path: Path) -> None:
        society = make_society(tmp_path, [])
        society._promote_to_foundation(artifact("a", "def rep : Nat := 1"))
        society._promote_to_foundation(artifact("b", "def rep : Nat := 2"))

        assert society.foundation.holds("a") is True
        assert society.foundation.holds("b") is False
        assert society.foundation.holds("never") is False
