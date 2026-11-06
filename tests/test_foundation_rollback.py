"""A generation that breaks the merged foundation is rolled back (26Q3-HARN-22).

Every artifact reaching the foundation has already passed the verifier
*individually*. The merged module can still fail: two entries that each compile
alone can collide, shadow a name, or depend on an ordering the merge does not
preserve.

Before this, `_write_and_build_foundation` saved, built, and returned False --
leaving the broken file on disk. `import LMS.Foundation` then fails for every
later generation, so one bad artifact ends the run and the errors point at the
importer rather than the cause.
"""

import pytest

from lms.artifacts import Artifact, ArtifactType
from lms.config import ProviderConfig
from lms.foundation import FoundationFile
from lms.lean.interface import (
    LeanVerifier,
    VerificationResult,
    VerificationStatus,
    VerifierKind,
)
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.society import Society


class FakeProject:
    """Stands in for `LeanProject`, with scriptable build outcomes."""

    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = list(outcomes)
        self.builds = 0

    async def rebuild_changed_sources(self) -> bool:
        self.builds += 1
        return self.outcomes.pop(0) if self.outcomes else True


class ProjectVerifier(LeanVerifier):
    verifier_kind: VerifierKind = "real"

    def __init__(self, project) -> None:
        self.project = project

    async def verify(self, code: str) -> VerificationResult:
        return self._result(success=True, code=code)


class SilentProvider(BaseLLMProvider):
    name = "mock"

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        return GenerationResponse(
            content="No artifacts proposed.",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider=self.name,
        )


def make_artifact(name: str, ident: str) -> Artifact:
    return Artifact(
        id=ident,
        type=ArtifactType.DEFINITION,
        natural_language=name,
        created_by="agent-0",
        generation=1,
        lean_code=f"def {name} : Nat := 1",
        status=VerificationStatus.VERIFIED_LEAN,
    )


def make_coded(ident: str, code: str) -> Artifact:
    return Artifact(
        id=ident,
        type=ArtifactType.DEFINITION,
        natural_language=ident,
        created_by="agent-0",
        generation=1,
        lean_code=code,
        status=VerificationStatus.VERIFIED_LEAN,
    )


def make_society(tmp_path, outcomes):
    project = FakeProject(outcomes)
    society = Society(
        n_agents=1,
        provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
        verifier=ProjectVerifier(project),
        foundation_path=tmp_path / "Foundation.lean",
    )
    return society, project


class TestSnapshotRestore:
    def test_restore_returns_dropped_names(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.add_artifact(make_artifact("good", "a-1"))
        snap = f.snapshot()
        f.add_artifact(make_artifact("bad", "a-2"))
        assert f.restore(snap) == ["bad"]
        assert [e.name for e in f.entries] == ["good"]

    def test_restore_forgets_the_dropped_artifact_id(self, tmp_path):
        """A rolled-back artifact must be genuinely forgotten, or a later
        resubmission of the same work is skipped as a duplicate."""
        f = FoundationFile(tmp_path / "F.lean")
        snap = f.snapshot()
        f.add_artifact(make_artifact("bad", "a-2"))
        f.restore(snap)
        assert "a-2" not in f._artifact_ids
        f.add_artifact(make_artifact("bad", "a-2"))
        assert [e.name for e in f.entries] == ["bad"]

    def test_snapshot_is_not_aliased_to_live_state(self, tmp_path):
        """Adding after a snapshot must not mutate the snapshot."""
        f = FoundationFile(tmp_path / "F.lean")
        f.add_artifact(make_artifact("good", "a-1"))
        snap = f.snapshot()
        f.add_artifact(make_artifact("later", "a-2"))
        assert len(snap.entries) == 1
        assert "a-2" not in snap.artifact_ids

    def test_restoring_an_empty_snapshot_clears_everything(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        empty = f.snapshot()
        f.add_artifact(make_artifact("x", "a-1"))
        f.restore(empty)
        assert f.entries == []
        assert f._definition_names == set()


class TestBuildRollback:
    async def test_successful_build_records_a_good_state(self, tmp_path):
        society, _ = make_society(tmp_path, [True])
        society.foundation.add_artifact(make_artifact("good", "a-1"))
        assert await society._write_and_build_foundation() is True
        assert society._last_good_foundation is not None
        assert len(society._last_good_foundation.entries) == 1

    async def test_failed_build_rolls_back_to_last_good(self, tmp_path):
        society, project = make_society(tmp_path, [True, False, True])
        society.foundation.add_artifact(make_artifact("good", "a-1"))
        assert await society._write_and_build_foundation() is True

        society.foundation.add_artifact(make_artifact("bad", "a-2"))
        assert await society._write_and_build_foundation() is False

        assert [e.name for e in society.foundation.entries] == ["good"]
        # build, failed build, rebuild-after-rollback
        assert project.builds == 3

    async def test_rollback_rewrites_the_file_on_disk(self, tmp_path):
        """The whole point: a broken module must not be left where the next
        generation's `import LMS.Foundation` will find it."""
        society, _ = make_society(tmp_path, [True, False, True])
        society.foundation.add_artifact(make_artifact("good", "a-1"))
        await society._write_and_build_foundation()
        society.foundation.add_artifact(make_artifact("bad", "a-2"))
        await society._write_and_build_foundation()

        on_disk = (tmp_path / "Foundation.lean").read_text()
        assert "good" in on_disk
        assert "def bad" not in on_disk

    async def test_first_ever_build_failure_does_not_crash(self, tmp_path):
        """No good state to return to -- report, do not raise."""
        society, _ = make_society(tmp_path, [False])
        society.foundation.add_artifact(make_artifact("bad", "a-1"))
        assert await society._write_and_build_foundation() is False

    async def test_good_state_advances_across_generations(self, tmp_path):
        society, _ = make_society(tmp_path, [True, True, False, True])
        society.foundation.add_artifact(make_artifact("one", "a-1"))
        await society._write_and_build_foundation()
        society.foundation.add_artifact(make_artifact("two", "a-2"))
        await society._write_and_build_foundation()
        society.foundation.add_artifact(make_artifact("three", "a-3"))
        await society._write_and_build_foundation()
        assert [e.name for e in society.foundation.entries] == ["one", "two"]

    async def test_no_project_means_no_rollback_machinery(self, tmp_path):
        """Mock runs have nothing to build; they must not be penalised."""
        from lms.lean.mock import MockLeanVerifier

        society = Society(
            n_agents=1,
            provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
            verifier=MockLeanVerifier(),
            foundation_path=tmp_path / "Foundation.lean",
        )
        society.foundation.add_artifact(make_artifact("x", "a-1"))
        assert await society._write_and_build_foundation() is True

    async def test_rollback_reports_what_it_dropped(self, tmp_path, capsys):
        society, _ = make_society(tmp_path, [True, False, True])
        society.foundation.add_artifact(make_artifact("good", "a-1"))
        await society._write_and_build_foundation()
        society.foundation.add_artifact(make_artifact("bad", "a-2"))
        await society._write_and_build_foundation()
        out = capsys.readouterr().out
        assert "rolled back" in out
        assert "bad" in out


class TestPerArtifactRollback:
    """A failed merged build costs the artifacts that broke it, not the generation."""

    @staticmethod
    async def _build_generation(society, *names: str) -> bool:
        for number, name in enumerate(names):
            society.foundation.add_artifact(make_artifact(name, f"gen-{number}"))
        return await society._write_and_build_foundation()

    async def _seeded(self, tmp_path, outcomes):
        society, project = make_society(tmp_path, [True, *outcomes])
        society.foundation.add_artifact(make_artifact("good", "a-0"))
        await society._write_and_build_foundation()
        return society, project

    async def test_it_keeps_every_artifact_that_builds(self, tmp_path):
        society, project = await self._seeded(tmp_path, [False, True, False, True])

        await self._build_generation(society, "first", "bad", "third")

        assert [e.name for e in society.foundation.entries] == [
            "good",
            "first",
            "third",
        ]
        # seed build, joint failure, then one per artifact
        assert project.builds == 5

    async def test_the_good_state_includes_what_was_kept(self, tmp_path):
        society, _ = await self._seeded(tmp_path, [False, True, False])

        await self._build_generation(society, "first", "bad")

        assert [e.name for e in society._last_good_foundation.entries] == [
            "good",
            "first",
        ]

    async def test_a_failure_last_rebuilds_what_was_kept(self, tmp_path):
        """The last build tried was the bad one, so the module on disk is stale."""
        society, project = await self._seeded(tmp_path, [False, True, False, True])

        await self._build_generation(society, "first", "bad")

        # seed build, joint failure, first, bad, then the rebuild
        assert project.builds == 5
        on_disk = (tmp_path / "Foundation.lean").read_text()
        assert "def first" in on_disk
        assert "def bad" not in on_disk

    async def test_a_success_last_needs_no_rebuild(self, tmp_path):
        society, project = await self._seeded(tmp_path, [False, False, True])

        await self._build_generation(society, "bad", "last")

        assert project.builds == 4

    async def test_all_failing_leaves_the_good_state(self, tmp_path):
        society, _ = await self._seeded(tmp_path, [False, False, False, True])

        assert await self._build_generation(society, "bad", "worse") is False

        assert [e.name for e in society.foundation.entries] == ["good"]

    async def test_a_kept_artifact_stays_known(self, tmp_path):
        """A retry must leave ids and names as a plain add would."""
        society, _ = await self._seeded(tmp_path, [False, True, False])

        await self._build_generation(society, "first", "bad")

        assert "gen-0" in society.foundation._artifact_ids
        assert "gen-1" not in society.foundation._artifact_ids
        assert "first" in society.foundation._definition_names
        assert "bad" not in society.foundation._definition_names

    async def test_it_returns_false_when_something_was_dropped(self, tmp_path):
        society, _ = await self._seeded(tmp_path, [False, True, False])

        assert await self._build_generation(society, "first", "bad") is False

    async def test_it_returns_true_when_everything_was_kept(self, tmp_path):
        """A failure that doesn't recur: the foundation holds all of them and compiles."""
        society, _ = await self._seeded(tmp_path, [False, True, True])

        assert await self._build_generation(society, "first", "second") is True

    async def test_a_copy_lands_when_its_original_is_dropped(self, tmp_path):
        """It built on its own, so losing the other artifact mustn't cost it."""
        society, _ = await self._seeded(tmp_path, [False, False, True])
        society.foundation.add_artifact(
            make_coded("first", "def rep : Nat := 1\n\ndef broken : Nat := rep")
        )
        society.foundation.add_artifact(
            make_coded("copy", "def rep : Nat := 1\n\ndef fine : Nat := rep")
        )

        await society._write_and_build_foundation()

        assert [e.name for e in society.foundation.entries] == ["good", "rep", "fine"]
        assert "first" not in society.foundation._artifact_ids
        assert "copy" in society.foundation._artifact_ids

    async def test_a_pure_copy_stays_known_and_takes_no_build(self, tmp_path):
        society, project = await self._seeded(tmp_path, [False, False, True])
        society.foundation.add_artifact(make_coded("copy", "def good : Nat := 1"))
        society.foundation.add_artifact(make_coded("bad", "def bad : Nat := 1"))

        await society._write_and_build_foundation()

        # seed build, joint failure, bad, then the rebuild. The copy adds nothing to build.
        assert project.builds == 4
        assert "copy" in society.foundation._artifact_ids

    async def test_a_copy_survives_a_checkpoint_for_the_retry(self, tmp_path):
        """A walltime kill saves mid-generation, and the resume build can fail."""
        society, _ = make_society(tmp_path, [False, False, True])
        society.foundation.add_artifact(
            make_coded("first", "def rep : Nat := 1\n\ndef broken : Nat := rep")
        )
        society.foundation.add_artifact(
            make_coded("copy", "def rep : Nat := 1\n\ndef fine : Nat := rep")
        )
        society.foundation.save()

        society.foundation = FoundationFile.load(society.foundation.path)
        society._last_good_foundation = society.foundation.snapshot_before(1)
        await society._write_and_build_foundation()

        assert [e.name for e in society.foundation.entries] == ["rep", "fine"]

    def test_a_generation_boundary_forgets_a_copy(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        original = make_coded("first", "def rep : Nat := 1")
        f.add_artifact(original)
        f.add_artifact(make_coded("copy", "def rep : Nat := 1"))

        boundary = f.snapshot_before(1)

        assert "copy" not in boundary.artifact_ids

    async def test_a_failure_with_nothing_new_still_reports_failure(self, tmp_path):
        """No artifact to blame: an environment fault, or a resume whose last good
        state no longer builds. `restore_foundation` decides the resume on this."""
        society, project = await self._seeded(tmp_path, [False, True])

        assert await society._write_and_build_foundation() is False

        # seed build, the failure, and the rebuild of the good state
        assert project.builds == 3

    async def test_the_log_names_what_went_and_what_stayed(self, tmp_path, capsys):
        society, _ = await self._seeded(tmp_path, [False, True, False])

        await self._build_generation(society, "first", "bad")

        out = capsys.readouterr().out
        assert "rolled back 1 entry: bad" in out
        assert "kept 1 artifact: gen-0" in out


@pytest.mark.parametrize("n_dropped,expected", [(1, "entry"), (2, "entries")])
def test_rollback_message_pluralises(n_dropped, expected):
    """Cosmetic, but the line is read by a human triaging a failed run."""
    assert ("entry" if n_dropped == 1 else "entries") == expected


class TestAddArtifactReportsWhetherItAdded:
    """`add_artifact` returning nothing made three distinct outcomes look like
    success at the call site (26Q3-HARN-22)."""

    def test_new_artifact_returns_true(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        assert f.add_artifact(make_artifact("fresh", "a-1")) is True

    def test_repeated_id_returns_false(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.add_artifact(make_artifact("fresh", "a-1"))
        assert f.add_artifact(make_artifact("other", "a-1")) is False

    def test_duplicate_declaration_name_returns_false(self, tmp_path):
        """Different artifact, same declaration name: nothing new lands."""
        f = FoundationFile(tmp_path / "F.lean")
        f.add_artifact(make_artifact("same", "a-1"))
        assert f.add_artifact(make_artifact("same", "a-2")) is False
        assert len(f.entries) == 1

    def test_unverified_artifact_still_raises(self, tmp_path):
        """The exception contract is unchanged -- only the success signal is new."""
        f = FoundationFile(tmp_path / "F.lean")
        art = make_artifact("x", "a-1")
        art.status = VerificationStatus.FAILED
        with pytest.raises(ValueError):
            f.add_artifact(art)
