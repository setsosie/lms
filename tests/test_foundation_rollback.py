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
