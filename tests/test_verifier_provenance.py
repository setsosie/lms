"""Provenance tests for 26Q3-HARN-01.

The invariant under test: an artifact can only reach `VERIFIED_LEAN` by way of a
verifier that actually ran Lean. `MockLeanVerifier` is a regex; nothing it
produces may ever be counted toward the calibration numerator.

This is the root cause of the retracted roadmap numbers — a mock run scored 92%
verified and nothing in the record distinguished it from a real Lean result.
"""

import json

import pytest

from lms.artifacts import Artifact, ArtifactLibrary, ArtifactType
from lms.lean.interface import VerificationResult, VerificationStatus
from lms.lean.mock import MockLeanVerifier
from lms.lean.real import RealLeanVerifier

# Code the mock's regex happily accepts.
GOOD_LOOKING_CODE = "theorem goal_thm : True := trivial"


def make_artifact(
    status: VerificationStatus = VerificationStatus.UNVERIFIED,
    lean_code: str | None = None,
    artifact_id: str = "a1",
) -> Artifact:
    """An artifact with the required fields filled in."""
    return Artifact(
        id=artifact_id,
        type=ArtifactType.THEOREM,
        natural_language="trivially true",
        created_by="agent-0",
        generation=0,
        lean_code=lean_code,
        status=status,
    )


class TestVerificationResultProvenance:
    """`VerificationResult` must carry who produced it."""

    def test_result_requires_verifier_kind_and_id(self):
        """A result cannot be constructed without declaring its origin."""
        with pytest.raises(TypeError):
            VerificationResult(success=True, code=GOOD_LOOKING_CODE)  # type: ignore[call-arg]

    def test_mock_kind_success_maps_to_heuristic(self):
        result = VerificationResult(
            success=True,
            code=GOOD_LOOKING_CODE,
            verifier_kind="mock",
            verifier_id="MockLeanVerifier",
        )
        assert result.status is VerificationStatus.VERIFIED_HEURISTIC

    @pytest.mark.parametrize("kind", ["real", "mcp"])
    def test_lean_kinds_success_maps_to_verified_lean(self, kind):
        result = VerificationResult(
            success=True,
            code=GOOD_LOOKING_CODE,
            verifier_kind=kind,
            verifier_id=f"{kind}-verifier",
        )
        assert result.status is VerificationStatus.VERIFIED_LEAN

    @pytest.mark.parametrize("kind", ["mock", "real", "mcp"])
    def test_failure_maps_to_failed_regardless_of_kind(self, kind):
        result = VerificationResult(
            success=False,
            code="garbage",
            error="nope",
            verifier_kind=kind,
            verifier_id=f"{kind}-verifier",
        )
        assert result.status is VerificationStatus.FAILED

    def test_unknown_kind_is_rejected(self):
        """A typo in the kind must not silently become a Lean-grade result."""
        with pytest.raises(ValueError):
            VerificationResult(
                success=True,
                code=GOOD_LOOKING_CODE,
                verifier_kind="reall",  # type: ignore[arg-type]
                verifier_id="x",
            )


class TestMockCanNeverVerifyLean:
    """The central acceptance criterion of the card."""

    @pytest.mark.asyncio
    async def test_mock_declares_itself_mock(self):
        verifier = MockLeanVerifier()
        assert verifier.verifier_kind == "mock"
        assert "Mock" in verifier.verifier_id

    @pytest.mark.asyncio
    async def test_mock_success_is_only_heuristic(self):
        result = await MockLeanVerifier().verify(GOOD_LOOKING_CODE)
        assert result.success is True
        assert result.status is VerificationStatus.VERIFIED_HEURISTIC
        assert result.status is not VerificationStatus.VERIFIED_LEAN

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code",
        [
            GOOD_LOOKING_CODE,
            "def f (n : Nat) : Nat := n",
            "lemma l : 1 = 1 := rfl",
            "structure S where x : Nat",
            "axiom bad : False",
            "example : True := trivial",
            "",
            "sorry",
            "not lean at all",
        ],
    )
    async def test_no_input_makes_the_mock_emit_verified_lean(self, code):
        """Exhaustive over the mock's accept and reject paths."""
        result = await MockLeanVerifier().verify(code)
        assert result.status is not VerificationStatus.VERIFIED_LEAN

    @pytest.mark.asyncio
    async def test_mock_verified_artifact_is_not_counted_verified(self):
        """End-to-end: mock result -> artifact -> library.get_verified()."""
        result = await MockLeanVerifier().verify(GOOD_LOOKING_CODE)
        artifact = make_artifact(lean_code=GOOD_LOOKING_CODE, status=result.status)

        assert artifact.verified is False

        library = ArtifactLibrary()
        library.add(artifact)
        assert library.get_verified() == []


class TestRealVerifierDeclaresLeanKind:
    def test_real_declares_itself_real(self):
        try:
            verifier = RealLeanVerifier()
        except FileNotFoundError:
            pytest.skip("no Lean toolchain on this machine")
        assert verifier.verifier_kind == "real"


class TestArtifactVerifiedProperty:
    """`verified` is a read-only view onto `status`, never a settable flag."""

    def test_verified_true_only_for_verified_lean(self):
        assert make_artifact(status=VerificationStatus.VERIFIED_LEAN).verified is True

    @pytest.mark.parametrize(
        "status",
        [
            VerificationStatus.UNVERIFIED,
            VerificationStatus.VERIFIED_HEURISTIC,
            VerificationStatus.FAILED,
        ],
    )
    def test_verified_false_for_everything_else(self, status):
        assert make_artifact(status=status).verified is False

    def test_verified_cannot_be_assigned(self):
        """Nothing may promote an artifact by writing the boolean."""
        artifact = make_artifact()
        with pytest.raises(AttributeError):
            artifact.verified = True  # type: ignore[misc]

    def test_default_status_is_unverified(self):
        assert make_artifact().status is VerificationStatus.UNVERIFIED


class TestLegacyLoadDemotion:
    """Historical runs must not be silently promoted to Lean-grade."""

    def test_legacy_verified_true_demotes_to_heuristic(self):
        legacy = {
            "id": "old-1",
            "type": "theorem",
            "natural_language": "from a 2025 mock run",
            "lean_code": GOOD_LOOKING_CODE,
            "verified": True,
            "created_by": "agent-0-mock",
            "generation": 0,
        }
        artifact = Artifact.from_dict(legacy)
        assert artifact.status is VerificationStatus.VERIFIED_HEURISTIC
        assert artifact.verified is False

    def test_legacy_verified_false_is_unverified(self):
        legacy = {
            "id": "old-2",
            "type": "theorem",
            "natural_language": "failed back then",
            "verified": False,
            "created_by": "agent-0-mock",
            "generation": 0,
        }
        assert Artifact.from_dict(legacy).status is VerificationStatus.UNVERIFIED

    def test_explicit_status_field_wins_over_legacy_boolean(self):
        """New records round-trip at full fidelity."""
        d = {
            "id": "new-1",
            "type": "theorem",
            "natural_language": "real lean result",
            "verified": True,
            "status": "verified_lean",
            "created_by": "agent-0",
            "generation": 0,
        }
        assert Artifact.from_dict(d).status is VerificationStatus.VERIFIED_LEAN

    def test_archived_mock_run_scores_zero_verified(self):
        """Gate A in miniature: a 100%-verified mock library reloads as 0."""
        legacy_artifacts = [
            {
                "id": f"old-{i}",
                "type": "theorem",
                "natural_language": "mock-verified",
                "verified": True,
                "created_by": "agent-0-mock",
                "generation": 0,
            }
            for i in range(10)
        ]
        library = ArtifactLibrary()
        for d in legacy_artifacts:
            library.add(Artifact.from_dict(d))

        assert len(library) == 10
        assert library.get_verified() == []


class TestSerializationRoundTrip:
    def test_status_survives_round_trip(self):
        original = make_artifact(
            lean_code=GOOD_LOOKING_CODE,
            status=VerificationStatus.VERIFIED_LEAN,
        )
        restored = Artifact.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored.status is VerificationStatus.VERIFIED_LEAN
        assert restored.verified is True

    def test_to_dict_emits_both_status_and_legacy_boolean(self):
        d = make_artifact(status=VerificationStatus.VERIFIED_HEURISTIC).to_dict()
        assert d["status"] == "verified_heuristic"
        assert d["verified"] is False, (
            "legacy readers must not see heuristic as verified"
        )


class TestExperimentMetadata:
    """`metadata.json` must record which machinery produced the run."""

    def _society(self, verifier):
        from lms.config import ProviderConfig
        from lms.providers.base import BaseLLMProvider, GenerationResponse, TokenUsage
        from lms.society import Society

        class _Provider(BaseLLMProvider):
            name = "mock"

            async def generate(self, messages, system_prompt=None, max_tokens=4096):
                return GenerationResponse(
                    content="",
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                    provider=self.name,
                )

        return Society(
            n_agents=1,
            provider=_Provider(ProviderConfig(api_key="k", model="m")),
            verifier=verifier,
        )

    def test_metadata_reports_mock_kind(self):
        meta = self._society(MockLeanVerifier()).verifier_metadata()
        assert meta["kind"] == "mock"
        assert meta["id"] == "MockLeanVerifier"
        # A regex has no toolchain to report.
        assert meta["lean_version"] is None
        assert meta["mathlib_rev"] is None

    def test_metadata_has_all_four_provenance_keys(self):
        meta = self._society(MockLeanVerifier()).verifier_metadata()
        assert set(meta) == {"kind", "id", "lean_version", "mathlib_rev"}

    def test_metadata_survives_json(self):
        meta = self._society(MockLeanVerifier()).verifier_metadata()
        assert json.loads(json.dumps(meta)) == meta


class TestDefaultFoundationPathIsAnchored:
    """The shared corpus must not depend on the process's cwd."""

    def test_default_path_is_absolute(self):
        # Imported here so the conftest autouse patch is already in place.
        import lms.society

        # The patched value is what a Society would actually use; the module
        # default is what ships. Both must be absolute.
        assert lms.society.DEFAULT_FOUNDATION_PATH.is_absolute()

    def test_shipped_default_points_at_the_repo_corpus(self):
        import importlib

        import lms.society

        shipped = importlib.reload(lms.society).DEFAULT_FOUNDATION_PATH
        try:
            assert shipped.is_absolute()
            assert shipped.parts[-3:] == ("lean", "LMS", "Foundation.lean")
        finally:
            importlib.reload(lms.society)
