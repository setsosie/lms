"""Admissibility and gate-governed promotion (post-`committee_fix_c`).

`committee_fix_c` (2026-08-20, 20 generations, real verifier) produced ten
"verified" artifacts, three of which contained nothing but the scribe's own
prompt scaffold, `-- Your LEAN 4 code here`. A file of comments compiles with
zero errors and zero sorries, so the verifier reported success; one of those
three then closed the Yoneda Lemma milestone in the dependency graph and
released everything downstream of it.

Two defects, tested separately here:

* nothing checked that a submission introduced any declaration *before* Lean
  ran — `TestContentViolation`, `TestVerifyAdmissible`;
* the T2 gate did detect it post-compile and nothing acted on the verdict —
  `TestGateBlockedPromotion`.
"""

import pytest

from lms.artifacts import Artifact, ArtifactType
from lms.config import ProviderConfig
from lms.gates.base import GateOutcome, GateResult
from lms.gates.novelty import default_novelty_classifier
from lms.lean.interface import (
    LeanVerifier,
    VerificationResult,
    VerificationStatus,
    VerifierKind,
)
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.society import Society

# Verbatim from `prompts.py` (SCRIBE_SYSTEM_PROMPT_V1) and
# `working_group.py`. Three artifacts in `committee_fix_c` were exactly this.
SCRIBE_PLACEHOLDER = "-- Your LEAN 4 code here"

# The shape agents actually emitted: body indented two spaces because the
# YAML block-scalar strip dedents only the first line.
INDENTED_STRUCTURE = """import LMS.Foundation
  open LMS.Foundation

  structure Category (C : Type u) where
    Hom : C → C → Type v
"""


class CountingVerifier(LeanVerifier):
    """Accepts anything, and records whether Lean was actually reached."""

    verifier_kind: VerifierKind = "real"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def verify(self, code: str) -> VerificationResult:
        self.calls.append(code)
        return self._result(success=True, code=code)


class MockVerifier(LeanVerifier):
    verifier_kind: VerifierKind = "mock"

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


def make_society(tmp_path, verifier=None):
    return Society(
        n_agents=1,
        provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
        verifier=verifier if verifier is not None else CountingVerifier(),
        foundation_path=tmp_path / "Foundation.lean",
    )


def make_artifact(code, status=VerificationStatus.VERIFIED_LEAN):
    return Artifact(
        id="a-1",
        type=ArtifactType.THEOREM,
        natural_language="test",
        created_by="agent-0",
        generation=1,
        lean_code=code,
        status=status,
    )


class TestContentViolation:
    @pytest.mark.parametrize(
        "code",
        [
            SCRIBE_PLACEHOLDER,
            "",
            "   \n\t\n",
            "-- just a comment\n-- and another",
            "/- a block comment -/",
            "import Mathlib\nopen CategoryTheory",
            "example : True := trivial",
            "axiom cheat : False",
        ],
    )
    def test_contentless_submissions_rejected(self, code, tmp_path):
        society = make_society(tmp_path)
        assert society._content_violation(code) is not None

    @pytest.mark.parametrize(
        "code",
        [
            "theorem foo : True := trivial",
            "def f (n : Nat) : Nat := n + 1",
            "structure Category (C : Type u) where\n  Hom : C → C → Type v",
            INDENTED_STRUCTURE,
        ],
    )
    def test_real_declarations_admitted(self, code, tmp_path):
        society = make_society(tmp_path)
        assert society._content_violation(code) is None

    def test_message_names_the_missing_thing(self, tmp_path):
        society = make_society(tmp_path)
        message = society._content_violation(SCRIBE_PLACEHOLDER)
        # The scribe sees this as its repair prompt, so it has to say what to
        # emit, not merely that something was wrong.
        assert "named declaration" in message
        assert "theorem" in message


class TestVerifyAdmissible:
    async def test_placeholder_never_reaches_lean(self, tmp_path):
        verifier = CountingVerifier()
        society = make_society(tmp_path, verifier)
        result = await society._verify_admissible(SCRIBE_PLACEHOLDER)
        assert not result.success
        assert result.status is VerificationStatus.FAILED
        assert verifier.calls == [], "contentless code must not consume a Lean run"

    async def test_rejection_keeps_verifier_provenance(self, tmp_path):
        """A rejection is still attributable to the run's machinery."""
        society = make_society(tmp_path)
        result = await society._verify_admissible(SCRIBE_PLACEHOLDER)
        assert result.verifier_kind == "real"

    async def test_real_code_reaches_lean(self, tmp_path):
        verifier = CountingVerifier()
        society = make_society(tmp_path, verifier)
        result = await society._verify_admissible("theorem foo : True := trivial")
        assert result.success
        assert verifier.calls == ["theorem foo : True := trivial"]

    async def test_no_verifier_is_a_rejection_not_a_crash(self, tmp_path):
        society = make_society(tmp_path)
        society.verifier = None
        result = await society._verify_admissible("theorem foo : True := trivial")
        assert not result.success
        assert "No verifier configured" in (result.error or "")


class TestGateBlockedPromotion:
    def _artifact_with(self, *outcomes):
        artifact = make_artifact("theorem foo : True := trivial")
        artifact.gate_results = [
            GateResult(gate=f"T2.check{i}", outcome=o, reason="test")
            for i, o in enumerate(outcomes)
        ]
        return artifact

    def test_failed_gate_blocks(self, tmp_path):
        society = make_society(tmp_path)
        artifact = self._artifact_with(GateOutcome.PASSED, GateOutcome.FAILED)
        assert society._blocked_by_gates(artifact)

    def test_inconclusive_does_not_block(self, tmp_path):
        """`T2.duplicate` is INCONCLUSIVE whenever no duplicate checker is
        injected, which is every run today. Blocking on it would promote
        nothing, ever."""
        society = make_society(tmp_path)
        artifact = self._artifact_with(GateOutcome.PASSED, GateOutcome.INCONCLUSIVE)
        assert not society._blocked_by_gates(artifact)

    def test_no_gate_results_does_not_block(self, tmp_path):
        society = make_society(tmp_path)
        assert not society._blocked_by_gates(self._artifact_with())

    def test_block_reason_recorded_in_notes(self, tmp_path):
        society = make_society(tmp_path)
        artifact = self._artifact_with(GateOutcome.FAILED)
        society._note_gate_block(artifact)
        assert "Not promoted" in (artifact.notes or "")
        assert "T2.check0" in (artifact.notes or "")

    async def test_status_is_untouched_by_gate_failure(self, tmp_path):
        """Promotion is gated; `status` is not. "Lean accepted it" stays true
        and separately recorded — collapsing the two is what the gate
        machinery exists to avoid."""
        society = make_society(tmp_path)
        artifact = make_artifact(SCRIBE_PLACEHOLDER)
        await society._apply_gates(artifact)
        assert artifact.status is VerificationStatus.VERIFIED_LEAN
        assert society._blocked_by_gates(artifact)


class TestNoveltyClassifierWiring:
    def test_mock_verifier_gets_no_classifier(self, tmp_path):
        """No Lean project → no local Mathlib to search. "Absent from Mathlib"
        must not be assertable when nothing could look."""
        society = make_society(tmp_path, MockVerifier())
        assert society.novelty_classifier is None

    def test_projectless_real_verifier_gets_no_classifier(self, tmp_path):
        society = make_society(tmp_path, CountingVerifier())
        assert society.novelty_classifier is None

    def test_factory_returns_none_without_verifier(self):
        assert default_novelty_classifier(None) is None

    async def test_classifier_errors_do_not_end_the_run(self, tmp_path):
        """A search backend raising is a hole in the audit, not a failed
        artifact — `novelty_level` stays None, which reads as "never
        classified" rather than as a novelty claim."""

        class ExplodingClassifier:
            mathlib_rev = None

            def classify(self, *args, **kwargs):
                raise RuntimeError("loogle unreachable")

        society = make_society(tmp_path)
        society.novelty_classifier = ExplodingClassifier()
        artifact = make_artifact("theorem foo : True := trivial")
        await society._apply_novelty_gate(artifact)
        assert artifact.novelty_level is None
        assert "loogle unreachable" in artifact.novelty_evidence[0]

    async def test_classifier_stamps_verdict_onto_artifact(self, tmp_path):
        from lms.novelty import NoveltyLevel, NoveltyResult

        class StubClassifier:
            mathlib_rev = "abc123"

            def classify(self, lean_statement, informal=None):
                return NoveltyResult(
                    level=NoveltyLevel.N0,
                    confidence=0.95,
                    evidence=["name: CategoryTheory.Category"],
                    mathlib_rev=self.mathlib_rev,
                )

        society = make_society(tmp_path)
        society.novelty_classifier = StubClassifier()
        artifact = make_artifact("structure Category (C : Type u) where\n  Hom : C → C")
        await society._apply_novelty_gate(artifact)
        assert artifact.novelty_level == "N0"
        assert artifact.novelty_confidence == 0.95
        assert artifact.novelty_evidence == ["name: CategoryTheory.Category"]
