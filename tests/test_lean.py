"""Tests for LEAN verification interface."""

import pytest

from lms.lean.interface import LeanVerifier, VerificationResult
from lms.lean.mock import MockLeanVerifier


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_success_result(self):
        """VerificationResult holds success information."""
        result = VerificationResult(
            success=True,
            code="theorem test : True := trivial",
            error=None,
        )
        assert result.success is True
        assert result.error is None

    def test_create_failure_result(self):
        """VerificationResult holds failure information."""
        result = VerificationResult(
            success=False,
            code="theorem broken : False := sorry",
            error="Type mismatch: expected True, got False",
        )
        assert result.success is False
        assert "Type mismatch" in result.error


class TestLeanVerifier:
    """Tests for the abstract LeanVerifier class."""

    def test_cannot_instantiate_directly(self):
        """LeanVerifier is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            LeanVerifier()  # type: ignore

    def test_subclass_must_implement_verify(self):
        """Subclasses must implement verify method."""

        class IncompleteVerifier(LeanVerifier):
            pass

        with pytest.raises(TypeError):
            IncompleteVerifier()  # type: ignore


class TestMockLeanVerifier:
    """Tests for MockLeanVerifier."""

    def test_initialization(self):
        """MockLeanVerifier can be instantiated."""
        verifier = MockLeanVerifier()
        assert verifier is not None

    @pytest.mark.asyncio
    async def test_verify_valid_lemma(self):
        """Mock verifier accepts well-formed lemma syntax."""
        verifier = MockLeanVerifier()
        code = "lemma even_add (a b : Nat) : Even a -> Even b -> Even (a + b) := by omega"
        result = await verifier.verify(code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_valid_theorem(self):
        """Mock verifier accepts well-formed theorem syntax."""
        verifier = MockLeanVerifier()
        code = "theorem main_thm : True := trivial"
        result = await verifier.verify(code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_sorry_fails(self):
        """Mock verifier rejects proofs with sorry."""
        verifier = MockLeanVerifier()
        code = "theorem incomplete : False := sorry"
        result = await verifier.verify(code)
        assert result.success is False
        assert "sorry" in result.error.lower()

    @pytest.mark.asyncio
    async def test_verify_empty_code_fails(self):
        """Mock verifier rejects empty code."""
        verifier = MockLeanVerifier()
        result = await verifier.verify("")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verify_malformed_code_fails(self):
        """Mock verifier rejects obviously malformed code."""
        verifier = MockLeanVerifier()
        result = await verifier.verify("this is not lean code at all")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verify_def_accepted(self):
        """Mock verifier accepts well-formed definitions."""
        verifier = MockLeanVerifier()
        code = "def double (n : Nat) : Nat := n + n"
        result = await verifier.verify(code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_configurable_success_rate(self):
        """Mock verifier can be configured with custom success rate."""
        verifier = MockLeanVerifier(success_rate=0.0)
        code = "lemma test : True := trivial"
        # With 0% success rate, everything should fail (except it still checks syntax)
        # Let's make a truly random verifier
        verifier_random = MockLeanVerifier(success_rate=0.0, always_check_syntax=False)
        result = await verifier_random.verify(code)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verification_result_includes_code(self):
        """Verification result includes the original code."""
        verifier = MockLeanVerifier()
        code = "theorem test : True := trivial"
        result = await verifier.verify(code)
        assert result.code == code
