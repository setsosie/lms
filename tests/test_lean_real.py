"""Tests for real LEAN 4 verifier."""

import pytest
import shutil

from lms.lean.real import RealLeanVerifier


# Skip all tests if LEAN is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("lean") is None and not shutil.which("/home/stsosie/.elan/bin/lean"),
    reason="LEAN 4 not installed",
)


class TestRealLeanVerifier:
    """Tests for RealLeanVerifier using actual LEAN 4."""

    def test_initialization(self):
        """RealLeanVerifier can be instantiated."""
        verifier = RealLeanVerifier()
        assert verifier is not None

    @pytest.mark.asyncio
    async def test_verify_valid_theorem(self):
        """Verifier accepts valid theorem."""
        verifier = RealLeanVerifier()
        code = "theorem test : True := trivial"
        result = await verifier.verify(code)
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_verify_valid_arithmetic(self):
        """Verifier accepts valid arithmetic theorem."""
        verifier = RealLeanVerifier()
        # Note: 'lemma' requires Mathlib; use 'theorem' in base LEAN 4
        code = "theorem test_arith : 1 + 1 = 2 := rfl"
        result = await verifier.verify(code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_valid_def(self):
        """Verifier accepts valid definition."""
        verifier = RealLeanVerifier()
        code = "def double (n : Nat) : Nat := n + n"
        result = await verifier.verify(code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_type_error(self):
        """Verifier rejects type errors."""
        verifier = RealLeanVerifier()
        code = "theorem broken : False := trivial"
        result = await verifier.verify(code)
        assert result.success is False
        assert "type" in result.error.lower() or "mismatch" in result.error.lower()

    @pytest.mark.asyncio
    async def test_verify_syntax_error(self):
        """Verifier rejects syntax errors."""
        verifier = RealLeanVerifier()
        code = "theorem incomplete : True :="
        result = await verifier.verify(code)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verify_sorry_fails(self):
        """Verifier rejects proofs with sorry."""
        verifier = RealLeanVerifier()
        code = "theorem incomplete : True := sorry"
        result = await verifier.verify(code)
        # sorry compiles but we should detect and reject it
        assert result.success is False
        assert "sorry" in result.error.lower()

    @pytest.mark.asyncio
    async def test_verify_empty_code_fails(self):
        """Verifier rejects empty code."""
        verifier = RealLeanVerifier()
        result = await verifier.verify("")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verify_multiple_declarations(self):
        """Verifier handles multiple declarations."""
        verifier = RealLeanVerifier()
        code = """
def add_one (n : Nat) : Nat := n + 1

theorem add_one_succ (n : Nat) : add_one n = n + 1 := rfl
"""
        result = await verifier.verify(code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_result_includes_code(self):
        """Verification result includes original code."""
        verifier = RealLeanVerifier()
        code = "theorem test : True := trivial"
        result = await verifier.verify(code)
        assert result.code == code

    @pytest.mark.asyncio
    async def test_verify_with_mathlib_import_note(self):
        """Verifier handles code that might need imports gracefully."""
        verifier = RealLeanVerifier()
        # This uses Nat which is built-in, should work
        code = "theorem nat_zero : (0 : Nat) = 0 := rfl"
        result = await verifier.verify(code)
        assert result.success is True
