"""Tests for real LEAN 4 verifier."""

import shutil
import subprocess
from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType, VerificationStatus
from lms.foundation import FoundationFile
from lms.lean.real import RealLeanVerifier
from lms.novelty.mathlib_search import ExactProbeBackend, StageOutcome
from tests._lean_env import LEAN_MISSING, LEAN_MISSING_REASON


# Skip all tests if LEAN is not installed
pytestmark = pytest.mark.skipif(LEAN_MISSING, reason=LEAN_MISSING_REASON)


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


class TestFoundationUniverses:
    """The saved foundation compiles when artifacts bind their own universes."""

    @staticmethod
    def _foundation(tmp_path: Path, *codes: str) -> FoundationFile:
        foundation = FoundationFile(tmp_path / "Foundation.lean")
        for number, code in enumerate(codes):
            foundation.add_artifact(
                Artifact(
                    id=f"artifact-{number}",
                    type=ArtifactType.DEFINITION,
                    natural_language="test",
                    lean_code=code,
                    created_by="agent-0",
                    generation=0,
                    status=VerificationStatus.VERIFIED_LEAN,
                )
            )
        return foundation

    @staticmethod
    def _errors(foundation: FoundationFile) -> str:
        """Compile the saved file; the header's library imports need no toolchain build."""
        foundation.save()
        kept = [
            line
            for line in foundation.path.read_text().splitlines()
            if not line.startswith("import ")
        ]
        foundation.path.write_text("\n".join(kept) + "\n")
        proc = subprocess.run(
            ["lean", str(foundation.path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.stdout + proc.stderr if proc.returncode else ""

    def test_an_exotic_universe_line_compiles(self, tmp_path: Path) -> None:
        foundation = self._foundation(
            tmp_path,
            "universe u₁ u₂\n\ndef pairUp (A : Type u₁) (B : Type u₂) : Type (max u₁ u₂) := A × B",
        )
        assert self._errors(foundation) == ""

    def test_two_artifacts_can_bind_the_same_name(self, tmp_path: Path) -> None:
        foundation = self._foundation(
            tmp_path,
            "universe u₁\n\ndef first (A : Type u₁) : A → A := fun a => a",
            "universe u₁\n\ndef second (A : Type u₁) : A → A := fun a => a",
        )
        assert self._errors(foundation) == ""

    def test_a_later_artifact_can_name_the_level_itself(self, tmp_path: Path) -> None:
        foundation = self._foundation(
            tmp_path,
            "universe u₁\n\ndef first (A : Type u₁) : A → A := fun a => a",
            "def second.{u₁} (A : Type u₁) : A → A := fun a => a",
        )
        assert self._errors(foundation) == ""


class TestExactProbeContext:
    """A probe sees what the statement's own context gives it, and no more.

    Core Lean has `exact?` and these goals need nothing from the library, so the
    probes run without their import. A wrong close here reads as a false N0.
    """

    @staticmethod
    def run_probe(tmp_path, statement: str) -> StageOutcome:
        source = ExactProbeBackend.probe_source(statement)
        assert source is not None
        path = tmp_path / "probe.lean"
        path.write_text(source.replace("import Mathlib\n", ""))
        proc = subprocess.run(
            [shutil.which("lean") or "lean", "-DautoImplicit=false", str(path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=120,
        )
        return ExactProbeBackend.read_output(proc.stdout + proc.stderr)

    def test_it_cannot_use_a_variable_the_statement_leaves_out(self, tmp_path):
        outcome = self.run_probe(
            tmp_path, "variable (p : Prop) (h : p)\n\ntheorem t : p := sorry\n"
        )
        assert outcome.available is True
        assert outcome.closed_by is None

    def test_it_can_use_a_variable_the_statement_includes(self, tmp_path):
        outcome = self.run_probe(
            tmp_path,
            "variable (p : Prop) (h : p)\ninclude h\n\ntheorem t : p := sorry\n",
        )
        assert outcome.closed_by is not None

    def test_it_cannot_close_a_goal_by_calling_itself(self, tmp_path):
        outcome = self.run_probe(tmp_path, "theorem t (n : Nat) : n = n + 1 := sorry\n")
        assert outcome.available is True
        assert outcome.closed_by is None

    def test_a_statement_of_data_type_cannot_read_as_closed(self, tmp_path):
        # `exact n` would show only that some term exists, not that this one does.
        outcome = self.run_probe(tmp_path, "theorem t (n : Nat) : Nat := sorry\n")
        assert outcome.available is False
        assert outcome.closed_by is None

    def test_another_commands_close_is_not_the_probes(self, tmp_path):
        outcome = self.run_probe(
            tmp_path,
            "open Nat in #check (by exact? : 1 = 1)\n\n"
            "theorem foo : ∀ n : Nat, n = 0 := sorry\n",
        )
        assert outcome.available is False
        assert outcome.closed_by is None

    def test_an_explicit_universe_list_elaborates(self, tmp_path):
        outcome = self.run_probe(
            tmp_path, "theorem t.{u} (α : Type u) (a : α) : a = a := sorry\n"
        )
        assert outcome.closed_by is not None

    def test_scopes_closed_before_the_statement_still_parse(self, tmp_path):
        outcome = self.run_probe(
            tmp_path,
            "noncomputable section\nend\n\nsection A\nend A\n\n"
            "theorem t : True := sorry\n",
        )
        assert outcome.available is True
        assert outcome.closed_by is not None
