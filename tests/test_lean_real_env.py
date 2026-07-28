"""Tests for 26Q3-HARN-07: the verifier must invoke Lean inside the project env.

`RealLeanVerifier` ran bare `lean <file>` with no `LEAN_PATH` and no `cwd`, so it
could only check import-free Lean. Anything with `import Mathlib...` failed with
`unknown module prefix` — an environment error indistinguishable in the result
from a genuine proof failure. Every real agent proof has imports.

These tests mock the subprocess so they assert the *command vector*, and
therefore run on a machine with no Lean toolchain installed.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from lms.lean.real import RealLeanVerifier

IMPORTING_CODE = """import Mathlib.Logic.Basic
theorem t (p : Prop) : ¬¬p ↔ p := not_not"""


class _FakeProc:
    """Stands in for an asyncio subprocess that succeeded silently."""

    def __init__(self) -> None:
        self.returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"", b"")

    def kill(self) -> None:  # pragma: no cover - only on timeout
        pass


class _Recorder:
    """Captures every create_subprocess_exec call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *args, **kwargs) -> _FakeProc:
        self.calls.append((args, kwargs))
        return _FakeProc()

    @property
    def lean_invocation(self) -> tuple[tuple, dict]:
        """The call that actually runs Lean on the snippet.

        `ensure_built` may shell out to `lake build` first; the Lean run is the
        last call either way.
        """
        return self.calls[-1]


class TestVerifierUsesProjectEnvironment:
    async def test_uses_lake_env_when_project_configured(self, tmp_path):
        """With a project, Lean must run under `lake env` from the project dir."""
        verifier = RealLeanVerifier(lean_path="/fake/lean", project_dir=tmp_path)
        recorder = _Recorder()

        with patch("asyncio.create_subprocess_exec", new=recorder):
            await verifier.verify(IMPORTING_CODE)

        args, kwargs = recorder.lean_invocation
        # Compare the basename: `lake` may resolve to an absolute path under
        # $ELAN_HOME/bin rather than being found on PATH.
        assert Path(args[0]).name == "lake", f"expected lake, got {args[0]!r}"
        assert args[1:3] == ("env", "lean"), f"expected `lake env lean`, got {args[:3]}"
        assert args[3].endswith(".lean")
        assert kwargs.get("cwd") == tmp_path, (
            "lake env must run from the Lean project directory"
        )

    async def test_falls_back_to_bare_lean_without_project(self, tmp_path):
        """No project configured → unchanged behavior, so existing tests hold."""
        verifier = RealLeanVerifier(lean_path="/fake/lean")
        recorder = _Recorder()

        with patch("asyncio.create_subprocess_exec", new=recorder):
            await verifier.verify("theorem t : True := trivial")

        args, kwargs = recorder.lean_invocation
        assert args[0] == "/fake/lean"
        assert kwargs.get("cwd") is None

    async def test_failure_still_rejected_under_lake_env(self, tmp_path):
        """The fix must not turn the oracle into a yes-machine."""
        verifier = RealLeanVerifier(lean_path="/fake/lean", project_dir=tmp_path)

        class _FailingProc(_FakeProc):
            def __init__(self) -> None:
                self.returncode = 1

            async def communicate(self) -> tuple[bytes, bytes]:
                return (b"", b"error: type mismatch")

        async def fail(*args, **kwargs) -> _FailingProc:
            return _FailingProc()

        with patch("asyncio.create_subprocess_exec", new=fail):
            result = await verifier.verify(IMPORTING_CODE)

        assert result.success is False
        assert result.error is not None
        assert "type mismatch" in result.error


class TestLakeDiscovery:
    def test_lake_located_alongside_lean(self, tmp_path):
        """`lake` must be found the same way `lean` is, not assumed on PATH."""
        verifier = RealLeanVerifier(lean_path="/fake/lean", project_dir=tmp_path)
        assert hasattr(verifier, "lake_path")
        assert verifier.lake_path


@pytest.mark.parametrize("code", ["", "   "])
async def test_empty_code_still_rejected_before_any_subprocess(code, tmp_path):
    """Empty input must not reach the compiler at all."""
    verifier = RealLeanVerifier(lean_path="/fake/lean", project_dir=tmp_path)
    recorder = _Recorder()

    with patch("asyncio.create_subprocess_exec", new=recorder):
        result = await verifier.verify(code)

    assert result.success is False
    assert recorder.calls == []
