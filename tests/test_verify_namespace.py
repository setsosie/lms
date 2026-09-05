"""Tests for 26Q3-HARN-13: verify an artifact in the namespace it is stored in.

The oracle used to elaborate candidates at top level while the foundation
stored them inside `namespace LMS.Foundation` — so a declaration named after
anything Lean core binds (`Functor`, `Option`, `Prod`, ...) failed verification
for a collision that does not exist at the destination.
"""

import shutil

import pytest

from lms.foundation import FOUNDATION_NAMESPACE, FoundationFile, split_imports
from lms.lean.real import RealLeanVerifier

_LEAN_MISSING = shutil.which("lean") is None and not shutil.which(
    "/home/stsosie/.elan/bin/lean"
)


class TestWrapShape:
    """AC-2 / AC-4: wrapper structure, no Lean toolchain required."""

    def test_imports_hoisted_above_namespace(self):
        """AC-2: import lines stay above the namespace line."""
        code = (
            "import Mathlib.Logic.Basic\n"
            "import Mathlib.Tactic.Common\n"
            "\n"
            "theorem foo : True := trivial\n"
        )
        wrapped = RealLeanVerifier._wrap_in_storage_namespace(code)
        lines = wrapped.split("\n")
        ns_index = lines.index(f"namespace {FOUNDATION_NAMESPACE}")
        import_indices = [
            i for i, line in enumerate(lines) if line.strip().startswith("import ")
        ]
        assert import_indices, "imports must be preserved"
        assert all(i < ns_index for i in import_indices)

    def test_wrapper_opens_and_closes_storage_namespace(self):
        code = "def d : Nat := 1"
        wrapped = RealLeanVerifier._wrap_in_storage_namespace(code)
        lines = wrapped.split("\n")
        assert f"namespace {FOUNDATION_NAMESPACE}" in lines
        assert f"end {FOUNDATION_NAMESPACE}" in lines
        assert lines.index(f"namespace {FOUNDATION_NAMESPACE}") < lines.index(
            "def d : Nat := 1"
        )
        assert lines.index("def d : Nat := 1") < lines.index(
            f"end {FOUNDATION_NAMESPACE}"
        )

    def test_no_imports_means_no_leading_blank_block(self):
        """No imports must not leave an empty line at the top of the file.

        The wrapper now also binds the foundation's universes (26Q3-HARN-21),
        so the namespace is no longer the first line -- but the invariant this
        test exists for, that an absent import block leaves no blank gap, is
        unchanged.
        """
        wrapped = RealLeanVerifier._wrap_in_storage_namespace("def d : Nat := 1")
        assert not wrapped.startswith("\n")
        lines = wrapped.split("\n")
        assert lines[0].startswith("universe ")
        assert lines.index(f"namespace {FOUNDATION_NAMESPACE}") < lines.index(
            "def d : Nat := 1"
        )

    def test_verifier_and_store_share_one_constant(self):
        """AC-4: header, footer, and verifier all derive from one constant."""
        assert FoundationFile.NAMESPACE is FOUNDATION_NAMESPACE
        assert f"namespace {FOUNDATION_NAMESPACE}" in FoundationFile.FOUNDATION_HEADER
        assert f"end {FOUNDATION_NAMESPACE}" in FoundationFile.FOUNDATION_FOOTER
        wrapped = RealLeanVerifier._wrap_in_storage_namespace("def d : Nat := 1")
        assert f"namespace {FOUNDATION_NAMESPACE}" in wrapped

    def test_split_imports_partitions_and_preserves_order(self):
        code = "import A\ndef x : Nat := 1\nimport B\ndef y : Nat := 2"
        imports, body = split_imports(code)
        assert imports == ["import A", "import B"]
        assert body == ["def x : Nat := 1", "def y : Nat := 2"]


@pytest.mark.skipif(_LEAN_MISSING, reason="LEAN 4 not installed")
class TestVerifyInStorageNamespace:
    """AC-1 / AC-3 / AC-6 against the real compiler (core-only, no Mathlib)."""

    @pytest.mark.asyncio
    async def test_core_name_collision_verifies(self):
        """AC-1: `structure Functor` — bound by Lean core at top level —
        verifies, because it elaborates as `LMS.Foundation.Functor`,
        which is where an accepted artifact actually lands."""
        verifier = RealLeanVerifier()
        code = "structure Functor where\n  obj : Nat -> Nat"
        result = await verifier.verify(code)
        assert result.success is True, result.error

    @pytest.mark.asyncio
    async def test_own_namespace_still_verifies(self):
        """AC-3: code carrying its own namespace nests legally."""
        verifier = RealLeanVerifier()
        code = "namespace Foo\ndef bar : Nat := 1\nend Foo"
        result = await verifier.verify(code)
        assert result.success is True, result.error

    @pytest.mark.asyncio
    async def test_plain_theorem_still_verifies(self):
        """AC-6 spot check: previously-verifying code keeps verifying."""
        verifier = RealLeanVerifier()
        result = await verifier.verify("theorem test : True := trivial")
        assert result.success is True, result.error

    @pytest.mark.asyncio
    async def test_genuine_error_still_fails(self):
        """The wrap must not swallow real failures."""
        verifier = RealLeanVerifier()
        result = await verifier.verify("theorem broken : False := trivial")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_result_reports_original_code(self):
        """The wrapper is verification plumbing; results carry the
        agent's code verbatim, which is what the foundation strips/stores."""
        verifier = RealLeanVerifier()
        code = "def unwrapped_probe : Nat := 7"
        result = await verifier.verify(code)
        assert result.code == code
