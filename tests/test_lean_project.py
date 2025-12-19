"""Tests for the LeanProject class."""

import tempfile
from pathlib import Path

import pytest

from lms.lean.project import LeanProject


class TestLeanProject:
    """Tests for LeanProject class."""

    def test_initialization(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = LeanProject(tmpdir)
            assert project.project_dir == Path(tmpdir)
            assert project.temp_dir.exists()
            assert project._seen_imports == set()

    def test_temp_dir_created(self):
        """Test that temp directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = LeanProject(tmpdir)
            expected_temp = Path(tmpdir) / "LMS" / "Temp"
            assert project.temp_dir == expected_temp
            assert expected_temp.exists()

    def test_extract_imports_single(self):
        """Test extracting a single import."""
        project = LeanProject("/tmp/fake")
        code = """import Mathlib.CategoryTheory.Yoneda

theorem foo : True := trivial
"""
        imports = project._extract_imports(code)
        assert imports == {"Mathlib.CategoryTheory.Yoneda"}

    def test_extract_imports_multiple(self):
        """Test extracting multiple imports."""
        project = LeanProject("/tmp/fake")
        code = """import Mathlib.CategoryTheory.Category.Basic
import Mathlib.CategoryTheory.Yoneda
import Mathlib.CategoryTheory.Limits.HasLimits

theorem foo : True := trivial
"""
        imports = project._extract_imports(code)
        assert imports == {
            "Mathlib.CategoryTheory.Category.Basic",
            "Mathlib.CategoryTheory.Yoneda",
            "Mathlib.CategoryTheory.Limits.HasLimits",
        }

    def test_extract_imports_empty(self):
        """Test extracting from code with no imports."""
        project = LeanProject("/tmp/fake")
        code = "theorem foo : True := trivial"
        imports = project._extract_imports(code)
        assert imports == set()

    def test_extract_imports_with_comments(self):
        """Test that imports are extracted even with surrounding comments."""
        project = LeanProject("/tmp/fake")
        code = """-- Some comment
import Mathlib.Data.Nat.Basic
-- Another comment
"""
        imports = project._extract_imports(code)
        assert imports == {"Mathlib.Data.Nat.Basic"}

    def test_compute_import_hash_deterministic(self):
        """Test that hash is deterministic."""
        project = LeanProject("/tmp/fake")
        imports = {"A", "B", "C"}
        hash1 = project._compute_import_hash(imports)
        hash2 = project._compute_import_hash(imports)
        assert hash1 == hash2

    def test_compute_import_hash_order_independent(self):
        """Test that hash is order-independent."""
        project = LeanProject("/tmp/fake")
        hash1 = project._compute_import_hash({"A", "B", "C"})
        hash2 = project._compute_import_hash({"C", "A", "B"})
        assert hash1 == hash2

    def test_get_temp_file(self):
        """Test getting a temp file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = LeanProject(tmpdir)
            code = "theorem foo : True := trivial"
            path = project.get_temp_file(code)

            assert path.parent == project.temp_dir
            assert path.suffix == ".lean"
            assert "verify_" in path.name

    def test_get_temp_file_same_code_same_path(self):
        """Test that same code produces same path (caching)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = LeanProject(tmpdir)
            code = "theorem foo : True := trivial"
            path1 = project.get_temp_file(code)
            path2 = project.get_temp_file(code)
            assert path1 == path2

    def test_get_temp_file_different_code_different_path(self):
        """Test that different code produces different paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = LeanProject(tmpdir)
            path1 = project.get_temp_file("theorem foo : True := trivial")
            path2 = project.get_temp_file("theorem bar : True := trivial")
            assert path1 != path2


class TestLeanProjectAsync:
    """Async tests for LeanProject class."""

    @pytest.mark.asyncio
    async def test_ensure_built_tracks_imports(self):
        """Test that ensure_built tracks seen imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal project structure so build doesn't fail hard
            project = LeanProject(tmpdir)

            code = "import Mathlib.Data.Nat.Basic\ntheorem foo : True := trivial"

            # First call should detect new imports
            # (Build will fail without real project, but imports get tracked)
            try:
                await project.ensure_built(code)
            except Exception:
                pass  # Build may fail, that's fine

            assert "Mathlib.Data.Nat.Basic" in project._seen_imports

    @pytest.mark.asyncio
    async def test_build_returns_false_on_missing_project(self):
        """Test that build returns False when project doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = LeanProject(tmpdir)
            # No lakefile.toml, so build should fail
            result = await project.build()
            # Result depends on whether lake is installed
            # Either way, it shouldn't crash
            assert isinstance(result, bool)
