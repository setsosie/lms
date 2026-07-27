"""Tests for 26Q3-CHORE-01: the test environment itself.

The suite is only a trustworthy regression signal if `uv sync` installs the
plugins the config assumes. `asyncio_mode = "auto"` is set in pyproject.toml,
so pytest-asyncio must be in `[dependency-groups] dev` (which `uv sync`
installs), not `[project.optional-dependencies] dev` (which it does not).
"""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_test_deps_in_dependency_group(pyproject):
    """pytest and pytest-asyncio must be installed by plain `uv sync`."""
    dev_group = pyproject["dependency-groups"]["dev"]
    names = {dep.split(">")[0].split("=")[0].strip() for dep in dev_group}
    assert "pytest" in names
    assert "pytest-asyncio" in names


def test_no_duplicate_test_deps_in_optional_dependencies(pyproject):
    """One source of truth: test deps must not also sit in optional-dependencies."""
    optional_dev = (
        pyproject.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    )
    names = {dep.split(">")[0].split("=")[0].strip() for dep in optional_dev}
    assert "pytest" not in names
    assert "pytest-asyncio" not in names


async def test_async_tests_execute():
    """An async test must actually run, not fail as 'not natively supported'."""
    assert True
