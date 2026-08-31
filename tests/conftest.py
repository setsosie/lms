"""Shared pytest fixtures and safety guards."""

import subprocess
from pathlib import Path

import pytest

import lms.society


@pytest.fixture(autouse=True, scope="session")
def guard_tracked_lean_corpus():
    """Fail the run loudly if the suite dirties the tracked `lean/` corpus.

    Issue #19: tests once overwrote the committed WC-3 corpus silently, and
    the damage was invisible unless someone checked `git status` for paths
    unrelated to their change. `isolate_default_foundation` below makes that
    specific write impossible; this guard catches any future regression by a
    different path. Skips quietly where git is unavailable (a source tarball,
    a stripped CI image) — the guard is belt-and-braces, not a dependency.
    """
    repo_root = Path(__file__).resolve().parent.parent

    def snapshot() -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain", "lean/"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout if proc.returncode == 0 else None

    before = snapshot()
    yield
    after = snapshot()
    if before is not None and after is not None and after != before:
        raise AssertionError(
            "The test suite modified the tracked Lean corpus under lean/ "
            f"(git status delta:\n{after}). Some test is writing to the real "
            "repository instead of tmp_path — see issue #19."
        )


@pytest.fixture(autouse=True)
def isolate_default_foundation(tmp_path, monkeypatch):
    """Keep the test suite out of the tracked Lean corpus.

    `Society` writes accumulated definitions to `DEFAULT_FOUNDATION_PATH` when
    no `foundation_path` is passed. That default points at the real
    `lean/LMS/Foundation.lean`, so running the suite used to overwrite the
    committed WC-3 category-theory corpus — replacing the Yoneda development
    with whatever a mock agent produced that run, on every `pytest` invocation.

    Redirecting the default per-test makes the damage impossible rather than
    merely unlikely, and costs nothing for tests that pass an explicit path.
    """
    monkeypatch.setattr(
        lms.society,
        "DEFAULT_FOUNDATION_PATH",
        tmp_path / "default-foundation" / "LMS" / "Foundation.lean",
    )
