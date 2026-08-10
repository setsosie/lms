"""Shared pytest fixtures and safety guards."""

import pytest

import lms.society


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
