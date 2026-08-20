"""Tests for `lean:` payload extraction (26Q3-HARN-02).

Every archived payload in `experiments/stacks_ch4_phase1/artifacts.json` begins
with the literal characters `"|\\n  "` -- a YAML block-scalar marker captured as
source code. Under the real verifier those artifacts fail on line 1 for a reason
that has nothing to do with the mathematics, so both the failure counts and the
success counts from those runs measure the parser.

The canary at the bottom is the load-bearing test: it asserts the property
directly rather than checking any particular input, so a future regex change
that reintroduces packaging fails here even for inputs nobody thought to add.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lms.agent import Agent, _clean_lean_code

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "malformed_lean"

PACKAGING_PREFIXES = ("|", ">", "```")


def _fixture_payloads() -> list[tuple[str, str]]:
    """Real `lean_code` values recorded by the December runs."""
    return sorted((p.name, p.read_text()) for p in FIXTURE_DIR.glob("*.txt"))


# --- Table-driven cases -------------------------------------------------


@pytest.mark.parametrize(
    ("label", "raw", "expected"),
    [
        (
            "yaml_block_scalar",
            "|\n  import Mathlib\n  theorem t : True := trivial",
            "import Mathlib\ntheorem t : True := trivial",
        ),
        (
            "block_scalar_strip_chomp",
            "|-\n  theorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "block_scalar_keep_chomp",
            "|+\n  theorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "block_scalar_indent_indicator",
            "|2\n  theorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "block_scalar_indent_and_chomp",
            "|2-\n  theorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "folded_scalar",
            ">\n  theorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "bare_fence",
            "```\ntheorem t : True := trivial\n```",
            "theorem t : True := trivial",
        ),
        (
            "lean_tagged_fence",
            "```lean\ntheorem t : True := trivial\n```",
            "theorem t : True := trivial",
        ),
        (
            "lean4_tagged_fence",
            "```lean4\ntheorem t : True := trivial\n```",
            "theorem t : True := trivial",
        ),
        (
            "block_scalar_wrapping_a_fence",
            "|\n  ```lean\n  theorem t : True := trivial\n  ```",
            "theorem t : True := trivial",
        ),
        (
            "fence_with_trailing_prose",
            "```lean\ntheorem t : True := trivial\n```\nThis proves it.",
            "theorem t : True := trivial",
        ),
        (
            "unclosed_fence_is_still_usable",
            "```lean\ntheorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "clean_code_passes_through",
            "theorem t : True := trivial",
            "theorem t : True := trivial",
        ),
        (
            "blank_lines_inside_block_survive",
            "|\n  import Mathlib\n  \n  theorem t : True := trivial",
            "import Mathlib\n\ntheorem t : True := trivial",
        ),
        (
            "interior_indentation_is_relative",
            "|\n  theorem t : True := by\n    trivial",
            "theorem t : True := by\n  trivial",
        ),
        ("packaging_only_yields_none", "|\n  ", None),
        ("empty_fence_yields_none", "```lean\n```", None),
        ("none_passes_through", None, None),
    ],
)
def test_clean_lean_code(label: str, raw: str | None, expected: str | None) -> None:
    assert _clean_lean_code(raw) == expected


def test_crlf_is_normalized() -> None:
    """Windows line endings must not survive into Lean source."""
    assert _clean_lean_code("|\r\n  theorem t : True := trivial\r\n") == (
        "theorem t : True := trivial"
    )


# --- Real recorded payloads ---------------------------------------------


def test_fixtures_are_present() -> None:
    """Guard against the fixture directory silently emptying."""
    assert len(_fixture_payloads()) >= 5


@pytest.mark.parametrize(("name", "raw"), _fixture_payloads())
def test_recorded_payloads_lose_their_packaging(name: str, raw: str) -> None:
    """Each December payload starts as packaging and must not stay that way."""
    assert raw.startswith("|"), f"{name} was expected to carry the block-scalar bug"

    cleaned = _clean_lean_code(raw)

    assert cleaned is not None
    assert not cleaned.startswith(PACKAGING_PREFIXES)
    # Dedent must flatten the block, not merely shift it.
    assert not cleaned.startswith(" ")


def test_recorded_payload_recovers_its_import() -> None:
    """The most common December payload should begin at its import line."""
    raw = (FIXTURE_DIR / "definition-category_def-c68ecde2.txt").read_text()

    cleaned = _clean_lean_code(raw)

    assert cleaned is not None
    assert cleaned.startswith("import Mathlib.CategoryTheory.Category.Basic")


# --- Canary -------------------------------------------------------------


def _artifact_block(lean_payload: str) -> str:
    return (
        "<artifact>\n"
        "type: theorem\n"
        "name: t\n"
        "description: a description\n"
        f"lean: {lean_payload}\n"
        "references: []\n"
        "</artifact>"
    )


@pytest.mark.parametrize(("name", "raw"), _fixture_payloads())
def test_canary_no_extracted_lean_code_starts_with_packaging(
    name: str, raw: str
) -> None:
    """End-to-end: nothing the parser emits may begin with packaging.

    This runs through `_parse_artifacts` rather than calling the cleaner
    directly, so it also covers the wiring between the two.
    """
    agent = Agent(id="canary", provider=None)  # type: ignore[arg-type]

    proposed, _ = agent._parse_artifacts(_artifact_block(raw))

    assert len(proposed) == 1
    lean_code = proposed[0].lean_code
    assert lean_code is not None
    assert not lean_code.startswith(PACKAGING_PREFIXES)


def test_raw_capture_is_retained_for_diagnosis() -> None:
    """`lean_code_raw` keeps extraction bugs visible in the record."""
    agent = Agent(id="canary", provider=None)  # type: ignore[arg-type]

    proposed, _ = agent._parse_artifacts(
        _artifact_block("|\n  theorem t : True := trivial")
    )

    assert proposed[0].lean_code == "theorem t : True := trivial"
    assert proposed[0].lean_code_raw is not None
    assert proposed[0].lean_code_raw.startswith("|")


class TestFoundationImportRewrite:
    """`import LMS.Foundation.X` names a module that cannot exist — the
    foundation is one file — so Lean dies at line 1 with "object file does
    not exist" before the code's content is judged (committee_yolo_a). The
    cleaner rewrites such imports to the umbrella module."""

    def test_submodule_import_is_rewritten(self) -> None:
        cleaned = _clean_lean_code(
            "import LMS.Foundation.Category\n\nstructure F where\n  x : Nat"
        )
        assert cleaned is not None
        assert cleaned.splitlines()[0] == "import LMS.Foundation"
        assert "LMS.Foundation.Category" not in cleaned

    def test_rewrite_dedupes_against_existing_umbrella(self) -> None:
        cleaned = _clean_lean_code(
            "import LMS.Foundation\nimport LMS.Foundation.Category\n\ndef f : Nat := 1"
        )
        assert cleaned is not None
        assert cleaned.count("import LMS.Foundation") == 1

    def test_qualified_identifier_in_body_is_untouched(self) -> None:
        code = "import LMS.Foundation\n\ndef f (c : LMS.Foundation.Category) := c"
        assert _clean_lean_code(code) == code

    def test_mathlib_imports_are_not_rewritten(self) -> None:
        code = "import Mathlib.Data.Equiv.Basic\n\ndef f : Nat := 1"
        assert _clean_lean_code(code) == code
