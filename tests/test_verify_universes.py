"""The verifier must bind the same universes the foundation does (26Q3-HARN-21).

`FoundationFile.add_artifact` strips an entry's own `universe` lines because
the foundation header already binds `u v w` above its `namespace`. The verifier
wrapped candidates in the namespace *without* those universes, so it was
strictly stricter than the destination: code that would compile once stored was
rejected at verification.

Observed 2026-08-21 in the Sonnet-team simulation. A complete and correct
`IsPullback` plus a uniqueness-up-to-iso theorem failed on
`error: unknown universe level 'v'`, purely because the submission omitted a
`universe u v` line. Re-submitting the identical mathematics with that one line
added verified.
"""

from lms.foundation import (
    FOUNDATION_UNIVERSES,
    FoundationFile,
    declared_universe_names,
)
from lms.lean.real import RealLeanVerifier

# The shape that failed: uses `Category.{u,v}` without declaring `u v`.
UNDECLARED_UNIVERSES = """import LMS.Foundation

open LMS.Foundation

structure IsPullback (C : Category.{u,v}) (X Y Z P : C.Obj) : Prop where
  square : True
"""

SELF_DECLARED = """import LMS.Foundation

open LMS.Foundation

universe u v

structure IsPullback (C : Category.{u,v}) (X Y Z P : C.Obj) : Prop where
  square : True
"""


def wrap(code: str) -> str:
    return RealLeanVerifier._wrap_in_storage_namespace(code)


class TestDeclaredUniverseNames:
    def test_finds_declared_names(self):
        assert declared_universe_names(["universe u v"]) == {"u", "v"}

    def test_multiple_lines_accumulate(self):
        assert declared_universe_names(["universe u", "universe w"]) == {"u", "w"}

    def test_indented_declaration_counts(self):
        assert declared_universe_names(["  universe u"]) == {"u"}

    def test_no_declaration_is_empty(self):
        assert declared_universe_names(["def f := 1"]) == set()

    def test_does_not_match_identifier_prefix(self):
        """`universes` and `universe_foo` are not `universe` declarations."""
        assert declared_universe_names(["def universe_helper := 1"]) == set()


class TestWrapperBindsFoundationUniverses:
    def test_undeclared_universes_are_supplied(self):
        """The regression: the candidate never declared `u v`, so the wrapper
        must, or Lean reports `unknown universe level`."""
        wrapped = wrap(UNDECLARED_UNIVERSES)
        assert f"universe {' '.join(FOUNDATION_UNIVERSES)}" in wrapped

    def test_universe_line_precedes_namespace(self):
        """Lean binds universes at top level; inside the namespace is too late
        for the namespace header itself and reads as a different scope."""
        wrapped = wrap(UNDECLARED_UNIVERSES)
        assert wrapped.index("universe ") < wrapped.index("namespace LMS.Foundation")

    def test_imports_still_precede_universes(self):
        """Lean rejects an `import` that follows any other command."""
        wrapped = wrap(UNDECLARED_UNIVERSES)
        assert wrapped.index("import LMS.Foundation") < wrapped.index("universe ")

    def test_self_declared_names_are_not_duplicated(self):
        """Declaring `u` twice is `error: a universe level named 'u' has
        already been declared` -- the same bug pointing the other way."""
        wrapped = wrap(SELF_DECLARED)
        assert wrapped.count("universe u v\n") == 1
        # `w` is still missing from the candidate, so it is supplied alone.
        assert "universe w" in wrapped

    def test_candidate_declaring_everything_gets_no_extra_line(self):
        code = "universe u v w\n\ndef f : Type u := PUnit\n"
        wrapped = wrap(code)
        assert wrapped.count("universe") == 1

    def test_exotic_universe_names_are_preserved(self):
        """A candidate using a name outside the header keeps its own
        declaration; the header names are added alongside it."""
        code = "universe x\n\ndef f : Type x := PUnit\n"
        wrapped = wrap(code)
        assert "universe x" in wrapped
        assert f"universe {' '.join(FOUNDATION_UNIVERSES)}" in wrapped


class TestVerifierAndFoundationAgree:
    def test_header_declares_exactly_the_shared_constant(self):
        """The header and the wrapper must not drift -- that divergence is the
        whole defect. Both read `FOUNDATION_UNIVERSES`."""
        header = FoundationFile.FOUNDATION_HEADER
        assert f"universe {' '.join(FOUNDATION_UNIVERSES)}" in header

    def test_header_binds_universes_above_its_namespace(self):
        header = FoundationFile.FOUNDATION_HEADER
        assert header.index("universe ") < header.index("namespace LMS.Foundation")

    def test_every_header_universe_is_supplied_by_the_wrapper(self):
        """The property that makes verification and storage agree: anything the
        destination binds, the oracle binds too."""
        wrapped = wrap(UNDECLARED_UNIVERSES)
        supplied = declared_universe_names(wrapped.split("\n"))
        assert set(FOUNDATION_UNIVERSES) <= supplied
