"""Tests for the T4 axiom/sorry gate (26Q3-HARN-03, Gate 2)."""

import pytest

from lms.gates.axioms import ALLOWED_AXIOMS, AxiomGate, AxiomProber, ProbeError
from lms.gates.base import GateOutcome
from lms.gates.lean_source import extract_declarations, strip_comments


def result_for(results, gate):
    matching = [r for r in results if r.gate == gate]
    assert len(matching) == 1, f"expected exactly one {gate} result"
    return matching[0]


class FakeProber:
    """Stands in for AxiomProber with canned or raising behavior."""

    def __init__(self, axiom_sets=None, error=None):
        self.axiom_sets = axiom_sets or {}
        self.error = error
        self.calls: list[list[str]] = []

    async def probe(self, code, full_names):
        self.calls.append(full_names)
        if self.error is not None:
            raise self.error
        return {name: self.axiom_sets.get(name, []) for name in full_names}


class FakeRunner:
    """Stands in for LeanProbeRunner with a canned compile result."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.code_seen: str | None = None

    async def run(self, code):
        self.code_seen = code
        return (self.returncode, self.stdout, self.stderr)


class TestStripComments:
    def test_removes_line_comments(self):
        assert "sorry" not in strip_comments("theorem t : True := trivial -- sorry")

    def test_removes_nested_block_comments(self):
        code = "/- outer /- sorry -/ still comment -/ def x := 1"
        assert "sorry" not in strip_comments(code)
        assert "def x" in strip_comments(code)

    def test_preserves_line_count(self):
        code = "a\n/- b\nc -/\nd"
        assert strip_comments(code).count("\n") == code.count("\n")


class TestExtractDeclarations:
    def test_namespaced_full_names(self):
        code = (
            "namespace LMS.Foundation\n"
            "theorem foo : True := trivial\n"
            "end LMS.Foundation\n"
        )
        decls = extract_declarations(code)
        assert [d.full_name for d in decls] == ["LMS.Foundation.foo"]

    def test_example_is_anonymous(self):
        decls = extract_declarations("example : True := trivial")
        assert decls[0].keyword == "example"
        assert decls[0].anonymous

    def test_anonymous_instance(self):
        decls = extract_declarations("instance : Inhabited Nat := ⟨0⟩")
        assert decls[0].keyword == "instance"
        assert decls[0].name is None

    def test_modifiers_and_attributes(self):
        code = "@[simp]\nprivate noncomputable def LMS.helper : Nat := 0"
        decls = extract_declarations(code)
        assert decls[0].keyword == "def"
        assert decls[0].name == "LMS.helper"

    def test_axiom_declaration_found(self):
        decls = extract_declarations("axiom bad : False")
        assert decls[0].keyword == "axiom"
        assert decls[0].name == "bad"


class TestStaticChecks:
    async def test_sorry_rejected(self):
        results = await AxiomGate().check("theorem t : True := sorry")
        r = result_for(results, "T4.sorry")
        assert r.outcome is GateOutcome.FAILED

    async def test_sorry_in_identifier_not_flagged(self):
        results = await AxiomGate().check("def sorry_free : Nat := 0")
        assert result_for(results, "T4.sorry").outcome is GateOutcome.PASSED

    async def test_sorry_in_comment_not_flagged(self):
        results = await AxiomGate().check("-- sorry\ndef x : Nat := 0")
        assert result_for(results, "T4.sorry").outcome is GateOutcome.PASSED

    async def test_new_axiom_rejected_with_name(self):
        results = await AxiomGate().check("axiom convenient : 1 = 2")
        r = result_for(results, "T4.axiom_decl")
        assert r.outcome is GateOutcome.FAILED
        assert "convenient" in (r.detail or "")

    async def test_native_decide_rejected(self):
        code = "theorem t : 2 + 2 = 4 := by native_decide"
        r = result_for(await AxiomGate().check(code), "T4.native_decide")
        assert r.outcome is GateOutcome.FAILED

    async def test_clean_code_passes_static_checks(self):
        code = "theorem t : True := trivial"
        results = await AxiomGate().check(code)
        for gate in ("T4.sorry", "T4.axiom_decl", "T4.native_decide"):
            assert result_for(results, gate).outcome is GateOutcome.PASSED


class TestAxiomAudit:
    async def test_no_prober_is_inconclusive_not_passing(self):
        results = await AxiomGate(prober=None).check("theorem t : True := trivial")
        r = result_for(results, "T4.axiom_audit")
        assert r.outcome is GateOutcome.INCONCLUSIVE

    async def test_standard_axioms_pass(self):
        prober = FakeProber(axiom_sets={"t": sorted(ALLOWED_AXIOMS)})
        results = await AxiomGate(prober=prober).check("theorem t : True := trivial")
        assert result_for(results, "T4.axiom_audit").outcome is GateOutcome.PASSED
        assert prober.calls == [["t"]]

    async def test_offending_axiom_named_in_failure(self):
        prober = FakeProber(axiom_sets={"t": ["propext", "myBadAxiom"]})
        results = await AxiomGate(prober=prober).check("theorem t : True := trivial")
        r = result_for(results, "T4.axiom_audit")
        assert r.outcome is GateOutcome.FAILED
        assert "myBadAxiom" in (r.detail or "")

    async def test_probe_error_is_inconclusive(self):
        prober = FakeProber(error=ProbeError("boom"))
        results = await AxiomGate(prober=prober).check("theorem t : True := trivial")
        r = result_for(results, "T4.axiom_audit")
        assert r.outcome is GateOutcome.INCONCLUSIVE
        assert "boom" in (r.detail or "")

    async def test_no_named_declarations_is_inconclusive(self):
        prober = FakeProber()
        results = await AxiomGate(prober=prober).check("example : True := trivial")
        r = result_for(results, "T4.axiom_audit")
        assert r.outcome is GateOutcome.INCONCLUSIVE
        assert prober.calls == []  # nothing to probe

    async def test_audit_uses_namespace_qualified_names(self):
        prober = FakeProber(axiom_sets={"LMS.Foundation.foo": []})
        code = (
            "namespace LMS.Foundation\n"
            "theorem foo : True := trivial\n"
            "end LMS.Foundation\n"
        )
        results = await AxiomGate(prober=prober).check(code)
        assert result_for(results, "T4.axiom_audit").outcome is GateOutcome.PASSED
        assert prober.calls == [["LMS.Foundation.foo"]]


class TestAxiomProber:
    async def test_parses_depends_on_axioms(self):
        stdout = "'foo' depends on axioms: [propext, Classical.choice]\n"
        prober = AxiomProber(FakeRunner(stdout=stdout))
        assert await prober.probe("code", ["foo"]) == {
            "foo": ["propext", "Classical.choice"]
        }

    async def test_parses_no_axioms_form(self):
        prober = AxiomProber(FakeRunner(stdout="'foo' does not depend on any axioms"))
        assert await prober.probe("code", ["foo"]) == {"foo": []}

    async def test_appends_print_axioms_lines(self):
        runner = FakeRunner(stdout="'a.b' does not depend on any axioms")
        await AxiomProber(runner).probe("theorem x : True := trivial", ["a.b"])
        assert "#print axioms a.b" in (runner.code_seen or "")

    async def test_compile_failure_raises(self):
        prober = AxiomProber(FakeRunner(returncode=1, stderr="unknown constant"))
        with pytest.raises(ProbeError, match="unknown constant"):
            await prober.probe("code", ["foo"])

    async def test_missing_answer_raises(self):
        prober = AxiomProber(FakeRunner(stdout=""))
        with pytest.raises(ProbeError, match="foo"):
            await prober.probe("code", ["foo"])
