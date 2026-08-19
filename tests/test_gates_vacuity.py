"""Tests for the T2 non-vacuity gate (26Q3-HARN-03, Gate 3)."""

from lms.artifacts import Artifact, ArtifactLibrary, ArtifactType
from lms.config import ProviderConfig
from lms.gates.base import GateOutcome, GateResult
from lms.gates.lean_source import extract_declarations, parse_theorem_signature
from lms.gates.vacuity import VacuityGate, build_witness_probe
from lms.lean.interface import (
    LeanVerifier,
    VerificationResult,
    VerificationStatus,
    VerifierKind,
)
from lms.metrics import gate_failure_histogram, gate_inconclusive_histogram
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.society import Society

# The December-run degenerate artifact from the card: typechecks, anonymous,
# introduces nothing, recorded as a formalization of "Definition: Category".
TRIVIAL_EXAMPLE = (
    "example {C : Type*} [CategoryTheory.Category C] (X : C) : X ⟶ X := 𝟙 X"
)


def result_for(results, gate):
    matching = [r for r in results if r.gate == gate]
    assert len(matching) == 1, f"expected exactly one {gate} result"
    return matching[0]


class FakeWitnessProber:
    def __init__(self, answer):
        self.answer = answer
        self.probes: list[str] = []

    async def probe(self, probe_code):
        self.probes.append(probe_code)
        return self.answer


class TestNamedDeclaration:
    async def test_trivial_example_rejected(self):
        """Regression required by the card: the degenerate `example` is worth
        zero and must fail the gate."""
        results = await VacuityGate().check(TRIVIAL_EXAMPLE)
        r = result_for(results, "T2.named_declaration")
        assert r.outcome is GateOutcome.FAILED

    async def test_named_theorem_passes(self):
        results = await VacuityGate().check("theorem foo : True := trivial")
        assert result_for(results, "T2.named_declaration").outcome is GateOutcome.PASSED

    async def test_axiom_only_submission_fails(self):
        # An axiom is named but asserts rather than proves; it cannot be the
        # named declaration that satisfies non-vacuity (T4 rejects it anyway).
        results = await VacuityGate().check("axiom foo : True")
        assert result_for(results, "T2.named_declaration").outcome is GateOutcome.FAILED


class TestDuplicateDelegation:
    async def test_unwired_checker_is_inconclusive(self):
        results = await VacuityGate().check("theorem foo : True := trivial")
        r = result_for(results, "T2.duplicate")
        assert r.outcome is GateOutcome.INCONCLUSIVE
        assert "26Q3-HARN-04" in r.reason

    async def test_injected_checker_result_passes_through(self):
        async def checker(code):
            return GateResult(
                gate="T2.duplicate",
                outcome=GateOutcome.FAILED,
                reason="alpha-equivalent to Mathlib.Foo.bar",
            )

        gate = VacuityGate(duplicate_checker=checker)
        results = await gate.check("theorem foo : True := trivial")
        assert result_for(results, "T2.duplicate").outcome is GateOutcome.FAILED


class TestHypothesisSatisfiability:
    async def test_no_theorems_passes(self):
        results = await VacuityGate().check("def x : Nat := 0")
        r = result_for(results, "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.PASSED

    async def test_hypothesis_free_theorem_passes(self):
        results = await VacuityGate().check("theorem t : 1 = 1 := rfl")
        r = result_for(results, "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.PASSED

    async def test_false_hypothesis_fails(self):
        code = "theorem t (h : False) : 1 = 2 := h.elim"
        r = result_for(await VacuityGate().check(code), "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.FAILED
        assert "False" in r.reason

    async def test_contradictory_pair_fails(self):
        code = "theorem t (h : 1 = 2) (h2 : ¬ 1 = 2) : False := h2 h"
        r = result_for(await VacuityGate().check(code), "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.FAILED

    async def test_hypotheses_without_prober_inconclusive(self):
        code = "theorem t (n : Nat) (h : n = 0) : n + 1 = 1 := by omega"
        r = result_for(await VacuityGate().check(code), "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.INCONCLUSIVE
        assert "t" in (r.detail or "")

    async def test_witness_found_passes(self):
        code = "theorem t (n : Nat) (h : n = 0) : n + 1 = 1 := by omega"
        gate = VacuityGate(witness_prober=FakeWitnessProber(True))
        r = result_for(await gate.check(code), "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.PASSED

    async def test_witness_not_found_is_inconclusive_not_failed(self):
        code = "theorem t (n : Nat) (h : n = 0) : n + 1 = 1 := by omega"
        gate = VacuityGate(witness_prober=FakeWitnessProber(None))
        r = result_for(await gate.check(code), "T2.hypothesis_satisfiability")
        assert r.outcome is GateOutcome.INCONCLUSIVE


class TestWitnessProbeConstruction:
    def test_existential_over_full_telescope(self):
        code = "theorem t {C : Type} [Inhabited C] (h : 1 = 1) : True := trivial"
        decl = next(d for d in extract_declarations(code) if d.name == "t")
        sig = parse_theorem_signature(code, decl)
        assert sig is not None
        probe = build_witness_probe(code, sig)
        assert probe is not None
        assert "example : ∃ (C : Type) (inst_1 : Inhabited C) (h : 1 = 1), True" in (
            probe
        )
        # The probe carries the original code so names stay in scope.
        assert probe.startswith(code)

    def test_default_value_binder_defeats_probe(self):
        code = "theorem t (n : Nat := 3) : n = n := rfl"
        decl = next(d for d in extract_declarations(code) if d.name == "t")
        sig = parse_theorem_signature(code, decl)
        assert sig is not None
        assert build_witness_probe(code, sig) is None


class StubLeanVerifier(LeanVerifier):
    """Accepts anything with Lean-grade provenance (no project → no prober)."""

    verifier_kind: VerifierKind = "real"

    async def verify(self, code: str) -> VerificationResult:
        return self._result(success=True, code=code)


class SilentProvider(BaseLLMProvider):
    name = "mock"

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        return GenerationResponse(
            content="No artifacts proposed.",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider=self.name,
        )


def make_artifact(code, status):
    return Artifact(
        id="a-1",
        type=ArtifactType.THEOREM,
        natural_language="test",
        created_by="agent-0",
        generation=1,
        lean_code=code,
        status=status,
    )


class TestSocietyWiring:
    async def test_gates_attach_to_verified_artifact(self, tmp_path):
        society = Society(
            n_agents=1,
            provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
            verifier=StubLeanVerifier(),
            foundation_path=tmp_path / "Foundation.lean",
        )
        artifact = make_artifact(
            "theorem foo : True := trivial", VerificationStatus.VERIFIED_LEAN
        )
        await society._apply_gates(artifact)
        gates_seen = {r.gate for r in artifact.gate_results}
        assert "T4.sorry" in gates_seen
        assert "T2.named_declaration" in gates_seen
        # No Lean project behind the stub verifier → audit must be
        # INCONCLUSIVE, never silently passing.
        audit = next(r for r in artifact.gate_results if r.gate == "T4.axiom_audit")
        assert audit.outcome is GateOutcome.INCONCLUSIVE
        assert not artifact.gates_passed

    async def test_failed_artifact_gets_no_gate_results(self, tmp_path):
        society = Society(
            n_agents=1,
            provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
            verifier=StubLeanVerifier(),
            foundation_path=tmp_path / "Foundation.lean",
        )
        artifact = make_artifact("nonsense", VerificationStatus.FAILED)
        await society._apply_gates(artifact)
        assert artifact.gate_results == []


class TestSerializationAndHistogram:
    def test_gate_results_round_trip(self, tmp_path):
        artifact = make_artifact(
            "example : True := trivial", VerificationStatus.VERIFIED_LEAN
        )
        artifact.gate_results = [
            GateResult(
                gate="T2.named_declaration",
                outcome=GateOutcome.FAILED,
                reason="example-only",
            ),
            GateResult(
                gate="T2.duplicate",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="not wired",
            ),
        ]
        library = ArtifactLibrary()
        library.add(artifact)
        path = tmp_path / "artifacts.json"
        library.save(path)

        loaded = ArtifactLibrary.load(path)
        results = loaded.get("a-1").gate_results
        assert [r.gate for r in results] == ["T2.named_declaration", "T2.duplicate"]
        assert results[0].outcome is GateOutcome.FAILED
        assert not loaded.get("a-1").gates_passed

        assert gate_failure_histogram(loaded) == {"T2.named_declaration": 1}
        assert gate_inconclusive_histogram(loaded) == {"T2.duplicate": 1}

    def test_gates_passed_requires_results(self):
        artifact = make_artifact(
            "theorem foo : True := trivial", VerificationStatus.VERIFIED_LEAN
        )
        assert not artifact.gates_passed  # gates never ran
        artifact.gate_results = [
            GateResult(gate="T4.sorry", outcome=GateOutcome.PASSED, reason="ok")
        ]
        assert artifact.gates_passed
