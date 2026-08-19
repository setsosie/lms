"""Machine gates from `specs/faithfulness_protocol.md` §4.

Gate 2 (T4, axiom/sorry audit): `lms.gates.axioms`
Gate 3 (T2, non-vacuity): `lms.gates.vacuity`
Gate 4 (novelty N0/N1): `lms.gates.novelty` — 26Q3-HARN-04, separate card.
"""

from lms.gates.axioms import AxiomGate, AxiomProber
from lms.gates.base import (
    Gate,
    GateOutcome,
    GateResult,
    GateRunner,
    LeanProbeRunner,
)
from lms.gates.vacuity import VacuityGate, WitnessProber
from lms.lean.interface import LeanVerifier

__all__ = [
    "AxiomGate",
    "AxiomProber",
    "Gate",
    "GateOutcome",
    "GateResult",
    "GateRunner",
    "LeanProbeRunner",
    "VacuityGate",
    "WitnessProber",
    "default_gate_runner",
]


def default_gate_runner(verifier: LeanVerifier | None) -> GateRunner:
    """The standard T4 + T2 gate stack, probing through `verifier`'s toolchain.

    With no lean-grade verifier (mock runs, no project), the Lean-backed
    sub-checks degrade to INCONCLUSIVE rather than silently passing.
    """
    runner = LeanProbeRunner.from_verifier(verifier)
    return GateRunner(
        [
            AxiomGate(prober=AxiomProber(runner) if runner else None),
            VacuityGate(witness_prober=WitnessProber(runner) if runner else None),
        ]
    )
