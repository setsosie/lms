"""Machine gates from `specs/faithfulness_protocol.md` §4.

Gate 2 (T4, axiom/sorry audit): `lms.gates.axioms`
Gate 3 (T2, non-vacuity): `lms.gates.vacuity`
Gate 4 (novelty N0/N1): `lms.gates.novelty` — kept out of this package's
namespace *and* out of `default_gate_runner`. It imports `lms.artifacts` (to
stamp the verdict on the artifact), and `lms.artifacts` imports this package
for `GateResult`, so re-exporting it here is a circular import. Import
`apply_novelty_gate` / `default_novelty_classifier` from `lms.gates.novelty`
directly. It is also synchronous and network-bound, so `Society` runs it off
the event loop rather than inside the gate runner.
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
