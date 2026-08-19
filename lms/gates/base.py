"""Gate protocol and result types for the faithfulness-protocol machine gates.

Gate numbering follows `docs/planning/calibration-program.md` §2; the T-check
names follow `specs/faithfulness_protocol.md` §4. Each gate emits one
`GateResult` per sub-check (e.g. `T4.sorry`, `T2.named_declaration`) so the
per-run failure histogram has enough resolution to read its *shape* — the
distribution of failure reasons is a primary calibration output, not debug
output.

`INCONCLUSIVE` is a first-class outcome: T2's general form is undecidable, and
an audit that could not run must never be recorded as a pass. Inconclusive
results route to D4 human review.
"""

import asyncio
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from lms.lean.interface import LEAN_GRADE_KINDS, LeanVerifier

__all__ = [
    "Gate",
    "GateOutcome",
    "GateResult",
    "GateRunner",
    "LeanProbeRunner",
]


class GateOutcome(Enum):
    """Tri-state gate verdict.

    A boolean would force undecidable checks to lie in one direction or the
    other. `INCONCLUSIVE` is routed to D4 human review and is never counted
    as passing.
    """

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class GateResult:
    """Verdict of a single gate sub-check on one artifact."""

    gate: str
    outcome: GateOutcome
    reason: str
    detail: str | None = None

    @property
    def passed(self) -> bool:
        """Strict pass — `INCONCLUSIVE` does not count."""
        return self.outcome is GateOutcome.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "outcome": self.outcome.value,
            # Redundant with `outcome`, kept so a jq one-liner over
            # artifacts.json can filter on a boolean.
            "passed": self.passed,
            "reason": self.reason,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GateResult":
        return cls(
            gate=d["gate"],
            outcome=GateOutcome(d["outcome"]),
            reason=d.get("reason", ""),
            detail=d.get("detail"),
        )


class Gate(Protocol):
    """A machine gate: takes Lean source, returns one result per sub-check."""

    name: str

    async def check(self, code: str) -> list[GateResult]: ...


class GateRunner:
    """Runs a fixed sequence of gates over one artifact's code."""

    def __init__(self, gates: Sequence[Gate]) -> None:
        self.gates = list(gates)

    async def run(self, code: str) -> list[GateResult]:
        results: list[GateResult] = []
        for gate in self.gates:
            results.extend(await gate.check(code))
        return results


class LeanProbeRunner:
    """Runs a Lean snippet and captures its stdout.

    The verifier's `verify()` discards stdout, but `#print axioms` and witness
    probes answer *through* stdout, so gates need their own invocation. Reuses
    the configured verifier's toolchain paths and project so a probe compiles
    in exactly the environment the artifact was verified in.
    """

    def __init__(
        self,
        command_prefix: Sequence[str],
        cwd: Path | None,
        temp_dir: Path | None,
        timeout: float,
    ) -> None:
        self.command_prefix = tuple(command_prefix)
        self.cwd = cwd
        self.temp_dir = temp_dir
        self.timeout = timeout

    @classmethod
    def from_verifier(cls, verifier: LeanVerifier | None) -> "LeanProbeRunner | None":
        """Build a probe runner from a lean-grade verifier, else None.

        Duck-typed against `RealLeanVerifier`: a mock verifier (or a real one
        without a project, which could not resolve Mathlib imports anyway)
        yields None, and gates that need Lean report INCONCLUSIVE.
        """
        if verifier is None or verifier.verifier_kind not in LEAN_GRADE_KINDS:
            return None
        project = getattr(verifier, "project", None)
        lake_path = getattr(verifier, "lake_path", None)
        flags = getattr(verifier, "STRICTNESS_FLAGS", ())
        if project is None or lake_path is None:
            return None
        return cls(
            command_prefix=(lake_path, "env", "lean", *flags),
            cwd=project.project_dir,
            temp_dir=getattr(project, "temp_dir", None),
            timeout=getattr(verifier, "timeout", 30.0) * 2,
        )

    async def run(self, code: str) -> tuple[int, str, str]:
        """Compile `code`; return (returncode, stdout, stderr).

        Raises OSError/asyncio.TimeoutError to the caller — gates translate
        those into INCONCLUSIVE rather than swallowing them.
        """
        if self.temp_dir is not None:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            prefix="gate_probe_",
            dir=self.temp_dir,
            delete=False,
        ) as f:
            f.write(code)
            probe_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command_prefix,
                str(probe_path),
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except TimeoutError:
                proc.kill()
                raise
            return (
                proc.returncode if proc.returncode is not None else -1,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        finally:
            probe_path.unlink(missing_ok=True)
