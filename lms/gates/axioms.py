"""Gate 2 (T4): axiom/sorry audit.

Static checks reject `sorry`, new `axiom` declarations and `native_decide` in
the source; the post-compile check runs `#print axioms` on every named
declaration and requires the axiom set to stay inside Lean's three standard
axioms. Numbering per `calibration-program.md` §2, definition per
`specs/faithfulness_protocol.md` §4 T4.
"""

import re
import time

from lms.gates.base import GateOutcome, GateResult, LeanProbeRunner
from lms.gates.lean_source import extract_declarations, strip_comments

__all__ = ["ALLOWED_AXIOMS", "AxiomGate", "AxiomProber", "ProbeError"]

#: The axioms all of Mathlib's classical mathematics rests on. Anything else
#: in a dependency cone means the artifact assumed what it claims to prove.
ALLOWED_AXIOMS: frozenset[str] = frozenset(
    {"propext", "Classical.choice", "Quot.sound"}
)

_SORRY_RE = re.compile(r"\bsorry\b")
_NATIVE_DECIDE_RE = re.compile(r"\bnative_decide\b")

# `'Foo.bar' depends on axioms: [propext, Classical.choice]`
_AXIOMS_RE = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]")
# `'Foo.bar' does not depend on any axioms`
_NO_AXIOMS_RE = re.compile(r"'([^']+)' does not depend on any axioms")


class ProbeError(Exception):
    """The axiom probe could not produce a usable answer."""


class AxiomProber:
    """Answers "which axioms does each declaration depend on?" via Lean."""

    def __init__(self, runner: LeanProbeRunner) -> None:
        self.runner = runner

    async def probe(self, code: str, full_names: list[str]) -> dict[str, list[str]]:
        """Compile `code` + `#print axioms` per name; parse the answers.

        Raises ProbeError when the probe fails to compile or an expected
        answer is missing — callers map that to INCONCLUSIVE.
        """
        probe_lines = "\n".join(f"#print axioms {name}" for name in full_names)
        probe_code = f"{code}\n\n{probe_lines}\n"
        try:
            returncode, stdout, stderr = await self.runner.run(probe_code)
        except (TimeoutError, OSError) as exc:
            raise ProbeError(f"probe did not run: {exc}") from exc

        if returncode != 0:
            message = (stderr.strip() or stdout.strip())[:500]
            raise ProbeError(f"probe failed to compile: {message}")

        found: dict[str, list[str]] = {}
        for name, axiom_list in _AXIOMS_RE.findall(stdout):
            axioms = [a.strip() for a in axiom_list.split(",") if a.strip()]
            found[name] = axioms
        for name in _NO_AXIOMS_RE.findall(stdout):
            found[name] = []

        missing = [n for n in full_names if n not in found]
        if missing:
            raise ProbeError(f"no `#print axioms` answer for {missing} in probe output")
        return found


class AxiomGate:
    """T4 in four sub-checks: sorry, axiom declarations, native_decide, audit."""

    name = "T4"

    def __init__(self, prober: AxiomProber | None = None) -> None:
        self.prober = prober

    async def check(self, code: str) -> list[GateResult]:
        results = [
            self._check_sorry(code),
            self._check_axiom_decl(code),
            self._check_native_decide(code),
        ]
        results.append(await self._check_axiom_audit(code))
        return results

    def _check_sorry(self, code: str) -> GateResult:
        stripped = strip_comments(code)
        match = _SORRY_RE.search(stripped)
        if match:
            return GateResult(
                gate="T4.sorry",
                outcome=GateOutcome.FAILED,
                reason="code contains `sorry` — incomplete proof",
            )
        return GateResult(
            gate="T4.sorry", outcome=GateOutcome.PASSED, reason="no `sorry`"
        )

    def _check_axiom_decl(self, code: str) -> GateResult:
        axioms = [d for d in extract_declarations(code) if d.keyword == "axiom"]
        if axioms:
            names = ", ".join(d.name or "<anonymous>" for d in axioms)
            return GateResult(
                gate="T4.axiom_decl",
                outcome=GateOutcome.FAILED,
                reason="agent-authored code declares new axiom(s)",
                detail=names,
            )
        return GateResult(
            gate="T4.axiom_decl",
            outcome=GateOutcome.PASSED,
            reason="no new axiom declarations",
        )

    def _check_native_decide(self, code: str) -> GateResult:
        if _NATIVE_DECIDE_RE.search(strip_comments(code)):
            return GateResult(
                gate="T4.native_decide",
                outcome=GateOutcome.FAILED,
                reason="`native_decide` trusts the compiler, not the kernel",
            )
        return GateResult(
            gate="T4.native_decide",
            outcome=GateOutcome.PASSED,
            reason="no `native_decide`",
        )

    async def _check_axiom_audit(self, code: str) -> GateResult:
        named = [
            d
            for d in extract_declarations(code)
            if d.full_name is not None and d.keyword != "axiom"
        ]
        if not named:
            return GateResult(
                gate="T4.axiom_audit",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="no named declarations to audit",
            )
        if self.prober is None:
            return GateResult(
                gate="T4.axiom_audit",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="no Lean toolchain available for `#print axioms` audit; "
                "routed to D4",
            )

        names = [d.full_name for d in named if d.full_name is not None]
        started = time.monotonic()
        try:
            axiom_sets = await self.prober.probe(code, names)
        except ProbeError as exc:
            return GateResult(
                gate="T4.axiom_audit",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="axiom audit could not run; routed to D4",
                detail=str(exc),
            )
        elapsed = time.monotonic() - started

        violations = {
            name: sorted(set(axioms) - ALLOWED_AXIOMS)
            for name, axioms in axiom_sets.items()
            if set(axioms) - ALLOWED_AXIOMS
        }
        if violations:
            listed = "; ".join(
                f"{name}: {', '.join(extra)}" for name, extra in violations.items()
            )
            return GateResult(
                gate="T4.axiom_audit",
                outcome=GateOutcome.FAILED,
                reason="declaration depends on non-standard axiom(s)",
                detail=f"{listed} (audit {elapsed:.1f}s)",
            )
        return GateResult(
            gate="T4.axiom_audit",
            outcome=GateOutcome.PASSED,
            reason="axiom sets within {propext, Classical.choice, Quot.sound}",
            detail=f"audited {len(names)} declaration(s) in {elapsed:.1f}s",
        )
