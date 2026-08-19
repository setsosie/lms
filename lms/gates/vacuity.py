"""Gate 3 (T2): non-vacuity.

Rejects artifacts that typecheck but assert nothing: anonymous `example`-only
submissions, and theorems whose hypotheses are contradictory (vacuously true).
The general problem is undecidable — this implements the tractable subset and
reports everything else as INCONCLUSIVE, which routes to D4 human review.

The Mathlib-duplicate sub-check is *delegated*: statement-level novelty search
belongs to Gate 4 (26Q3-HARN-04). Until a duplicate checker is injected, that
sub-check reports INCONCLUSIVE rather than pretending it looked.
"""

from collections.abc import Awaitable, Callable

from lms.gates.base import GateOutcome, GateResult, LeanProbeRunner
from lms.gates.lean_source import (
    BinderGroup,
    TheoremSignature,
    extract_declarations,
    parse_theorem_signature,
)

__all__ = ["VacuityGate", "WitnessProber", "build_witness_probe"]

#: Signature of an injected Mathlib-duplicate checker (26Q3-HARN-04):
#: takes Lean source, returns a finished GateResult for `T2.duplicate`.
DuplicateChecker = Callable[[str], Awaitable[GateResult]]

# Tactics tried, in order, to discharge the witness goal. Best-effort: a
# failure here proves nothing (the hypotheses may be satisfiable by a witness
# no tactic finds) and maps to INCONCLUSIVE, never to FAILED.
_WITNESS_TACTICS = "first | trivial | decide | simp | exact?"


def _binder_as_explicit(group: BinderGroup, index: int) -> str | None:
    """Render a binder group as an explicit `(names : type)`, if possible."""
    if ":=" in group.type:
        return None  # default value — not a plain telescope entry
    if group.names is None:
        if group.bracket == "[":
            # Anonymous instance binder: invent a name to ∃-bind it.
            return f"(inst_{index} : {group.type})"
        return None
    return f"({group.names} : {group.type})"


def build_witness_probe(code: str, sig: TheoremSignature) -> str | None:
    """Build `example : ∃ <telescope>, True` probing hypothesis satisfiability.

    ∃-binding the *entire* telescope — data, instances and hypotheses alike —
    asks whether the theorem's context is inhabited at all, which sidesteps
    the undecidable question of which binders are "the hypotheses". Returns
    None when any binder cannot be rendered explicitly.
    """
    if not sig.binders:
        return None
    rendered: list[str] = []
    for i, group in enumerate(sig.binders):
        explicit = _binder_as_explicit(group, i)
        if explicit is None:
            return None
        rendered.append(explicit)
    telescope = " ".join(rendered)
    return f"{code}\n\nexample : ∃ {telescope}, True := by {_WITNESS_TACTICS}\n"


class WitnessProber:
    """Attempts to compile the witness probe; success means satisfiable."""

    def __init__(self, runner: LeanProbeRunner) -> None:
        self.runner = runner

    async def probe(self, probe_code: str) -> bool | None:
        """True: witness found. None: probe failed or found nothing.

        Never returns False — an automatic tactic failing to find a witness
        is not evidence the hypotheses are contradictory.
        """
        try:
            returncode, _stdout, _stderr = await self.runner.run(probe_code)
        except (TimeoutError, OSError):
            return None
        return True if returncode == 0 else None


def _negation_pairs(types: list[str]) -> list[str]:
    """Hypothesis types P for which some other hypothesis is textually ¬P."""
    normalized = {t.replace(" ", "") for t in types}
    contradicted = []
    for t in types:
        bare = t.replace(" ", "")
        for neg in (f"¬{bare}", f"¬({bare})", f"{bare}→False", f"({bare})→False"):
            if neg in normalized:
                contradicted.append(t)
                break
    return contradicted


class VacuityGate:
    """T2 in three sub-checks: named declaration, duplicate, satisfiability."""

    name = "T2"

    def __init__(
        self,
        witness_prober: WitnessProber | None = None,
        duplicate_checker: DuplicateChecker | None = None,
    ) -> None:
        self.witness_prober = witness_prober
        self.duplicate_checker = duplicate_checker

    async def check(self, code: str) -> list[GateResult]:
        return [
            self._check_named_declaration(code),
            await self._check_duplicate(code),
            await self._check_hypothesis_satisfiability(code),
        ]

    def _check_named_declaration(self, code: str) -> GateResult:
        decls = extract_declarations(code)
        named = [d for d in decls if d.name is not None and d.keyword != "axiom"]
        if named:
            return GateResult(
                gate="T2.named_declaration",
                outcome=GateOutcome.PASSED,
                reason="introduces named declaration(s)",
                detail=", ".join(d.full_name or "" for d in named),
            )
        if any(d.keyword == "example" for d in decls):
            return GateResult(
                gate="T2.named_declaration",
                outcome=GateOutcome.FAILED,
                reason="`example`-only submission introduces no named "
                "declaration — worth zero under honest accounting",
            )
        return GateResult(
            gate="T2.named_declaration",
            outcome=GateOutcome.FAILED,
            reason="no named declaration found in code",
        )

    async def _check_duplicate(self, code: str) -> GateResult:
        if self.duplicate_checker is None:
            return GateResult(
                gate="T2.duplicate",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="Mathlib duplicate search delegated to Gate 4 "
                "(26Q3-HARN-04), not yet wired; routed to D4",
            )
        return await self.duplicate_checker(code)

    async def _check_hypothesis_satisfiability(self, code: str) -> GateResult:
        theorems = [
            d
            for d in extract_declarations(code)
            if d.keyword in ("theorem", "lemma") and d.name is not None
        ]
        if not theorems:
            return GateResult(
                gate="T2.hypothesis_satisfiability",
                outcome=GateOutcome.PASSED,
                reason="no theorem-form declarations to check",
            )

        unparsed: list[str] = []
        undecided: list[str] = []
        for decl in theorems:
            sig = parse_theorem_signature(code, decl)
            if sig is None:
                unparsed.append(decl.name or "<anonymous>")
                continue

            hyp_types = [g.type for g in sig.binders]
            if any(t.strip() == "False" for t in hyp_types):
                return GateResult(
                    gate="T2.hypothesis_satisfiability",
                    outcome=GateOutcome.FAILED,
                    reason=f"`{sig.name}` hypothesizes `False` — vacuously true",
                )
            contradicted = _negation_pairs(hyp_types)
            if contradicted:
                return GateResult(
                    gate="T2.hypothesis_satisfiability",
                    outcome=GateOutcome.FAILED,
                    reason=f"`{sig.name}` carries contradictory hypotheses",
                    detail=f"P and ¬P both hypothesized: {contradicted[0]}",
                )

            if not sig.binders:
                continue  # no hypotheses — nothing to be vacuous about

            probe = build_witness_probe(code, sig)
            if probe is None or self.witness_prober is None:
                undecided.append(sig.name)
                continue
            witnessed = await self.witness_prober.probe(probe)
            if witnessed is not True:
                undecided.append(sig.name)

        if unparsed:
            return GateResult(
                gate="T2.hypothesis_satisfiability",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="could not parse theorem signature(s); routed to D4",
                detail=", ".join(unparsed),
            )
        if undecided:
            return GateResult(
                gate="T2.hypothesis_satisfiability",
                outcome=GateOutcome.INCONCLUSIVE,
                reason="hypothesis satisfiability not decided (no witness "
                "found automatically); routed to D4",
                detail=", ".join(undecided),
            )
        return GateResult(
            gate="T2.hypothesis_satisfiability",
            outcome=GateOutcome.PASSED,
            reason="hypotheses satisfiable (or absent) for all theorems",
        )
