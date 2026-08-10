"""Abstract interface for LEAN verification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, get_args

# Which machinery produced a result. "mock" is a regex; "real" and "mcp" both
# put the code through the Lean compiler.
VerifierKind = Literal["mock", "real", "mcp"]

VERIFIER_KINDS: tuple[str, ...] = get_args(VerifierKind)

# Kinds whose success actually means "Lean accepted this".
LEAN_GRADE_KINDS: frozenset[str] = frozenset({"real", "mcp"})


class VerificationStatus(Enum):
    """How thoroughly an artifact has been checked.

    A plain boolean cannot carry provenance, which is how a run verified by
    `MockLeanVerifier`'s regex came to be recorded as 92% verified and used to
    calibrate a multi-thousand-statement roadmap. Only `VERIFIED_LEAN` counts.
    """

    UNVERIFIED = "unverified"
    VERIFIED_HEURISTIC = "verified_heuristic"
    VERIFIED_LEAN = "verified_lean"
    FAILED = "failed"


@dataclass
class VerificationResult:
    """Result of verifying LEAN code.

    Attributes:
        success: Whether the code was verified successfully
        code: The original code that was verified
        verifier_kind: Which machinery produced this result
        verifier_id: Human-readable identity of the verifier instance
        error: Error message if verification failed, None otherwise
    """

    success: bool
    code: str
    verifier_kind: VerifierKind
    verifier_id: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.verifier_kind not in VERIFIER_KINDS:
            raise ValueError(
                f"unknown verifier_kind {self.verifier_kind!r}; "
                f"expected one of {sorted(VERIFIER_KINDS)}"
            )

    @property
    def status(self) -> VerificationStatus:
        """Derived status — the mock has no path to `VERIFIED_LEAN`.

        This is a property rather than a stored field so that no caller can
        construct a result whose status disagrees with its provenance.
        """
        if not self.success:
            return VerificationStatus.FAILED
        if self.verifier_kind in LEAN_GRADE_KINDS:
            return VerificationStatus.VERIFIED_LEAN
        return VerificationStatus.VERIFIED_HEURISTIC


class LeanVerifier(ABC):
    """Abstract base class for LEAN verification.

    Implementations can either mock verification for development
    or use the actual LEAN 4 proof assistant.

    Subclasses declare their `verifier_kind` so that every result they emit is
    stamped with its provenance; use `_result` to build results rather than
    constructing `VerificationResult` directly.
    """

    #: Set by each concrete verifier. Governs the status its results can reach.
    verifier_kind: VerifierKind

    @property
    def verifier_id(self) -> str:
        """Identity of this verifier, recorded alongside every artifact."""
        return type(self).__name__

    def _result(
        self,
        success: bool,
        code: str,
        error: str | None = None,
    ) -> VerificationResult:
        """Build a result stamped with this verifier's provenance."""
        return VerificationResult(
            success=success,
            code=code,
            error=error,
            verifier_kind=self.verifier_kind,
            verifier_id=self.verifier_id,
        )

    def toolchain_info(self) -> dict[str, str | None]:
        """Toolchain provenance for the experiment metadata block.

        Verifiers that invoke a real toolchain override this to report the
        versions the result actually depends on.
        """
        return {"lean_version": None, "mathlib_rev": None}

    @abstractmethod
    async def verify(self, code: str) -> VerificationResult:
        """Verify LEAN code.

        Args:
            code: LEAN 4 code to verify

        Returns:
            VerificationResult with success status and any errors
        """
        pass
