"""Abstract interface for LEAN verification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of verifying LEAN code.

    Attributes:
        success: Whether the code was verified successfully
        code: The original code that was verified
        error: Error message if verification failed, None otherwise
    """

    success: bool
    code: str
    error: str | None = None


class LeanVerifier(ABC):
    """Abstract base class for LEAN verification.

    Implementations can either mock verification for development
    or use the actual LEAN 4 proof assistant.
    """

    @abstractmethod
    async def verify(self, code: str) -> VerificationResult:
        """Verify LEAN code.

        Args:
            code: LEAN 4 code to verify

        Returns:
            VerificationResult with success status and any errors
        """
        pass
