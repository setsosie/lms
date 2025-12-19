"""Mock LEAN verifier for development and testing."""

import re
import random

from lms.lean.interface import LeanVerifier, VerificationResult


class MockLeanVerifier(LeanVerifier):
    """Mock LEAN verifier that simulates verification.

    Uses heuristics to determine if code looks valid:
    - Checks for proper LEAN 4 keywords (theorem, lemma, def, etc.)
    - Rejects code containing 'sorry' (incomplete proofs)
    - Rejects empty or obviously malformed code

    This is meant for rapid prototyping. Replace with RealLeanVerifier
    before running actual experiments.
    """

    # LEAN 4 keywords that indicate valid code structure
    VALID_KEYWORDS = re.compile(
        r"^\s*(theorem|lemma|def|axiom|example|structure|inductive|class)\s+",
        re.MULTILINE,
    )

    def __init__(
        self,
        success_rate: float = 1.0,
        always_check_syntax: bool = True,
    ) -> None:
        """Initialize mock verifier.

        Args:
            success_rate: Probability of accepting valid-looking code (0.0-1.0)
            always_check_syntax: If True, always check syntax even with random success
        """
        self.success_rate = success_rate
        self.always_check_syntax = always_check_syntax

    async def verify(self, code: str) -> VerificationResult:
        """Verify LEAN code using heuristics.

        Args:
            code: LEAN 4 code to verify

        Returns:
            VerificationResult based on heuristic checks
        """
        # Empty code always fails
        if not code or not code.strip():
            return VerificationResult(
                success=False,
                code=code,
                error="Empty code provided",
            )

        # Check for sorry (incomplete proof)
        if "sorry" in code.lower():
            return VerificationResult(
                success=False,
                code=code,
                error="Code contains 'sorry' - incomplete proof",
            )

        # Check for valid LEAN structure
        if not self.VALID_KEYWORDS.search(code):
            return VerificationResult(
                success=False,
                code=code,
                error="Code does not appear to be valid LEAN 4 syntax",
            )

        # If always_check_syntax is False and success_rate < 1.0, use random
        if not self.always_check_syntax and random.random() > self.success_rate:
            return VerificationResult(
                success=False,
                code=code,
                error="Random verification failure (mock mode)",
            )

        # If we get here with always_check_syntax=True, syntax was valid
        # Apply success rate for randomness if configured
        if self.always_check_syntax and self.success_rate < 1.0:
            if random.random() > self.success_rate:
                return VerificationResult(
                    success=False,
                    code=code,
                    error="Random verification failure (mock mode)",
                )

        return VerificationResult(
            success=True,
            code=code,
            error=None,
        )
