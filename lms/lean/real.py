"""Real LEAN 4 verifier using the actual LEAN compiler."""

import asyncio
import shutil
from pathlib import Path

from lms.lean.interface import LeanVerifier, VerificationResult
from lms.lean.project import LeanProject


class RealLeanVerifier(LeanVerifier):
    """Verifier that uses the actual LEAN 4 compiler.

    Runs LEAN 4 on the provided code and checks for compilation errors.
    Also detects and rejects use of 'sorry' (incomplete proofs).

    When a project_dir is provided, automatically rebuilds when new
    imports are detected to prevent 'object file does not exist' errors.
    """

    def __init__(
        self,
        lean_path: str | None = None,
        timeout: float = 30.0,
        project_dir: Path | str | None = None,
    ) -> None:
        """Initialize the real LEAN verifier.

        Args:
            lean_path: Path to LEAN executable. If None, searches PATH
                       and common installation locations.
            timeout: Maximum time in seconds to wait for LEAN.
            project_dir: Path to Lean project root. If provided, enables
                         automatic rebuilding when new imports are detected.
        """
        self.lean_path = lean_path or self._find_lean()
        self.timeout = timeout

        # Optional project manager for auto-rebuild
        self.project: LeanProject | None = None
        if project_dir:
            self.project = LeanProject(project_dir)

    def _find_lean(self) -> str:
        """Find the LEAN 4 executable.

        Returns:
            Path to LEAN executable

        Raises:
            FileNotFoundError: If LEAN is not found
        """
        # Check PATH first
        lean = shutil.which("lean")
        if lean:
            return lean

        # Check common elan installation
        elan_path = Path.home() / ".elan" / "bin" / "lean"
        if elan_path.exists():
            return str(elan_path)

        raise FileNotFoundError(
            "LEAN 4 not found. Install via elan: "
            "https://github.com/leanprover/elan"
        )

    async def verify(self, code: str) -> VerificationResult:
        """Verify LEAN code using the real compiler.

        Args:
            code: LEAN 4 code to verify

        Returns:
            VerificationResult with compilation status
        """
        # Empty code always fails
        if not code or not code.strip():
            return VerificationResult(
                success=False,
                code=code,
                error="Empty code provided",
            )

        # Check for sorry (incomplete proofs) - we reject these
        if "sorry" in code:
            return VerificationResult(
                success=False,
                code=code,
                error="Code contains 'sorry' - incomplete proof not allowed",
            )

        # Auto-rebuild if new imports detected (prevents .olean errors)
        if self.project:
            await self.project.ensure_built(code)
            temp_path = self.project.get_temp_file(code)
            temp_path.write_text(code)
        else:
            # Fallback to system temp
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".lean",
                delete=False,
            ) as f:
                f.write(code)
                temp_path = Path(f.name)

        try:
            # Run LEAN
            proc = await asyncio.create_subprocess_exec(
                self.lean_path,
                str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return VerificationResult(
                    success=False,
                    code=code,
                    error=f"LEAN verification timed out after {self.timeout}s",
                )

            # Check result
            if proc.returncode == 0:
                return VerificationResult(
                    success=True,
                    code=code,
                    error=None,
                )
            else:
                error_msg = stderr.decode("utf-8").strip()
                if not error_msg:
                    error_msg = stdout.decode("utf-8").strip()
                if not error_msg:
                    error_msg = f"LEAN returned exit code {proc.returncode}"

                return VerificationResult(
                    success=False,
                    code=code,
                    error=error_msg,
                )

        finally:
            # Clean up temp file (unless using project manager which handles cleanup)
            if not self.project:
                temp_path.unlink(missing_ok=True)
