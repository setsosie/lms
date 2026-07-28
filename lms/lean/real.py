"""Real LEAN 4 verifier using the actual LEAN compiler."""

import asyncio
import os
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

        # With a project, Lean runs under `lake env` so that the package's
        # LEAN_PATH is exported; bare `lean` cannot resolve a single import.
        self.lake_path = self._find_lake()

    def _find_lean(self) -> str:
        """Find the LEAN 4 executable.

        Returns:
            Path to LEAN executable

        Raises:
            FileNotFoundError: If LEAN is not found
        """
        lean = shutil.which("lean")
        if lean:
            return lean

        for candidate in self._toolchain_candidates("lean"):
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "LEAN 4 not found. Install via elan: https://github.com/leanprover/elan"
        )

    @staticmethod
    def _toolchain_candidates(exe: str) -> list[Path]:
        """Elan bin locations to search, most specific first.

        `ELAN_HOME` is honored by elan at runtime but is undocumented, and on a
        cluster it routinely points somewhere other than `~/.elan` to keep
        multi-GB toolchains off the home quota.
        """
        roots = []
        elan_home = os.environ.get("ELAN_HOME")
        if elan_home:
            roots.append(Path(elan_home))
        roots.append(Path.home() / ".elan")
        return [root / "bin" / exe for root in roots]

    def _find_lake(self) -> str:
        """Find the `lake` executable.

        Unlike `_find_lean` this does not raise when absent: `lake` is only
        needed when a project is configured, and failing at construction time
        would break import-free verification on a machine without a toolchain.
        An unresolved name surfaces as a clear error at exec time instead.
        """
        lake = shutil.which("lake")
        if lake:
            return lake

        for candidate in self._toolchain_candidates("lake"):
            if candidate.exists():
                return str(candidate)

        return "lake"

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

        # `lake env lean` exports the package's LEAN_PATH so imports resolve.
        # Bare `lean` has no search path and fails on any import with
        # "unknown module prefix", which is indistinguishable in the result
        # from a genuine proof failure.
        if self.project:
            command: tuple[str, ...] = (
                self.lake_path,
                "env",
                "lean",
                str(temp_path),
            )
            cwd: Path | None = self.project.project_dir
        else:
            command = (self.lean_path, str(temp_path))
            cwd = None

        try:
            # Run LEAN
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                return VerificationResult(
                    success=False,
                    code=code,
                    error=(
                        f"Could not execute {command[0]!r}. A Lean toolchain "
                        "with `lake` on PATH (or under $ELAN_HOME/bin) is "
                        "required to verify code with imports."
                    ),
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
