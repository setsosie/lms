"""MCP-based LEAN verifier using the lean-lsp MCP server.

This verifier uses the MCP lean-lsp tools for verification, providing
richer error messages and integration with the Lean project.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from lms.lean.interface import LeanVerifier, VerificationResult
from lms.lean.project import LeanProject


class MCPLeanVerifier(LeanVerifier):
    """Verifier that uses MCP lean-lsp tools.

    This verifier writes code to a file in the Lean project and uses
    the lean_diagnostic_messages tool to check for errors.

    When new imports are detected, automatically rebuilds the project
    to ensure .olean files are available.

    Note: This requires the lean-lsp MCP server to be running and
    a valid Lean project at the specified path.
    """

    def __init__(
        self,
        project_path: Path | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the MCP LEAN verifier.

        Args:
            project_path: Path to Lean project. If None, uses ./lean
            timeout: Maximum time to wait for verification
        """
        # Always use absolute paths to avoid cwd issues
        self.project_path = (project_path or Path("lean")).resolve()
        self.timeout = timeout
        self._temp_file: Path | None = None
        self._mcp_available = False
        # Project manager for auto-rebuild on new imports
        self.project = LeanProject(self.project_path)

    async def _check_mcp_available(self) -> bool:
        """Check if MCP lean-lsp is available."""
        # This will be called from the agent context where MCP tools are available
        # For now, we'll assume it's available if we're using this verifier
        return True

    async def verify(self, code: str) -> VerificationResult:
        """Verify LEAN code using MCP tools.

        Writes the code to a temp file in the Lean project and uses
        lean_diagnostic_messages to check for errors.

        Args:
            code: LEAN 4 code to verify

        Returns:
            VerificationResult with verification status
        """
        # Empty code always fails
        if not code or not code.strip():
            return VerificationResult(
                success=False,
                code=code,
                error="Empty code provided",
            )

        # Check for sorry (incomplete proofs)
        if "sorry" in code:
            return VerificationResult(
                success=False,
                code=code,
                error="Code contains 'sorry' - incomplete proof not allowed",
            )

        # Clean up leading pipe characters (YAML block scalar artifact)
        cleaned_code = code
        if cleaned_code.startswith("|"):
            cleaned_code = cleaned_code[1:].strip()

        # Add Foundation import if Foundation exists (ensures compatibility)
        # Must handle import ordering carefully - all imports must be at the top
        foundation_path = self.project_path / "LMS" / "Foundation.lean"
        if foundation_path.exists() and "import LMS.Foundation" not in cleaned_code:
            cleaned_code = self._merge_foundation_import(cleaned_code)

        # Ensure imports are present for category theory (only if no Foundation)
        elif "CategoryTheory" in cleaned_code and "import" not in cleaned_code:
            cleaned_code = "import Mathlib.CategoryTheory.Category.Basic\n\n" + cleaned_code

        # Auto-rebuild if new imports detected (prevents .olean errors)
        await self.project.ensure_built(cleaned_code)

        # Write to temp file in the Lean project
        temp_dir = self.project_path / "LMS" / "Temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique temp file
        import uuid
        temp_file = temp_dir / f"verify_{uuid.uuid4().hex[:8]}.lean"

        try:
            temp_file.write_text(cleaned_code)
            self._temp_file = temp_file

            # Use absolute path for verification
            abs_temp_file = temp_file.resolve()
            result = await self._verify_with_lean(cleaned_code, abs_temp_file)
            return result

        finally:
            # Clean up
            if temp_file.exists():
                temp_file.unlink()

    def _merge_foundation_import(self, code: str) -> str:
        """Merge Foundation import with existing imports in code.

        In Lean 4, ALL imports must be at the very top of the file.
        This method extracts any imports from the code, combines them
        with the Foundation import, and reassembles properly.

        Args:
            code: The original LEAN code

        Returns:
            Code with Foundation import properly merged
        """
        lines = code.split("\n")
        imports = ["import LMS.Foundation"]
        other_lines = []
        in_imports = True

        for line in lines:
            stripped = line.strip()
            # Collect import statements
            if stripped.startswith("import "):
                if stripped != "import LMS.Foundation":
                    imports.append(stripped)
            # Stop collecting imports after first non-import, non-empty, non-comment line
            elif stripped and not stripped.startswith("--") and not stripped.startswith("/-"):
                in_imports = False
                other_lines.append(line)
            elif not in_imports:
                other_lines.append(line)
            # Skip empty lines and comments at the top (before first real code)
            elif in_imports and (not stripped or stripped.startswith("--") or stripped.startswith("/-")):
                # Keep comments that might be module docs
                if stripped.startswith("/-"):
                    other_lines.append(line)
                    in_imports = False

        # Remove duplicate imports
        unique_imports = list(dict.fromkeys(imports))

        # Reassemble: imports first, then open Foundation, then rest of code
        result_lines = unique_imports + ["", "open LMS.Foundation", ""] + other_lines

        return "\n".join(result_lines)

    async def _verify_with_lean(self, code: str, file_path: Path) -> VerificationResult:
        """Verify code using the Lean compiler directly.

        This is a fallback when MCP is not available in the current context.
        Uses subprocess to run lake build or lean directly.

        Args:
            code: The LEAN code
            file_path: Path to the temp file

        Returns:
            VerificationResult
        """
        try:
            # Try using lake env lean for verification
            proc = await asyncio.create_subprocess_exec(
                "lake", "env", "lean", str(file_path),
                cwd=self.project_path,
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
                    error=f"Verification timed out after {self.timeout}s",
                )

            # Parse output for errors
            output = stderr.decode("utf-8") + stdout.decode("utf-8")

            # Check for errors
            if proc.returncode != 0 or "error:" in output.lower():
                # Extract relevant error messages
                error_lines = []
                for line in output.split("\n"):
                    if "error:" in line.lower() or "Error:" in line:
                        error_lines.append(line.strip())

                error_msg = "\n".join(error_lines) if error_lines else output[:500]
                return VerificationResult(
                    success=False,
                    code=code,
                    error=error_msg,
                )

            return VerificationResult(
                success=True,
                code=code,
                error=None,
            )

        except FileNotFoundError:
            # lake not found, try direct lean
            return await self._verify_with_lean_direct(code, file_path)

    async def _verify_with_lean_direct(self, code: str, file_path: Path) -> VerificationResult:
        """Fallback: verify with lean directly (no lake)."""
        import shutil

        lean_path = shutil.which("lean")
        if not lean_path:
            return VerificationResult(
                success=False,
                code=code,
                error="LEAN not found. Install via elan.",
            )

        proc = await asyncio.create_subprocess_exec(
            lean_path, str(file_path),
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
                error=f"Verification timed out after {self.timeout}s",
            )

        if proc.returncode == 0:
            return VerificationResult(success=True, code=code, error=None)

        error_msg = stderr.decode("utf-8").strip() or stdout.decode("utf-8").strip()
        return VerificationResult(success=False, code=code, error=error_msg[:500])
