"""Real LEAN 4 verifier using the actual LEAN compiler."""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

from lms.foundation import (
    FOUNDATION_NAMESPACE,
    FOUNDATION_UNIVERSES,
    declared_universe_names,
    split_imports,
)
from lms.lean.interface import LeanVerifier, VerificationResult, VerifierKind
from lms.lean.project import LeanProject


class RealLeanVerifier(LeanVerifier):
    """Verifier that uses the actual LEAN 4 compiler.

    Runs LEAN 4 on the provided code and checks for compilation errors.
    Also detects and rejects use of 'sorry' (incomplete proofs).

    When a project_dir is provided, automatically rebuilds when new
    imports are detected to prevent 'object file does not exist' errors.
    """

    # Passed to `lean` itself, NOT set as `leanOptions` in lakefile.toml:
    # `lake env lean <file>` only exports the package environment, so library
    # options never reach a file compiled this way.
    #
    # With autoImplicit on (Lean's default), an identifier the file cannot
    # resolve becomes an auto-bound implicit variable instead of an error. A
    # missing `import` then surfaces as "Function expected at Category, but
    # this term has type ?m.1" somewhere further down, which reads like the
    # agent's mathematics is wrong when the real fault is that the foundation
    # never made it onto disk. Off, the same failure reads "unknown identifier".
    # Mathlib disables both for the same reason.
    STRICTNESS_FLAGS: tuple[str, ...] = (
        "-DautoImplicit=false",
        "-DrelaxedAutoImplicit=false",
    )

    verifier_kind: VerifierKind = "real"

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

    def toolchain_info(self) -> dict[str, str | None]:
        """Report the Lean toolchain and Mathlib revision this verifier binds.

        A `VERIFIED_LEAN` result is only meaningful relative to the toolchain
        that produced it, so the versions are recorded with the run rather than
        inferred later from whatever happens to be installed.
        """
        lean_version: str | None = None
        try:
            proc = subprocess.run(
                (self.lean_path, "--version"),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                lean_version = proc.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass

        return {
            "lean_version": lean_version,
            "mathlib_rev": self._mathlib_rev(),
        }

    def _mathlib_rev(self) -> str | None:
        """Pinned Mathlib revision from the project's lake manifest, if any."""
        if not self.project:
            return None

        manifest = self.project.project_dir / "lake-manifest.json"
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        for package in data.get("packages", []):
            if package.get("name") == "mathlib":
                rev = package.get("rev")
                return str(rev) if rev else None
        return None

    @staticmethod
    def _wrap_in_storage_namespace(code: str) -> str:
        """Wrap `code` in the namespace the foundation stores entries in.

        The candidate is elaborated as `FOUNDATION_NAMESPACE.<name>`, exactly
        where an accepted artifact lands (`FoundationFile.FOUNDATION_HEADER`).
        Verifying at top level instead rejected any declaration whose name Lean
        core already binds -- `Functor`, `Option`, `Prod` -- for a collision
        that does not exist at the destination (26Q3-HARN-13).

        Imports are hoisted above the wrapper: Lean rejects an `import` inside
        a `namespace`. Code carrying its own `namespace Foo ... end Foo` nests
        inside the wrapper, which is legal.

        The foundation's universe names are bound here too (26Q3-HARN-21).
        `FoundationFile.add_artifact` strips an entry's own `universe` lines
        because the header already binds them -- so a candidate that writes
        `Category.{u,v}` without declaring `u v` compiles once stored but was
        rejected at verification with `unknown universe level 'v'`. The
        verifier was strictly stricter than the destination, and the gap cost
        correct work. Only the names the candidate has *not* declared are
        added; declaring one twice is an error in the other direction.
        """
        imports, body = split_imports(code)
        missing = [
            u for u in FOUNDATION_UNIVERSES if u not in declared_universe_names(body)
        ]
        pieces: list[str] = []
        if imports:
            pieces.extend(imports)
            pieces.append("")
        if missing:
            pieces.append(f"universe {' '.join(missing)}")
            pieces.append("")
        pieces.append(f"namespace {FOUNDATION_NAMESPACE}")
        pieces.append("")
        pieces.extend(body)
        pieces.append("")
        pieces.append(f"end {FOUNDATION_NAMESPACE}")
        pieces.append("")
        return "\n".join(pieces)

    async def verify(self, code: str) -> VerificationResult:
        """Verify LEAN code using the real compiler.

        The code is elaborated inside the namespace it will be stored in --
        see `_wrap_in_storage_namespace`. Results report the original,
        unwrapped code.

        Args:
            code: LEAN 4 code to verify

        Returns:
            VerificationResult with compilation status
        """
        # Empty code always fails
        if not code or not code.strip():
            return self._result(
                success=False,
                code=code,
                error="Empty code provided",
            )

        # Check for sorry (incomplete proofs) - we reject these
        if "sorry" in code:
            return self._result(
                success=False,
                code=code,
                error="Code contains 'sorry' - incomplete proof not allowed",
            )

        # What Lean actually sees: the candidate inside its storage namespace.
        wrapped = self._wrap_in_storage_namespace(code)

        # Auto-rebuild if new imports detected (prevents .olean errors)
        if self.project:
            await self.project.ensure_built(wrapped)
            temp_path = self.project.get_temp_file(wrapped)
            temp_path.write_text(wrapped)
        else:
            # Fallback to system temp
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".lean",
                delete=False,
            ) as f:
                f.write(wrapped)
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
                *self.STRICTNESS_FLAGS,
                str(temp_path),
            )
            cwd: Path | None = self.project.project_dir
        else:
            command = (self.lean_path, *self.STRICTNESS_FLAGS, str(temp_path))
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
                return self._result(
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
                return self._result(
                    success=False,
                    code=code,
                    error=f"LEAN verification timed out after {self.timeout}s",
                )

            # Check result
            if proc.returncode == 0:
                return self._result(
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

                return self._result(
                    success=False,
                    code=code,
                    error=error_msg,
                )

        finally:
            # Clean up temp file (unless using project manager which handles cleanup)
            if not self.project:
                temp_path.unlink(missing_ok=True)
