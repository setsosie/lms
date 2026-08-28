"""Lean project management for LMS.

Handles project building, import tracking, and cache management to avoid
.olean file errors when new imports are added.
"""

import asyncio
import hashlib
import re
from pathlib import Path


class LeanProject:
    """Manages the Lean project and ensures build consistency.

    Tracks imports across verification attempts and triggers rebuilds
    when new imports are detected, preventing 'object file does not exist'
    errors that plagued earlier experiments.
    """

    def __init__(self, project_dir: Path | str) -> None:
        """Initialize the Lean project manager.

        Args:
            project_dir: Path to the Lean project root (contains lakefile.toml)
        """
        # Resolved, not stored as given. `RealLeanVerifier` runs Lean with
        # `cwd=project_dir` (so `lake env` finds the package) while passing the
        # temp file path derived from this attribute. If both stay relative,
        # Lean is handed `lean/.lake/verify-temp/x.lean` *from inside* `lean/`
        # and reports `no such file or directory` — which is indistinguishable
        # in the result from a genuine proof failure.
        self.project_dir = Path(project_dir).resolve()
        # Deliberately OUTSIDE the library source tree. The lean_lib globs
        # `LMS.+`, so a scratch file under LMS/ becomes build input and
        # `lake build` starts compiling whatever an agent last emitted.
        self.temp_dir = self.project_dir / ".lake" / "verify-temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Track imports we've seen to detect when rebuild is needed
        self._seen_imports: set[str] = set()
        self._last_build_hash: str | None = None

    def _extract_imports(self, code: str) -> set[str]:
        """Extract import statements from Lean code.

        Args:
            code: Lean source code

        Returns:
            Set of import paths (e.g., {"Mathlib.CategoryTheory.Yoneda"})
        """
        # Match: import Foo.Bar.Baz
        pattern = r"^\s*import\s+([A-Za-z][A-Za-z0-9_.]*)"
        imports = set()
        for match in re.finditer(pattern, code, re.MULTILINE):
            imports.add(match.group(1))
        return imports

    def _compute_import_hash(self, imports: set[str]) -> str:
        """Compute hash of current import set for change detection."""
        sorted_imports = sorted(imports)
        return hashlib.md5("|".join(sorted_imports).encode()).hexdigest()

    async def ensure_built(self, code: str) -> bool:
        """Ensure the project is built with all required imports.

        Detects new imports and triggers a rebuild if necessary.

        Args:
            code: Lean code about to be verified

        Returns:
            True if project is ready, False if build failed
        """
        imports = self._extract_imports(code)
        new_imports = imports - self._seen_imports

        if new_imports:
            # New imports detected - need to rebuild
            self._seen_imports.update(imports)
            current_hash = self._compute_import_hash(self._seen_imports)

            if current_hash != self._last_build_hash:
                success = await self.build()
                if success:
                    self._last_build_hash = current_hash
                return success

        return True

    async def rebuild_changed_sources(self) -> bool:
        """Rebuild unconditionally, bypassing the import-change heuristic.

        `ensure_built` rebuilds only when a *new* import name appears, which is
        the wrong test for `LMS.Foundation`: its name never changes while its
        contents change every generation. Left to that heuristic the stale
        `.olean` is reused for the whole run, so `import LMS.Foundation`
        resolves to a module that predates every artifact in it.

        Returns:
            True if the build succeeded.
        """
        success = await self.build()
        if success:
            # Let the import heuristic re-evaluate from scratch: the tree it
            # last built against no longer exists.
            self._last_build_hash = None
        return success

    async def build(self, clean: bool = False) -> bool:
        """Run lake build in the project directory.

        Args:
            clean: If True, run lake clean first (slower but more thorough)

        Returns:
            True if build succeeded, False otherwise
        """
        # A box without a Lean toolchain has no `lake` on PATH, and
        # create_subprocess_exec then raises instead of returning a status.
        # That is a failed build, not a crash worth taking the harness down.
        try:
            if clean:
                clean_proc = await asyncio.create_subprocess_exec(
                    "lake",
                    "clean",
                    cwd=self.project_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await clean_proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                "lake",
                "build",
                cwd=self.project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            print("Lake build warning: `lake` not found on PATH")
            return False

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Log build error but don't fail - verification will catch it
            error_msg = stderr.decode("utf-8") if stderr else stdout.decode("utf-8")
            print(f"Lake build warning: {error_msg[:200]}")

        return proc.returncode == 0

    def get_temp_file(self, code: str) -> Path:
        """Get a temp file path for verification.

        Uses content hash to enable caching.

        Args:
            code: Lean code to verify

        Returns:
            Path to temp file (may or may not exist yet)
        """
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        return self.temp_dir / f"verify_{code_hash}.lean"

    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up old temp files.

        Args:
            max_age_hours: Remove files older than this

        Returns:
            Number of files removed
        """
        import time

        count = 0
        cutoff = time.time() - (max_age_hours * 3600)

        for f in self.temp_dir.glob("verify_*.lean"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                count += 1

        return count
