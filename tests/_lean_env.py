"""Whether this machine can actually run Lean.

The suite used to gate its Lean-dependent modules on

    shutil.which("lean") is None and not shutil.which("/home/<developer>/.elan/bin/lean")

which was wrong in two ways. It hardcoded one developer's home directory, so
the fallback silently did nothing on any other machine. And it tested for the
*binary*, not for a usable toolchain: `elan` installs a `lean` shim that exists
before `elan default stable` has ever been run, so on a freshly provisioned box
the guard reports Lean present and every test in those modules fails with
"no default toolchain configured" instead of skipping.

Probing `lean --version` answers the question the tests actually care about.
"""

import functools
import shutil
import subprocess
from pathlib import Path


def _candidate_lean() -> str | None:
    """Path to a `lean` executable, preferring PATH over a default elan install."""
    return shutil.which("lean") or shutil.which(
        str(Path.home() / ".elan" / "bin" / "lean")
    )


@functools.lru_cache(maxsize=1)
def lean_available() -> bool:
    """True when a Lean toolchain is installed *and* configured well enough to run.

    Cached: without this, every guarded test would spawn its own subprocess.
    """
    exe = _candidate_lean()
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


LEAN_MISSING = not lean_available()
LEAN_MISSING_REASON = "no usable Lean 4 toolchain on this machine"
