#!/usr/bin/env bash
# Verification script for 26Q3-HARN-07: verifier invokes Lean with the project env.
#
# Discrimination note: every check below fails at the merge base, where
# RealLeanVerifier ran bare `lean` with no LEAN_PATH and had no `lake_path`.
set -uo pipefail

fail=0
section() { printf '\n== %s ==\n' "$1"; }
check() {
  if eval "$2" >/dev/null 2>&1; then
    printf '  ok   %s\n' "$1"
  else
    printf '  FAIL %s\n' "$1"
    fail=1
  fi
}

section "1. Verifier exposes a discovered lake path"
check "RealLeanVerifier has lake_path" \
  "uv run python -c \"
from lms.lean.real import RealLeanVerifier
v = RealLeanVerifier(lean_path='/fake/lean')
assert v.lake_path, 'lake_path empty'
\""

section "2. Project-configured verifier invokes 'lake env lean' from the project dir"
check "command vector is lake env lean, cwd is the project" \
  "uv run python -c \"
import asyncio, tempfile
from pathlib import Path
from unittest.mock import patch
from lms.lean.real import RealLeanVerifier

calls = []
class P:
    returncode = 0
    async def communicate(self): return (b'', b'')
async def rec(*a, **k):
    calls.append((a, k)); return P()

with tempfile.TemporaryDirectory() as d:
    v = RealLeanVerifier(lean_path='/fake/lean', project_dir=Path(d))
    with patch('asyncio.create_subprocess_exec', new=rec):
        asyncio.run(v.verify('import Mathlib.Logic.Basic\ntheorem t : True := trivial'))
    args, kwargs = calls[-1]
    assert Path(args[0]).name == 'lake', args
    assert args[1:3] == ('env', 'lean'), args
    assert kwargs.get('cwd') == Path(d), kwargs
\""

section "3. Without a project, behavior is unchanged (bare lean, no cwd)"
check "falls back to bare lean" \
  "uv run python -c \"
import asyncio
from pathlib import Path
from unittest.mock import patch
from lms.lean.real import RealLeanVerifier

calls = []
class P:
    returncode = 0
    async def communicate(self): return (b'', b'')
async def rec(*a, **k):
    calls.append((a, k)); return P()

v = RealLeanVerifier(lean_path='/fake/lean')
with patch('asyncio.create_subprocess_exec', new=rec):
    asyncio.run(v.verify('theorem t : True := trivial'))
args, kwargs = calls[-1]
assert args[0] == '/fake/lean', args
assert kwargs.get('cwd') is None, kwargs
\""

section "4. Tests and lint"
check "pytest tests/test_lean_real_env.py" "uv run pytest tests/test_lean_real_env.py -q"
check "pytest (full suite)" "uv run pytest -q"
check "ruff on changed files" "uv run ruff check lms/lean/real.py tests/test_lean_real_env.py"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-07: PASS"
else
  echo "26Q3-HARN-07: FAIL"
fi
exit "$fail"
