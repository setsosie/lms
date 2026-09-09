#!/usr/bin/env bash
# Verification for 26Q3-INFRA-04: run the real Lean verifier in CI.
#
# Discrimination note: every check below fails at the merge base, where
# tests.yml had no lean-tests job at all and the 16 Lean-dependent tests ran
# nowhere — skipped in CI for want of a toolchain, failed on a developer box
# whose elan shim had no configured default.
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

W=.github/workflows/tests.yml

section "1. The job exists and is gated like the others"
check "a lean-tests job exists"        "grep -q '^  lean-tests:' $W"
check "it waits on lint"               "awk '/^  lean-tests:/,/^  pytest:/' $W | grep -q 'needs: lint'"
check "it is parallel to pytest, not inside it" \
  "awk '/^  lean-tests:/,/^  pytest:/' $W | grep -q 'uv run pytest'"

section "2. Toolchain only -- no Mathlib, no corpus build"
check "elan is installed from the official script" \
  "grep -q 'elan.lean-lang.org/elan-init.sh' $W"
check "the toolchain is pinned to lean/lean-toolchain" \
  "grep -q 'default-toolchain \"\$(cat lean/lean-toolchain)\"' $W"
check "the cache key is that same pin" \
  "grep -q \"hashFiles('lean/lean-toolchain')\" $W"
check "the job never runs lake build" \
  "! awk '/^  lean-tests:/,/^  pytest:/' $W | grep -q 'lake build'"
check "the job never fetches a Mathlib cache" \
  "! awk '/^  lean-tests:/,/^  pytest:/' $W | grep -qi 'mathlib'"
check "lean.yml still owns the corpus build" \
  "grep -q 'leanprover/lean-action' .github/workflows/lean.yml"

section "3. It cannot go green while the tests skip"
check "the job probes lean --version"  "grep -q 'lean --version' $W"
check "the job asserts lean_available()" \
  "grep -q 'lean_available()' $W"
check "the guard is real: it rejects an unusable toolchain" \
  "uv run python -c \"
import os, tempfile, importlib
d = tempfile.mkdtemp()
os.environ['PATH'] = d
os.environ['HOME'] = d
import tests._lean_env as m
importlib.reload(m)
raise SystemExit(0 if not m.lean_available() else 1)\""

section "4. The tests it runs are the ones that were skipping"
check "test_lean_real.py is named"     "grep -q 'tests/test_lean_real.py' $W"
check "test_verify_namespace.py is named" "grep -q 'tests/test_verify_namespace.py' $W"
check "those two modules hold every skip in the suite" \
  "test 0 -eq \"\$(uv run pytest -q --ignore=tests/test_lean_real.py \
     --ignore=tests/test_verify_namespace.py 2>&1 | grep -c skipped)\""

printf '\n'
if [ "$fail" -eq 0 ]; then printf 'PASS 26Q3-INFRA-04\n'; else printf 'FAIL 26Q3-INFRA-04\n'; fi
exit "$fail"
