#!/usr/bin/env bash
# Verification script for 26Q3-HARN-09: verified work reaches the next generation.
#
# Behavioural assertions live in tests/test_foundation_persistence.py, not here.
# This script only checks that the pieces exist and imports resolve, then runs
# the tests that do the real work. Assertions written inline in a shell script
# are untested code checking tested code -- no fixtures, no useful failure
# output, and they drift from the implementation silently.
#
# Discrimination note: at the merge base `Society.persist_foundation`,
# `Society.reset_foundation` and `LeanProject.rebuild_changed_sources` did not
# exist, `foundation.save()` ran only from `Society.save()` at a 10-generation
# checkpoint, `lean` was invoked without strictness flags, and
# `get_context_for_agent` emitted no declaration signature.
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

TESTS=tests/test_foundation_persistence.py

section "1. The pieces exist and import"
check "Society.persist_foundation" \
  "uv run python -c 'from lms.society import Society; Society.persist_foundation'"
check "Society.reset_foundation" \
  "uv run python -c 'from lms.society import Society; Society.reset_foundation'"
check "LeanProject.rebuild_changed_sources" \
  "uv run python -c 'from lms.lean.project import LeanProject; LeanProject.rebuild_changed_sources'"
check "RealLeanVerifier.STRICTNESS_FLAGS" \
  "uv run python -c 'from lms.lean.real import RealLeanVerifier; RealLeanVerifier.STRICTNESS_FLAGS'"
check "test module is present" "test -f $TESTS"

section "2. Behaviour (tests/test_foundation_persistence.py)"
check "foundation reaches disk and is recompiled" \
  "uv run pytest $TESTS -q -k 'persist or generation_writes'"
check "runs start independent of each other" \
  "uv run pytest $TESTS -q -k 'reset'"
check "autoImplicit is off on both paths" \
  "uv run pytest $TESTS -q -k 'strictness or header'"
check "agents can see the declaration signature" \
  "uv run pytest $TESTS -q -k 'signature'"
check "strictness flags reach the lean command vector" \
  "uv run pytest tests/test_lean_real_env.py -q -k 'strictness'"
check "the whole module" "uv run pytest $TESTS -q"

section "3. Suite and hygiene"
check "pytest (full suite)" "uv run pytest -q"
check "pytest leaves lean/ clean" "git diff --quiet -- lean/"
check "ruff on changed source files" \
  "uv run ruff check lms/society.py lms/run.py lms/lean/real.py lms/lean/project.py $TESTS"
check "mypy adds nothing new in changed files" \
  "test \$(uv run mypy lms/society.py lms/foundation.py lms/lean/real.py lms/lean/project.py 2>&1 | grep -cE '^lms/(society|foundation|lean/real|lean/project)\.py') -le 4"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-09: PASS"
else
  echo "26Q3-HARN-09: FAIL"
fi
exit "$fail"
