#!/usr/bin/env bash
# Verification script for 26Q3-HARN-10: foundation names need opening.
#
# Behaviour lives in tests/test_foundation_namespace.py. See
# scripts/verify/README.md for why none of it is asserted inline here.
#
# Discrimination note: at the merge base `FoundationFile.get_preamble` and
# `NAMESPACE` did not exist, `get_context_for_agent` emitted only the import
# line, and `agent_system_goal` was v2.5.0 -- which claimed the foundation was
# "imported automatically" and whose WILL SUCCEED example used `def Cone`.
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

TESTS=tests/test_foundation_namespace.py

section "1. The pieces exist and import"
check "FoundationFile.get_preamble" \
  "uv run python -c 'from lms.foundation import FoundationFile; FoundationFile.get_preamble'"
check "FoundationFile.NAMESPACE" \
  "uv run python -c 'from lms.foundation import FoundationFile; FoundationFile.NAMESPACE'"
check "agent_system_goal resolves to v2.6.0" \
  "uv run python -c 'from lms.prompts import get_all_versions; assert get_all_versions()[\"agent_system_goal\"] == \"2.6.0\"'"
check "test module is present" "test -f $TESTS"

section "2. Behaviour (tests/test_foundation_namespace.py)"
check "agents are told to open the namespace, and why" \
  "uv run pytest $TESTS -q -k 'preamble or open or explains'"
check "NAMESPACE cannot drift from the written file" \
  "uv run pytest $TESTS -q -k 'namespace_constant'"
check "goal prompt corrected and re-versioned" \
  "uv run pytest $TESTS -q -k 'prompt'"
check "the whole module" "uv run pytest $TESTS -q"

section "3. Suite and hygiene"
check "pytest (full suite)" "uv run pytest -q"
check "pytest leaves lean/ clean" "git diff --quiet -- lean/"
check "ruff on changed source files" \
  "uv run ruff check lms/foundation.py lms/prompts.py $TESTS"
check "mypy adds nothing new in changed files" \
  "test \$(uv run mypy lms/foundation.py lms/prompts.py 2>&1 | grep -cE '^lms/(foundation|prompts)\.py') -eq 0"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-10: PASS"
else
  echo "26Q3-HARN-10: FAIL"
fi
exit "$fail"
