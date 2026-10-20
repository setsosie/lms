#!/usr/bin/env bash
# Verification script for 26Q3-HARN-02: `lean_code` reaches the verifier as Lean.
#
# Discrimination note: every check below fails at the merge base, where
# `_clean_lean_code` did not exist, `_parse_artifacts` stored the raw regex
# capture, and `Artifact` had no `lean_code_raw`.
#
# The behaviour checks live in tests/test_lean_extraction.py and
# tests/test_reextract_lean_code.py. This script used to assert them inline,
# two of them against experiments/stacks_ch4_phase1/, which is not checked in.
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

section "1. The work exists"
check "_clean_lean_code is defined" "grep -q '^def _clean_lean_code' lms/agent.py"
check "Artifact carries lean_code_raw" "grep -q 'lean_code_raw' lms/artifacts.py"
check "reextract script exists" "test -f scripts/reextract_lean_code.py"
check "at least 5 real payloads are checked in as fixtures" \
  "test \$(ls tests/fixtures/malformed_lean/*.txt | wc -l) -ge 5"

section "2. Behaviour"
check "cleaner, parser wiring, recorded payloads, canary" \
  "uv run pytest tests/test_lean_extraction.py -q"
check "re-extraction: input untouched, stale verdicts reset, raw round trip" \
  "uv run pytest tests/test_reextract_lean_code.py -q"

section "3. Tests and lint"
check "pytest (full suite)" "uv run pytest -q"
check "pytest leaves lean/ clean" "git diff --quiet -- lean/"
check "ruff on changed source files" \
  "uv run ruff check lms/agent.py lms/artifacts.py tests/test_lean_extraction.py tests/test_reextract_lean_code.py scripts/reextract_lean_code.py"
# Scoped to the files this card touches: mypy follows imports, and the repo
# carries pre-existing errors elsewhere that this card does not claim to fix.
check "mypy reports nothing in the changed source files" \
  "! uv run mypy lms/agent.py lms/artifacts.py 2>&1 | grep -qE '^lms/(agent|artifacts)\.py'"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-02: PASS"
else
  echo "26Q3-HARN-02: FAIL"
fi
exit "$fail"
