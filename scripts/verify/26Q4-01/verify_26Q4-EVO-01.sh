#!/usr/bin/env bash
# Verification for 26Q4-EVO-01: per-run prompt override hook.
# Run: bash scripts/verify/26Q4-01/verify_26Q4-EVO-01.sh
#
# Discrimination note: at the merge base `grep -rn "prompt-file\|apply_prompt_overrides" lms/`
# is empty — CURRENT_PROMPTS was a hardcoded module global with no injection
# point, so a bred prompt could not reach a run without editing source. The
# wiring greps and the whole tests/test_prompt_overrides.py suite fail there.
# The hygiene section is expected green at the base.
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

section "1. Wiring: the override path exists end to end"
check "prompts.py exposes apply/load/active/clear" \
  "grep -q 'def apply_prompt_overrides' lms/prompts.py && \
   grep -q 'def load_prompt_overrides' lms/prompts.py && \
   grep -q 'def active_overrides' lms/prompts.py && \
   grep -q 'def clear_prompt_overrides' lms/prompts.py"
check "run.py takes --prompt-file with an LMS_PROMPT_FILE fallback" \
  "grep -q -- '--prompt-file' lms/run.py && grep -q 'LMS_PROMPT_FILE' lms/run.py"
check "metadata records the applied overrides" \
  "grep -q '\"prompt_overrides\": active_overrides()' lms/run.py"
check "a resume under overrides is recorded under its own key" \
  "grep -q 'prompt_overrides_resume' lms/run.py"

section "2. Behaviour: tests prove the card's claims"
check "override reaches get_prompt; provenance versions; loud failures" \
  "uv run pytest tests/test_prompt_overrides.py -q"

section "3. Hygiene"
check "full test suite passes" "uv run pytest -q"
check "ruff clean on touched files" \
  "uv run ruff check lms/prompts.py lms/run.py tests/test_prompt_overrides.py"
check "mypy clean on touched files" "uv run mypy lms/prompts.py lms/run.py"

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf 'PASS 26Q4-EVO-01\n'
else
  printf 'FAIL 26Q4-EVO-01\n'
fi
exit "$fail"
