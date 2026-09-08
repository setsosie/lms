#!/usr/bin/env bash
# Verification for 26Q3-INFRA-02: per-request token cap configurable from env.
# Run: bash scripts/verify/26Q3-01/verify_26Q3-INFRA-02.sh
#
# Discrimination note: at the merge base `grep LMS_OPENAI_MAX_TOKENS lms/` is
# empty — `Config.from_env` never read a cap variable, so every ProviderConfig
# took the 64k dataclass default and every request asked a local server for
# 64,000 completion tokens. The wiring greps and the whole
# TestMaxTokensFromEnv suite (`-k` matches nothing → pytest exits 5) fail
# there. The hygiene section is expected green at the base.
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

section "1. Wiring: Config.from_env reads a cap per provider"
check "LMS_ANTHROPIC_MAX_TOKENS read in config.py" \
  "grep -q 'LMS_ANTHROPIC_MAX_TOKENS' lms/config.py"
check "LMS_OPENAI_MAX_TOKENS read in config.py" \
  "grep -q 'LMS_OPENAI_MAX_TOKENS' lms/config.py"
check "LMS_GOOGLE_MAX_TOKENS read in config.py" \
  "grep -q 'LMS_GOOGLE_MAX_TOKENS' lms/config.py"
check "DEFAULT_MAX_TOKENS no longer asserts a Claude limit as global" \
  "! grep -q 'limited by Claude' lms/config.py"
check ".env.example documents the cap variables" \
  "grep -q 'LMS_OPENAI_MAX_TOKENS' .env.example"
check "runbook Step 5 sets the cap" \
  "grep -q 'LMS_OPENAI_MAX_TOKENS=16384' docs/infrastructure/cluster-runbook-calibration.md"

section "2. Behaviour: tests prove the card's claims"
check "env cap reaches ProviderConfig and the request payload; bad values fail at load" \
  "uv run pytest tests/test_local_serving.py -q -k 'TestMaxTokensFromEnv'"

section "3. Hygiene"
check "full test suite passes" "uv run pytest -q"
check "ruff clean on touched modules" "uv run ruff check lms/config.py tests/test_local_serving.py"
# Same pattern as verify_26Q3-HARN-05: pre-existing errors elsewhere in the
# import graph are not this card's; the touched file itself must stay clean.
# import-not-found is excluded — on a box where mypy runs outside the venv it
# cannot see dotenv, and that is the environment's defect, not the file's.
check "mypy adds nothing new in lms/config.py" \
  "test \$(uv run mypy lms/config.py 2>&1 | grep -E '^lms/config\.py:[0-9]+: error:' | grep -cv 'import-not-found') -eq 0"

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf 'PASS 26Q3-INFRA-02\n'
else
  printf 'FAIL 26Q3-INFRA-02\n'
fi
exit "$fail"
