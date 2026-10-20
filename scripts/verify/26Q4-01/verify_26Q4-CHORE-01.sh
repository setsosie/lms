#!/bin/bash
# Verification for 26Q4-CHORE-01: Move the HARN-02 verify script's assertions into pytest
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-01.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
#
# Discrimination: at main @ 0c3ad12 the HARN-02 script holds 13 inline-Python
# checks and tests/test_reextract_lean_code.py does not exist.
set -e

# 1. The assertions moved
test -f tests/test_reextract_lean_code.py
# Not `! grep`: set -e ignores a negated status, so those lines could never fail.
if grep -q "uv run python" scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh; then exit 1; fi
if grep -q "experiments/stacks_ch4_phase1" scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh; then exit 1; fi

# 2. Behaviour
uv run pytest tests/test_reextract_lean_code.py tests/test_lean_extraction.py -q

# 3. The rewritten script still passes
bash scripts/verify/26Q3-01/verify_26Q3-HARN-02.sh

# 4. Hygiene
uv run ruff check tests/test_reextract_lean_code.py

echo "26Q4-CHORE-01: verification passed"
