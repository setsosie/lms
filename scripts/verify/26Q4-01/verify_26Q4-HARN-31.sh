#!/bin/bash
# Verification for 26Q4-HARN-31: The foundation keeps the `open` and `variable` lines its entries verified under
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-31.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Behaviour (the real-Lean class skips without a usable toolchain; CI's real-Lean job runs it)
uv run pytest tests/test_foundation_preamble.py tests/test_foundation_imports.py \
  tests/test_foundation_universes.py tests/test_foundation_rollback.py \
  tests/test_resume_foundation.py tests/test_foundation_api_exposure.py -q
uv run pytest tests/test_lean_real.py -q -k "TestFoundationPreamble or TestFoundationCollisions"

# 2. The pieces are where a run reaches them
test -f tests/fixtures/foundation_preamble/q3_verified.json
grep -q 'entry.context = context' lms/foundation.py
grep -q '"context": list(self.context)' lms/foundation.py
grep -q 'context=tuple(d.get("context", ()))' lms/foundation.py
grep -q 'for line in entry.context' lms/foundation.py

# 3. Quality
uv run ruff check lms/foundation.py tests/test_foundation_preamble.py tests/test_lean_real.py
uv run mypy lms/foundation.py

echo "26Q4-HARN-31: verification passed"
