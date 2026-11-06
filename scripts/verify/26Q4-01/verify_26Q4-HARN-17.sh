#!/bin/bash
# Verification for 26Q4-HARN-17: A name collision can't discard a generation's verified work
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-17.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Rejection and identical duplicates
uv run pytest tests/test_foundation_collisions.py -q

# 2. Per-artifact rollback, with the neighbours that read the same state
uv run pytest tests/test_foundation_rollback.py tests/test_foundation.py tests/test_seed.py tests/test_resume_foundation.py -q

# 3. All three promotion paths go through the helper
test "$(grep -c '_promote_to_foundation(artifact)' lms/society.py)" -eq 3

# 4. Quality
uv run ruff check lms/foundation.py lms/society.py tests/test_foundation_collisions.py
uv run mypy lms/foundation.py lms/society.py

echo "26Q4-HARN-17: verification passed"
