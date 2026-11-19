#!/bin/bash
# Verification for 26Q4-HARN-24: A tag closes only on an artifact that is that tag's statement
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-24.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Files exist
test -f tests/test_tag_closure.py

# 2. The rule and its use at every closure site
grep -q "^def tag_name_matches" lms/society.py
grep -q "def _closes_tag" lms/society.py

# 3. Tests pass
uv run pytest tests/test_tag_closure.py -q

# 4. Code quality
uv run ruff check lms/society.py tests/test_tag_closure.py
uv run mypy lms/society.py

echo "26Q4-HARN-24: verification passed"
