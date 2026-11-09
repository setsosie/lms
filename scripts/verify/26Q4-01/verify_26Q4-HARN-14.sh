#!/bin/bash
# Verification for 26Q4-HARN-14: Tell an exhausted exact? from a miss
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-14.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The measurement script wraps the classifier's own probe and reads outcomes right
uv run pytest tests/test_measure_exact_budget.py -q

# 2. The result is recorded where the reader of a miss will look
grep -q "R4b's 18 misses were all genuine" docs/runbooks/2026-10-08-box-restore.md
grep -q "measure_exact_budget.py" docs/runbooks/2026-10-08-box-restore.md
grep -q "193,525" docs/planning/tasks/26Q4-01/26Q4-HARN-14-exhausted-exact-probe-vs-miss.md

# 3. Quality
uv run ruff check scripts/research/measure_exact_budget.py tests/test_measure_exact_budget.py

echo "26Q4-HARN-14: verification passed"
