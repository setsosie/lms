#!/bin/bash
# Verification for 26Q4-HARN-18: Rollback reopens the goal tags it undid
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-18.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The reconciliation
uv run pytest tests/test_goal_reopens_tags.py -q

# 2. Neighbours that read the same state
uv run pytest tests/test_foundation_rollback.py tests/test_foundation_collisions.py tests/test_resume_foundation.py tests/test_dependency.py tests/test_goals.py -q

# 3. Quality
uv run ruff check lms/goals.py lms/foundation.py lms/society.py tests/test_goal_reopens_tags.py
uv run mypy lms/goals.py lms/foundation.py lms/society.py

echo "26Q4-HARN-18: verification passed"
