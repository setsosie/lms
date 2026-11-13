#!/bin/bash
# Verification for 26Q4-HARN-23: A goal file names its own seed, and the kernel tracks start bare
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-23.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Behaviour
uv run pytest tests/test_goal_seed.py tests/test_goals.py tests/test_goal_files.py -q

# 2. The kernel goal files and their generator carry the bare seed
grep -q '"seed": ""' goals/stacks_kernel_track_a.json
grep -q '"seed": ""' goals/stacks_kernel_track_b.json
grep -q '"seed": ""' scripts/goals/extract_stacks_goal.py
grep -q 'seeded_tags=data.get' lms/goals.py

# 3. Quality
uv run ruff check lms/goals.py scripts/goals/extract_stacks_goal.py tests/test_goal_seed.py
uv run mypy lms/goals.py

echo "26Q4-HARN-23: verification passed"
