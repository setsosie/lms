#!/bin/bash
# Verification for 26Q4-HARN-13: Decide what a decisive N1 means at Gate 4
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-13.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The chosen rule is pinned on the recorded statements
grep -q "class TestNoN1CountsWithoutReview" tests/test_novelty.py
grep -q "What counts during calibration" specs/faithfulness_protocol.md

# 2. Tests pass: the gate, the CVFN report and the arm comparison read the same rule
uv run pytest tests/test_novelty.py tests/test_accounting.py tests/test_arms.py -q

# 3. Code quality
uv run ruff check lms/ tests/test_novelty.py tests/test_accounting.py tests/test_arms.py
uv run mypy lms/

echo "26Q4-HARN-13: verification passed"
