#!/bin/bash
# Verification for 26Q4-HARN-16: Foundation merge keeps universes the header doesn't bind
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-16.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The merge and its round trips
uv run pytest tests/test_foundation_universes.py -q

# 2. Neighbours that read the same state
uv run pytest tests/test_verify_universes.py tests/test_foundation_rollback.py tests/test_resume_foundation.py -q

# 3. Real Lean, which skips where no toolchain exists; CI's real-Lean job runs it
uv run pytest tests/test_lean_real.py -k TestFoundationUniverses -q

# 4. Quality
uv run ruff check lms/foundation.py tests/test_foundation_universes.py
uv run mypy lms/foundation.py

echo "26Q4-HARN-16: verification passed"
