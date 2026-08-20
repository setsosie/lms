#!/bin/bash
# Verification for 26Q3-HARN-15: A same-tag failure clobbers a task's DONE status
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-15.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. DONE is terminal in the graph
grep -q "refusing demotion" lms/dependency.py

# 2. The explicit rollback door exists
grep -q "def revert_done" lms/dependency.py

# 3. Behavior proven in pytest
uv run pytest tests/test_dependency.py -k DoneIsTerminal -q
uv run pytest tests/test_society.py -k SameTagFailure -q

echo "26Q3-HARN-15: verification passed"
