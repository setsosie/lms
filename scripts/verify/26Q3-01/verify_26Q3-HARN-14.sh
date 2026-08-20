#!/bin/bash
# Verification for 26Q3-HARN-14: Committee groups need a Lean verify-feedback loop
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-14.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The repair seam exists on WorkingGroup
grep -q "def repair" lms/working_group.py

# 2. Repair spend is distinguishable from session spend in the ledger
grep -q "group_repair" lms/working_group.py

# 3. The knob exists and is threaded
grep -q "max_repair_attempts" lms/config.py
grep -q "max_repair_attempts" lms/society.py

# 4. Behavior is proven in pytest
uv run pytest tests/test_working_group.py -k repair -q
uv run pytest tests/test_society.py -k "repair or zero_repair" -q

echo "26Q3-HARN-14: verification passed"
