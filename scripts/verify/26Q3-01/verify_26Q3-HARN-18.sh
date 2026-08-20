#!/bin/bash
# Verification for 26Q3-HARN-18: `--agents` is inert in committee mode
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-18.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. Society scales the working-group cast from its size
grep -q "_committee_members_per_role" lms/society.py

# 2. The scaled cast is passed into the group config
grep -q "members_per_role=self._committee_members_per_role()" lms/society.py

# 3. Behavior proven in pytest
uv run pytest tests/test_society.py -k CastScaling -q

echo "26Q3-HARN-18: verification passed"
