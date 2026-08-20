#!/bin/bash
# Verification for 26Q3-HARN-16: Committee mode cannot measure reuse
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-16.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. The scribe is asked to cite references
grep -q "references" lms/working_group.py

# 2. The committee path links references (mirrors the iterative fix)
grep -q "add_reference" lms/society.py
test "$(grep -c add_reference lms/society.py)" -ge 3

# 3. Behavior proven in pytest
uv run pytest tests/test_working_group.py -k ArtifactReferences -q
uv run pytest tests/test_society.py -k CommitteeReuse -q

echo "26Q3-HARN-16: verification passed"
