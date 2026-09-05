#!/bin/bash
# Verification for 26Q3-HARN-26: Closing leaves — a proved child resolves its parent sketch
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-26.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. Files exist
test -f tests/test_society_resolve.py
test -f tests/test_dependency_leaves.py
test -f tests/test_sketch.py

# 2. Required symbols present
grep -q "def close_leaf" lms/dependency.py
grep -q "def leaves_of" lms/dependency.py
grep -q "def strip_children" lms/sketch.py
grep -q "def _resolve_parent" lms/society.py
grep -q "artifacts_resolved" lms/society.py
grep -q "RESOLVE FAILED" lms/society.py
grep -q "Prior reduction" lms/society.py

# 3. Resolution is accounted for, not free
grep -q 'outcome="resolve"' lms/society.py

# 4. Import smoke
uv run python -c "from lms.sketch import strip_children; from lms.dependency import DependencyGraph; DependencyGraph.close_leaf; DependencyGraph.leaves_of"

# 5. Tests pass
uv run pytest tests/test_society_resolve.py tests/test_dependency_leaves.py tests/test_sketch.py tests/test_society_sketch.py -q --tb=short

# 6. Code quality
uv run ruff check lms/sketch.py lms/dependency.py lms/society.py lms/artifacts.py
uv run mypy lms/sketch.py lms/dependency.py

echo "26Q3-HARN-26: verification passed"
