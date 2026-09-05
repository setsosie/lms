#!/bin/bash
# Verification for 26Q3-HARN-27: Source-anchored dependency edges — the task graph is a DAG, not a chain
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-27.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. Files exist
test -f tests/test_dependency_edges.py
test -f tests/test_extract_stacks_goal.py
test -f tests/test_goal_files.py

# 2. Required symbols present
grep -q "requires" lms/goals.py
grep -q "def resolve_refs" scripts/goals/extract_stacks_goal.py
grep -q "begin{proof}" scripts/goals/extract_stacks_goal.py
grep -q "requires" goals/README.md

# 3. The checked-in goal files carry edges
grep -q '"requires"' goals/stacks_kernel_track_a.json
grep -q '"requires"' goals/stacks_kernel_track_b.json

# 4. Inference survives as the fallback; society untouched
grep -q "def _infer_dependencies" lms/dependency.py
git diff --quiet origin/main -- lms/society.py

# 5. Import smoke
uv run python -c "from lms.goals import Goal, StacksDefinition; StacksDefinition(tag='A', section='1.1', name='A', content='', requires=[])"

# 6. Tests pass
uv run pytest tests/test_dependency_edges.py tests/test_extract_stacks_goal.py tests/test_goal_files.py tests/test_goals.py tests/test_dependency.py -q --tb=short

# 7. Code quality
uv run ruff check lms/goals.py lms/dependency.py scripts/goals/extract_stacks_goal.py
uv run mypy lms/goals.py lms/dependency.py

echo "26Q3-HARN-27: verification passed"
