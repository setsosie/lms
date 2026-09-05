#!/bin/bash
# Verification for 26Q3-HARN-28: Theorem cards — a task carries its exact Lean statement, and proofs are pinned to it
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-28.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. Files exist
test -f lms/statement.py
test -f scripts/goals/arc_to_goal.py
test -f tests/test_statement.py
test -f tests/test_dependency_cards.py
test -f tests/test_society_cards.py

# 2. Required symbols present
grep -q "def statement_header" lms/statement.py
grep -q "def pin_statement" lms/statement.py
grep -q "lean_statement" lms/goals.py
grep -q "def goal_from_arc" lms/goals.py
grep -q "def set_statement" lms/dependency.py
grep -q "pinned_to" lms/artifacts.py
grep -q "pin_statement" lms/society.py

# 3. One Lean scanner: the top-level ':=' scan lives in lean_source, not copied
grep -q "_scan_for_assign\|def find_assign" lms/gates/lean_source.py
! grep -q "def _scan_for_assign" lms/foundation.py

# 4. Scribe prompt names the pinned declaration
grep -q "exact name" lms/working_group.py

# 5. Import smoke
uv run python -c "from lms.statement import statement_header, pin_statement; from lms.goals import goal_from_arc"

# 6. Tests pass
uv run pytest tests/test_statement.py tests/test_dependency_cards.py tests/test_society_cards.py tests/test_goals.py tests/test_dependency.py -q --tb=short

# 7. Code quality
uv run ruff check lms/statement.py lms/goals.py lms/dependency.py lms/society.py scripts/goals/arc_to_goal.py
uv run mypy lms/statement.py lms/goals.py lms/dependency.py

echo "26Q3-HARN-28: verification passed"
