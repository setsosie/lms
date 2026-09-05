#!/bin/bash
# Verification for 26Q3-HARN-25: Proof sketches — an artifact may leave named child lemmas open
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-25.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. Files exist
test -f lms/sketch.py
test -f tests/test_sketch.py
test -f tests/test_dependency_leaves.py
test -f tests/test_society_sketch.py

# 2. Required symbols present
grep -q "def split_sketch" lms/sketch.py
grep -q 'SKETCH = "sketch"' lms/lean/interface.py
grep -q "allow_sorry" lms/lean/interface.py
grep -q "allow_sorry" lms/lean/real.py
grep -q "allow_sorry" lms/lean/mock.py
grep -q "allow_sorry" lms/lean/mcp.py
grep -q "def add_open_leaves" lms/dependency.py
grep -q "artifacts_sketched" lms/society.py
grep -q "open_children" lms/artifacts.py

# 3. Prompts describe the sketch form and no longer forbid sorry outright
grep -q "sketch" lms/working_group.py
! grep -q "Do NOT use \`sorry\`. Only propose complete" lms/working_group.py

# 4. Import smoke
uv run python -c "from lms.sketch import split_sketch, Sketch, OpenChild"

# 5. Tests pass
uv run pytest tests/test_sketch.py tests/test_dependency_leaves.py tests/test_society_sketch.py tests/test_lean.py -q --tb=short

# 6. Code quality
uv run ruff check lms/sketch.py lms/dependency.py lms/society.py lms/lean/
uv run mypy lms/sketch.py lms/dependency.py

echo "26Q3-HARN-25: verification passed"
