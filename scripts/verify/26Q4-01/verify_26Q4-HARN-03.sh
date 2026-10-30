#!/bin/bash
# Verification for 26Q4-HARN-03: Arm-comparison report — pooled isolated arm, distinct statements, paired tests
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-03.sh
set -e

# 1. Files exist
test -f lms/arms.py
test -f scripts/compare_arms.py
test -f tests/test_arms.py

# 2. Required content present
grep -q "def statement_outcomes" lms/arms.py
grep -q "def arm_report" lms/arms.py
grep -q "def paired_comparison" lms/arms.py
grep -q "untagged_verified" lms/arms.py
grep -q "^def counts_as_novel\|^counts_as_novel" lms/accounting.py

# 3. Import and CLI smoke
uv run python -c "from lms.arms import arm_report, paired_comparison"
uv run python scripts/compare_arms.py --help

# 4. Tests pass
uv run pytest tests/test_arms.py tests/test_accounting.py -v --tb=short

# 5. Code quality
uv run ruff check lms/arms.py lms/accounting.py scripts/compare_arms.py tests/test_arms.py
uv run mypy lms/arms.py

echo "26Q4-HARN-03: verification passed"
