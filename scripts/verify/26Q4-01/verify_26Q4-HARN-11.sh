#!/bin/bash
# Verification for 26Q4-HARN-11: Fixed probe set from the Stacks category chapter, scored by penalized cost
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-11.sh
set -e

# 1. Files exist
test -f data/probes/categories_v1.json
test -f lms/probes.py
test -f scripts/probes/run_baseline.py

# 2. Required content present
grep -q "par2_cost" lms/probes.py
grep -q "require_clean_cutoff" lms/probes.py
grep -q -- "--dev-only" scripts/probes/run_baseline.py

# 3. Tests pass
uv run pytest tests/test_probes.py -v --tb=short -m "not lean"

# 4. Code quality
uv run ruff check lms/probes.py scripts/probes/run_baseline.py
uv run mypy lms/probes.py

echo "26Q4-HARN-11: verification passed"
