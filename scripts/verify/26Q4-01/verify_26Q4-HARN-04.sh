#!/bin/bash
# Verification for 26Q4-HARN-04: Reset the foundation to its seed at every generation (the R9 arm)
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-04.sh
set -e

# 1. Required content present
grep -q "reset_foundation_each_generation" lms/society.py
grep -q -- "--reset-each-generation" lms/run.py
grep -q "reset_each_generation" lms/run.py
grep -q "RESET_EACH_GENERATION" scripts/slurm/lms_run.sbatch

# 2. Import and CLI smoke
uv run python -c "from lms.society import Society; Society.reset_foundation"
uv run python -m lms.run --help | grep -q -- "--reset-each-generation"

# 3. Tests pass
uv run pytest tests/test_foundation_persistence.py tests/test_society.py -v --tb=short

# 4. Code quality
uv run ruff check lms/society.py lms/run.py
uv run mypy lms/society.py lms/run.py

echo "26Q4-HARN-04: verification passed"
