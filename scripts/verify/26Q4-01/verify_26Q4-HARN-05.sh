#!/bin/bash
# Verification for 26Q4-HARN-05: Best-of-k mode — k independent attempts per available node, Lean selects (the K9 arm)
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-05.sh
set -e

# 1. Files exist
test -f lms/best_of_k.py
test -f tests/test_best_of_k.py

# 2. Required content present
grep -q -- "--best-of-k" lms/run.py
grep -q "best_of_k.json" lms/best_of_k.py
grep -q "BEST_OF_K" scripts/slurm/lms_run.sbatch

# 3. Import and CLI smoke
uv run python -c "import lms.best_of_k"
uv run python -m lms.run --help | grep -q -- "--best-of-k"

# 4. Tests pass
uv run pytest tests/test_best_of_k.py tests/test_society.py -v --tb=short

# 5. Code quality
uv run ruff check lms/best_of_k.py lms/society.py lms/run.py tests/test_best_of_k.py
uv run mypy lms/best_of_k.py lms/society.py lms/run.py

echo "26Q4-HARN-05: verification passed"
