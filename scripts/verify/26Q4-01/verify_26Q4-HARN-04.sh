#!/bin/bash
# Verification for 26Q4-HARN-04: Reset the foundation to its seed at every generation (the R9 arm)
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-04.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The flag, its command-line switch, its metadata field and its batch knob
grep -q "reset_foundation_each_generation" lms/society.py
grep -q -- "--reset-each-generation" lms/run.py
grep -q '"reset_each_generation"' lms/run.py
grep -q "RESET_EACH_GENERATION" scripts/slurm/lms_run.sbatch
uv run python -m lms.run --help | grep -q -- "--reset-each-generation"

# 2. The reset, what it leaves alone, the flag-off regression, and the wiring
uv run pytest tests/test_foundation_persistence.py tests/test_society.py -q

# 3. The batch script passes the flag only when asked
uv run pytest tests/test_run_script.py -q

# 4. The neighbours it touches: rollback, tag reopening, resume
uv run pytest tests/test_foundation_rollback.py tests/test_goal_reopens_tags.py tests/test_resume_foundation.py -q

# 5. Quality
uv run ruff check lms/society.py lms/run.py tests/test_foundation_persistence.py tests/test_society.py tests/test_run_script.py
uv run mypy lms/society.py lms/run.py

echo "26Q4-HARN-04: verification passed"
