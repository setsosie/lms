#!/bin/bash
# Verification for 26Q4-HARN-01: A resumed run keeps its foundation where Lean can import it
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-01.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
#
# Discrimination: at main @ 0c3ad12 the test file does not exist, FOUNDATION_SNAPSHOT
# and RESUME are absent, and 6 of the 7 tests fail once the constant is added.
set -e

# 1. The checkpoint carries the foundation, and resume puts it back
grep -q 'FOUNDATION_SNAPSHOT = "foundation.json"' lms/society.py
grep -q 'async def restore_foundation' lms/society.py
grep -q 'await society.restore_foundation()' lms/run.py
grep -q 'def save_metadata' lms/foundation.py

# 2. The batch script can resume
grep -q 'RESUME="${RESUME:-0}"' scripts/slurm/lms_run.sbatch
grep -q 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"' scripts/slurm/lms_run.sbatch
bash -n scripts/slurm/lms_run.sbatch

# 3. The runbook no longer forbids it
! grep -q 'Do not use' docs/runbooks/2026-10-08-box-restore.md
grep -q 'RESUME=1' docs/runbooks/2026-10-08-box-restore.md

# 4. Behaviour
uv run pytest tests/test_resume_foundation.py -q

# 5. Hygiene
uv run ruff check lms/society.py lms/foundation.py lms/run.py tests/test_resume_foundation.py
uv run mypy lms/society.py lms/foundation.py lms/run.py

echo "26Q4-HARN-01: verification passed"
