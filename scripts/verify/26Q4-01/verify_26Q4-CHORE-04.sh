#!/bin/bash
# Verification for 26Q4-CHORE-04: Rounds at different commits can wait in the queue together
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-04.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The kit and its fake server are here
test -f scripts/slurm/queue/lms_q_lib.sh
test -f scripts/slurm/queue/lms_q_job.sbatch
test -f scripts/slurm/queue/lms_all_run.sh
test -f tests/slurm_queue/fakebox.sh

# 2. The checks pass on the fake server
uv run pytest tests/test_queue_kit.py -q

# 3. The procedure is written down
test -f docs/runbooks/2026-11-10-queued-rounds.md
grep -q "afterany" docs/runbooks/2026-11-10-queued-rounds.md

# 4. Quality
uv run ruff check tests/test_queue_kit.py

echo "26Q4-CHORE-04: verification passed"
