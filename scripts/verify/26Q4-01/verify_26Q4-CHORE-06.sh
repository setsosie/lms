#!/bin/bash
# Verification for 26Q4-CHORE-06: LMS jobs run on one GPU each, in up to four lanes
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-06.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The kit parses
bash -n scripts/slurm/queue/lms_q_lib.sh
bash -n scripts/slurm/queue/lms_q_job.sbatch

# 2. The pieces are where a round reaches them
grep -q 'gpu:nvidia_h100_nvl:1' scripts/slurm/queue/lms_q_lib.sh
grep -q '^q_lanes()' scripts/slurm/queue/lms_q_lib.sh
grep -q '^q_lane()' scripts/slurm/queue/lms_q_lib.sh
grep -q 'LMS_Q_LOCK' scripts/slurm/queue/lms_q_job.sbatch
grep -q 'lake build --no-build' scripts/slurm/queue/lms_q_job.sbatch

# 3. The checks pass on the fake server
uv run pytest tests/test_queue_kit.py -q

# 4. The procedure is written down
grep -q '^## Lanes' docs/runbooks/2026-11-10-queued-rounds.md

echo "26Q4-CHORE-06: verification passed"
