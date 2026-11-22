#!/bin/bash
# Verification for 26Q4-CHORE-07: Three runs share one GPU, each in its own worktree
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-07.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The kit parses
bash -n scripts/slurm/queue/lms_q_lib.sh
bash -n scripts/slurm/queue/lms_q_pack.sh

# 2. The pieces are where a round reaches them
grep -q '^q_slots()' scripts/slurm/queue/lms_q_lib.sh
grep -q '^q_submit_pack()' scripts/slurm/queue/lms_q_lib.sh
grep -q '^q_job_log()' scripts/slurm/queue/lms_q_lib.sh
grep -q 'Q_GPU_PACK=' scripts/slurm/queue/lms_q_lib.sh
test -f tests/slurm_queue/test_pack.sh

# 3. The checks pass on the fake server, packs and the kit as before
uv run pytest tests/test_queue_kit.py -q

# 4. The procedure is written down
grep -q '^## Packs' docs/runbooks/2026-11-10-queued-rounds.md

echo "26Q4-CHORE-07: verification passed"
