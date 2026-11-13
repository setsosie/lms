#!/bin/bash
# Verification for 26Q4-CHORE-05: A vLLM start that hangs in NCCL is retried, not fatal
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-05.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The script parses and behaves, against stub programs
bash -n scripts/slurm/lms_run.sbatch
uv run pytest tests/test_run_script.py -q

# 2. The pieces are in the tracked script
grep -q 'START_LADDER' scripts/slurm/lms_run.sbatch
grep -q 'START_TIMEOUT' scripts/slurm/lms_run.sbatch
grep -q 'disable-custom-all-reduce' scripts/slurm/lms_run.sbatch
grep -q 'NCCL_P2P_DISABLE=1' scripts/slurm/lms_run.sbatch
grep -q 'SERVED_BY=' scripts/slurm/lms_run.sbatch
grep -q 'setsid' scripts/slurm/lms_run.sbatch
grep -q 'nvidia-smi topo -m' scripts/slurm/lms_run.sbatch
grep -q 'scontrol show job -d' scripts/slurm/lms_run.sbatch

# 3. Quality
uv run ruff check tests/test_run_script.py

echo "26Q4-CHORE-05: verification passed"
