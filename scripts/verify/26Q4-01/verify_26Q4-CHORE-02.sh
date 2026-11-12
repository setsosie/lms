#!/bin/bash
# Verification for 26Q4-CHORE-02: The run script serves on a node with no CUDA toolkit, and takes a budget
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-02.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The script parses and behaves, against stub programs
bash -n scripts/slurm/lms_run.sbatch
uv run pytest tests/test_run_script.py -q

# 2. The pieces are in the tracked script, not a patched copy
grep -q 'VLLM_USE_FLASHINFER_SAMPLER' scripts/slurm/lms_run.sbatch
grep -q 'LMS_EXTRA_VLLM_ARGS' scripts/slurm/lms_run.sbatch
grep -q 'MAX_TOKENS' scripts/slurm/lms_run.sbatch
grep -q 'gres=gpu:nvidia_h100_nvl:4' scripts/slurm/lms_run.sbatch

# 3. Quality
uv run ruff check tests/test_run_script.py

echo "26Q4-CHORE-02: verification passed"
