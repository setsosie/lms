#!/bin/bash
# Verification for 26Q4-HARN-22: Each run samples from its own seed
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-22.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Behaviour
uv run pytest tests/test_sampling_seed.py tests/test_thinking.py tests/test_providers.py -q

# 2. The pieces are where a run reaches them
grep -q 'sampling\["seed"\]' lms/providers/openai.py
grep -q '"sampling_seed": seed' lms/run.py
grep -q 'LMS_SEED' lms/run.py
uv run python -m lms.run --help | grep -q -- '--seed'

# 3. Quality
uv run ruff check lms/config.py lms/providers/openai.py lms/run.py tests/test_sampling_seed.py
uv run mypy lms/config.py lms/providers/openai.py lms/run.py

echo "26Q4-HARN-22: verification passed"
