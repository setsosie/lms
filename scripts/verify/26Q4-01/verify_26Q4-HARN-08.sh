#!/bin/bash
# Verification for 26Q4-HARN-08: Local provider pool and solve correlation — heterogeneous populations (the H9 arm)
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-08.sh
set -e

# 1. Files exist
test -f tests/test_provider_pool.py

# 2. Required content present
grep -q -- "--provider-pool" lms/run.py
grep -q "def load_provider_pool" lms/config.py
grep -q "def solve_correlation" lms/arms.py

# 3. Import and CLI smoke
uv run python -c "from lms.config import load_provider_pool; from lms.arms import solve_correlation"
uv run python -m lms.run --help | grep -q -- "--provider-pool"

# 4. Tests pass
uv run pytest tests/test_provider_pool.py tests/test_arms.py tests/test_config.py -v --tb=short

# 5. Code quality
uv run ruff check lms/config.py lms/providers/openai.py lms/run.py lms/arms.py scripts/compare_arms.py tests/test_provider_pool.py
uv run mypy lms/config.py lms/providers/openai.py lms/run.py lms/arms.py

echo "26Q4-HARN-08: verification passed"
