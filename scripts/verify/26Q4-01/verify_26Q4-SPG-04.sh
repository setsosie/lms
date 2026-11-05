#!/bin/bash
# Verification for 26Q4-SPG-04: Gradient-tuned soft prompt, as an option alongside evolution
# Run: bash scripts/verify/26Q4-01/verify_26Q4-SPG-04.sh
set -e

# 1. Files exist
test -f lms/tuning_corpus.py
test -f scripts/soft_prompt/tune.py

# 2. Required content present
grep -q "def filter_corpus" lms/tuning_corpus.py

# 3. Tests pass
uv run pytest tests/test_tuning_corpus.py -v --tb=short

# 4. Code quality
uv run ruff check lms/tuning_corpus.py scripts/soft_prompt/tune.py
uv run mypy lms/tuning_corpus.py

echo "26Q4-SPG-04: verification passed"
