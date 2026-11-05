#!/bin/bash
# Verification for 26Q4-SPG-02: Soft-prompt merge operators — averaging, multiple parents, per-token crossover
# Run: bash scripts/verify/26Q4-01/verify_26Q4-SPG-02.sh
set -e

# 1. Files exist
test -f lms/genome_merge.py
test -f scripts/research/soft_prompt_merge_sim.py

# 2. Required content present
grep -q "def merge_mean" lms/genome_merge.py
grep -q "def merge_per_token" lms/genome_merge.py
grep -q "def population_stats" lms/genome_merge.py

# 3. Tests pass
uv run pytest tests/test_genome_merge.py -v --tb=short

# 4. Code quality
uv run ruff check lms/genome_merge.py
uv run mypy lms/genome_merge.py

echo "26Q4-SPG-02: verification passed"
