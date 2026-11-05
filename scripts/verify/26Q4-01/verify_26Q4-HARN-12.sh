#!/bin/bash
# Verification for 26Q4-HARN-12: Committee prompts come from the registry, so overrides and recorded versions are true
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-12.sh
set -e

# 1. Required content present
grep -q "get_prompt" lms/working_group.py
grep -q "get_prompt" lms/planning.py
! grep -q "_SYSTEM_PROMPT = " lms/working_group.py || exit 1
! grep -q "_SYSTEM_PROMPT = " lms/planning.py || exit 1

# 2. Tests pass
uv run pytest tests/test_committee_prompts.py -v --tb=short

# 3. Code quality
uv run ruff check lms/prompts.py lms/working_group.py lms/planning.py
uv run mypy lms/prompts.py lms/working_group.py lms/planning.py

echo "26Q4-HARN-12: verification passed"
