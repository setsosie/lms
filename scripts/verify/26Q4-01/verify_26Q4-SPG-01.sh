#!/bin/bash
# Verification for 26Q4-SPG-01: Washout probe — does a soft prompt still change the model at our context lengths?
# Run: bash scripts/verify/26Q4-01/verify_26Q4-SPG-01.sh
set -e

# 1. Files exist
test -f lms/soft_prompt_metrics.py
test -f scripts/soft_prompt/washout_probe.py

# 2. Required content present
grep -q "next_token_kl" lms/soft_prompt_metrics.py
grep -q "inputs_embeds" scripts/soft_prompt/washout_probe.py

# 3. Tests pass
uv run pytest tests/test_soft_prompt_metrics.py -v --tb=short

# 4. Code quality
uv run ruff check lms/soft_prompt_metrics.py scripts/soft_prompt/washout_probe.py
uv run mypy lms/soft_prompt_metrics.py

echo "26Q4-SPG-01: verification passed"
