#!/bin/bash
# Verification for 26Q4-SPG-03: Soft-prompt evolution loop, selected by Lean on the probe set
# Run: bash scripts/verify/26Q4-01/verify_26Q4-SPG-03.sh
set -e

# 1. Files exist
test -f lms/soft_prompt_evolution.py
test -f scripts/soft_prompt/evolve.py

# 2. Required content present
grep -q "def marginal_coverage" lms/soft_prompt_evolution.py
grep -q "prompt_embeds" lms/providers/openai.py
grep -q -- "--confirm" scripts/soft_prompt/evolve.py

# 3. Tests pass
uv run pytest tests/test_soft_prompt_evolution.py tests/test_soft_prompt_requests.py -v --tb=short

# 4. Code quality
uv run ruff check lms/soft_prompt_evolution.py scripts/soft_prompt/evolve.py lms/providers/openai.py lms/providers/base.py
uv run mypy lms/soft_prompt_evolution.py lms/providers/openai.py lms/providers/base.py

echo "26Q4-SPG-03: verification passed"
