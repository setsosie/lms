#!/bin/bash
# Verification for 26Q4-HARN-21: A thinking model's scratch work is not its answer
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-21.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The split, the provider and a working group end to end
uv run pytest tests/test_thinking.py -q

# 2. The neighbours that read provider replies
uv run pytest tests/test_providers.py tests/test_working_group.py -q

# 3. The provider splits every reply
grep -q "split_thinking(message.content" lms/providers/openai.py
uv run python -c "from lms.providers.base import GenerationResponse, split_thinking"

# 4. Quality
uv run ruff check lms/providers/base.py lms/providers/openai.py tests/test_thinking.py
uv run mypy lms/providers/base.py lms/providers/openai.py

echo "26Q4-HARN-21: verification passed"
