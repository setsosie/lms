#!/bin/bash
# Verification for 26Q4-HARN-30: Thinking is recorded, and its budget is a setting
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-30.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Behaviour
uv run pytest tests/test_thinking_budget.py tests/test_thinking.py tests/test_traces.py tests/test_trace_store_wiring.py tests/test_sampling_seed.py tests/test_providers.py tests/test_local_serving.py -q

# 2. The pieces are where a run reaches them
grep -q '"thinking_token_budget"' lms/providers/openai.py
grep -q 'max_retries=config.max_retries' lms/providers/openai.py
grep -q '"thinking": t.thinking' lms/traces.py
grep -q 'LMS_OPENAI_TIMEOUT' lms/config.py
grep -q 'LMS_OPENAI_MAX_RETRIES' lms/config.py
grep -q 'LMS_THINKING_BUDGET' lms/run.py
grep -q '"thinking_budget": thinking_budget' lms/run.py
uv run python -m lms.run --help | grep -q -- '--thinking-budget'

# 3. Quality
uv run ruff check lms/config.py lms/providers/openai.py lms/run.py lms/traces.py lms/agent.py lms/planning.py lms/working_group.py tests/test_thinking_budget.py
uv run mypy lms/config.py lms/providers/openai.py lms/run.py lms/traces.py lms/agent.py lms/planning.py lms/working_group.py

echo "26Q4-HARN-30: verification passed"
