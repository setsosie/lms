#!/bin/bash
set -e

# 1. Files exist
test -f tests/test_local_serving.py

# 2. base_url plumbed into config + provider
grep -q "base_url" lms/config.py
grep -q "LMS_OPENAI_BASE_URL" lms/config.py
grep -q "base_url" lms/providers/openai.py

# 3. Tests pass
uv run pytest tests/test_local_serving.py -v --tb=short

# 4. Code quality
uv run ruff check lms/config.py lms/providers/openai.py
uv run mypy lms/config.py lms/providers/openai.py

echo "26Q2-INFRA-01: verification passed"
