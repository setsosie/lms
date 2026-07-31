#!/usr/bin/env bash
# Verification script for 26Q3-INFRA-01: local OpenAI-compatible serving path.
#
# Discrimination note: every check below fails at the merge base, where
# `ProviderConfig` had no `base_url` field, `Config.from_env` never read
# `LMS_OPENAI_BASE_URL`, and `grep -rn base_url lms/` returned nothing — so
# `OpenAIProvider` could only ever reach api.openai.com.
set -uo pipefail

fail=0
section() { printf '\n== %s ==\n' "$1"; }
check() {
  if eval "$2" >/dev/null 2>&1; then
    printf '  ok   %s\n' "$1"
  else
    printf '  FAIL %s\n' "$1"
    fail=1
  fi
}

section "1. ProviderConfig carries the endpoint"
check "base_url defaults to None (hosted OpenAI unchanged)" \
  "uv run python -c \"
from lms.config import ProviderConfig
assert ProviderConfig(api_key='k', model='m').base_url is None
\""

check "base_url is settable" \
  "uv run python -c \"
from lms.config import ProviderConfig
c = ProviderConfig(api_key='k', model='m', base_url='http://localhost:8000/v1')
assert c.base_url == 'http://localhost:8000/v1'
\""

section "2. Environment round-trip"
# env_path points at a nonexistent file so the repo's real .env cannot leak in.
check "LMS_OPENAI_BASE_URL reaches ProviderConfig" \
  "uv run python -c \"
import os
from pathlib import Path
from unittest import mock
from lms.config import Config
env = {'OPENAI_API_KEY': 'dummy', 'LMS_OPENAI_BASE_URL': 'http://localhost:8000/v1'}
with mock.patch.dict(os.environ, env, clear=True):
    cfg = Config.from_env(env_path=Path('/nonexistent/.env'))
assert cfg.openai.base_url == 'http://localhost:8000/v1'
\""

check "an empty LMS_OPENAI_BASE_URL is treated as unset" \
  "uv run python -c \"
import os
from pathlib import Path
from unittest import mock
from lms.config import Config
env = {'OPENAI_API_KEY': 'dummy', 'LMS_OPENAI_BASE_URL': ''}
with mock.patch.dict(os.environ, env, clear=True):
    cfg = Config.from_env(env_path=Path('/nonexistent/.env'))
assert cfg.openai.base_url is None, 'empty string must not reach the SDK'
\""

section "3. The client actually points at the endpoint"
check "base_url threads into AsyncOpenAI" \
  "uv run python -c \"
from lms.config import ProviderConfig
from lms.providers.openai import OpenAIProvider
p = OpenAIProvider(ProviderConfig(api_key='k', model='m', base_url='http://localhost:8000/v1'))
assert str(p.client.base_url).rstrip('/') == 'http://localhost:8000/v1'
\""

check "unset base_url keeps the hosted default" \
  "uv run python -c \"
from lms.config import ProviderConfig
from lms.providers.openai import OpenAIProvider
p = OpenAIProvider(ProviderConfig(api_key='k', model='m'))
assert str(p.client.base_url).rstrip('/') == 'https://api.openai.com/v1'
\""

section "4. Escape-hatch providers untouched"
check "no base_url wiring leaked into the Anthropic provider" \
  "! grep -q base_url lms/providers/anthropic.py"

check "no base_url wiring leaked into the Google provider" \
  "! grep -q base_url lms/providers/google.py"

section "5. Test suite"
check "tests/test_local_serving.py passes (incl. end-to-end vs a stub /v1)" \
  "uv run pytest tests/test_local_serving.py -q"

check "provider and config suites still pass" \
  "uv run pytest tests/test_providers.py tests/test_config.py -q"

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf 'PASS 26Q3-INFRA-01\n'
else
  printf 'FAIL 26Q3-INFRA-01\n'
fi
exit "$fail"
