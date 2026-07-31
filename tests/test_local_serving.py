"""Tests for the local OpenAI-compatible serving path (26Q3-INFRA-01).

vLLM, SGLang and Ollama all expose an OpenAI-compatible `/v1`, so pointing the
harness at a self-hosted model is a `base_url` away. These tests pin that switch.

Every test that touches `Config.from_env` passes an empty `env_path`: the
repo-root `.env` otherwise leaks through `load_dotenv` even under
`clear=True`, because an empty environ is exactly the case `load_dotenv`
is designed to fill.
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from lms.config import Config, ProviderConfig
from lms.providers.openai import OpenAIProvider

HOSTED_DEFAULT = "https://api.openai.com/v1"


@pytest.fixture
def empty_env(tmp_path: Path) -> Path:
    """An env file with no variables, to neutralize the repo's real `.env`."""
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    return env_file


class TestProviderConfigBaseURL:
    def test_base_url_defaults_to_none(self):
        """Hosted OpenAI stays the default when nothing is configured."""
        config = ProviderConfig(api_key="k", model="m")
        assert config.base_url is None

    def test_accepts_base_url(self):
        config = ProviderConfig(
            api_key="k", model="m", base_url="http://localhost:8000/v1"
        )
        assert config.base_url == "http://localhost:8000/v1"


class TestConfigFromEnv:
    def test_reads_base_url_from_env(self, empty_env: Path):
        env = {
            "OPENAI_API_KEY": "dummy-key-vllm-ignores-this",
            "LMS_OPENAI_MODEL": "Qwen/Qwen3-Coder-30B-A3B",
            "LMS_OPENAI_BASE_URL": "http://localhost:8000/v1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.base_url == "http://localhost:8000/v1"
        assert config.openai.model == "Qwen/Qwen3-Coder-30B-A3B"

    def test_base_url_absent_leaves_none(self, empty_env: Path):
        """Unset env var must not invent a base_url."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.base_url is None


class TestOpenAIProviderClient:
    def test_base_url_threads_into_client(self):
        config = ProviderConfig(
            api_key="k", model="m", base_url="http://localhost:8000/v1"
        )
        provider = OpenAIProvider(config)
        assert str(provider.client.base_url).rstrip("/") == "http://localhost:8000/v1"

    def test_unset_base_url_keeps_hosted_default(self):
        config = ProviderConfig(api_key="k", model="m")
        provider = OpenAIProvider(config)
        assert str(provider.client.base_url).rstrip("/") == HOSTED_DEFAULT
