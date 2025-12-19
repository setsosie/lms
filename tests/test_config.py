"""Tests for configuration management."""

import os
from pathlib import Path
from unittest import mock

import pytest

from lms.config import Config, ProviderConfig


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_create_provider_config(self):
        """ProviderConfig holds api_key and model."""
        config = ProviderConfig(api_key="test-key", model="test-model")
        assert config.api_key == "test-key"
        assert config.model == "test-model"


class TestConfig:
    """Tests for main Config class."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = Config()
        assert config.n_agents == 3
        assert config.n_generations == 6
        assert config.default_provider == "anthropic"
        assert config.anthropic is None
        assert config.openai is None
        assert config.google is None

    def test_from_env_with_anthropic_key(self):
        """Config loads Anthropic credentials from environment."""
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "LMS_ANTHROPIC_MODEL": "claude-3-opus",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env()

        assert config.anthropic is not None
        assert config.anthropic.api_key == "sk-ant-test"
        assert config.anthropic.model == "claude-3-opus"

    def test_from_env_with_openai_key(self):
        """Config loads OpenAI credentials from environment."""
        env = {
            "OPENAI_API_KEY": "sk-openai-test",
            "LMS_OPENAI_MODEL": "gpt-4-turbo",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env()

        assert config.openai is not None
        assert config.openai.api_key == "sk-openai-test"
        assert config.openai.model == "gpt-4-turbo"

    def test_from_env_with_google_key(self):
        """Config loads Google credentials from environment."""
        env = {
            "GOOGLE_API_KEY": "google-test-key",
            "LMS_GOOGLE_MODEL": "gemini-pro",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env()

        assert config.google is not None
        assert config.google.api_key == "google-test-key"
        assert config.google.model == "gemini-pro"

    def test_from_env_uses_default_models(self):
        """Config uses default models when not specified."""
        env = {"ANTHROPIC_API_KEY": "test-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env()

        assert config.anthropic.model == "claude-opus-4-5-20251101"

    def test_from_env_loads_default_provider(self):
        """Config loads default provider from environment."""
        env = {
            "ANTHROPIC_API_KEY": "test",
            "LMS_DEFAULT_PROVIDER": "openai",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env()

        assert config.default_provider == "openai"

    def test_get_provider_config_returns_config(self):
        """get_provider_config returns the provider's config."""
        config = Config(
            anthropic=ProviderConfig(api_key="ant-key", model="claude"),
        )
        provider = config.get_provider_config("anthropic")
        assert provider.api_key == "ant-key"

    def test_get_provider_config_raises_for_missing(self):
        """get_provider_config raises ValueError for unconfigured provider."""
        config = Config()
        with pytest.raises(ValueError, match="not configured"):
            config.get_provider_config("anthropic")

    def test_available_providers_empty(self):
        """available_providers returns empty list when none configured."""
        config = Config()
        assert config.available_providers() == []

    def test_available_providers_lists_configured(self):
        """available_providers lists all configured providers."""
        config = Config(
            anthropic=ProviderConfig(api_key="a", model="m"),
            google=ProviderConfig(api_key="g", model="m"),
        )
        providers = config.available_providers()
        assert "anthropic" in providers
        assert "google" in providers
        assert "openai" not in providers
