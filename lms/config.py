"""Configuration management for LMS."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 64k tokens - limited by Claude Opus 4.5's max output tokens
DEFAULT_MAX_TOKENS = 64_000


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    api_key: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class Config:
    """Main configuration for LMS experiments."""

    # Provider settings
    anthropic: Optional[ProviderConfig] = None
    openai: Optional[ProviderConfig] = None
    google: Optional[ProviderConfig] = None
    default_provider: str = "anthropic"

    # Experiment settings
    n_agents: int = 3
    n_generations: int = 6
    experiments_dir: Path = field(default_factory=lambda: Path("experiments"))

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        config = cls()

        # Load Anthropic config
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            config.anthropic = ProviderConfig(
                api_key=anthropic_key,
                model=os.getenv("LMS_ANTHROPIC_MODEL", "claude-sonnet-4-5-20250514"),
            )

        # Load OpenAI config
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            config.openai = ProviderConfig(
                api_key=openai_key,
                model=os.getenv("LMS_OPENAI_MODEL", "gpt-5.2"),
            )

        # Load Google config
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            config.google = ProviderConfig(
                api_key=google_key,
                model=os.getenv("LMS_GOOGLE_MODEL", "gemini-3"),
            )

        config.default_provider = os.getenv("LMS_DEFAULT_PROVIDER", "anthropic")

        return config

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        configs = {
            "anthropic": self.anthropic,
            "openai": self.openai,
            "google": self.google,
        }

        provider_config = configs.get(provider)
        if provider_config is None:
            raise ValueError(
                f"Provider '{provider}' not configured. "
                f"Set {provider.upper()}_API_KEY in .env"
            )
        return provider_config

    def available_providers(self) -> list[str]:
        """List providers that have API keys configured."""
        providers = []
        if self.anthropic:
            providers.append("anthropic")
        if self.openai:
            providers.append("openai")
        if self.google:
            providers.append("google")
        return providers
