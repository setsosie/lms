"""Configuration management for LMS."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Per-request completion cap when LMS_<PROVIDER>_MAX_TOKENS is unset. 64k
# matches the hosted Claude Opus 4.5 output limit; a self-hosted endpoint
# should set the env var to fit its served context window instead (issue #17:
# vLLM 400s any request where prompt_tokens + max_tokens > max_model_len).
DEFAULT_MAX_TOKENS = 64_000


def _max_tokens_from_env(var: str) -> int:
    """Read a per-request completion cap from the environment.

    Blank is treated as unset, matching every other read in `Config.from_env`.
    Anything else must parse as a positive integer: a bad value raises here,
    at config load, rather than surfacing generations later as an HTTP 400
    that reads like a malformed request.
    """
    raw = os.getenv(var) or None
    if raw is None:
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{var} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{var} must be positive, got {value}")
    return value


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    api_key: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    # OpenAI-compatible endpoint (vLLM, SGLang, Ollama). None = the hosted API.
    base_url: str | None = None


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

    # Working Group settings
    use_working_groups: bool = False  # Enable working group mode
    n_working_groups: int = 3  # Number of parallel groups per generation
    group_size: int = 3  # Members per group (1 chair + 1 scribe + N-2 researchers)
    max_turns_per_group: int = 5  # Max conversation turns per group
    max_repair_attempts: int = 2  # Scribe repair turns after a failed verify
    use_planning_panel: bool = True  # Use planning panel for task allocation

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        config = cls()

        # Every read below uses `or`, not os.getenv's default argument: a `.env`
        # copied from `.env.example` leaves keys bare, which yields "" rather
        # than an unset variable. An empty string is never a valid model name,
        # provider name, or URL, so treat blank as absent throughout.

        # Load Anthropic config
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            config.anthropic = ProviderConfig(
                api_key=anthropic_key,
                model=os.getenv("LMS_ANTHROPIC_MODEL") or "claude-opus-4-5-20251101",
                max_tokens=_max_tokens_from_env("LMS_ANTHROPIC_MAX_TOKENS"),
            )

        # Load OpenAI config
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            config.openai = ProviderConfig(
                api_key=openai_key,
                model=os.getenv("LMS_OPENAI_MODEL") or "gpt-5.2",
                max_tokens=_max_tokens_from_env("LMS_OPENAI_MAX_TOKENS"),
                # Blank here would hand the SDK a relative URL.
                base_url=os.getenv("LMS_OPENAI_BASE_URL") or None,
            )

        # Load Google config
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            config.google = ProviderConfig(
                api_key=google_key,
                model=os.getenv("LMS_GOOGLE_MODEL") or "gemini-3",
                max_tokens=_max_tokens_from_env("LMS_GOOGLE_MAX_TOKENS"),
            )

        config.default_provider = os.getenv("LMS_DEFAULT_PROVIDER") or "anthropic"

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
