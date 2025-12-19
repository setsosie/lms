"""LLM Provider abstraction for LMS."""

from lms.config import ProviderConfig
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.providers.anthropic import AnthropicProvider
from lms.providers.openai import OpenAIProvider
from lms.providers.google import GoogleProvider

__all__ = [
    "BaseLLMProvider",
    "GenerationResponse",
    "Message",
    "TokenUsage",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "create_provider",
]


def create_provider(name: str, config: ProviderConfig) -> BaseLLMProvider:
    """Create an LLM provider by name.

    Args:
        name: Provider name ('anthropic', 'openai', or 'google')
        config: Provider configuration with API key and model

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider name is unknown
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "google": GoogleProvider,
    }

    provider_class = providers.get(name)
    if provider_class is None:
        raise ValueError(f"Unknown provider: {name}")

    return provider_class(config)
