"""OpenAI GPT provider."""

import httpx
from openai import AsyncOpenAI

from lms.config import ProviderConfig
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI GPT models."""

    name: str = "openai"

    def __init__(self, config: ProviderConfig, timeout: float = 300.0) -> None:
        """Initialize the OpenAI provider.

        Args:
            config: Provider configuration with API key and model
            timeout: Request timeout in seconds (default 5 minutes)
        """
        super().__init__(config)
        # Use explicit timeout to prevent infinite hangs
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResponse:
        """Generate a response using GPT.

        Uses async client for proper concurrency with asyncio.gather.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response (defaults to config.max_tokens)

        Returns:
            GenerationResponse with content and token usage
        """
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        api_messages = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        api_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            max_completion_tokens=effective_max_tokens,
        )

        # Extract cache tokens if available
        cache_read = 0
        if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
            cache_read = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )
        self._track_usage(usage)

        return GenerationResponse(
            content=response.choices[0].message.content,
            usage=usage,
            provider=self.name,
        )
