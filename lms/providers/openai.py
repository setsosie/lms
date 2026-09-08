"""OpenAI GPT provider."""

from typing import cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from lms.config import ProviderConfig
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI GPT models."""

    name: str = "openai"

    def __init__(self, config: ProviderConfig, timeout: float = 300.0) -> None:
        """Initialize the OpenAI provider.

        Args:
            config: Provider configuration with API key, model, and optional base_url
            timeout: Request timeout in seconds (default 5 minutes)
        """
        super().__init__(config)
        # Use explicit timeout to prevent infinite hangs.
        # base_url=None is the SDK's own default, so an unset config leaves
        # hosted-OpenAI behavior untouched; setting it points the same code path
        # at a local vLLM/SGLang/Ollama `/v1`.
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
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
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.config.max_tokens
        )

        raw_messages: list[dict[str, str]] = []

        if system_prompt:
            raw_messages.append({"role": "system", "content": system_prompt})

        raw_messages.extend([{"role": m.role, "content": m.content} for m in messages])
        api_messages = cast("list[ChatCompletionMessageParam]", raw_messages)

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            max_completion_tokens=effective_max_tokens,
        )

        # Extract cache tokens if available. Some OpenAI-compatible servers omit
        # `usage` altogether, or send null counts inside it; report zeros rather
        # than raising. A missing token count is a hole in accounting, not a
        # failed generation, and it must not take down the agent loop.
        api_usage = response.usage
        cache_read = 0
        if api_usage is not None and getattr(api_usage, "prompt_tokens_details", None):
            cache_read = (
                getattr(api_usage.prompt_tokens_details, "cached_tokens", 0) or 0
            )

        usage = TokenUsage(
            input_tokens=(api_usage.prompt_tokens or 0) if api_usage else 0,
            output_tokens=(api_usage.completion_tokens or 0) if api_usage else 0,
            cache_read_tokens=cache_read,
        )
        self._track_usage(usage)

        return GenerationResponse(
            # None happens on refusals and tool-only turns; the harness treats
            # content as text everywhere, so empty is the faithful mapping.
            content=response.choices[0].message.content or "",
            usage=usage,
            provider=self.name,
        )
