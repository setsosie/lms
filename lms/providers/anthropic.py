"""Anthropic Claude provider."""

import httpx
from anthropic import AsyncAnthropic

from lms.config import ProviderConfig
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""

    name: str = "anthropic"

    def __init__(self, config: ProviderConfig, timeout: float = 600.0) -> None:
        """Initialize the Anthropic provider.

        Args:
            config: Provider configuration with API key and model
            timeout: Request timeout in seconds (default 10 minutes for long generations)
        """
        super().__init__(config)
        # Use explicit timeout to prevent infinite hangs
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResponse:
        """Generate a response using Claude.

        Uses async streaming to handle large max_tokens values and
        allow proper concurrency with asyncio.gather.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response (defaults to config.max_tokens)

        Returns:
            GenerationResponse with content and token usage
        """
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        kwargs = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": effective_max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Use async streaming for proper concurrency
        content_parts = []
        input_tokens = 0
        output_tokens = 0

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                content_parts.append(text)
            # Get final message for usage stats
            final_message = await stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens
            # Cache tokens (if available)
            cache_read = getattr(final_message.usage, 'cache_read_input_tokens', 0) or 0
            cache_write = getattr(final_message.usage, 'cache_creation_input_tokens', 0) or 0

        content = "".join(content_parts)

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        self._track_usage(usage)

        return GenerationResponse(
            content=content,
            usage=usage,
            provider=self.name,
        )
