"""Google Gemini provider."""

import google.generativeai as genai
from google.generativeai.types import RequestOptions

from lms.config import ProviderConfig
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage


class GoogleProvider(BaseLLMProvider):
    """Provider for Google Gemini models."""

    name: str = "google"

    def __init__(self, config: ProviderConfig, timeout: int = 600) -> None:
        """Initialize the Google provider.

        Args:
            config: Provider configuration with API key and model
            timeout: Request timeout in seconds (default 600 = 10 minutes)
        """
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)
        self.timeout = timeout

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResponse:
        """Generate a response using Gemini.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response (defaults to config.max_tokens)

        Returns:
            GenerationResponse with content and token usage
        """
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Build conversation content
        contents = []

        if system_prompt:
            contents.append({"role": "user", "parts": [f"System: {system_prompt}"]})
            contents.append({"role": "model", "parts": ["Understood."]})

        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [msg.content]})

        response = await self.model.generate_content_async(
            contents,
            generation_config=genai.types.GenerationConfig(max_output_tokens=effective_max_tokens),
            request_options=RequestOptions(timeout=self.timeout),
            tool_config={"function_calling_config": {"mode": "NONE"}},  # Disable function calling
        )

        # Extract token counts from Gemini response
        usage_metadata = response.usage_metadata
        usage = TokenUsage(
            input_tokens=usage_metadata.prompt_token_count,
            output_tokens=usage_metadata.candidates_token_count,
        )
        self._track_usage(usage)

        # Handle cases where response.text might fail (e.g., invalid function call)
        try:
            content = response.text
        except ValueError as e:
            # Gemini sometimes returns invalid responses - return empty content
            content = f"[Gemini error: {e}]"

        return GenerationResponse(
            content=content,
            usage=usage,
            provider=self.name,
        )
