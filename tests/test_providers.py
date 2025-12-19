"""Tests for LLM provider abstraction."""

from unittest import mock

import pytest

from lms.config import ProviderConfig
from lms.providers.base import BaseLLMProvider, Message
from lms.providers.anthropic import AnthropicProvider
from lms.providers.openai import OpenAIProvider
from lms.providers.google import GoogleProvider


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Message holds role and content."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestBaseLLMProvider:
    """Tests for the abstract base provider."""

    def test_cannot_instantiate_directly(self):
        """BaseLLMProvider is abstract and cannot be instantiated."""
        config = ProviderConfig(api_key="test", model="test")
        with pytest.raises(TypeError):
            BaseLLMProvider(config)  # type: ignore

    def test_subclass_must_implement_generate(self):
        """Subclasses must implement generate method."""

        class IncompleteProvider(BaseLLMProvider):
            pass

        config = ProviderConfig(api_key="test", model="test")
        with pytest.raises(TypeError):
            IncompleteProvider(config)  # type: ignore


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_initialization(self):
        """AnthropicProvider initializes with config."""
        config = ProviderConfig(api_key="sk-ant-test", model="claude-3-opus")
        provider = AnthropicProvider(config)
        assert provider.config == config
        assert provider.name == "anthropic"

    def test_uses_async_client(self):
        """AnthropicProvider uses AsyncAnthropic for true async support."""
        from anthropic import AsyncAnthropic

        config = ProviderConfig(api_key="sk-ant-test", model="claude-3-opus")
        provider = AnthropicProvider(config)
        assert isinstance(provider.client, AsyncAnthropic)

    @pytest.mark.asyncio
    async def test_generate_calls_api(self):
        """generate() calls the Anthropic API with correct parameters."""
        config = ProviderConfig(api_key="sk-ant-test", model="claude-3-opus")
        provider = AnthropicProvider(config)

        # Mock the async streaming response
        mock_final_message = mock.MagicMock()
        mock_final_message.usage.input_tokens = 10
        mock_final_message.usage.output_tokens = 20

        # Create async iterator for text_stream
        async def async_text_stream():
            for text in ["Hello ", "from ", "Claude"]:
                yield text

        mock_stream = mock.MagicMock()
        mock_stream.text_stream = async_text_stream()
        mock_stream.get_final_message = mock.AsyncMock(return_value=mock_final_message)

        # Mock async context manager
        mock_stream.__aenter__ = mock.AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = mock.AsyncMock(return_value=False)

        with mock.patch.object(
            provider.client.messages, "stream", return_value=mock_stream
        ) as mock_stream_call:
            messages = [Message(role="user", content="Hi")]
            result = await provider.generate(messages)

            mock_stream_call.assert_called_once()
            call_kwargs = mock_stream_call.call_args.kwargs
            assert call_kwargs["model"] == "claude-3-opus"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
            assert result.content == "Hello from Claude"
            assert result.usage.input_tokens == 10
            assert result.usage.output_tokens == 20


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_initialization(self):
        """OpenAIProvider initializes with config."""
        config = ProviderConfig(api_key="sk-openai-test", model="gpt-4")
        provider = OpenAIProvider(config)
        assert provider.config == config
        assert provider.name == "openai"

    def test_uses_async_client(self):
        """OpenAIProvider uses AsyncOpenAI for true async support."""
        from openai import AsyncOpenAI

        config = ProviderConfig(api_key="sk-openai-test", model="gpt-4")
        provider = OpenAIProvider(config)
        assert isinstance(provider.client, AsyncOpenAI)

    @pytest.mark.asyncio
    async def test_generate_calls_api(self):
        """generate() calls the OpenAI API with correct parameters."""
        config = ProviderConfig(api_key="sk-openai-test", model="gpt-4")
        provider = OpenAIProvider(config)

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="Hello from GPT"))]
        mock_response.usage = mock.MagicMock(prompt_tokens=15, completion_tokens=25)

        # Mock as async method
        with mock.patch.object(
            provider.client.chat.completions, "create",
            new_callable=mock.AsyncMock,
            return_value=mock_response
        ) as mock_create:
            messages = [Message(role="user", content="Hi")]
            result = await provider.generate(messages)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
            assert result.content == "Hello from GPT"
            assert result.usage.input_tokens == 15
            assert result.usage.output_tokens == 25


class TestGoogleProvider:
    """Tests for GoogleProvider."""

    def test_initialization(self):
        """GoogleProvider initializes with config."""
        config = ProviderConfig(api_key="google-test", model="gemini-pro")
        provider = GoogleProvider(config)
        assert provider.config == config
        assert provider.name == "google"

    @pytest.mark.asyncio
    async def test_generate_calls_api(self):
        """generate() calls the Google API with correct parameters."""
        config = ProviderConfig(api_key="google-test", model="gemini-pro")
        provider = GoogleProvider(config)

        mock_response = mock.MagicMock()
        mock_response.text = "Hello from Gemini"
        mock_response.usage_metadata = mock.MagicMock(
            prompt_token_count=12, candidates_token_count=18
        )

        with mock.patch.object(
            provider.model, "generate_content_async", return_value=mock_response
        ) as mock_generate:
            messages = [Message(role="user", content="Hi")]
            result = await provider.generate(messages)

            mock_generate.assert_called_once()
            assert result.content == "Hello from Gemini"
            assert result.usage.input_tokens == 12
            assert result.usage.output_tokens == 18


class TestProviderFactory:
    """Tests for provider factory function."""

    def test_create_anthropic_provider(self):
        """Factory creates AnthropicProvider for 'anthropic'."""
        from lms.providers import create_provider

        config = ProviderConfig(api_key="test", model="test")
        provider = create_provider("anthropic", config)
        assert isinstance(provider, AnthropicProvider)

    def test_create_openai_provider(self):
        """Factory creates OpenAIProvider for 'openai'."""
        from lms.providers import create_provider

        config = ProviderConfig(api_key="test", model="test")
        provider = create_provider("openai", config)
        assert isinstance(provider, OpenAIProvider)

    def test_create_google_provider(self):
        """Factory creates GoogleProvider for 'google'."""
        from lms.providers import create_provider

        config = ProviderConfig(api_key="test", model="test")
        provider = create_provider("google", config)
        assert isinstance(provider, GoogleProvider)

    def test_create_unknown_provider_raises(self):
        """Factory raises ValueError for unknown provider."""
        from lms.providers import create_provider

        config = ProviderConfig(api_key="test", model="test")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown", config)
