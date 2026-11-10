"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lms.config import ProviderConfig


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # 'user', 'assistant', or 'system'
    content: str


@dataclass
class TokenUsage:
    """Token usage statistics from an API call."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0  # Tokens read from cache (cheaper)
    cache_write_tokens: int = 0  # Tokens written to cache (Anthropic)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def effective_input_tokens(self) -> int:
        """Non-cached input tokens (the expensive ones)."""
        return self.input_tokens - self.cache_read_tokens


# What ends a thinking block. Models that think (Qwen3, DeepSeek-R1 and others)
# emit their scratch work first, then this tag, then the answer.
_THINKING_END = "</think>"
_THINKING_START = "<think>"


def split_thinking(text: str) -> tuple[str, str]:
    """Split a reply into (thinking, answer).

    Some chat templates supply the opening tag themselves, so the reply starts
    inside the thinking block and only the closing tag appears in it. The
    answer is what follows the last closing tag: thinking may quote the tag,
    and a model's own reference parser splits there too. A reply with no
    closing tag is all answer, which is every model that does not think and
    also a thinking reply cut off before it finished. The two cannot be told
    apart here.

    Args:
        text: The reply exactly as the server returned it.

    Returns:
        The thinking, stripped, and the answer with only the newlines that
        separated it from the tag removed.
    """
    thinking, tag, answer = text.rpartition(_THINKING_END)
    if not tag:
        return "", text
    return thinking.strip().removeprefix(_THINKING_START).strip(), answer.lstrip("\n")


@dataclass
class GenerationResponse:
    """Response from an LLM generation call.

    Like a letter received from a mathematician - contains the content
    and a record of the correspondence cost.

    `content` is the answer only. `thinking` is the scratch work a thinking
    model produced before it, which nothing downstream may parse, quote back
    to the model or count as a proposal. Its tokens are in `usage` all the
    same: the model spent them.
    """

    content: str
    usage: TokenUsage
    provider: str
    thinking: str = ""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the provider with configuration.

        Args:
            config: Provider configuration with API key and model
        """
        self.config = config
        self.total_tokens_used = 0

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response (defaults to config.max_tokens)

        Returns:
            GenerationResponse with content and token usage
        """
        pass

    def _track_usage(self, usage: TokenUsage) -> None:
        """Track cumulative token usage."""
        self.total_tokens_used += usage.total_tokens
