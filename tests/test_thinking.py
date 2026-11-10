"""A thinking model's scratch work is not its answer.

Qwen3.6 opens every reply inside a thinking block: the chat template supplies
the opening tag, so the reply the server returns is scratch work, `</think>`,
then the answer. In round R5e's run the scratch work was 87% of the first reply
and held all 29 of its Lean code fences; the answer held none, only the
`<artifact>` block.

The harness read the whole reply as the answer. A working group took the first
code fence of each reply for its blackboard, which is the first thing the model
tried while thinking, and pasted every reply into the next prompt, thinking
included, until the server refused the prompt.
"""

from __future__ import annotations

from unittest import mock

import pytest

from lms.config import ProviderConfig
from lms.providers.base import GenerationResponse, split_thinking
from lms.providers.openai import OpenAIProvider
from lms.working_group import create_working_group

SCRATCH = (
    "Let me think. SCRATCH first try:\n"
    "```lean\ntheorem scratch_attempt : False := by simp\n```\n"
    "That fails. SCRATCH second try:\n"
    "```lean\ntheorem scratch_again : 1 = 2 := rfl\n```\n"
    "I could say CONSENSUS REACHED but the others have not seen it yet.\n"
)
ANSWER_CODE = "theorem the_answer : 1 + 1 = 2 := rfl"
ANSWER = f"Here is my proposal.\n```lean\n{ANSWER_CODE}\n```\n"
SCRIBE_ANSWER = (
    "<artifact>\ntype: lemma\nname: the_answer\nstacks_tag: 0014\n"
    f"description: one plus one\nlean: |\n  {ANSWER_CODE}\nreferences: []\n</artifact>\n"
)


def reply(answer: str, scratch: str = SCRATCH) -> str:
    """A reply as the server returns it: no opening tag, then `</think>`."""
    return f"{scratch}</think>\n\n{answer}"


class TestSplitThinking:
    def test_a_reply_with_no_thinking_is_all_answer(self) -> None:
        assert split_thinking("Hello from GPT") == ("", "Hello from GPT")

    def test_scratch_work_before_the_closing_tag_is_not_the_answer(self) -> None:
        thinking, answer = split_thinking(reply(ANSWER))
        assert answer == ANSWER
        assert "SCRATCH" in thinking
        assert "SCRATCH" not in answer

    def test_an_opening_tag_is_dropped_with_the_thinking(self) -> None:
        thinking, answer = split_thinking("<think>\nhm\n</think>\nyes")
        assert thinking == "hm"
        assert answer == "yes"

    def test_the_answer_starts_after_the_last_closing_tag(self) -> None:
        # Thinking that quotes the tag: the answer is what follows the last one.
        thinking, answer = split_thinking(
            "I must end with </think> soon\n</think>\nyes"
        )
        assert answer == "yes"
        assert thinking.startswith("I must end with </think> soon")

    def test_a_reply_that_stops_after_thinking_has_an_empty_answer(self) -> None:
        # Scratch work is not promoted to an answer because nothing followed it.
        assert split_thinking("only scratch\n</think>\n") == ("only scratch", "")

    def test_the_answer_keeps_its_own_layout(self) -> None:
        _, answer = split_thinking("t</think>\n\n  indented first line\nsecond\n")
        assert answer == "  indented first line\nsecond\n"


def provider_returning(*contents: str, reasoning: str | None = None) -> OpenAIProvider:
    """A real provider whose HTTP client answers with `contents`, in order."""
    provider = OpenAIProvider(ProviderConfig(api_key="EMPTY", model="lms-generalist"))
    responses = []
    for content in contents:
        message = mock.MagicMock(content=content, reasoning_content=reasoning)
        response = mock.MagicMock()
        response.choices = [mock.MagicMock(message=message)]
        response.usage = mock.MagicMock(
            prompt_tokens=100, completion_tokens=900, prompt_tokens_details=None
        )
        responses.append(response)
    provider.client.chat.completions.create = mock.AsyncMock(side_effect=responses)  # type: ignore[method-assign]
    return provider


class TestOpenAIProviderThinking:
    @pytest.mark.asyncio
    async def test_content_is_the_answer_and_thinking_is_kept_beside_it(self) -> None:
        provider = provider_returning(reply(ANSWER))
        result = await provider.generate([])
        assert result.content == ANSWER
        assert "SCRATCH first try" in result.thinking

    @pytest.mark.asyncio
    async def test_thinking_tokens_are_still_counted(self) -> None:
        # The model spent them, so the cost ledger must see them.
        provider = provider_returning(reply(ANSWER))
        result = await provider.generate([])
        assert result.usage.output_tokens == 900
        assert provider.total_tokens_used == 1000

    @pytest.mark.asyncio
    async def test_a_plain_reply_is_unchanged(self) -> None:
        provider = provider_returning("Hello from GPT")
        result = await provider.generate([])
        assert result.content == "Hello from GPT"
        assert result.thinking == ""

    @pytest.mark.asyncio
    async def test_thinking_a_server_already_separated_is_kept(self) -> None:
        # vLLM's reasoning parsers return it as `reasoning_content`.
        provider = provider_returning("Hello", reasoning="server-side thinking")
        result = await provider.generate([])
        assert result.content == "Hello"
        assert result.thinking == "server-side thinking"

    def test_a_response_has_no_thinking_unless_given(self) -> None:
        response = GenerationResponse(content="x", usage=mock.MagicMock(), provider="p")
        assert response.thinking == ""


class TestWorkingGroupSeesAnswersOnly:
    """The committee path end to end, through the real provider."""

    async def run_group(self):
        # Chair, researcher, chair summary, scribe: every reply thinks first.
        provider = provider_returning(
            reply("Agenda: prove one plus one. No code from me."),
            reply(ANSWER),
            reply("We agree. CONSENSUS REACHED."),
            reply(SCRIBE_ANSWER),
            reply(SCRIBE_ANSWER),
            reply(SCRIBE_ANSWER),
        )
        group = create_working_group(
            group_id=0,
            task_tag="0014",
            task_name="one plus one",
            task_content="Show 1 + 1 = 2.",
            provider=provider,
            max_turns=3,
        )
        artifact = await group.run_session()
        sent = [
            "\n".join(str(m["content"]) for m in call.kwargs["messages"])
            for call in provider.client.chat.completions.create.call_args_list  # type: ignore[attr-defined]
        ]
        return group, artifact, sent

    @pytest.mark.asyncio
    async def test_the_blackboard_holds_the_proposal_not_the_first_scratch_fence(
        self,
    ) -> None:
        group, _, _ = await self.run_group()
        assert group.state.blackboard == ANSWER_CODE

    @pytest.mark.asyncio
    async def test_no_later_prompt_carries_an_earlier_reply_s_thinking(self) -> None:
        _, _, sent = await self.run_group()
        assert len(sent) >= 3
        assert all("SCRATCH" not in prompt for prompt in sent)
        # The answers themselves do carry forward.
        assert any("Here is my proposal." in prompt for prompt in sent[2:])

    @pytest.mark.asyncio
    async def test_the_artifact_is_the_scribe_s_answer(self) -> None:
        _, artifact, _ = await self.run_group()
        assert artifact is not None
        assert artifact["lean"].strip() == ANSWER_CODE
        assert "scratch_attempt" not in artifact["raw"]

    @pytest.mark.asyncio
    async def test_consensus_is_not_read_out_of_thinking(self) -> None:
        # Every reply's thinking toys with the phrase; only the chair's answer says it.
        group, _, _ = await self.run_group()
        said_it = [m for m in group.state.messages if "CONSENSUS REACHED" in m.content]
        assert len(said_it) == 1
