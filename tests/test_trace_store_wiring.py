"""Every LLM exchange the cost ledger records also lands in the trace store.

The ledger proves a call was paid for; the trace store keeps what was said.
Same calls, same tokens: in every mode the transcripts must account for
exactly the spend the ledger does.
"""

import json

import pytest

from tests.test_accounting import (
    ARTIFACT_RESPONSE,
    GROUP_RESPONSES,
    PANEL_RESPONSES,
    REVIEW_APPROVE,
    StubLeanVerifier,
    make_committee_society,
    make_society,
)


def transcript_tokens(society) -> int:
    return sum(log.total_tokens() for log in society.trace_store.conversations)


@pytest.mark.asyncio
async def test_flat_mode_transcribes_every_ledgered_call(tmp_path):
    society = make_society(n_agents=2, responses=[ARTIFACT_RESPONSE], tmp_path=tmp_path)
    await society.run_generation(0)

    kinds = {log.metadata.get("kind") for log in society.trace_store.conversations}
    assert kinds == {"propose", "review"}
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_committee_mode_transcribes_panel_group_and_review_calls(tmp_path):
    society = make_committee_society(
        tmp_path,
        PANEL_RESPONSES + GROUP_RESPONSES + [REVIEW_APPROVE],
        use_planning_panel=True,
    )
    await society.run_generation(0)

    # 4 panel + 4 group + 1 review exchanges, one log each
    assert len(society.trace_store.conversations) == 9
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_iterative_session_is_one_log_holding_every_turn(tmp_path):
    society = make_society(
        n_agents=1,
        responses=[ARTIFACT_RESPONSE],
        verifier=StubLeanVerifier(),
        tmp_path=tmp_path,
    )
    society.iterative_mode = True
    await society.run_generation(0)

    [log] = society.trace_store.conversations
    # system prompt, task, verified attempt, writeup request, writeup
    assert [t.role for t in log.turns] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert (log.agent_id, log.generation, log.success) == (
        society.agents[0].id,
        0,
        True,
    )
    assert log.metadata["kind"] == "iterative"
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_checkpoint_writes_the_transcripts(tmp_path):
    society = make_society(n_agents=1, responses=[ARTIFACT_RESPONSE], tmp_path=tmp_path)
    await society.run_generation(0)
    society.save(tmp_path / "run")

    saved = json.loads((tmp_path / "run" / "conversations.json").read_text())
    assert saved and saved[0]["turns"][-1]["content"] == ARTIFACT_RESPONSE
