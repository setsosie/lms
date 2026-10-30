"""Every LLM exchange the cost ledger records also lands in the trace store.

The ledger proves a call was paid for; the trace store keeps what was said.
Same calls, same tokens: in every mode the transcripts must account for
exactly the spend the ledger does.
"""

import json

import pytest

from lms.lean.interface import LeanVerifier, VerificationResult, VerifierKind
from tests.test_accounting import (
    ARTIFACT_RESPONSE,
    GROUP_RESPONSES,
    PANEL_RESPONSES,
    REVIEW_APPROVE,
    StubLeanVerifier,
    make_committee_society,
    make_society,
)


class FailingVerifier(LeanVerifier):
    """Rejects the first `failures` submissions and accepts every later one."""

    verifier_kind: VerifierKind = "real"

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0

    async def verify(self, code: str) -> VerificationResult:
        self.calls += 1
        if self.calls <= self.failures:
            return self._result(
                success=False, code=code, error="unknown identifier 'foo'"
            )
        return self._result(success=True, code=code)


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
    # each request's prompt tokens on the turn that sent it, completion on the reply
    assert [t.tokens for t in log.turns] == [0, 100, 50, 100, 50]
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_checkpoint_writes_the_transcripts(tmp_path):
    society = make_society(n_agents=1, responses=[ARTIFACT_RESPONSE], tmp_path=tmp_path)
    await society.run_generation(0)
    society.save(tmp_path / "run")

    saved = json.loads((tmp_path / "run" / "conversations.json").read_text())
    assert saved and saved[0]["turns"][-1]["content"] == ARTIFACT_RESPONSE


@pytest.mark.asyncio
async def test_iterative_log_keeps_artifact_ids_and_verifier_errors(tmp_path):
    society = make_society(
        n_agents=1,
        responses=[ARTIFACT_RESPONSE],
        verifier=FailingVerifier(failures=1),
        tmp_path=tmp_path,
    )
    society.iterative_mode = True
    await society.run_generation(0)

    [log] = society.trace_store.conversations
    assert log.success is True
    assert log.errors_encountered == ["unknown identifier 'foo'"]
    # one artifact per attempt: the rejected one and the verified retry
    assert len(log.artifacts_produced) == 2
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_iterative_log_records_a_session_that_never_verifies(tmp_path):
    society = make_society(
        n_agents=1,
        responses=[ARTIFACT_RESPONSE],
        verifier=FailingVerifier(failures=99),
        tmp_path=tmp_path,
    )
    society.iterative_mode = True
    society.max_attempts = 2
    await society.run_generation(0)

    [log] = society.trace_store.conversations
    # two failed attempts, then the writeup the last failure asks for
    assert [t.role for t in log.turns] == ["system"] + ["user", "assistant"] * 3
    assert log.success is False
    assert log.errors_encountered == ["unknown identifier 'foo'"] * 2
    assert len(log.artifacts_produced) == 2
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_iterative_log_records_retries_that_produce_no_artifact(tmp_path):
    society = make_society(n_agents=1, verifier=StubLeanVerifier(), tmp_path=tmp_path)
    society.iterative_mode = True
    society.max_attempts = 2
    await society.run_generation(0)

    [log] = society.trace_store.conversations
    assert [t.role for t in log.turns] == ["system"] + ["user", "assistant"] * 2
    assert (log.success, log.artifacts_produced, log.errors_encountered) == (
        False,
        [],
        [],
    )
    assert transcript_tokens(society) == society.ledger.total_tokens


@pytest.mark.asyncio
async def test_committee_transcripts_carry_the_generation_the_ledger_does(tmp_path):
    society = make_committee_society(
        tmp_path,
        PANEL_RESPONSES + GROUP_RESPONSES + [REVIEW_APPROVE],
        use_planning_panel=True,
    )
    await society.run_generation(3)

    by_kind = {
        (log.metadata["kind"], log.generation)
        for log in society.trace_store.conversations
    }
    assert {generation for _, generation in by_kind} == {3}, by_kind
    assert {r.generation for r in society.ledger.records} == {3}
