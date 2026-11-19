"""Thinking is recorded, and how much of it a reply may spend is a setting.

In batch q2 about 80% of Qwen3.6's output tokens were thinking, roughly half of
each run's budget, and none of it reached `conversations.json`. Nothing could
vary it, and three runs ended on `Request timed out.`: a reply longer than the
fixed 300 s timeout was sent twice more, 300 s each, before the run ended.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
from openai.types.chat import ChatCompletion

import lms.run
from lms.config import Config, ProviderConfig
from lms.lean.mock import MockLeanVerifier
from lms.providers.openai import OpenAIProvider
from lms.society import GenerationResult, Society
from lms.traces import ConversationLog, TraceStore
from tests import test_accounting as acc
from tests.test_local_serving import _StubEndpoint
from tests.test_sampling_seed import _main_kwargs

SCRATCH = "SCRATCH: try simp, no, try omega."
ANSWER = acc.ARTIFACT_RESPONSE
# Without a reasoning parser the thinking comes ahead of `</think>`; vLLM
# 0.33's parser moves it to `reasoning`; older servers used `reasoning_content`.
SHAPES: dict[str, dict[str, str]] = {
    "think-tag": {"content": f"{SCRATCH}\n</think>\n\n{ANSWER}"},
    "reasoning": {"content": ANSWER, "reasoning": SCRATCH},
    "reasoning_content": {"content": ANSWER, "reasoning_content": SCRATCH},
}


def _provider(message: dict[str, str] | None = None, **cfg: Any) -> OpenAIProvider:
    """A real provider whose client returns `message`, parsed as the SDK does."""
    provider = OpenAIProvider(ProviderConfig(api_key="EMPTY", model="m", **cfg))
    message = {"role": "assistant", **(message or {"content": "ok"})}
    choice = {"index": 0, "finish_reason": "stop", "message": message}
    usage = {"prompt_tokens": 1, "completion_tokens": 9, "total_tokens": 10}
    reply = ChatCompletion.model_validate(
        {"id": "c", "object": "chat.completion", "created": 0, "model": "m"}
        | {"choices": [choice], "usage": usage}
    )
    provider.client.chat.completions.create = mock.AsyncMock(return_value=reply)  # type: ignore[method-assign]
    return provider


# --- The request ----------------------------------------------------------------


@pytest.mark.parametrize("budget", [None, 0, 2048])
async def test_the_budget_goes_with_every_request_only_when_set(budget) -> None:
    provider = _provider(thinking_budget=budget, seed=7)
    await provider.generate([])
    await provider.generate([])
    expected = None if budget is None else {"thinking_token_budget": budget}
    for call in provider.client.chat.completions.create.call_args_list:  # type: ignore[attr-defined]
        assert call.kwargs.get("extra_body") == expected
        assert isinstance(call.kwargs["seed"], int)


@pytest.mark.parametrize("budget", [None, 1024])
async def test_the_server_reads_the_budget_as_a_top_level_field(budget) -> None:
    with _StubEndpoint() as endpoint:
        cfg = ProviderConfig(
            "k", "m", base_url=endpoint.base_url, thinking_budget=budget
        )
        await OpenAIProvider(cfg, timeout=10.0).generate([])
    payload = endpoint.requests[0]["payload"]
    assert payload.get("thinking_token_budget") == budget
    assert "extra_body" not in payload


def test_timeout_and_retries_reach_the_client() -> None:
    default = OpenAIProvider(ProviderConfig("k", "m")).client
    assert (default.timeout.read, default.max_retries) == (300.0, 2)  # type: ignore[union-attr]
    set_ = OpenAIProvider(
        ProviderConfig("k", "m", request_timeout_s=1800, max_retries=0)
    ).client
    assert (set_.timeout.read, set_.timeout.connect, set_.max_retries) == (1800, 30, 0)  # type: ignore[union-attr]


def _openai_from_env(tmp_path: Path, **env: str) -> ProviderConfig:
    (tmp_path / "empty.env").write_text("")  # keeps the repo's own .env out
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", **env}, clear=True):
        config = Config.from_env(env_path=tmp_path / "empty.env")
    assert config.openai is not None
    return config.openai


def test_the_environment_sets_timeout_and_retries(tmp_path: Path) -> None:
    cfg = _openai_from_env(
        tmp_path, LMS_OPENAI_TIMEOUT="1800", LMS_OPENAI_MAX_RETRIES="0"
    )
    assert (cfg.request_timeout_s, cfg.max_retries) == (1800.0, 0)
    blank = _openai_from_env(tmp_path, LMS_OPENAI_TIMEOUT="", LMS_OPENAI_MAX_RETRIES="")
    assert (blank.request_timeout_s, blank.max_retries) == (300.0, 2)


@pytest.mark.parametrize(
    "var,bad",
    [("LMS_OPENAI_TIMEOUT", b) for b in ["abc", "0", "-5", "inf", "nan"]]
    + [("LMS_OPENAI_MAX_RETRIES", b) for b in ["x", "-1", "1.5"]],
)
def test_a_bad_timeout_or_retry_count_fails_at_config_load(tmp_path, var, bad) -> None:
    with pytest.raises(ValueError, match=var):
        _openai_from_env(tmp_path, **{var: bad})


# --- The trace ------------------------------------------------------------------


def test_thinking_round_trips_and_old_files_still_load(tmp_path: Path) -> None:
    store = TraceStore()
    store.add_conversation(log := ConversationLog(agent_id="a", generation=0))
    log.add_exchange("prompt", "answer", None, thinking=SCRATCH)
    log.add_exchange("prompt", "answer", None)
    store.save(tmp_path)
    [loaded] = TraceStore.load(tmp_path).conversations
    assert [t.thinking for t in loaded.turns] == ["", SCRATCH, "", ""]
    assert [t.content for t in loaded.turns] == ["prompt", "answer"] * 2

    old = {
        "agent_id": "a",
        "generation": 0,
        "turns": [{"role": "user", "content": "x"}],
    }
    (tmp_path / "conversations.json").write_text(json.dumps([old]))
    assert TraceStore.load(tmp_path).conversations[0].turns[0].thinking == ""


@pytest.mark.parametrize("shape", sorted(SHAPES))
async def test_a_run_s_saved_trace_holds_each_reply_s_thinking(tmp_path, shape) -> None:
    """Through the real provider, a society, and conversations.json."""
    society = Society(
        n_agents=2,
        provider=_provider(SHAPES[shape]),
        verifier=MockLeanVerifier(),
        foundation_path=tmp_path / "Foundation.lean",
    )
    await society.run_generation(0)
    society.trace_store.save(tmp_path)
    logs = TraceStore.load(tmp_path).conversations
    replies = [t for log in logs for t in log.turns if t.role == "assistant"]
    assert replies and all(t.thinking == SCRATCH for t in replies)
    assert not any("SCRATCH" in t.content for t in replies)


def _flat(tmp_path: Path) -> Society:
    return acc.make_society(n_agents=2, responses=[ANSWER], tmp_path=tmp_path)


def _iterative(tmp_path: Path) -> Society:
    society = acc.make_society(1, [ANSWER], acc.StubLeanVerifier(), tmp_path)
    society.iterative_mode = True
    return society


def _committee(tmp_path: Path) -> Society:
    responses = acc.PANEL_RESPONSES + acc.GROUP_RESPONSES + [acc.REVIEW_APPROVE]
    return acc.make_committee_society(tmp_path, responses, use_planning_panel=True)


@pytest.mark.parametrize("make", [_flat, _iterative, _committee])
async def test_every_call_site_records_its_reply_s_thinking(tmp_path, make) -> None:
    """Propose, review, iterative attempts and writeups, planning, groups."""
    society = make(tmp_path)
    provider = cast(acc.MockProvider, society.provider)
    answer = provider.generate

    async def thinks(*args: Any, **kwargs: Any) -> Any:
        response = await answer(*args, **kwargs)
        return replace(response, thinking=f"scratch {provider.call_count}")

    provider.generate = thinks  # type: ignore[method-assign]
    await society.run_generation(0)
    turns = [t for log in society.trace_store.conversations for t in log.turns]
    thoughts = sorted(t.thinking for t in turns if t.role == "assistant")
    assert provider.call_count > 0
    assert thoughts == sorted(f"scratch {n + 1}" for n in range(provider.call_count))
    assert all(t.thinking == "" for t in turns if t.role != "assistant")


# --- The run's record -----------------------------------------------------------

SERVED = ProviderConfig(
    "k", "m", max_tokens=32_768, request_timeout_s=1800, max_retries=0
)


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[ProviderConfig]:
    """Runs that build their providers and stop: the configs they were given."""
    configs: list[ProviderConfig] = []

    def create(name: str, cfg: ProviderConfig) -> acc.MockProvider:
        configs.append(cfg)
        return acc.MockProvider(cfg)

    async def generation(self: Society, n: int) -> GenerationResult:
        self.current_generation = n + 1
        return GenerationResult(n, 0, 0, 0, fresh_creations=0)

    monkeypatch.setattr("lms.run.create_provider", create)
    monkeypatch.setattr(Society, "run_generation", generation)
    return configs


KEYS = ("thinking_budget", "request_max_tokens", "request_timeout_s", "max_retries")


async def test_a_run_and_its_resume_record_their_settings(tmp_path, made) -> None:
    out, cfg = tmp_path / "run", Config(openai=SERVED)
    await lms.run.run_experiment(1, 1, "openai", out, cfg, 10**6, thinking_budget=2048)
    resumed = Config(openai=replace(SERVED, request_timeout_s=3600, max_retries=1))
    await lms.run.resume_experiment(out, 2, resumed, thinking_budget=4096)
    meta = json.loads((out / "metadata.json").read_text())
    assert [c.thinking_budget for c in made] == [2048, 4096]
    assert meta["max_tokens"] == 10**6  # the run's budget keeps its key
    assert [meta[k] for k in KEYS] == [2048, 32_768, 1800, 0]
    assert [meta[f"{k}_resume"] for k in KEYS] == [4096, 32_768, 3600, 1]


async def test_no_budget_is_recorded_and_a_mixed_run_records_limits_per_provider(
    tmp_path, made
) -> None:
    config = Config(anthropic=ProviderConfig("a", "c"), openai=SERVED)
    await lms.run.run_experiment(2, 1, "openai", tmp_path, config, mixed_providers=True)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["thinking_budget"] is None and made[0].thinking_budget is None
    assert [p["request_timeout_s"] for p in meta["providers"]] == [300.0, 1800]


# --- The command line -----------------------------------------------------------


@pytest.mark.parametrize(
    "argv,env,expected",
    [
        (["--thinking-budget", "2048"], None, 2048),
        (["--thinking-budget", "2048", "--resume", "."], None, 2048),
        (["--thinking-budget", "0"], None, 0),
        ([], "1024", 1024),
        (["--thinking-budget", "4096"], "1024", 4096),
        ([], None, None),
    ],
)
def test_the_budget_reaches_the_run(tmp_path, monkeypatch, argv, env, expected) -> None:
    monkeypatch.delenv("LMS_THINKING_BUDGET", raising=False)
    if env is not None:
        monkeypatch.setenv("LMS_THINKING_BUDGET", env)
    argv = [str(tmp_path) if a == "." else a for a in argv]
    assert _main_kwargs(tmp_path, monkeypatch, argv)["thinking_budget"] == expected


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("raw", ["x", "-1", "1.5", "2k"])
def test_a_bad_budget_stops_the_run(tmp_path, monkeypatch, flag, raw) -> None:
    monkeypatch.delenv("LMS_THINKING_BUDGET", raising=False)
    if not flag:
        monkeypatch.setenv("LMS_THINKING_BUDGET", raw)
    with pytest.raises(SystemExit):
        _main_kwargs(
            tmp_path, monkeypatch, [f"--thinking-budget={raw}"] if flag else []
        )
