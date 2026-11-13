"""Each run samples from its own seed.

Batch q1's arms rehearsal ran nine single-agent I9 runs, each against a freshly
started vLLM server. Eight finished, and all eight were the same run token for
token: 11 calls, 294,028 prompt and 51,529 completion tokens each. Requests
carried no seed, so every fresh server sampled from its engine seed (0) in the
same order. The arm meant to be nine independent agents was one agent, counted
eight times.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import lms.run
from lms.config import Config, ProviderConfig
from lms.providers.openai import OpenAIProvider
from lms.society import GenerationResult, Society
from tests.test_society import MockProvider


def _reply() -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


def _provider(seed: int | None) -> OpenAIProvider:
    provider = OpenAIProvider(
        ProviderConfig(api_key="EMPTY", model="lms-generalist", seed=seed)
    )
    provider.client.chat.completions.create = mock.AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda **_: _reply()
    )
    return provider


def _seeds(provider: OpenAIProvider) -> list[object]:
    calls = provider.client.chat.completions.create.call_args_list  # type: ignore[attr-defined]
    return [call.kwargs.get("seed", "absent") for call in calls]


# --- The provider ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_seeded_provider_sends_a_seed_with_every_request() -> None:
    provider = _provider(seed=1234)
    for _ in range(3):
        await provider.generate([])
    seeds = _seeds(provider)
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 3, "the same seed twice repeats a sample"


@pytest.mark.asyncio
async def test_no_two_requests_in_a_run_share_a_seed() -> None:
    """A society can hold one provider per agent, all from the run's seed.
    Agents with the same prompt and the same seed would answer alike."""
    first, second = _provider(seed=1234), _provider(seed=1234)
    for _ in range(3):
        await first.generate([])
        await second.generate([])
    seeds = _seeds(first) + _seeds(second)
    assert len(set(seeds)) == 6


@pytest.mark.asyncio
async def test_different_runs_get_different_seeds() -> None:
    a, b = _provider(seed=1), _provider(seed=2_000_000_000)
    await a.generate([])
    await b.generate([])
    assert _seeds(a) != _seeds(b)


@pytest.mark.asyncio
async def test_an_unseeded_provider_sends_no_seed() -> None:
    """The hosted API path is unchanged unless a seed is configured."""
    provider = _provider(seed=None)
    await provider.generate([])
    assert _seeds(provider) == ["absent"]


@pytest.mark.asyncio
async def test_seeds_stay_in_range() -> None:
    provider = _provider(seed=2**31 - 1)
    for _ in range(3):
        await provider.generate([])
    assert all(0 <= s < 2**31 for s in _seeds(provider))  # type: ignore[operator]


# --- The run --------------------------------------------------------------------


@pytest.fixture
def quiet_run(monkeypatch: pytest.MonkeyPatch) -> list[ProviderConfig]:
    """Runs that build their providers and stop: the configs they were given."""
    made: list[ProviderConfig] = []

    def _create(name: str, cfg: ProviderConfig) -> MockProvider:
        made.append(cfg)
        return MockProvider(cfg)

    async def _generation(self: Society, generation: int) -> GenerationResult:
        self.current_generation = generation + 1
        return GenerationResult(
            generation=generation,
            artifacts_created=0,
            artifacts_verified=0,
            artifacts_referenced=0,
            fresh_creations=0,
        )

    monkeypatch.setattr("lms.run.create_provider", _create)
    monkeypatch.setattr(Society, "run_generation", _generation)
    return made


async def _run(tmp_path: Path, name: str, **kwargs: object) -> dict:
    out = tmp_path / name
    await lms.run.run_experiment(
        n_agents=1,
        n_generations=1,
        provider_name="openai",
        output_dir=out,
        config=Config(openai=ProviderConfig(api_key="k", model="m")),
        verifier_type="mock",
        **kwargs,  # type: ignore[arg-type]
    )
    return json.loads((out / "metadata.json").read_text())


@pytest.mark.asyncio
async def test_a_run_without_a_seed_draws_one_and_records_it(
    tmp_path: Path, quiet_run: list[ProviderConfig]
) -> None:
    first = await _run(tmp_path, "a")
    second = await _run(tmp_path, "b")
    assert isinstance(first["sampling_seed"], int)
    assert first["sampling_seed"] != second["sampling_seed"]
    assert [c.seed for c in quiet_run] == [
        first["sampling_seed"],
        second["sampling_seed"],
    ]


@pytest.mark.asyncio
async def test_a_given_seed_reaches_the_provider_and_the_record(
    tmp_path: Path, quiet_run: list[ProviderConfig]
) -> None:
    metadata = await _run(tmp_path, "a", sampling_seed=7)
    assert metadata["sampling_seed"] == 7
    assert [c.seed for c in quiet_run] == [7]


@pytest.mark.asyncio
async def test_a_resumed_run_samples_afresh_and_records_it(
    tmp_path: Path, quiet_run: list[ProviderConfig]
) -> None:
    """Under its own key, like the other `_resume` fields: the original run's
    seed stays as it was."""
    await _run(tmp_path, "a", sampling_seed=7)
    await lms.run.resume_experiment(
        checkpoint_dir=tmp_path / "a",
        target_generations=2,
        config=Config(openai=ProviderConfig(api_key="k", model="m")),
        verifier_type="mock",
        sampling_seed=9,
    )
    metadata = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert metadata["sampling_seed"] == 7
    assert metadata["sampling_seed_resume"] == 9
    assert [c.seed for c in quiet_run] == [7, 9]


# --- The command line -----------------------------------------------------------


def _main_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> dict[str, object]:
    config = Config(openai=ProviderConfig(api_key="k", model="m"))
    monkeypatch.setattr(
        lms.run.Config, "from_env", classmethod(lambda cls, path=None: config)
    )
    calls: list[dict[str, object]] = []

    async def _fake(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(lms.run, "run_experiment", _fake)
    monkeypatch.setattr(lms.run, "resume_experiment", _fake)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lms.run", "--provider", "openai", "--output", str(tmp_path), *argv],
    )
    lms.run.main()
    (kwargs,) = calls
    return kwargs


@pytest.mark.parametrize("resume", [False, True])
def test_the_seed_flag_reaches_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    monkeypatch.delenv("LMS_SEED", raising=False)
    argv = ["--seed", "42"] + (["--resume", str(tmp_path)] if resume else [])
    assert _main_kwargs(tmp_path, monkeypatch, argv)["sampling_seed"] == 42


def test_the_environment_can_set_the_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch job sets it without touching the run script."""
    monkeypatch.setenv("LMS_SEED", "43")
    assert _main_kwargs(tmp_path, monkeypatch, [])["sampling_seed"] == 43


def test_the_flag_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LMS_SEED", "43")
    assert _main_kwargs(tmp_path, monkeypatch, ["--seed", "44"])["sampling_seed"] == 44


def test_no_seed_anywhere_leaves_the_run_to_draw_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LMS_SEED", raising=False)
    assert _main_kwargs(tmp_path, monkeypatch, [])["sampling_seed"] is None


@pytest.mark.parametrize("raw", ["x", "-1", str(2**31)])
def test_a_bad_seed_in_the_environment_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("LMS_SEED", raw)
    with pytest.raises(SystemExit):
        _main_kwargs(tmp_path, monkeypatch, [])
