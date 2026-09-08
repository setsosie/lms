"""Tests for the local OpenAI-compatible serving path (26Q3-INFRA-01).

vLLM, SGLang and Ollama all expose an OpenAI-compatible `/v1`, so pointing the
harness at a self-hosted model is a `base_url` away. These tests pin that switch.

Every test that touches `Config.from_env` passes an empty `env_path`: the
repo-root `.env` otherwise leaks through `load_dotenv` even under
`clear=True`, because an empty environ is exactly the case `load_dotenv`
is designed to fill.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from lms.config import DEFAULT_MAX_TOKENS, Config, ProviderConfig
from lms.providers.base import Message
from lms.providers.openai import OpenAIProvider

HOSTED_DEFAULT = "https://api.openai.com/v1"


@pytest.fixture
def empty_env(tmp_path: Path) -> Path:
    """An env file with no variables, to neutralize the repo's real `.env`."""
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    return env_file


class _StubEndpoint:
    """A minimal OpenAI-compatible `/v1` server, standing in for vLLM.

    vLLM/SGLang/Ollama differ from hosted OpenAI in exactly the ways this stub
    reproduces: any API key is accepted, and the model name is whatever the
    server was launched with. Running the real SDK over real TCP against it is
    what makes the end-to-end claim testable without a GPU.
    """

    def __init__(self, *, report_usage: bool = True) -> None:
        self.requests: list[dict] = []
        # Ollama and some vLLM configurations answer without a usage block.
        self.report_usage = report_usage
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "payload": json.loads(body),
                    }
                )
                payload = json.dumps(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "local-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "theorem stub : True",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                        }
                        if outer.report_usage
                        else None,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                """Silence the default stderr access log."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "_StubEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def stub_endpoint() -> Iterator[_StubEndpoint]:
    with _StubEndpoint() as server:
        yield server


class TestProviderConfigBaseURL:
    def test_base_url_defaults_to_none(self):
        """Hosted OpenAI stays the default when nothing is configured."""
        config = ProviderConfig(api_key="k", model="m")
        assert config.base_url is None

    def test_accepts_base_url(self):
        config = ProviderConfig(
            api_key="k", model="m", base_url="http://localhost:8000/v1"
        )
        assert config.base_url == "http://localhost:8000/v1"


class TestConfigFromEnv:
    def test_reads_base_url_from_env(self, empty_env: Path):
        env = {
            "OPENAI_API_KEY": "dummy-key-vllm-ignores-this",
            "LMS_OPENAI_MODEL": "Qwen/Qwen3-Coder-30B-A3B",
            "LMS_OPENAI_BASE_URL": "http://localhost:8000/v1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.base_url == "http://localhost:8000/v1"
        assert config.openai.model == "Qwen/Qwen3-Coder-30B-A3B"

    def test_base_url_absent_leaves_none(self, empty_env: Path):
        """Unset env var must not invent a base_url."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.base_url is None

    def test_empty_base_url_is_treated_as_unset(self, empty_env: Path):
        """`.env.example` ships the key bare; a copied file must not break the client."""
        env = {"OPENAI_API_KEY": "sk-test", "LMS_OPENAI_BASE_URL": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.base_url is None
        assert (
            str(OpenAIProvider(config.openai).client.base_url).rstrip("/")
            == HOSTED_DEFAULT
        )

    def test_blank_model_falls_back_to_the_default(self, empty_env: Path):
        """Blank means absent for every read, not just base_url — "" is no model name."""
        env = {"OPENAI_API_KEY": "sk-test", "LMS_OPENAI_MODEL": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.model == "gpt-5.2"


class TestMaxTokensFromEnv:
    """26Q3-INFRA-02: the per-request completion cap follows the endpoint.

    A vLLM server at `--max-model-len 65536` rejects any request where
    `prompt_tokens + max_tokens > max_model_len`, so a Claude-sized 64k
    default cap 400s every agent prompt. The operator knows the served
    window; the env var lets the harness respect it.
    """

    def test_env_reaches_provider_config(self, empty_env: Path):
        env = {"OPENAI_API_KEY": "dummy", "LMS_OPENAI_MAX_TOKENS": "8192"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.max_tokens == 8192

    def test_each_provider_reads_its_own_var(self, empty_env: Path):
        env = {
            "ANTHROPIC_API_KEY": "a",
            "OPENAI_API_KEY": "o",
            "GOOGLE_API_KEY": "g",
            "LMS_ANTHROPIC_MAX_TOKENS": "1000",
            "LMS_OPENAI_MAX_TOKENS": "2000",
            "LMS_GOOGLE_MAX_TOKENS": "3000",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.anthropic is not None
        assert config.openai is not None
        assert config.google is not None
        assert config.anthropic.max_tokens == 1000
        assert config.openai.max_tokens == 2000
        assert config.google.max_tokens == 3000

    def test_unset_keeps_the_default(self, empty_env: Path):
        """Hosted-provider behavior is unchanged when the var is absent."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.max_tokens == DEFAULT_MAX_TOKENS

    def test_blank_is_treated_as_unset(self, empty_env: Path):
        """`.env.example` ships keys bare; a copied file must not break loading."""
        env = {"OPENAI_API_KEY": "sk", "LMS_OPENAI_MAX_TOKENS": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        assert config.openai is not None
        assert config.openai.max_tokens == DEFAULT_MAX_TOKENS

    @pytest.mark.parametrize("bad", ["64k", "8192.5", "-1", "0"])
    def test_bad_value_fails_at_config_load(self, empty_env: Path, bad: str):
        """A bad cap must fail here, not as an HTTP 400 on the first generation."""
        env = {"OPENAI_API_KEY": "sk", "LMS_OPENAI_MAX_TOKENS": bad}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="LMS_OPENAI_MAX_TOKENS"):
                Config.from_env(env_path=empty_env)

    async def test_cap_reaches_the_request_payload(
        self, stub_endpoint: _StubEndpoint, empty_env: Path
    ):
        """The acceptance criterion: the configured cap is what the server sees."""
        env = {
            "OPENAI_API_KEY": "dummy",
            "LMS_OPENAI_MODEL": "lms-generalist",
            "LMS_OPENAI_BASE_URL": stub_endpoint.base_url,
            "LMS_OPENAI_MAX_TOKENS": "8192",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        provider = OpenAIProvider(config.get_provider_config("openai"), timeout=10.0)
        await provider.generate([Message(role="user", content="formalize this")])

        assert len(stub_endpoint.requests) == 1
        payload = stub_endpoint.requests[0]["payload"]
        assert payload["max_completion_tokens"] == 8192


class TestOpenAIProviderClient:
    def test_base_url_threads_into_client(self):
        config = ProviderConfig(
            api_key="k", model="m", base_url="http://localhost:8000/v1"
        )
        provider = OpenAIProvider(config)
        assert str(provider.client.base_url).rstrip("/") == "http://localhost:8000/v1"

    def test_unset_base_url_keeps_hosted_default(self):
        config = ProviderConfig(api_key="k", model="m")
        provider = OpenAIProvider(config)
        assert str(provider.client.base_url).rstrip("/") == HOSTED_DEFAULT


class TestEndToEndAgainstLocalEndpoint:
    """The acceptance criterion: a local model is usable via env config alone."""

    async def test_env_config_reaches_a_local_endpoint(
        self, stub_endpoint: _StubEndpoint, empty_env: Path
    ):
        env = {
            "OPENAI_API_KEY": "dummy-key",  # vLLM accepts any key
            "LMS_OPENAI_MODEL": "Qwen/Qwen3-Coder-30B-A3B",
            "LMS_OPENAI_BASE_URL": stub_endpoint.base_url,
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_path=empty_env)

        # timeout caps the SDK's 300s read timeout x2 retries: a stub that
        # accepted and never answered would otherwise stall CI for ~15 minutes.
        provider = OpenAIProvider(config.get_provider_config("openai"), timeout=10.0)
        response = await provider.generate(
            [Message(role="user", content="formalize this")]
        )

        assert response.content == "theorem stub : True"
        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 7

        # The request reached the local server, not api.openai.com.
        assert len(stub_endpoint.requests) == 1
        request = stub_endpoint.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["authorization"] == "Bearer dummy-key"
        assert request["payload"]["model"] == "Qwen/Qwen3-Coder-30B-A3B"

    async def test_server_omitting_usage_does_not_crash_the_agent_loop(self):
        """A missing token count is a hole in accounting, not a failed generation."""
        with _StubEndpoint(report_usage=False) as endpoint:
            config = ProviderConfig(
                api_key="local", model="m", base_url=endpoint.base_url
            )
            response = await OpenAIProvider(config, timeout=10.0).generate(
                [Message(role="user", content="formalize this")]
            )

        assert response.content == "theorem stub : True"
        assert response.usage.total_tokens == 0
