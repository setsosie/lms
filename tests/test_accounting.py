"""Tests for per-statement cost accounting (26Q3-HARN-05).

The invariant that matters: every token the society spends appears in the
ledger exactly once — attributed to a statement or to the overhead bucket —
so `sum(attempt tokens) == society.total_tokens_used`. Scope: flat and
iterative modes (working-group mode is 26Q3-HARN-12's card).
"""

import json

import pytest

from lms.accounting import (
    OVERHEAD_KEY,
    AttemptRecord,
    CostLedger,
    calculate_cvfn,
    cvfn_report,
    statement_key,
)
from lms.agent import Agent
from lms.artifacts import ArtifactLibrary
from lms.config import ProviderConfig
from lms.lean.interface import (
    LeanVerifier,
    VerificationResult,
    VerifierKind,
)
from lms.lean.mock import MockLeanVerifier
from lms.providers.base import (
    BaseLLMProvider,
    GenerationResponse,
    Message,
    TokenUsage,
)
from lms.society import Society


class MockProvider(BaseLLMProvider):
    """Mock LLM provider with fixed token usage per call."""

    name = "mock"

    def __init__(self, config: ProviderConfig, responses: list[str] | None = None):
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        if self.responses:
            content = self.responses[self.call_count % len(self.responses)]
        else:
            content = "No artifacts proposed."
        self.call_count += 1
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        return GenerationResponse(content=content, usage=usage, provider=self.name)


class StubLeanVerifier(LeanVerifier):
    """Accepts any code with Lean-grade provenance (see test_society)."""

    verifier_kind: VerifierKind = "real"

    async def verify(self, code: str) -> VerificationResult:
        return self._result(success=True, code=code)


ARTIFACT_RESPONSE = """
I'll prove a basic lemma about addition.

<artifact>
type: lemma
name: add_comm
description: Addition is commutative
lean: lemma add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b
references: []
</artifact>
"""


def make_society(n_agents=2, responses=None, verifier=None, tmp_path=None, **kwargs):
    config = ProviderConfig(api_key="test", model="test")
    provider = MockProvider(config, responses=responses)
    return Society(
        n_agents=n_agents,
        provider=provider,
        verifier=verifier if verifier is not None else MockLeanVerifier(),
        foundation_path=(tmp_path / "Foundation.lean") if tmp_path else None,
        **kwargs,
    )


class TestStatementKey:
    def test_source_anchor_wins(self):
        key = statement_key("0013", "lemma foo : True := trivial", "Some claim")
        assert key == "tag:0013"

    def test_lean_declaration_name(self):
        key = statement_key(
            None, "lemma add_comm (a b : Nat) : a + b = b + a := by omega", None
        )
        assert key == "decl:add_comm"

    def test_natural_language_slug(self):
        key = statement_key(None, None, "Addition is commutative!")
        assert key == "nl:addition-is-commutative"

    def test_nothing_identifiable_is_overhead(self):
        assert statement_key(None, None, None) == OVERHEAD_KEY
        assert statement_key(None, "-- just a comment", "") == OVERHEAD_KEY

    def test_stable_across_retries(self):
        first = statement_key(None, "theorem t1 : True := trivial", "x")
        retry = statement_key(None, "theorem t1 : True := by trivial", "x reworded")
        assert first == retry


class TestCostLedger:
    def rec(self, key="decl:foo", outcome="failed", **kw):
        defaults = dict(
            statement_key=key,
            agent_id="agent-0",
            generation=0,
            prompt_tokens=100,
            completion_tokens=50,
            wall_clock_s=1.5,
            outcome=outcome,
        )
        defaults.update(kw)
        return AttemptRecord(**defaults)

    def test_totals_and_overhead(self):
        ledger = CostLedger()
        ledger.record(self.rec())
        ledger.record_overhead("agent-1", 0, 10, 5, 0.1, outcome="no_artifact")
        assert ledger.total_tokens == 165
        assert ledger.overhead_tokens == 15

    def test_retry_chain_rolls_up(self):
        ledger = CostLedger()
        ledger.record(self.rec(outcome="failed", gate_failed="lean_verify"))
        ledger.record(self.rec(outcome="verified_lean"))
        ledger.record(self.rec(outcome="writeup"))
        costs = ledger.per_statement()
        assert costs["decl:foo"].attempts == 3
        assert costs["decl:foo"].tokens == 450
        assert costs["decl:foo"].verified is True
        assert ledger.gate_failure_histogram() == {"lean_verify": 1}

    def test_save_load_round_trip(self, tmp_path):
        ledger = CostLedger()
        ledger.record(self.rec(outcome="failed", gate_failed="lean_verify"))
        ledger.record_overhead("agent-1", 2, 10, 5, 0.1)
        path = tmp_path / "attempts.json"
        ledger.save(path)
        loaded = CostLedger.load(path)
        assert [r.to_dict() for r in loaded.records] == [
            r.to_dict() for r in ledger.records
        ]


class TestConservationFlat:
    """Flat (non-iterative) mode: propose + review spend all reach the ledger."""

    @pytest.mark.asyncio
    async def test_zero_artifact_responses_land_in_overhead(self, tmp_path):
        society = make_society(n_agents=2, tmp_path=tmp_path)
        await society.run_generation(0)

        assert society.total_tokens_used > 0
        assert society.ledger.total_tokens == society.total_tokens_used
        # Nothing identifiable was produced, so it is all overhead — the
        # exact spend that used to vanish from attribution entirely.
        assert society.ledger.overhead_tokens == society.total_tokens_used

    @pytest.mark.asyncio
    async def test_conservation_with_artifacts_and_reviews(self, tmp_path):
        society = make_society(
            n_agents=2, responses=[ARTIFACT_RESPONSE], tmp_path=tmp_path
        )
        await society.run_generation(0)

        assert society.ledger.total_tokens == society.total_tokens_used
        keys = {r.statement_key for r in society.ledger.records}
        assert "decl:add_comm" in keys
        outcomes = {r.outcome for r in society.ledger.records}
        assert "proposed" in outcomes
        assert "review" in outcomes

    @pytest.mark.asyncio
    async def test_wall_clock_recorded(self, tmp_path):
        society = make_society(n_agents=1, tmp_path=tmp_path)
        result = await society.run_generation(0)
        assert result.wall_clock_s > 0
        assert all(r.wall_clock_s >= 0 for r in society.ledger.records)


class TestConservationIterative:
    """Iterative mode: every call — attempts, retries, writeups — is ledgered."""

    @pytest.mark.asyncio
    async def test_conservation_on_success(self, tmp_path):
        society = make_society(
            n_agents=2,
            responses=[ARTIFACT_RESPONSE],
            verifier=StubLeanVerifier(),
            tmp_path=tmp_path,
        )
        society.iterative_mode = True
        result = await society.run_generation(0)

        # attempt + writeup per agent, all 150 tokens each
        assert society.total_tokens_used == 2 * 2 * 150
        assert society.ledger.total_tokens == society.total_tokens_used
        assert society.ledger.overhead_tokens == 0
        outcomes = [r.outcome for r in society.ledger.records]
        assert outcomes.count("verified_lean") == 2
        assert outcomes.count("writeup") == 2
        assert result.wall_clock_s > 0

    @pytest.mark.asyncio
    async def test_zero_artifact_attempts_are_overhead_not_dropped(self, tmp_path):
        society = make_society(
            n_agents=1, verifier=StubLeanVerifier(), tmp_path=tmp_path
        )
        society.iterative_mode = True
        society.max_attempts = 2

        result = await society.run_generation(0)

        # Two calls parsed to nothing; the spend is real and recorded.
        assert society.total_tokens_used == 2 * 150
        assert society.ledger.total_tokens == society.total_tokens_used
        assert society.ledger.overhead_tokens == society.total_tokens_used
        assert all(r.outcome == "no_artifact" for r in society.ledger.records)
        assert result.artifacts_created == 0

    @pytest.mark.asyncio
    async def test_attempts_are_not_artifacts_created(self, tmp_path):
        """society.py used to count len(response.attempts) as created."""
        society = make_society(
            n_agents=2,
            responses=[ARTIFACT_RESPONSE],
            verifier=StubLeanVerifier(),
            tmp_path=tmp_path,
        )
        society.iterative_mode = True
        result = await society.run_generation(0)

        # One final artifact per agent enters the library.
        assert result.artifacts_created == 2
        assert len(society.library) == 2
        # The attempts are reported separately, not as artifacts.
        assert result.attempts_total == 2
        for stats in society.artifacts_by_agent.values():
            assert stats["created"] == 1


class TestRetryAttribution:
    @pytest.mark.asyncio
    async def test_retry_chain_shares_a_statement_key(self):
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config, responses=[ARTIFACT_RESPONSE])
        agent = Agent(id="agent-0", provider=provider, generation=1)
        ledger = CostLedger()

        calls = {"n": 0}

        async def verify_fn(code: str) -> VerificationResult:
            calls["n"] += 1
            if calls["n"] == 1:
                return VerificationResult(
                    success=False,
                    code=code,
                    verifier_kind="real",
                    verifier_id="test",
                    error="boom",
                )
            return VerificationResult(
                success=True, code=code, verifier_kind="real", verifier_id="test"
            )

        response = await agent.propose_iterative(
            library=ArtifactLibrary(),
            verify_fn=verify_fn,
            max_attempts=3,
            ledger=ledger,
        )

        assert response.success
        # failed attempt + verified attempt + writeup, one statement
        assert [r.outcome for r in ledger.records] == [
            "failed",
            "verified_lean",
            "writeup",
        ]
        assert len({r.statement_key for r in ledger.records}) == 1
        cost = ledger.per_statement()["decl:add_comm"]
        assert cost.attempts == 3
        assert cost.verified is True
        assert ledger.records[0].gate_failed == "lean_verify"
        # The whole chain is on the ledger, not just the success
        assert ledger.total_tokens == response.total_tokens


class TestCVFN:
    def test_calculate_cvfn_undefined_at_zero(self):
        assert calculate_cvfn(1_000_000, 0) is None
        assert calculate_cvfn(1_000_000, 4) == 250_000

    @pytest.mark.asyncio
    async def test_report_on_fresh_run(self, tmp_path):
        society = make_society(
            n_agents=2,
            responses=[ARTIFACT_RESPONSE],
            verifier=StubLeanVerifier(),
            tmp_path=tmp_path,
        )
        society.iterative_mode = True
        await society.run_generation(0)
        run_dir = tmp_path / "run"
        society.save(run_dir)

        report = cvfn_report(run_dir)
        assert report.ledger_present
        assert report.total_tokens == society.total_tokens_used
        assert report.verified_count == 2
        # No novelty labels yet (HARN-04), so the denominator is the
        # unfiltered verified count and says so.
        assert report.denominator_kind == "verified_lean_unfiltered"
        assert report.denominator == 2
        assert report.cvfn_tokens_per_statement == society.total_tokens_used / 2
        assert report.total_wall_clock_s > 0
        assert "CVFN" in report.format()

    def test_report_on_archived_run_without_ledger(self, tmp_path):
        """Runs predating the ledger fall back to society totals."""
        run_dir = tmp_path / "archived"
        run_dir.mkdir()
        (run_dir / "results.json").write_text(
            json.dumps(
                {
                    "n_agents": 15,
                    "generations": [
                        {"generation": 0, "artifacts_created": 60, "tokens_used": 100}
                    ],
                    "checkpoint": {
                        "current_generation": 5,
                        "total_tokens_used": 8_987_704,
                    },
                }
            )
        )
        (run_dir / "artifacts.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"id": "a1", "status": "verified_heuristic"},
                        {"id": "a2", "status": "failed"},
                    ]
                }
            )
        )

        report = cvfn_report(run_dir)
        assert not report.ledger_present
        assert report.total_tokens == 8_987_704
        # Mock-verified artifacts are not a CVFN denominator.
        assert report.verified_count == 0
        assert report.cvfn_tokens_per_statement is None
        assert "undefined" in report.format()
        assert report.status_histogram == {"verified_heuristic": 1, "failed": 1}

    def test_report_honors_novelty_labels(self, tmp_path):
        run_dir = tmp_path / "labelled"
        run_dir.mkdir()
        (run_dir / "results.json").write_text(
            json.dumps({"generations": [], "checkpoint": {"total_tokens_used": 900}})
        )
        (run_dir / "artifacts.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"id": "a1", "status": "verified_lean", "novelty_level": "N0"},
                        {"id": "a2", "status": "verified_lean", "novelty_level": "N1"},
                        {"id": "a3", "status": "failed", "novelty_level": "N1"},
                    ]
                }
            )
        )

        report = cvfn_report(run_dir)
        assert report.verified_count == 2
        assert report.denominator_kind == "verified_novel"
        # Only verified AND novel counts; the N0 re-proof does not.
        assert report.denominator == 1
        assert report.cvfn_tokens_per_statement == 900

    def test_report_ingests_review_minutes(self, tmp_path):
        run_dir = tmp_path / "reviewed"
        run_dir.mkdir()
        (run_dir / "results.json").write_text(
            json.dumps({"generations": [], "checkpoint": {"total_tokens_used": 10}})
        )
        (run_dir / "artifacts.json").write_text(json.dumps({"artifacts": []}))
        (run_dir / "review_log.json").write_text(
            json.dumps(
                [
                    {"statement_key": "tag:0013", "minutes": 4.5},
                    {"statement_key": "tag:0014", "minutes": 3.0},
                ]
            )
        )

        report = cvfn_report(run_dir)
        assert report.review_minutes == 7.5


class TestLedgerPersistence:
    @pytest.mark.asyncio
    async def test_ledger_survives_save_and_load(self, tmp_path):
        society = make_society(n_agents=1, tmp_path=tmp_path)
        await society.run_generation(0)
        run_dir = tmp_path / "run"
        society.save(run_dir)

        assert (run_dir / "attempts.json").exists()

        config = ProviderConfig(api_key="test", model="test")
        resumed = Society.load(
            run_dir,
            provider=MockProvider(config),
            verifier=MockLeanVerifier(),
        )
        assert resumed.ledger.total_tokens == society.ledger.total_tokens
        assert len(resumed.ledger) == len(society.ledger)
