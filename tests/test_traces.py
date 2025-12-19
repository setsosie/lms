"""Tests for conversation logging and reasoning traces."""

import json
from pathlib import Path

import pytest

from lms.traces import (
    ConversationTurn,
    ConversationLog,
    ReasoningTrace,
    TraceStore,
)


class TestConversationLog:
    """Tests for ConversationLog."""

    def test_create_log(self):
        """ConversationLog holds agent and generation info."""
        log = ConversationLog(
            agent_id="agent-0-anthropic",
            generation=5,
        )
        assert log.agent_id == "agent-0-anthropic"
        assert log.generation == 5
        assert log.turns == []
        assert log.success is False

    def test_add_turn(self):
        """Can add conversation turns."""
        log = ConversationLog(agent_id="agent-1", generation=0)
        log.add_turn("user", "Please propose an artifact", tokens=100)
        log.add_turn("assistant", "Here is my proposal...", tokens=500)

        assert len(log.turns) == 2
        assert log.turns[0].role == "user"
        assert log.turns[0].content == "Please propose an artifact"
        assert log.turns[1].role == "assistant"
        assert log.turns[1].tokens == 500

    def test_total_tokens(self):
        """Total tokens sums all turns."""
        log = ConversationLog(agent_id="agent-1", generation=0)
        log.add_turn("user", "prompt", tokens=100)
        log.add_turn("assistant", "response", tokens=500)
        log.add_turn("user", "follow up", tokens=50)
        log.add_turn("assistant", "more response", tokens=300)

        assert log.total_tokens() == 950

    def test_to_dict(self):
        """ConversationLog can be serialized to dict."""
        log = ConversationLog(
            agent_id="agent-0",
            generation=3,
            success=True,
            artifacts_produced=["artifact-123"],
            errors_encountered=["Type mismatch"],
        )
        log.add_turn("user", "Do the thing", tokens=50)

        data = log.to_dict()
        assert data["agent_id"] == "agent-0"
        assert data["generation"] == 3
        assert data["success"] is True
        assert data["artifacts_produced"] == ["artifact-123"]
        assert len(data["turns"]) == 1
        assert data["turns"][0]["content"] == "Do the thing"


class TestReasoningTrace:
    """Tests for ReasoningTrace."""

    def test_create_trace(self):
        """ReasoningTrace captures breakthrough moments."""
        trace = ReasoningTrace(
            trace_type="reference_success",
            agent_id="agent-2-google",
            generation=6,
            description="Successfully built on 2 previous artifacts",
            context="Referenced hom_functor artifact",
            outcome="Produced verified HomFunctor",
            related_artifacts=["artifact-123", "artifact-456"],
        )
        assert trace.trace_type == "reference_success"
        assert trace.generation == 6
        assert len(trace.related_artifacts) == 2

    def test_to_dict(self):
        """ReasoningTrace can be serialized."""
        trace = ReasoningTrace(
            trace_type="error_fix",
            agent_id="agent-0",
            generation=5,
            description="Fixed namespace collision",
            context="Previous error: ambiguous term",
            outcome="Error resolved",
        )
        data = trace.to_dict()
        assert data["trace_type"] == "error_fix"
        assert data["description"] == "Fixed namespace collision"


class TestTraceStore:
    """Tests for TraceStore."""

    def test_add_conversation(self):
        """Can add conversation logs."""
        store = TraceStore()
        log = ConversationLog(agent_id="agent-0", generation=0)
        log.add_turn("user", "test", tokens=10)

        store.add_conversation(log)
        assert len(store.conversations) == 1

    def test_add_trace(self):
        """Can add reasoning traces."""
        store = TraceStore()
        trace = ReasoningTrace(
            trace_type="breakthrough",
            agent_id="agent-1",
            generation=5,
            description="Key insight",
            context="context",
            outcome="outcome",
        )
        store.add_trace(trace)
        assert len(store.traces) == 1

    def test_detect_breakthrough(self):
        """Detects breakthroughs when reference leads to success."""
        store = TraceStore()
        log = ConversationLog(
            agent_id="agent-0",
            generation=5,
            success=True,
        )
        log.add_turn("assistant", "Building on artifact-123...")

        trace = store.detect_breakthrough(
            log,
            artifact_id="new-artifact",
            referenced_artifacts=["artifact-123"],
        )

        assert trace is not None
        assert trace.trace_type == "reference_success"
        assert "artifact-123" in trace.related_artifacts

    def test_no_breakthrough_on_failure(self):
        """No breakthrough detected for failed attempts."""
        store = TraceStore()
        log = ConversationLog(
            agent_id="agent-0",
            generation=5,
            success=False,
        )

        trace = store.detect_breakthrough(
            log,
            artifact_id="failed-artifact",
            referenced_artifacts=["artifact-123"],
        )

        assert trace is None

    def test_get_breakthroughs(self):
        """Can filter to only breakthrough traces."""
        store = TraceStore()
        store.add_trace(ReasoningTrace(
            trace_type="reference_success",
            agent_id="a", generation=1,
            description="d", context="c", outcome="o",
        ))
        store.add_trace(ReasoningTrace(
            trace_type="error_fix",
            agent_id="a", generation=2,
            description="d", context="c", outcome="o",
        ))
        store.add_trace(ReasoningTrace(
            trace_type="breakthrough",
            agent_id="a", generation=3,
            description="d", context="c", outcome="o",
        ))

        breakthroughs = store.get_breakthroughs()
        assert len(breakthroughs) == 2  # reference_success and breakthrough

    def test_get_by_generation(self):
        """Can filter conversations by generation."""
        store = TraceStore()
        store.add_conversation(ConversationLog(agent_id="a", generation=0))
        store.add_conversation(ConversationLog(agent_id="b", generation=0))
        store.add_conversation(ConversationLog(agent_id="a", generation=1))

        gen0 = store.get_by_generation(0)
        assert len(gen0) == 2

        gen1 = store.get_by_generation(1)
        assert len(gen1) == 1

    def test_summary(self):
        """Summary provides aggregate stats."""
        store = TraceStore()

        log1 = ConversationLog(agent_id="a", generation=0, success=True)
        log1.add_turn("user", "x", tokens=100)
        log2 = ConversationLog(agent_id="b", generation=0, success=False)
        log2.add_turn("user", "y", tokens=200)

        store.add_conversation(log1)
        store.add_conversation(log2)
        store.add_trace(ReasoningTrace(
            trace_type="breakthrough",
            agent_id="a", generation=0,
            description="d", context="c", outcome="o",
        ))

        summary = store.summary()
        assert summary["total_conversations"] == 2
        assert summary["successful_conversations"] == 1
        assert summary["total_traces"] == 1
        assert summary["breakthroughs"] == 1
        assert summary["total_tokens"] == 300

    def test_save_and_load(self, tmp_path: Path):
        """TraceStore can be saved and loaded."""
        store = TraceStore()

        log = ConversationLog(agent_id="agent-0", generation=5, success=True)
        log.add_turn("user", "Hello", tokens=10)
        log.add_turn("assistant", "Hi there", tokens=20)
        store.add_conversation(log)

        trace = ReasoningTrace(
            trace_type="error_fix",
            agent_id="agent-0",
            generation=5,
            description="Fixed it",
            context="was broken",
            outcome="now works",
        )
        store.add_trace(trace)

        # Save
        store.save(tmp_path)

        # Verify files exist
        assert (tmp_path / "conversations.json").exists()
        assert (tmp_path / "reasoning_traces.json").exists()

        # Load
        loaded = TraceStore.load(tmp_path)

        assert len(loaded.conversations) == 1
        assert loaded.conversations[0].agent_id == "agent-0"
        assert len(loaded.conversations[0].turns) == 2

        assert len(loaded.traces) == 1
        assert loaded.traces[0].trace_type == "error_fix"
