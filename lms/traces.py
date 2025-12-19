"""Conversation and reasoning trace storage for analysis.

Captures full agent conversations and identifies reasoning breakthroughs
for post-hoc analysis of collective dynamics.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: int = 0


@dataclass
class ConversationLog:
    """Full conversation log for an agent's session.

    Stores the complete back-and-forth, not just summaries.
    """

    agent_id: str
    generation: int
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Outcome tracking
    success: bool = False
    artifacts_produced: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)

    def add_turn(
        self, role: str, content: str, tokens: int = 0
    ) -> None:
        """Add a conversation turn."""
        self.turns.append(ConversationTurn(
            role=role,
            content=content,
            tokens=tokens,
        ))

    def total_tokens(self) -> int:
        """Total tokens across all turns."""
        return sum(t.tokens for t in self.turns)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "success": self.success,
            "artifacts_produced": self.artifacts_produced,
            "errors_encountered": self.errors_encountered,
            "metadata": self.metadata,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp,
                    "tokens": t.tokens,
                }
                for t in self.turns
            ],
        }


@dataclass
class ReasoningTrace:
    """A notable reasoning event worth tracking.

    Captures breakthroughs, error diagnoses, strategy shifts, etc.
    """

    trace_type: str  # "breakthrough", "error_fix", "strategy_shift", "reference_success"
    agent_id: str
    generation: int
    description: str
    context: str  # What led to this
    outcome: str  # What resulted
    related_artifacts: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_type": self.trace_type,
            "agent_id": self.agent_id,
            "generation": self.generation,
            "description": self.description,
            "context": self.context,
            "outcome": self.outcome,
            "related_artifacts": self.related_artifacts,
            "timestamp": self.timestamp,
        }


class TraceStore:
    """Storage for conversation logs and reasoning traces.

    Enables post-hoc analysis of collective dynamics:
    - What conversations led to breakthroughs?
    - How did agents diagnose and fix errors?
    - What reference patterns correlated with success?
    """

    def __init__(self) -> None:
        self.conversations: list[ConversationLog] = []
        self.traces: list[ReasoningTrace] = []

    def add_conversation(self, log: ConversationLog) -> None:
        """Add a conversation log."""
        self.conversations.append(log)

    def add_trace(self, trace: ReasoningTrace) -> None:
        """Add a reasoning trace."""
        self.traces.append(trace)

    def detect_breakthrough(
        self,
        log: ConversationLog,
        artifact_id: str,
        referenced_artifacts: list[str],
    ) -> ReasoningTrace | None:
        """Detect if this conversation represents a breakthrough.

        A breakthrough is when:
        - Agent references previous work AND succeeds
        - Agent fixes an error that blocked previous generations
        - Agent discovers a new strategy that works
        """
        if not log.success:
            return None

        # Check if referenced artifacts and succeeded
        if referenced_artifacts:
            # Find what the agent said about the references
            context = ""
            for turn in log.turns:
                if turn.role == "assistant" and any(
                    ref in turn.content for ref in referenced_artifacts
                ):
                    context = turn.content[:500]
                    break

            return ReasoningTrace(
                trace_type="reference_success",
                agent_id=log.agent_id,
                generation=log.generation,
                description=f"Successfully built on {len(referenced_artifacts)} previous artifacts",
                context=context,
                outcome=f"Produced verified artifact: {artifact_id}",
                related_artifacts=[artifact_id] + referenced_artifacts,
            )

        return None

    def detect_error_fix(
        self,
        log: ConversationLog,
        previous_error: str,
        fix_description: str,
    ) -> ReasoningTrace:
        """Record when an agent fixes a recurring error."""
        return ReasoningTrace(
            trace_type="error_fix",
            agent_id=log.agent_id,
            generation=log.generation,
            description=fix_description,
            context=f"Previous error: {previous_error[:200]}",
            outcome="Error resolved",
            related_artifacts=log.artifacts_produced,
        )

    def save(self, path: Path) -> None:
        """Save all traces to JSON files."""
        # Save conversations
        conversations_path = path / "conversations.json"
        conversations_path.write_text(json.dumps(
            [c.to_dict() for c in self.conversations],
            indent=2,
        ))

        # Save reasoning traces
        traces_path = path / "reasoning_traces.json"
        traces_path.write_text(json.dumps(
            [t.to_dict() for t in self.traces],
            indent=2,
        ))

    @classmethod
    def load(cls, path: Path) -> "TraceStore":
        """Load traces from JSON files."""
        store = cls()

        conversations_path = path / "conversations.json"
        if conversations_path.exists():
            data = json.loads(conversations_path.read_text())
            for conv_data in data:
                log = ConversationLog(
                    agent_id=conv_data["agent_id"],
                    generation=conv_data["generation"],
                    success=conv_data.get("success", False),
                    artifacts_produced=conv_data.get("artifacts_produced", []),
                    errors_encountered=conv_data.get("errors_encountered", []),
                    metadata=conv_data.get("metadata", {}),
                )
                for turn_data in conv_data.get("turns", []):
                    log.turns.append(ConversationTurn(
                        role=turn_data["role"],
                        content=turn_data["content"],
                        timestamp=turn_data.get("timestamp", ""),
                        tokens=turn_data.get("tokens", 0),
                    ))
                store.conversations.append(log)

        traces_path = path / "reasoning_traces.json"
        if traces_path.exists():
            data = json.loads(traces_path.read_text())
            for trace_data in data:
                store.traces.append(ReasoningTrace(
                    trace_type=trace_data["trace_type"],
                    agent_id=trace_data["agent_id"],
                    generation=trace_data["generation"],
                    description=trace_data["description"],
                    context=trace_data["context"],
                    outcome=trace_data["outcome"],
                    related_artifacts=trace_data.get("related_artifacts", []),
                    timestamp=trace_data.get("timestamp", ""),
                ))

        return store

    def get_breakthroughs(self) -> list[ReasoningTrace]:
        """Get all breakthrough traces."""
        return [t for t in self.traces if t.trace_type in ("breakthrough", "reference_success")]

    def get_by_generation(self, generation: int) -> list[ConversationLog]:
        """Get all conversations from a specific generation."""
        return [c for c in self.conversations if c.generation == generation]

    def summary(self) -> dict:
        """Get summary statistics."""
        successful = [c for c in self.conversations if c.success]
        return {
            "total_conversations": len(self.conversations),
            "successful_conversations": len(successful),
            "total_traces": len(self.traces),
            "breakthroughs": len(self.get_breakthroughs()),
            "total_tokens": sum(c.total_tokens() for c in self.conversations),
        }
