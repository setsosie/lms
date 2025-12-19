"""Working Groups for synchronous agent collaboration.

A Working Group is a small cluster of agents (typically 3-5) focused on a specific
mathematical goal (stacks_tag). Unlike the async correspondence model, agents
in a working group have a shared conversation history and 'blackboard' state.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lms.providers.base import Message


class Role(Enum):
    """Role within a working group."""

    CHAIR = "chair"  # Facilitates, doesn't write code
    SCRIBE = "scribe"  # Compiles final artifact
    RESEARCHER = "researcher"  # Proposes code, debates


class GroupStatus(Enum):
    """Current status of the working group."""

    FORMING = "forming"
    DISCUSSING = "discussing"
    DRAFTING = "drafting"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class GroupMessage:
    """A single message in the group chat."""

    sender_id: str
    role: Role
    content: str
    turn: int
    timestamp: float = field(default_factory=time.time)

    def to_provider_message(self) -> dict[str, str]:
        """Convert to LLM provider message format."""
        return {
            "role": "user",
            "content": f"[{self.role.value.upper()}] {self.sender_id}: {self.content}",
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sender_id": self.sender_id,
            "role": self.role.value,
            "content": self.content,
            "turn": self.turn,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GroupMessage:
        """Create from dictionary."""
        return cls(
            sender_id=data["sender_id"],
            role=Role(data["role"]),
            content=data["content"],
            turn=data["turn"],
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class WorkingGroupConfig:
    """Configuration for a working group."""

    group_id: int
    task_tag: str
    task_name: str
    task_content: str  # The full definition from Goal
    guidance: str  # From planning panel
    max_turns: int = 5
    members_per_role: dict[Role, int] = field(
        default_factory=lambda: {Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: 1}
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "group_id": self.group_id,
            "task_tag": self.task_tag,
            "task_name": self.task_name,
            "task_content": self.task_content,
            "guidance": self.guidance,
            "max_turns": self.max_turns,
            "members_per_role": {k.value: v for k, v in self.members_per_role.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkingGroupConfig:
        """Create from dictionary."""
        members = {Role(k): v for k, v in data.get("members_per_role", {}).items()}
        if not members:
            members = {Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: 1}
        return cls(
            group_id=data["group_id"],
            task_tag=data["task_tag"],
            task_name=data["task_name"],
            task_content=data["task_content"],
            guidance=data["guidance"],
            max_turns=data.get("max_turns", 5),
            members_per_role=members,
        )


@dataclass
class WorkingGroupState:
    """The evolving state of a working group session."""

    config: WorkingGroupConfig
    messages: list[GroupMessage] = field(default_factory=list)
    blackboard: str = ""  # Current shared draft
    current_turn: int = 0
    status: GroupStatus = GroupStatus.FORMING
    final_artifact: dict[str, Any] | None = None

    def add_message(self, sender_id: str, role: Role, content: str) -> None:
        """Add a message to the history."""
        self.messages.append(
            GroupMessage(
                sender_id=sender_id,
                role=role,
                content=content,
                turn=self.current_turn,
            )
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "blackboard": self.blackboard,
            "current_turn": self.current_turn,
            "status": self.status.value,
            "final_artifact": self.final_artifact,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingGroupState":
        """Create from dictionary for checkpoint resumption."""
        config = WorkingGroupConfig.from_dict(data["config"])
        return cls(
            config=config,
            messages=[GroupMessage.from_dict(m) for m in data["messages"]],
            blackboard=data["blackboard"],
            current_turn=data["current_turn"],
            status=GroupStatus(data["status"]),
            final_artifact=data.get("final_artifact"),
        )


# =============================================================================
# Role-specific system prompts
# =============================================================================

CHAIR_SYSTEM_PROMPT = """You are the CHAIR of an LMS Working Group.

Your role is to FACILITATE, not to write code:
1. Keep the group focused on the assigned task
2. Summarize agreements and disagreements
3. Identify when consensus is reached
4. Ask clarifying questions

You are neutral and ensure all voices are heard.

When the group has reached agreement on the code, say "CONSENSUS REACHED" and summarize the final approach."""


RESEARCHER_SYSTEM_PROMPT = """You are a RESEARCHER in an LMS Working Group.

Your role is to propose and critique code:
1. Write LEAN 4 code that addresses the task
2. Use existing Foundation.lean definitions correctly
3. Debate with colleagues - disagree if you see issues
4. Be specific about types, universes, and structure signatures

When you propose code, wrap it in ```lean code blocks.

Do NOT use `sorry`. Only propose complete, verifiable code.

If you agree with the current blackboard draft, say "I agree with the current proposal."
If you have changes, provide the updated code."""


SCRIBE_SYSTEM_PROMPT = """You are the SCRIBE of an LMS Working Group.

Your role is to compile the final artifact:
1. Take the agreed-upon code from the discussion
2. Format it as a proper <artifact> block
3. Ensure imports and namespace are correct
4. Add notes summarizing the group's key decisions

The artifact must be ready for LEAN verification.

Use this format:
<artifact>
type: definition|lemma|theorem
name: short_identifier
stacks_tag: TAG
description: Natural language description
lean: |
  -- Your LEAN 4 code here
notes: |
  Summary of group discussion and key decisions
</artifact>"""


class WorkingGroup:
    """Orchestrates a working group conversation."""

    def __init__(
        self,
        config: WorkingGroupConfig,
        provider,
        foundation_summary: str = "",
    ):
        """Initialize a working group.

        Args:
            config: Configuration for this group
            provider: LLM provider for generating responses
            foundation_summary: Summary of what's in Foundation.lean
        """
        self.config = config
        self.provider = provider
        self.foundation_summary = foundation_summary
        self.state = WorkingGroupState(config=config)

        # Create member IDs
        self.members: list[tuple[str, Role]] = []
        for role, count in config.members_per_role.items():
            for i in range(count):
                member_id = f"group-{config.group_id}-{role.value}-{i}"
                self.members.append((member_id, role))

    async def run_session(self) -> dict[str, Any] | None:
        """Run the full working group session.

        Returns:
            Final artifact dict or None if session failed
        """
        self.state.status = GroupStatus.DISCUSSING

        # Turn 0: Chair opens
        await self._chair_opening()

        # Turns 1 to N-1: Discussion
        for turn in range(1, self.config.max_turns - 1):
            self.state.current_turn = turn
            await self._discussion_round()

            # Check if we have consensus
            if self._has_consensus():
                break

        # Final turn: Scribe compiles
        self.state.status = GroupStatus.FINALIZING
        self.state.current_turn = self.config.max_turns
        await self._scribe_finalize()

        self.state.status = GroupStatus.COMPLETE
        return self.state.final_artifact

    async def _chair_opening(self) -> None:
        """Chair sets the agenda."""
        chair_id = self._get_member_by_role(Role.CHAIR)
        if not chair_id:
            return

        prompt = f"""# Working Group {self.config.group_id} - Opening

## Your Task
{self.config.task_tag}: {self.config.task_name}

{self.config.task_content}

## Guidance from Planning Panel
{self.config.guidance}

## Foundation Summary
{self.foundation_summary or "Foundation.lean contains verified category theory definitions."}

## Your Role
You are the CHAIR. Set the agenda for this discussion:
1. Summarize what we need to accomplish
2. Propose a strategy (e.g., "Let's first agree on the structure signature")
3. Ask specific questions to guide the Researchers

Do NOT write code. Facilitate the discussion."""

        response = await self.provider.generate(
            [
                {"role": "system", "content": CHAIR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        self.state.add_message(chair_id, Role.CHAIR, response)

    async def _discussion_round(self) -> None:
        """One round of discussion among all members."""
        # Each researcher responds
        researchers = [
            (member_id, role)
            for member_id, role in self.members
            if role == Role.RESEARCHER
        ]

        for member_id, role in researchers:
            context = self._build_context()

            prompt = f"""{context}

## Your Turn
You are a RESEARCHER. Based on the discussion so far:
1. Propose code or critique existing proposals
2. Reference Foundation.lean definitions correctly
3. Be specific about types, universes, and structure signatures

If you agree with the current blackboard draft, say "I agree with the current proposal."
If you have changes, provide the updated code in ```lean blocks."""

            response = await self.provider.generate(
                [
                    {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )

            self.state.add_message(member_id, Role.RESEARCHER, response)

            # Update blackboard if code was proposed
            if "```lean" in response:
                self._update_blackboard(response)

        # Chair summarizes
        await self._chair_summary()

    async def _chair_summary(self) -> None:
        """Chair summarizes the round and checks for consensus."""
        chair_id = self._get_member_by_role(Role.CHAIR)
        if not chair_id:
            return

        context = self._build_context()

        prompt = f"""{context}

## Your Turn (Chair Summary)
Summarize this round:
1. What was agreed upon?
2. What disagreements remain?
3. Is the group ready to finalize?

If ready, say "CONSENSUS REACHED" and summarize the agreed approach.
If not, pose the next question to resolve."""

        response = await self.provider.generate(
            [
                {"role": "system", "content": CHAIR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        self.state.add_message(chair_id, Role.CHAIR, response)

    async def _scribe_finalize(self) -> None:
        """Scribe compiles the final artifact."""
        scribe_id = self._get_member_by_role(Role.SCRIBE)
        if not scribe_id:
            return

        context = self._build_context()

        prompt = f"""{context}

## Your Turn (Final Compilation)
You are the SCRIBE. Compile the final artifact:
1. Use the agreed-upon code from the blackboard
2. Ensure all imports are correct
3. Format as a proper <artifact> block
4. Include notes summarizing the group's discussion

The artifact MUST be complete and ready for LEAN verification.

Use the <artifact> format with type, name, stacks_tag, description, lean, and notes fields."""

        response = await self.provider.generate(
            [
                {"role": "system", "content": SCRIBE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        self.state.add_message(scribe_id, Role.SCRIBE, response)

        # Parse the artifact
        self.state.final_artifact = self._parse_artifact(response)

    def _get_member_by_role(self, role: Role) -> str | None:
        """Get the first member with the given role."""
        for member_id, member_role in self.members:
            if member_role == role:
                return member_id
        return None

    def _build_context(self) -> str:
        """Build context from conversation history."""
        lines = [
            f"# Working Group {self.config.group_id}",
            f"## Task: {self.config.task_tag} - {self.config.task_name}",
            "",
            "## Conversation History",
        ]

        for msg in self.state.messages:
            lines.append(f"### [{msg.role.value.upper()}] Turn {msg.turn}")
            lines.append(msg.content)
            lines.append("")

        if self.state.blackboard:
            lines.append("## Current Blackboard (Draft Code)")
            lines.append("```lean")
            lines.append(self.state.blackboard)
            lines.append("```")

        return "\n".join(lines)

    def _update_blackboard(self, response: str) -> None:
        """Extract code from response and update blackboard."""
        match = re.search(r"```lean\n(.*?)```", response, re.DOTALL)
        if match:
            self.state.blackboard = match.group(1).strip()

    def _has_consensus(self) -> bool:
        """Check if the group has reached consensus."""
        if not self.state.messages:
            return False
        last_msg = self.state.messages[-1]
        return "CONSENSUS REACHED" in last_msg.content.upper()

    def _parse_artifact(self, response: str) -> dict[str, Any]:
        """Parse artifact from Scribe's response.

        Returns a dict with the artifact fields or a fallback structure.
        """
        artifact: dict[str, Any] = {
            "raw": response,
            "blackboard": self.state.blackboard,
            "group_id": self.config.group_id,
            "task_tag": self.config.task_tag,
        }

        # Try to parse structured artifact
        artifact_match = re.search(r"<artifact>(.*?)</artifact>", response, re.DOTALL)
        if artifact_match:
            content = artifact_match.group(1)

            # Parse fields
            type_match = re.search(r"type:\s*(\w+)", content)
            name_match = re.search(r"name:\s*([^\n]+)", content)
            tag_match = re.search(r"stacks_tag:\s*([^\n]+)", content)
            desc_match = re.search(r"description:\s*([^\n]+)", content)
            lean_match = re.search(r"lean:\s*\|?\n(.*?)(?=\n\w+:|$)", content, re.DOTALL)
            notes_match = re.search(r"notes:\s*\|?\n(.*?)(?=\n\w+:|$)", content, re.DOTALL)

            if type_match:
                artifact["type"] = type_match.group(1).strip()
            if name_match:
                artifact["name"] = name_match.group(1).strip()
            if tag_match:
                artifact["stacks_tag"] = tag_match.group(1).strip()
            if desc_match:
                artifact["description"] = desc_match.group(1).strip()
            if lean_match:
                artifact["lean"] = lean_match.group(1).strip()
            if notes_match:
                artifact["notes"] = notes_match.group(1).strip()

        # Fallback: use blackboard as lean code
        if "lean" not in artifact and self.state.blackboard:
            artifact["lean"] = self.state.blackboard
            artifact["type"] = "definition"
            artifact["name"] = f"working_group_{self.config.group_id}_output"
            artifact["stacks_tag"] = self.config.task_tag

        return artifact

    def get_transcript(self) -> str:
        """Get a human-readable transcript of the session."""
        lines = [
            f"# Working Group {self.config.group_id} Transcript",
            f"## Task: {self.config.task_tag} - {self.config.task_name}",
            f"## Status: {self.state.status.value}",
            "",
        ]

        for msg in self.state.messages:
            lines.append(f"### Turn {msg.turn} - {msg.role.value.upper()}")
            lines.append(f"*{msg.sender_id}*")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        if self.state.blackboard:
            lines.append("## Final Blackboard")
            lines.append("```lean")
            lines.append(self.state.blackboard)
            lines.append("```")

        return "\n".join(lines)


def create_working_group(
    group_id: int,
    task_tag: str,
    task_name: str,
    task_content: str,
    provider,
    guidance: str = "",
    foundation_summary: str = "",
    max_turns: int = 5,
) -> WorkingGroup:
    """Factory function to create a working group.

    Args:
        group_id: Unique group ID
        task_tag: Goal tag (e.g., "CH4-LIMITS")
        task_name: Human-readable task name
        task_content: Full task description
        provider: LLM provider
        guidance: Optional guidance from planning panel
        foundation_summary: Summary of Foundation.lean
        max_turns: Maximum discussion turns

    Returns:
        Configured WorkingGroup instance
    """
    config = WorkingGroupConfig(
        group_id=group_id,
        task_tag=task_tag,
        task_name=task_name,
        task_content=task_content,
        guidance=guidance or "Follow Foundation.lean patterns for structure definitions.",
        max_turns=max_turns,
    )
    return WorkingGroup(
        config=config,
        provider=provider,
        foundation_summary=foundation_summary,
    )
