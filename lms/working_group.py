"""Working Groups for synchronous agent collaboration.

A Working Group is a small cluster of agents (typically 3-5) focused on a specific
mathematical goal (stacks_tag). Unlike the async correspondence model, agents
in a working group have a shared conversation history and 'blackboard' state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from lms.agent import Agent
from lms.artifacts import Artifact
from lms.providers.base import Message

if TYPE_CHECKING:
    from lms.society import Society


class GroupStatus(Enum):
    """Current status of the working group."""
    FORMING = "forming"
    DISCUSSING = "discussing"
    DRAFTING = "drafting"
    VOTING = "voting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class GroupMessage:
    """A single message in the group chat."""
    sender_id: str
    role: str  # "Chair", "Scribe", "Researcher"
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_provider_message(self) -> Message:
        """Convert to LLM provider message format."""
        # Map internal roles to "user" (from perspective of other agents)
        # or "assistant" (if it was self). This logic happens in Agent.chat()
        # For the shared history, we store as is.
        return Message(
            role="user",  # Default to user, Agent.chat handles the 'self' logic
            content=f"[{self.role}] {self.sender_id}: {self.content}"
        )


@dataclass
class WorkingGroupState:
    """Mutable state of the working group."""
    messages: list[GroupMessage] = field(default_factory=list)
    blackboard: str = ""  # Shared draft/scratchpad
    status: GroupStatus = GroupStatus.FORMING
    current_round: int = 0
    
    def add_message(self, sender: Agent, role: str, content: str) -> None:
        """Add a message to the history."""
        self.messages.append(GroupMessage(
            sender_id=sender.id,
            role=role,
            content=content
        ))


class WorkingGroup:
    """A synchronous collaboration unit of agents."""

    def __init__(
        self,
        id: str,
        topic: str,
        goal_tag: str,
        members: list[Agent],
        chair: Agent,
        scribe: Agent,
        society: "Society",
    ) -> None:
        """Initialize a working group.

        Args:
            id: Unique Group ID
            topic: Natural language topic (e.g. "Localization of Rings")
            goal_tag: Stacks Project tag (e.g. "00CM")
            members: List of all agents in the group
            chair: The agent acting as Moderator/Chair
            scribe: The agent acting as Scribe/Librarian
            society: Reference to parent society (for verification/tools)
        """
        self.id = id
        self.topic = topic
        self.goal_tag = goal_tag
        self.members = members
        self.chair = chair
        self.scribe = scribe
        self.society = society
        self.state = WorkingGroupState()
        
        # Verify roles
        if chair not in members:
            raise ValueError("Chair must be a member of the group")
        if scribe not in members:
            raise ValueError("Scribe must be a member of the group")

    async def run_session(self, max_rounds: int = 5) -> Optional[Artifact]:
        """Run a synchronous collaboration session.
        
        A session consists of multiple rounds of conversation, ending in
        a formal proposal by the Scribe.
        
        Args:
            max_rounds: Maximum discussion rounds before forced drafting
            
        Returns:
            The final proposed Artifact (if successful), or None
        """
        self.state.status = GroupStatus.DISCUSSING
        
        # 1. Chair Opening
        await self._broadcast(self.chair, "Chair", 
            f"Welcome. Our goal is to formalize {self.goal_tag}: {self.topic}. "
            "Please review the Blackboard. I propose we start by agreeing on definitions."
        )
        
        # 2. Discussion Loop
        for i in range(max_rounds):
            self.state.current_round = i + 1
            
            # Everyone speaks in turn (simple round-robin for now)
            # In future: Chair decides who speaks next
            for agent in self.members:
                if agent == self.chair:
                    continue # Chair speaks separately/last to summarize
                    
                role = "Scribe" if agent == self.scribe else "Researcher"
                
                # Get agent's contribution
                response = await agent.chat(
                    history=[m.to_provider_message() for m in self.state.messages],
                    role_prompt=self._get_role_prompt(role),
                    blackboard=self.state.blackboard
                )
                
                # Add to history
                self.state.add_message(agent, role, response.content)
                
                # If they proposed a blackboard update, apply it
                # (Simple overwrite for now, diffing later)
                if response.blackboard_update:
                    self.state.blackboard = response.blackboard_update
                    self.state.add_message(agent, "System", 
                        f"updated the blackboard (length: {len(response.blackboard_update)} chars)"
                    )

            # Chair summarizes/steers at end of round
            chair_response = await self.chair.chat(
                history=[m.to_provider_message() for m in self.state.messages],
                role_prompt=self._get_role_prompt("Chair"),
                blackboard=self.state.blackboard
            )
            self.state.add_message(self.chair, "Chair", chair_response.content)
            
            # Check for consensus (Chair decides)
            if "CONSENSUS_REACHED" in chair_response.content:
                self.state.status = GroupStatus.DRAFTING
                break
        
        # 3. Drafting (Scribe only)
        self.state.status = GroupStatus.DRAFTING
        final_proposal = await self.scribe.propose_from_blackboard(
            blackboard=self.state.blackboard,
            goal_tag=self.goal_tag,
            history=[m.to_provider_message() for m in self.state.messages]
        )
        
        # 4. Final Output
        self.state.status = GroupStatus.COMPLETE
        return final_proposal

    async def _broadcast(self, agent: Agent, role: str, content: str) -> None:
        """Send a message to the group (forced)."""
        self.state.add_message(agent, role, content)

    def _get_role_prompt(self, role: str) -> str:
        """Get the system prompt addition for a specific role."""
        prompts = {
            "Chair": (
                f"You are the Chair of the working group on {self.goal_tag}. "
                "Your goal is CONSENSUS. "
                "1. Guide the discussion. "
                "2. Summarize points. "
                "3. If agreement is reached, say 'CONSENSUS_REACHED'. "
                "Do not write long code blocks. Focus on high-level decisions."
            ),
            "Scribe": (
                "You are the Scribe. Your goal is ACCURACY. "
                "1. Keep the Blackboard updated with the latest agreed definitions. "
                "2. Note down conflicts. "
                "3. You will be responsible for the final proposal."
            ),
            "Researcher": (
                "You are a Researcher. Your goal is RIGOR. "
                "1. Propose definitions and proofs. "
                "2. Critique others' proposals. "
                "3. Spot edge cases."
            )
        }
        return prompts.get(role, prompts["Researcher"])
