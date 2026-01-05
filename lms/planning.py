"""Planning panel for generation-level task allocation.

The planning panel is a small group (1 Chair + 3 voting members) that
decides how to allocate work to Working Groups each generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lms.dependency import DependencyGraph, DependencyNode
    from lms.textbook import Textbook


class Vote(Enum):
    """A vote on a planning proposal."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class WorkingGroupAssignment:
    """An assignment of a task to a working group."""

    group_id: int
    task_tag: str
    task_name: str
    priority: int  # 1 = highest
    guidance: str  # Specific advice from panel, e.g., "Use structure not class"
    backup_task: str | None = None  # If primary is solved quickly

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "group_id": self.group_id,
            "task_tag": self.task_tag,
            "task_name": self.task_name,
            "priority": self.priority,
            "guidance": self.guidance,
            "backup_task": self.backup_task,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkingGroupAssignment:
        """Create from dictionary."""
        return cls(
            group_id=data["group_id"],
            task_tag=data["task_tag"],
            task_name=data["task_name"],
            priority=data["priority"],
            guidance=data["guidance"],
            backup_task=data.get("backup_task"),
        )


@dataclass
class PlanningProposal:
    """A proposal from the Chair for how to allocate work."""

    assignments: list[WorkingGroupAssignment]
    rationale: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "assignments": [a.to_dict() for a in self.assignments],
            "rationale": self.rationale,
        }


@dataclass
class PanelVote:
    """A vote from a panel member on a proposal."""

    member_id: str
    vote: Vote
    comment: str  # Why they voted this way

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "member_id": self.member_id,
            "vote": self.vote.value,
            "comment": self.comment,
        }


@dataclass
class PlanningSession:
    """State for a planning panel session."""

    generation: int
    chair_id: str
    member_ids: list[str]  # 3 voting members

    # Inputs
    available_tasks: list[DependencyNode] = field(default_factory=list)
    last_gen_failures: list[str] = field(default_factory=list)  # Summary of what failed
    foundation_summary: str = ""  # What's in Foundation.lean

    # Process
    proposals: list[PlanningProposal] = field(default_factory=list)
    votes: list[PanelVote] = field(default_factory=list)
    discussion: list[dict] = field(default_factory=list)  # Chat history

    # Output
    final_assignments: list[WorkingGroupAssignment] = field(default_factory=list)
    approved: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "generation": self.generation,
            "chair_id": self.chair_id,
            "member_ids": self.member_ids,
            "available_tasks": [
                {"tag": t.tag, "name": t.name} for t in self.available_tasks
            ],
            "last_gen_failures": self.last_gen_failures,
            "proposals": [p.to_dict() for p in self.proposals],
            "votes": [v.to_dict() for v in self.votes],
            "final_assignments": [a.to_dict() for a in self.final_assignments],
            "approved": self.approved,
        }


# =============================================================================
# Planning Panel Prompts
# =============================================================================

PLANNING_CHAIR_SYSTEM_PROMPT = """You are the Chair of the LMS Planning Panel.

Your role is to allocate work to Working Groups for this generation. You do NOT write code.

You must:
1. Prioritize tasks that unblock the most downstream work
2. Learn from past failures - include specific guidance
3. Avoid assigning the same task to multiple groups
4. Consider dependencies - don't assign blocked tasks

You are decisive but open to feedback from panel members.

When proposing assignments, use this format:
<proposal>
<rationale>Why this allocation makes sense</rationale>
<assignments>
<group id="1" task="TAG" backup="BACKUP_TAG" priority="1">
Specific guidance for this group...
</group>
<group id="2" task="TAG" backup="BACKUP_TAG" priority="2">
Specific guidance for this group...
</group>
...
</assignments>
</proposal>
"""

PLANNING_MEMBER_SYSTEM_PROMPT = """You are a voting member of the LMS Planning Panel.

Your role is to review the Chair's proposal and vote. You do NOT write code.

You should:
1. Check for conflicts or duplications
2. Verify priorities make sense
3. Ensure guidance is actionable
4. Vote APPROVE if the proposal is reasonable, REJECT if it has serious flaws

Be constructive - if you REJECT, explain what should change.

Format your vote as:
<vote>
<decision>APPROVE|REJECT|ABSTAIN</decision>
<comment>Your reasoning...</comment>
</vote>
"""


class PlanningPanel:
    """Orchestrates the planning panel discussion."""

    def __init__(
        self,
        provider,
        graph: DependencyGraph,
        textbook: Textbook | None = None,
        foundation_summary: str = "",
        n_groups: int = 3,
    ):
        """Initialize the planning panel.

        Args:
            provider: LLM provider for generating responses
            graph: Dependency graph tracking task status
            textbook: Optional textbook for learning from past failures
            foundation_summary: Summary of what's in Foundation.lean
            n_groups: Number of working groups to assign tasks to
        """
        self.provider = provider
        self.graph = graph
        self.textbook = textbook
        self.foundation_summary = foundation_summary
        self.n_groups = n_groups

    async def run_session(self, generation: int) -> list[WorkingGroupAssignment]:
        """Run a planning panel session and return assignments.

        Args:
            generation: Current generation number

        Returns:
            List of WorkingGroupAssignment for this generation
        """
        # 1. Gather context
        available = self.graph.available_tasks()
        if not available:
            return []  # Nothing to do

        # Get recent failures from textbook
        recent_failures = self._get_recent_failures()

        # 2. Create session
        session = PlanningSession(
            generation=generation,
            chair_id="panel-chair",
            member_ids=["panel-member-1", "panel-member-2", "panel-member-3"],
            available_tasks=available,
            last_gen_failures=recent_failures,
            foundation_summary=self.foundation_summary,
        )

        # 3. Chair proposes
        proposal = await self._get_chair_proposal(session)
        session.proposals.append(proposal)

        # 4. Members discuss and vote
        for member_id in session.member_ids:
            vote = await self._get_member_vote(session, member_id, proposal)
            session.votes.append(vote)

        # 5. Tally votes
        approvals = sum(1 for v in session.votes if v.vote == Vote.APPROVE)

        if approvals >= 2:  # Majority
            session.final_assignments = proposal.assignments
            session.approved = True
        else:
            # Chair revises based on feedback
            revised = await self._get_revised_proposal(session)
            session.proposals.append(revised)
            session.final_assignments = revised.assignments
            session.approved = True  # Chair decides on tie

        return session.final_assignments

    def _get_recent_failures(self) -> list[str]:
        """Get summaries of recent failures from textbook."""
        if self.textbook is None:
            return []

        failures = []
        entries = getattr(self.textbook, "entries", [])
        for entry in entries[-20:]:  # Last 20 entries
            title = getattr(entry, "title", "")
            content = getattr(entry, "content", "")
            if "[FAILED]" in title or "verification_error" in str(content).lower():
                failures.append(f"{title}: {str(content)[:200]}...")
        return failures[:5]  # Limit to 5 most recent

    def _extract_content(self, response) -> str:
        """Extract content string from provider response.

        Handles both raw strings and GenerationResponse objects.
        """
        if hasattr(response, "content"):
            return response.content
        return str(response)

    async def _get_chair_proposal(
        self, session: PlanningSession
    ) -> PlanningProposal:
        """Get the Chair's initial proposal."""
        prompt = self._build_chair_prompt(session)
        response = await self.provider.generate(
            [
                {"role": "system", "content": PLANNING_CHAIR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        content = self._extract_content(response)
        return self._parse_proposal(content, session.available_tasks)

    async def _get_member_vote(
        self, session: PlanningSession, member_id: str, proposal: PlanningProposal
    ) -> PanelVote:
        """Get a member's vote on the proposal."""
        prompt = self._build_vote_prompt(session, proposal)
        response = await self.provider.generate(
            [
                {"role": "system", "content": PLANNING_MEMBER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        content = self._extract_content(response)
        return self._parse_vote(content, member_id)

    async def _get_revised_proposal(
        self, session: PlanningSession
    ) -> PlanningProposal:
        """Get Chair's revised proposal after feedback."""
        prompt = self._build_revision_prompt(session)
        response = await self.provider.generate(
            [
                {"role": "system", "content": PLANNING_CHAIR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        content = self._extract_content(response)
        return self._parse_proposal(content, session.available_tasks)

    def _build_chair_prompt(self, session: PlanningSession) -> str:
        """Build the prompt for the Chair."""
        available_str = "\n".join(
            [
                f"- {t.tag}: {t.name} (unlocks {len(t.unlocks)} tasks, requires: {t.requires})"
                for t in session.available_tasks[: self.n_groups * 2]  # Show 2x groups
            ]
        )

        failures_str = (
            "\n".join(session.last_gen_failures)
            if session.last_gen_failures
            else "No recent failures."
        )

        return f"""# Planning Panel - Generation {session.generation}

## Available Tasks (sorted by priority)
{available_str}

## Recent Failures to Learn From
{failures_str}

## Foundation Summary
{session.foundation_summary or "Foundation.lean contains verified category theory definitions."}

## Your Task
Propose assignments for {self.n_groups} working groups. Each group should:
1. Work on ONE primary task
2. Have a backup task if primary is completed quickly
3. Receive specific guidance based on past failures

Use the <proposal> format with <assignments> containing <group> elements.
"""

    def _build_vote_prompt(
        self, session: PlanningSession, proposal: PlanningProposal
    ) -> str:
        """Build the prompt for a voting member."""
        assignments_str = self._format_assignments(proposal.assignments)

        return f"""# Planning Panel Vote - Generation {session.generation}

## Chair's Proposal
{proposal.rationale}

## Proposed Assignments
{assignments_str}

## Available Tasks
{", ".join(t.tag for t in session.available_tasks[:10])}

## Your Task
Vote on this proposal. Consider:
1. Are the priorities correct?
2. Is the guidance helpful based on past failures?
3. Are there conflicts or duplications?

Use the <vote> format with <decision> and <comment>.
"""

    def _build_revision_prompt(self, session: PlanningSession) -> str:
        """Build the prompt for Chair to revise after rejection."""
        original = session.proposals[0] if session.proposals else None
        original_str = (
            self._format_assignments(original.assignments)
            if original
            else "No original proposal."
        )

        feedback = "\n".join(
            [f"- {v.member_id}: {v.vote.value} - {v.comment}" for v in session.votes]
        )

        return f"""# Planning Panel - Revision Required

## Original Proposal
{original_str}

## Panel Feedback
{feedback}

## Your Task
Revise your proposal based on the feedback. Address the concerns raised.

Use the <proposal> format with updated <assignments>.
"""

    def _format_assignments(self, assignments: list[WorkingGroupAssignment]) -> str:
        """Format assignments for display."""
        if not assignments:
            return "No assignments."

        lines = []
        for a in assignments:
            lines.append(f"**Group {a.group_id}**: {a.task_tag} ({a.task_name})")
            lines.append(f"  Priority: {a.priority}")
            lines.append(f"  Backup: {a.backup_task or 'None'}")
            lines.append(f"  Guidance: {a.guidance}")
            lines.append("")
        return "\n".join(lines)

    def _parse_proposal(
        self, response: str, available_tasks: list[DependencyNode]
    ) -> PlanningProposal:
        """Parse a proposal from LLM response.

        Args:
            response: Raw LLM response text
            available_tasks: List of available tasks for validation

        Returns:
            Parsed PlanningProposal
        """
        assignments = []
        rationale = ""

        # Extract rationale
        rationale_match = re.search(
            r"<rationale>(.*?)</rationale>", response, re.DOTALL
        )
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        # Extract group assignments
        group_pattern = re.compile(
            r'<group\s+id="(\d+)"\s+task="([^"]+)"'
            r'(?:\s+backup="([^"]*)")?'
            r'(?:\s+priority="(\d+)")?'
            r"\s*>(.*?)</group>",
            re.DOTALL,
        )

        # Build lookup for task names
        task_lookup = {t.tag: t.name for t in available_tasks}

        for match in group_pattern.finditer(response):
            group_id = int(match.group(1))
            task_tag = match.group(2).strip()
            backup_task = match.group(3).strip() if match.group(3) else None
            priority = int(match.group(4)) if match.group(4) else group_id
            guidance = match.group(5).strip()

            # Get task name from lookup or use tag
            task_name = task_lookup.get(task_tag, task_tag)

            assignments.append(
                WorkingGroupAssignment(
                    group_id=group_id,
                    task_tag=task_tag,
                    task_name=task_name,
                    priority=priority,
                    guidance=guidance,
                    backup_task=backup_task if backup_task else None,
                )
            )

        # If no structured assignments found, create default from available tasks
        if not assignments and available_tasks:
            for i, task in enumerate(available_tasks[: self.n_groups]):
                assignments.append(
                    WorkingGroupAssignment(
                        group_id=i + 1,
                        task_tag=task.tag,
                        task_name=task.name,
                        priority=i + 1,
                        guidance="Work on this task following Foundation.lean patterns.",
                        backup_task=None,
                    )
                )
            rationale = "Default assignment based on task priority."

        return PlanningProposal(assignments=assignments, rationale=rationale)

    def _parse_vote(self, response: str, member_id: str) -> PanelVote:
        """Parse a vote from LLM response.

        Args:
            response: Raw LLM response text
            member_id: ID of the voting member

        Returns:
            Parsed PanelVote
        """
        # Try to extract structured vote
        decision_match = re.search(
            r"<decision>\s*(APPROVE|REJECT|ABSTAIN)\s*</decision>",
            response,
            re.IGNORECASE,
        )
        comment_match = re.search(r"<comment>(.*?)</comment>", response, re.DOTALL)

        if decision_match:
            decision_str = decision_match.group(1).upper()
            vote = Vote[decision_str]
        else:
            # Fall back to keyword detection
            response_upper = response.upper()
            if "REJECT" in response_upper:
                vote = Vote.REJECT
            elif "ABSTAIN" in response_upper:
                vote = Vote.ABSTAIN
            else:
                vote = Vote.APPROVE

        comment = comment_match.group(1).strip() if comment_match else response[:500]

        return PanelVote(member_id=member_id, vote=vote, comment=comment)


def create_default_assignments(
    available_tasks: list[DependencyNode], n_groups: int = 3
) -> list[WorkingGroupAssignment]:
    """Create default assignments without LLM involvement.

    Useful for testing or when you want deterministic assignment.

    Args:
        available_tasks: List of available tasks sorted by priority
        n_groups: Number of groups to assign to

    Returns:
        List of WorkingGroupAssignment
    """
    assignments = []
    for i, task in enumerate(available_tasks[:n_groups]):
        # Find a backup task if available
        backup = None
        if len(available_tasks) > n_groups + i:
            backup = available_tasks[n_groups + i].tag

        assignments.append(
            WorkingGroupAssignment(
                group_id=i + 1,
                task_tag=task.tag,
                task_name=task.name,
                priority=i + 1,
                guidance=f"Focus on {task.name}. Use existing Foundation.lean definitions.",
                backup_task=backup,
            )
        )

    return assignments
