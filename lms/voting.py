"""Voting system for LMS.

Supports two types of votes:
- Prompt changes: Simple majority (>50%) required
- Constitutional amendments (goals.md): Supermajority (2/3) required
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum

from lms.prompts import get_prompt


class VoteResult(Enum):
    """Result of a vote on a proposal."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class GoalAmendment:
    """A proposal to amend the constitution (goals.md).

    Requires 2/3 supermajority to pass.

    Attributes:
        id: Unique identifier for this amendment
        description: Short description of the change
        diff: The actual change (added/removed lines)
        rationale: Explanation of why the change is beneficial
        proposed_by: Agent ID that proposed this change
        generation: Generation when proposed
        votes_for: Agent IDs that voted for
        votes_against: Agent IDs that voted against
        applied: Whether this amendment has been applied
    """

    id: str
    description: str
    diff: str
    rationale: str
    proposed_by: str
    generation: int
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    applied: bool = False


@dataclass
class PromptProposal:
    """A proposal to change a prompt.

    Attributes:
        id: Unique identifier for this proposal
        prompt_name: Name of prompt to change (e.g., 'agent_system')
        new_content: Proposed new content for the prompt
        rationale: Explanation of why the change is beneficial
        proposed_by: Agent ID that proposed this change
        generation: Generation when proposed
        votes_for: Agent IDs that voted for
        votes_against: Agent IDs that voted against
        applied: Whether this proposal has been applied
    """

    id: str
    prompt_name: str
    new_content: str
    rationale: str
    proposed_by: str
    generation: int
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    applied: bool = False


@dataclass
class DefinitionCandidate:
    """A candidate definition in a conflict vote.

    Attributes:
        artifact_id: ID of the artifact proposing this definition
        concept_name: Name of the concept (e.g., "Category", "Functor")
        lean_code: The Lean code for this definition
        proposed_by: Agent ID that created this
    """

    artifact_id: str
    concept_name: str
    lean_code: str
    proposed_by: str


@dataclass
class DefinitionConflict:
    """A vote to resolve conflicting definitions.

    When multiple agents propose incompatible definitions for the same
    mathematical concept (e.g., two different Category structures),
    the society votes on which one to adopt.

    Attributes:
        id: Unique identifier for this conflict
        concept_name: The concept being defined (e.g., "Category")
        candidates: List of competing definitions
        generation: Generation when conflict arose
        votes: Dict mapping agent_id -> candidate index they voted for
        resolved: Whether this conflict has been resolved
        winner_idx: Index of winning candidate (after resolution)
    """

    id: str
    concept_name: str
    candidates: list[DefinitionCandidate]
    generation: int
    votes: dict[str, int] = field(default_factory=dict)  # agent_id -> candidate index
    resolved: bool = False
    winner_idx: int | None = None

    def vote(self, agent_id: str, candidate_idx: int) -> None:
        """Cast a vote for a candidate definition.

        Args:
            agent_id: Agent casting the vote
            candidate_idx: Index of the candidate they prefer
        """
        if candidate_idx < 0 or candidate_idx >= len(self.candidates):
            return
        self.votes[agent_id] = candidate_idx

    def resolve(self, n_agents: int) -> int | None:
        """Resolve the conflict by counting votes.

        Uses plurality voting - the candidate with the most votes wins.
        If there's a tie, the first candidate (earliest submission) wins.

        Args:
            n_agents: Total number of agents who could vote

        Returns:
            Index of winning candidate, or None if not enough votes
        """
        if len(self.votes) < n_agents // 2:  # Need at least half to vote
            return None

        # Count votes for each candidate
        vote_counts = [0] * len(self.candidates)
        for candidate_idx in self.votes.values():
            vote_counts[candidate_idx] += 1

        # Find winner (first candidate wins ties)
        max_votes = max(vote_counts)
        self.winner_idx = vote_counts.index(max_votes)
        self.resolved = True
        return self.winner_idx

    def get_winner(self) -> DefinitionCandidate | None:
        """Get the winning candidate if resolved."""
        if self.resolved and self.winner_idx is not None:
            return self.candidates[self.winner_idx]
        return None


class VotingSystem:
    """Manages proposals and voting for prompts, amendments, and definition conflicts.

    Agents can propose changes to prompts (simple majority), to the
    constitution/goals (2/3 supermajority), or vote on conflicting definitions.
    """

    # Threshold for supermajority (2/3)
    SUPERMAJORITY = 2 / 3

    def __init__(self, n_agents: int) -> None:
        """Initialize the voting system.

        Args:
            n_agents: Total number of agents who can vote
        """
        self.n_agents = n_agents
        self.proposals: list[PromptProposal] = []
        self.amendments: list[GoalAmendment] = []
        self.conflicts: list[DefinitionConflict] = []
        self._applied_prompts: dict[str, str] = {}

    def create_conflict(
        self,
        concept_name: str,
        candidates: list[DefinitionCandidate],
        generation: int,
    ) -> DefinitionConflict:
        """Create a new definition conflict for voting.

        Args:
            concept_name: Name of the concept (e.g., "Category")
            candidates: List of competing definitions
            generation: Current generation

        Returns:
            The created DefinitionConflict
        """
        conflict = DefinitionConflict(
            id=f"conflict-{uuid.uuid4().hex[:8]}",
            concept_name=concept_name,
            candidates=candidates,
            generation=generation,
        )
        # Each candidate's author automatically votes for their own
        for i, candidate in enumerate(candidates):
            conflict.vote(candidate.proposed_by, i)
        self.conflicts.append(conflict)
        return conflict

    def vote_on_conflict(self, conflict_id: str, agent_id: str, candidate_idx: int) -> None:
        """Cast a vote on a definition conflict.

        Args:
            conflict_id: ID of the conflict
            agent_id: Agent casting the vote
            candidate_idx: Index of candidate they prefer
        """
        for conflict in self.conflicts:
            if conflict.id == conflict_id and not conflict.resolved:
                conflict.vote(agent_id, candidate_idx)
                return

    def resolve_conflict(self, conflict_id: str) -> DefinitionCandidate | None:
        """Resolve a conflict and return the winner.

        Args:
            conflict_id: ID of the conflict to resolve

        Returns:
            Winning candidate, or None if can't resolve yet
        """
        for conflict in self.conflicts:
            if conflict.id == conflict_id:
                winner_idx = conflict.resolve(self.n_agents)
                if winner_idx is not None:
                    return conflict.get_winner()
        return None

    def get_pending_conflicts(self) -> list[DefinitionConflict]:
        """Get all unresolved conflicts."""
        return [c for c in self.conflicts if not c.resolved]

    def get_conflict_for_concept(self, concept_name: str) -> DefinitionConflict | None:
        """Get an unresolved conflict for a concept if one exists."""
        for conflict in self.conflicts:
            if conflict.concept_name == concept_name and not conflict.resolved:
                return conflict
        return None

    def submit_proposal(
        self,
        prompt_name: str,
        new_content: str,
        rationale: str,
        proposed_by: str,
        generation: int,
    ) -> PromptProposal:
        """Submit a new prompt change proposal.

        The proposing agent automatically votes for their proposal.

        Args:
            prompt_name: Name of prompt to change
            new_content: Proposed new content
            rationale: Explanation for the change
            proposed_by: Agent ID submitting the proposal
            generation: Current generation number

        Returns:
            The created PromptProposal
        """
        proposal = PromptProposal(
            id=f"prop-{uuid.uuid4().hex[:8]}",
            prompt_name=prompt_name,
            new_content=new_content,
            rationale=rationale,
            proposed_by=proposed_by,
            generation=generation,
            votes_for=[proposed_by],  # Auto-vote for own proposal
        )
        self.proposals.append(proposal)
        return proposal

    def vote(self, proposal_id: str, agent_id: str, approve: bool) -> None:
        """Cast a vote on a proposal.

        Each agent can only vote once per proposal. Subsequent
        votes are ignored.

        Args:
            proposal_id: ID of proposal to vote on
            agent_id: Agent casting the vote
            approve: True to vote for, False to vote against
        """
        proposal = self._get_proposal(proposal_id)
        if proposal is None:
            return

        # Check if already voted
        if agent_id in proposal.votes_for or agent_id in proposal.votes_against:
            return

        if approve:
            proposal.votes_for.append(agent_id)
        else:
            proposal.votes_against.append(agent_id)

    def check_result(self, proposal_id: str) -> VoteResult:
        """Check the current result of a proposal.

        A proposal is approved if votes_for > n_agents / 2 (simple majority).
        A proposal is rejected if votes_against >= n_agents / 2.
        Otherwise the result is pending.

        Args:
            proposal_id: ID of proposal to check

        Returns:
            VoteResult indicating current status
        """
        proposal = self._get_proposal(proposal_id)
        if proposal is None:
            return VoteResult.REJECTED

        majority_threshold = self.n_agents / 2

        if len(proposal.votes_for) > majority_threshold:
            return VoteResult.APPROVED
        elif len(proposal.votes_against) >= majority_threshold:
            return VoteResult.REJECTED
        else:
            return VoteResult.PENDING

    def apply_proposal(self, proposal_id: str) -> None:
        """Apply an approved proposal.

        Updates the active prompt to use the new content.

        Args:
            proposal_id: ID of proposal to apply

        Raises:
            ValueError: If proposal is not approved
        """
        proposal = self._get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        if self.check_result(proposal_id) != VoteResult.APPROVED:
            raise ValueError(f"Proposal {proposal_id} is not approved")

        self._applied_prompts[proposal.prompt_name] = proposal.new_content
        proposal.applied = True

    def get_applied_proposals(self) -> list[PromptProposal]:
        """Get all proposals that have been applied.

        Returns:
            List of applied PromptProposal objects
        """
        return [p for p in self.proposals if p.applied]

    def get_current_prompt(self, prompt_name: str) -> str:
        """Get the current content of a prompt.

        Returns the modified content if a proposal has been applied,
        otherwise returns the original prompt content.

        Args:
            prompt_name: Name of prompt to get

        Returns:
            Current prompt content
        """
        if prompt_name in self._applied_prompts:
            return self._applied_prompts[prompt_name]

        # Return original prompt
        return get_prompt(prompt_name).content

    def _get_proposal(self, proposal_id: str) -> PromptProposal | None:
        """Get a proposal by ID.

        Args:
            proposal_id: ID to look up

        Returns:
            The proposal, or None if not found
        """
        for proposal in self.proposals:
            if proposal.id == proposal_id:
                return proposal
        return None

    # --- Constitutional Amendment Methods (Supermajority) ---

    def submit_amendment(
        self,
        description: str,
        diff: str,
        rationale: str,
        proposed_by: str,
        generation: int,
    ) -> GoalAmendment:
        """Submit a constitutional amendment (requires 2/3 supermajority).

        The proposing agent automatically votes for their amendment.

        Args:
            description: Short description of the change
            diff: The actual change (added/removed lines)
            rationale: Explanation for the change
            proposed_by: Agent ID submitting the amendment
            generation: Current generation number

        Returns:
            The created GoalAmendment
        """
        amendment = GoalAmendment(
            id=f"amend-{uuid.uuid4().hex[:8]}",
            description=description,
            diff=diff,
            rationale=rationale,
            proposed_by=proposed_by,
            generation=generation,
            votes_for=[proposed_by],  # Auto-vote for own amendment
        )
        self.amendments.append(amendment)
        return amendment

    def vote_amendment(self, amendment_id: str, agent_id: str, approve: bool) -> None:
        """Cast a vote on a constitutional amendment.

        Each agent can only vote once per amendment.

        Args:
            amendment_id: ID of amendment to vote on
            agent_id: Agent casting the vote
            approve: True to vote for, False to vote against
        """
        amendment = self._get_amendment(amendment_id)
        if amendment is None:
            return

        # Check if already voted
        if agent_id in amendment.votes_for or agent_id in amendment.votes_against:
            return

        if approve:
            amendment.votes_for.append(agent_id)
        else:
            amendment.votes_against.append(agent_id)

    def check_amendment_result(self, amendment_id: str) -> VoteResult:
        """Check the current result of a constitutional amendment.

        An amendment is approved if votes_for >= 2/3 of n_agents.
        An amendment is rejected if votes_against > 1/3 of n_agents
        (making supermajority impossible).

        Args:
            amendment_id: ID of amendment to check

        Returns:
            VoteResult indicating current status
        """
        amendment = self._get_amendment(amendment_id)
        if amendment is None:
            return VoteResult.REJECTED

        supermajority_needed = self.n_agents * self.SUPERMAJORITY
        blocking_minority = self.n_agents - supermajority_needed

        if len(amendment.votes_for) >= supermajority_needed:
            return VoteResult.APPROVED
        elif len(amendment.votes_against) > blocking_minority:
            return VoteResult.REJECTED
        else:
            return VoteResult.PENDING

    def apply_amendment(self, amendment_id: str) -> None:
        """Apply an approved constitutional amendment.

        Args:
            amendment_id: ID of amendment to apply

        Raises:
            ValueError: If amendment is not approved
        """
        amendment = self._get_amendment(amendment_id)
        if amendment is None:
            raise ValueError(f"Amendment {amendment_id} not found")

        if self.check_amendment_result(amendment_id) != VoteResult.APPROVED:
            raise ValueError(f"Amendment {amendment_id} is not approved")

        amendment.applied = True

    def get_applied_amendments(self) -> list[GoalAmendment]:
        """Get all amendments that have been applied.

        Returns:
            List of applied GoalAmendment objects
        """
        return [a for a in self.amendments if a.applied]

    def _get_amendment(self, amendment_id: str) -> GoalAmendment | None:
        """Get an amendment by ID.

        Args:
            amendment_id: ID to look up

        Returns:
            The amendment, or None if not found
        """
        for amendment in self.amendments:
            if amendment.id == amendment_id:
                return amendment
        return None
