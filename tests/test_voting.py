"""Tests for prompt voting system."""

import pytest

from lms.voting import (
    PromptProposal,
    GoalAmendment,
    VotingSystem,
    VoteResult,
)


class TestPromptProposal:
    """Tests for PromptProposal dataclass."""

    def test_create_proposal(self):
        """PromptProposal holds proposal details."""
        proposal = PromptProposal(
            id="prop-001",
            prompt_name="agent_system",
            new_content="Modified system prompt",
            rationale="Better instructions",
            proposed_by="agent-1",
            generation=2,
        )
        assert proposal.prompt_name == "agent_system"
        assert proposal.proposed_by == "agent-1"

    def test_proposal_defaults(self):
        """PromptProposal has sensible defaults."""
        proposal = PromptProposal(
            id="prop-001",
            prompt_name="agent_user",
            new_content="New user prompt",
            rationale="Clearer",
            proposed_by="agent-0",
            generation=0,
        )
        assert proposal.votes_for == []
        assert proposal.votes_against == []
        assert proposal.applied is False


class TestVotingSystem:
    """Tests for VotingSystem."""

    def test_create_voting_system(self):
        """VotingSystem can be created with agent count."""
        system = VotingSystem(n_agents=6)
        assert system.n_agents == 6

    def test_submit_proposal(self):
        """Agents can submit prompt change proposals."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New prompt content",
            rationale="This is better because...",
            proposed_by="agent-1",
            generation=0,
        )
        assert proposal.id is not None
        assert proposal in system.proposals

    def test_vote_for(self):
        """Agents can vote for a proposal."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=True)
        assert "agent-1" in proposal.votes_for

    def test_vote_against(self):
        """Agents can vote against a proposal."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=False)
        assert "agent-1" in proposal.votes_against

    def test_agent_can_only_vote_once(self):
        """Each agent can only vote once per proposal."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=True)
        system.vote(proposal.id, "agent-1", approve=False)  # Should not change

        assert "agent-1" in proposal.votes_for
        assert "agent-1" not in proposal.votes_against

    def test_proposer_auto_votes_for(self):
        """The proposing agent automatically votes for their proposal."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        assert "agent-0" in proposal.votes_for

    def test_check_result_approved(self):
        """Proposal is approved with simple majority."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        # agent-0 already voted for
        system.vote(proposal.id, "agent-1", approve=True)

        result = system.check_result(proposal.id)
        assert result == VoteResult.APPROVED  # 2 for, 0 against, 1 abstain = majority

    def test_check_result_rejected(self):
        """Proposal is rejected without majority."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=False)
        system.vote(proposal.id, "agent-2", approve=False)

        result = system.check_result(proposal.id)
        assert result == VoteResult.REJECTED

    def test_check_result_pending(self):
        """Result is pending until voting complete."""
        system = VotingSystem(n_agents=5)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        # Only 1 vote (proposer), need more
        result = system.check_result(proposal.id)
        assert result == VoteResult.PENDING

    def test_check_result_tie_rejected(self):
        """Ties result in rejection (need majority, not just plurality)."""
        system = VotingSystem(n_agents=4)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=True)
        system.vote(proposal.id, "agent-2", approve=False)
        system.vote(proposal.id, "agent-3", approve=False)

        result = system.check_result(proposal.id)
        # 2 for, 2 against = tie = rejected
        assert result == VoteResult.REJECTED

    def test_apply_approved_proposal(self):
        """Approved proposals can be applied."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=True)

        # Should be approved
        assert system.check_result(proposal.id) == VoteResult.APPROVED

        # Apply it
        system.apply_proposal(proposal.id)
        assert proposal.applied is True

    def test_cannot_apply_rejected_proposal(self):
        """Rejected proposals cannot be applied."""
        system = VotingSystem(n_agents=3)
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=False)
        system.vote(proposal.id, "agent-2", approve=False)

        with pytest.raises(ValueError, match="not approved"):
            system.apply_proposal(proposal.id)

    def test_get_applied_proposals(self):
        """Can retrieve list of applied proposals."""
        system = VotingSystem(n_agents=2)

        # Approved proposal
        p1 = system.submit_proposal(
            prompt_name="agent_system",
            new_content="New content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(p1.id, "agent-1", approve=True)
        system.apply_proposal(p1.id)

        # Rejected proposal
        p2 = system.submit_proposal(
            prompt_name="agent_user",
            new_content="Bad content",
            rationale="No",
            proposed_by="agent-0",
            generation=1,
        )
        system.vote(p2.id, "agent-1", approve=False)

        applied = system.get_applied_proposals()
        assert len(applied) == 1
        assert applied[0].id == p1.id

    def test_get_current_prompt(self):
        """Can get current prompt (original or modified)."""
        system = VotingSystem(n_agents=2)

        # Initially returns original
        original = system.get_current_prompt("agent_system")
        assert "mathematical researcher" in original.lower()

        # Submit and apply change
        proposal = system.submit_proposal(
            prompt_name="agent_system",
            new_content="You are a theorem prover.",
            rationale="More focused",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote(proposal.id, "agent-1", approve=True)
        system.apply_proposal(proposal.id)

        # Now returns modified
        current = system.get_current_prompt("agent_system")
        assert current == "You are a theorem prover."


class TestGoalAmendment:
    """Tests for GoalAmendment (constitutional changes)."""

    def test_create_amendment(self):
        """GoalAmendment holds amendment details."""
        amendment = GoalAmendment(
            id="amend-001",
            description="Add Milestone 8: Completions",
            diff="+ ## Milestone 8: Completions\n+ - [ ] Completion (00M9)",
            rationale="We need completion theory",
            proposed_by="agent-2",
            generation=5,
        )
        assert amendment.description == "Add Milestone 8: Completions"
        assert amendment.proposed_by == "agent-2"

    def test_amendment_defaults(self):
        """GoalAmendment has sensible defaults."""
        amendment = GoalAmendment(
            id="amend-001",
            description="Test",
            diff="+ new line",
            rationale="Because",
            proposed_by="agent-0",
            generation=0,
        )
        assert amendment.votes_for == []
        assert amendment.votes_against == []
        assert amendment.applied is False


class TestConstitutionalVoting:
    """Tests for supermajority voting on constitutional amendments."""

    def test_submit_amendment(self):
        """Agents can submit constitutional amendments."""
        system = VotingSystem(n_agents=6)
        amendment = system.submit_amendment(
            description="Add new milestone",
            diff="+ ## Milestone 8",
            rationale="Needed for completeness",
            proposed_by="agent-0",
            generation=0,
        )
        assert amendment.id is not None
        assert amendment in system.amendments

    def test_amendment_requires_supermajority(self):
        """Amendments require 2/3 supermajority to pass."""
        system = VotingSystem(n_agents=6)
        amendment = system.submit_amendment(
            description="Add milestone",
            diff="+ content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        # agent-0 auto-votes, add 2 more = 3/6 = 50% (not enough)
        system.vote_amendment(amendment.id, "agent-1", approve=True)
        system.vote_amendment(amendment.id, "agent-2", approve=True)

        result = system.check_amendment_result(amendment.id)
        assert result == VoteResult.PENDING  # Need 4/6 for supermajority

    def test_amendment_passes_with_supermajority(self):
        """Amendment passes with 2/3 votes."""
        system = VotingSystem(n_agents=6)
        amendment = system.submit_amendment(
            description="Add milestone",
            diff="+ content",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        # Need 4/6 = 66.7% for supermajority
        system.vote_amendment(amendment.id, "agent-1", approve=True)
        system.vote_amendment(amendment.id, "agent-2", approve=True)
        system.vote_amendment(amendment.id, "agent-3", approve=True)

        result = system.check_amendment_result(amendment.id)
        assert result == VoteResult.APPROVED

    def test_amendment_rejected_without_supermajority(self):
        """Amendment is rejected if it can't reach 2/3."""
        system = VotingSystem(n_agents=6)
        amendment = system.submit_amendment(
            description="Bad idea",
            diff="+ bad content",
            rationale="No good reason",
            proposed_by="agent-0",
            generation=0,
        )
        # 3 votes against means supermajority impossible
        system.vote_amendment(amendment.id, "agent-1", approve=False)
        system.vote_amendment(amendment.id, "agent-2", approve=False)
        system.vote_amendment(amendment.id, "agent-3", approve=False)

        result = system.check_amendment_result(amendment.id)
        assert result == VoteResult.REJECTED

    def test_apply_amendment(self):
        """Approved amendments can be applied."""
        system = VotingSystem(n_agents=3)
        amendment = system.submit_amendment(
            description="Add milestone",
            diff="+ ## Milestone 8: New",
            rationale="Reason",
            proposed_by="agent-0",
            generation=0,
        )
        # 2/3 of 3 = 2 needed
        system.vote_amendment(amendment.id, "agent-1", approve=True)

        result = system.check_amendment_result(amendment.id)
        assert result == VoteResult.APPROVED

        system.apply_amendment(amendment.id)
        assert amendment.applied is True

    def test_cannot_apply_unapproved_amendment(self):
        """Cannot apply amendment without supermajority."""
        system = VotingSystem(n_agents=3)
        amendment = system.submit_amendment(
            description="Test",
            diff="+ test",
            rationale="Test",
            proposed_by="agent-0",
            generation=0,
        )
        # Only proposer voted, not enough

        with pytest.raises(ValueError, match="not approved"):
            system.apply_amendment(amendment.id)

    def test_get_applied_amendments(self):
        """Can retrieve list of applied amendments."""
        system = VotingSystem(n_agents=3)

        a1 = system.submit_amendment(
            description="Good change",
            diff="+ good",
            rationale="Good",
            proposed_by="agent-0",
            generation=0,
        )
        system.vote_amendment(a1.id, "agent-1", approve=True)
        system.apply_amendment(a1.id)

        a2 = system.submit_amendment(
            description="Rejected change",
            diff="+ bad",
            rationale="Bad",
            proposed_by="agent-0",
            generation=1,
        )
        system.vote_amendment(a2.id, "agent-1", approve=False)
        system.vote_amendment(a2.id, "agent-2", approve=False)

        applied = system.get_applied_amendments()
        assert len(applied) == 1
        assert applied[0].id == a1.id
