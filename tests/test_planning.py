"""Tests for the planning panel module."""

import pytest

from lms.dependency import DependencyGraph, DependencyNode, TaskStatus
from lms.planning import (
    Vote,
    WorkingGroupAssignment,
    PlanningProposal,
    PanelVote,
    PlanningSession,
    PlanningPanel,
    create_default_assignments,
    PLANNING_CHAIR_SYSTEM_PROMPT,
    PLANNING_MEMBER_SYSTEM_PROMPT,
)


class TestVote:
    """Tests for Vote enum."""

    def test_vote_values(self):
        """Test vote enum has expected values."""
        assert Vote.APPROVE.value == "approve"
        assert Vote.REJECT.value == "reject"
        assert Vote.ABSTAIN.value == "abstain"

    def test_vote_from_string(self):
        """Test creating vote from string value."""
        assert Vote("approve") == Vote.APPROVE
        assert Vote("reject") == Vote.REJECT
        assert Vote("abstain") == Vote.ABSTAIN


class TestWorkingGroupAssignment:
    """Tests for WorkingGroupAssignment dataclass."""

    def test_create_assignment(self):
        """Test basic assignment creation."""
        assignment = WorkingGroupAssignment(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits and Colimits",
            priority=1,
            guidance="Use structure not class",
        )
        assert assignment.group_id == 1
        assert assignment.task_tag == "CH4-LIMITS"
        assert assignment.task_name == "Limits and Colimits"
        assert assignment.priority == 1
        assert assignment.guidance == "Use structure not class"
        assert assignment.backup_task is None

    def test_assignment_with_backup(self):
        """Test assignment with backup task."""
        assignment = WorkingGroupAssignment(
            group_id=2,
            task_tag="CH4-LIMITS",
            task_name="Limits",
            priority=1,
            guidance="Focus on products first",
            backup_task="CH4-ADJOINT",
        )
        assert assignment.backup_task == "CH4-ADJOINT"

    def test_to_dict(self):
        """Test serialization to dict."""
        assignment = WorkingGroupAssignment(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits",
            priority=2,
            guidance="Guidance text",
            backup_task="CH4-ADJOINT",
        )
        data = assignment.to_dict()
        assert data["group_id"] == 1
        assert data["task_tag"] == "CH4-LIMITS"
        assert data["task_name"] == "Limits"
        assert data["priority"] == 2
        assert data["guidance"] == "Guidance text"
        assert data["backup_task"] == "CH4-ADJOINT"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "group_id": 3,
            "task_tag": "CH5-TOP",
            "task_name": "Topology",
            "priority": 1,
            "guidance": "Some guidance",
            "backup_task": None,
        }
        assignment = WorkingGroupAssignment.from_dict(data)
        assert assignment.group_id == 3
        assert assignment.task_tag == "CH5-TOP"
        assert assignment.backup_task is None

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = WorkingGroupAssignment(
            group_id=1,
            task_tag="TEST",
            task_name="Test Task",
            priority=1,
            guidance="Test guidance",
            backup_task="BACKUP",
        )
        data = original.to_dict()
        restored = WorkingGroupAssignment.from_dict(data)
        assert restored.group_id == original.group_id
        assert restored.task_tag == original.task_tag
        assert restored.task_name == original.task_name
        assert restored.priority == original.priority
        assert restored.guidance == original.guidance
        assert restored.backup_task == original.backup_task


class TestPlanningProposal:
    """Tests for PlanningProposal dataclass."""

    def test_create_proposal(self):
        """Test basic proposal creation."""
        assignments = [
            WorkingGroupAssignment(
                group_id=1,
                task_tag="A",
                task_name="Task A",
                priority=1,
                guidance="Do A",
            )
        ]
        proposal = PlanningProposal(
            assignments=assignments,
            rationale="This is why we should do A first",
        )
        assert len(proposal.assignments) == 1
        assert proposal.rationale == "This is why we should do A first"

    def test_to_dict(self):
        """Test serialization to dict."""
        proposal = PlanningProposal(
            assignments=[
                WorkingGroupAssignment(
                    group_id=1,
                    task_tag="A",
                    task_name="Task A",
                    priority=1,
                    guidance="Do A",
                )
            ],
            rationale="Rationale text",
        )
        data = proposal.to_dict()
        assert "assignments" in data
        assert "rationale" in data
        assert len(data["assignments"]) == 1
        assert data["rationale"] == "Rationale text"


class TestPanelVote:
    """Tests for PanelVote dataclass."""

    def test_create_vote(self):
        """Test basic vote creation."""
        vote = PanelVote(
            member_id="member-1",
            vote=Vote.APPROVE,
            comment="Looks good to me",
        )
        assert vote.member_id == "member-1"
        assert vote.vote == Vote.APPROVE
        assert vote.comment == "Looks good to me"

    def test_to_dict(self):
        """Test serialization to dict."""
        vote = PanelVote(
            member_id="member-2",
            vote=Vote.REJECT,
            comment="I disagree because...",
        )
        data = vote.to_dict()
        assert data["member_id"] == "member-2"
        assert data["vote"] == "reject"
        assert data["comment"] == "I disagree because..."


class TestPlanningSession:
    """Tests for PlanningSession dataclass."""

    def test_create_session(self):
        """Test basic session creation."""
        session = PlanningSession(
            generation=5,
            chair_id="panel-chair",
            member_ids=["m1", "m2", "m3"],
        )
        assert session.generation == 5
        assert session.chair_id == "panel-chair"
        assert len(session.member_ids) == 3
        assert session.approved is False
        assert session.final_assignments == []

    def test_session_with_context(self):
        """Test session with full context."""
        task = DependencyNode(
            tag="A", name="Task A", chapter=1, section="1.0", status=TaskStatus.AVAILABLE
        )
        session = PlanningSession(
            generation=1,
            chair_id="chair",
            member_ids=["m1", "m2", "m3"],
            available_tasks=[task],
            last_gen_failures=["Previous attempt failed because..."],
            foundation_summary="Contains Category, Functor, etc.",
        )
        assert len(session.available_tasks) == 1
        assert len(session.last_gen_failures) == 1
        assert "Category" in session.foundation_summary

    def test_to_dict(self):
        """Test serialization to dict."""
        task = DependencyNode(
            tag="A", name="Task A", chapter=1, section="1.0"
        )
        session = PlanningSession(
            generation=1,
            chair_id="chair",
            member_ids=["m1", "m2"],
            available_tasks=[task],
        )
        session.approved = True
        data = session.to_dict()
        assert data["generation"] == 1
        assert data["chair_id"] == "chair"
        assert data["approved"] is True
        assert len(data["available_tasks"]) == 1


class TestPlanningPanelParsing:
    """Tests for PlanningPanel parsing methods."""

    @pytest.fixture
    def panel(self):
        """Create a panel for testing."""
        graph = DependencyGraph()
        graph.add_node(
            DependencyNode(
                tag="CH4-LIMITS",
                name="Limits and Colimits",
                chapter=4,
                section="4.4",
                status=TaskStatus.AVAILABLE,
            )
        )
        graph.add_node(
            DependencyNode(
                tag="CH4-ADJOINT",
                name="Adjoint Functors",
                chapter=4,
                section="4.5",
                status=TaskStatus.AVAILABLE,
            )
        )
        # Mock provider - won't be used in parse tests
        return PlanningPanel(
            provider=None,
            graph=graph,
            n_groups=2,
        )

    def test_parse_proposal_structured(self, panel):
        """Test parsing a well-structured proposal."""
        response = """
<proposal>
<rationale>We should focus on Limits first because it unlocks more tasks.</rationale>
<assignments>
<group id="1" task="CH4-LIMITS" backup="CH4-ADJOINT" priority="1">
Focus on defining the Limit structure using existing Foundation patterns.
</group>
<group id="2" task="CH4-ADJOINT" priority="2">
Work on adjoint functors after limits are done.
</group>
</assignments>
</proposal>
"""
        available = panel.graph.available_tasks()
        proposal = panel._parse_proposal(response, available)

        assert len(proposal.assignments) == 2
        assert proposal.rationale == "We should focus on Limits first because it unlocks more tasks."

        a1 = proposal.assignments[0]
        assert a1.group_id == 1
        assert a1.task_tag == "CH4-LIMITS"
        assert a1.backup_task == "CH4-ADJOINT"
        assert a1.priority == 1
        assert "Foundation patterns" in a1.guidance

        a2 = proposal.assignments[1]
        assert a2.group_id == 2
        assert a2.task_tag == "CH4-ADJOINT"
        assert a2.priority == 2

    def test_parse_proposal_no_backup(self, panel):
        """Test parsing proposal without backup tasks."""
        response = """
<proposal>
<rationale>Simple allocation</rationale>
<assignments>
<group id="1" task="CH4-LIMITS" priority="1">
Work on limits.
</group>
</assignments>
</proposal>
"""
        available = panel.graph.available_tasks()
        proposal = panel._parse_proposal(response, available)

        assert len(proposal.assignments) == 1
        assert proposal.assignments[0].backup_task is None

    def test_parse_proposal_fallback(self, panel):
        """Test fallback when no structured proposal found."""
        response = "I think we should work on limits and adjoint functors."
        available = panel.graph.available_tasks()
        proposal = panel._parse_proposal(response, available)

        # Should create default assignments from available tasks
        assert len(proposal.assignments) == min(panel.n_groups, len(available))
        assert proposal.rationale == "Default assignment based on task priority."

    def test_parse_vote_approve(self, panel):
        """Test parsing an approve vote."""
        response = """
<vote>
<decision>APPROVE</decision>
<comment>The proposal looks reasonable and addresses our priorities.</comment>
</vote>
"""
        vote = panel._parse_vote(response, "member-1")
        assert vote.vote == Vote.APPROVE
        assert "reasonable" in vote.comment
        assert vote.member_id == "member-1"

    def test_parse_vote_reject(self, panel):
        """Test parsing a reject vote."""
        response = """
<vote>
<decision>REJECT</decision>
<comment>The priorities are wrong. CH4-ADJOINT should come first.</comment>
</vote>
"""
        vote = panel._parse_vote(response, "member-2")
        assert vote.vote == Vote.REJECT
        assert "priorities are wrong" in vote.comment

    def test_parse_vote_abstain(self, panel):
        """Test parsing an abstain vote."""
        response = """
<vote>
<decision>ABSTAIN</decision>
<comment>I don't have enough context to vote.</comment>
</vote>
"""
        vote = panel._parse_vote(response, "member-3")
        assert vote.vote == Vote.ABSTAIN

    def test_parse_vote_fallback_approve(self, panel):
        """Test fallback vote detection for approve."""
        response = "I think this is fine. Let's go with it."
        vote = panel._parse_vote(response, "member-1")
        assert vote.vote == Vote.APPROVE

    def test_parse_vote_fallback_reject(self, panel):
        """Test fallback vote detection for reject."""
        response = "I REJECT this proposal because it makes no sense."
        vote = panel._parse_vote(response, "member-1")
        assert vote.vote == Vote.REJECT

    def test_parse_vote_lowercase_decision(self, panel):
        """Test parsing vote with lowercase decision."""
        response = """
<vote>
<decision>approve</decision>
<comment>Good plan.</comment>
</vote>
"""
        vote = panel._parse_vote(response, "member-1")
        assert vote.vote == Vote.APPROVE


class TestPlanningPanelPrompts:
    """Tests for prompt building methods."""

    @pytest.fixture
    def panel_with_tasks(self):
        """Create a panel with available tasks."""
        graph = DependencyGraph()
        graph.add_node(
            DependencyNode(
                tag="CH4-LIMITS",
                name="Limits and Colimits",
                chapter=4,
                section="4.4",
                unlocks=["CH5-TOP", "CH6-PRESHEAF"],
                status=TaskStatus.AVAILABLE,
            )
        )
        graph.add_node(
            DependencyNode(
                tag="CH4-ADJOINT",
                name="Adjoint Functors",
                chapter=4,
                section="4.5",
                unlocks=["CH5-CONT"],
                status=TaskStatus.AVAILABLE,
            )
        )
        return PlanningPanel(
            provider=None,
            graph=graph,
            foundation_summary="Contains Category, Functor, NatTrans",
            n_groups=2,
        )

    def test_build_chair_prompt(self, panel_with_tasks):
        """Test building chair prompt."""
        session = PlanningSession(
            generation=3,
            chair_id="chair",
            member_ids=["m1", "m2", "m3"],
            available_tasks=panel_with_tasks.graph.available_tasks(),
            last_gen_failures=["CH4-LIMITS failed: universe mismatch"],
            foundation_summary="Contains Category, Functor",
        )
        prompt = panel_with_tasks._build_chair_prompt(session)

        assert "Generation 3" in prompt
        assert "CH4-LIMITS" in prompt
        assert "Limits and Colimits" in prompt
        assert "unlocks 2 tasks" in prompt
        assert "universe mismatch" in prompt
        assert "2 working groups" in prompt

    def test_build_vote_prompt(self, panel_with_tasks):
        """Test building vote prompt."""
        session = PlanningSession(
            generation=1,
            chair_id="chair",
            member_ids=["m1", "m2", "m3"],
            available_tasks=panel_with_tasks.graph.available_tasks(),
        )
        proposal = PlanningProposal(
            assignments=[
                WorkingGroupAssignment(
                    group_id=1,
                    task_tag="CH4-LIMITS",
                    task_name="Limits",
                    priority=1,
                    guidance="Focus on products",
                )
            ],
            rationale="Limits are the priority",
        )
        prompt = panel_with_tasks._build_vote_prompt(session, proposal)

        assert "Limits are the priority" in prompt
        assert "Group 1" in prompt
        assert "CH4-LIMITS" in prompt
        assert "Focus on products" in prompt

    def test_build_revision_prompt(self, panel_with_tasks):
        """Test building revision prompt after rejection."""
        session = PlanningSession(
            generation=1,
            chair_id="chair",
            member_ids=["m1", "m2", "m3"],
        )
        session.proposals.append(
            PlanningProposal(
                assignments=[
                    WorkingGroupAssignment(
                        group_id=1,
                        task_tag="CH4-LIMITS",
                        task_name="Limits",
                        priority=1,
                        guidance="Original guidance",
                    )
                ],
                rationale="Original rationale",
            )
        )
        session.votes.append(
            PanelVote(member_id="m1", vote=Vote.REJECT, comment="Priorities wrong")
        )
        session.votes.append(
            PanelVote(member_id="m2", vote=Vote.APPROVE, comment="Looks good")
        )

        prompt = panel_with_tasks._build_revision_prompt(session)

        assert "Revision Required" in prompt
        assert "Original guidance" in prompt
        assert "Priorities wrong" in prompt
        assert "reject" in prompt.lower()


class TestCreateDefaultAssignments:
    """Tests for create_default_assignments helper."""

    def test_empty_tasks(self):
        """Test with no available tasks."""
        assignments = create_default_assignments([], n_groups=3)
        assert assignments == []

    def test_fewer_tasks_than_groups(self):
        """Test when there are fewer tasks than groups."""
        tasks = [
            DependencyNode(tag="A", name="Task A", chapter=1, section="1.0"),
            DependencyNode(tag="B", name="Task B", chapter=1, section="1.1"),
        ]
        assignments = create_default_assignments(tasks, n_groups=5)
        assert len(assignments) == 2
        assert assignments[0].task_tag == "A"
        assert assignments[1].task_tag == "B"

    def test_more_tasks_than_groups(self):
        """Test when there are more tasks than groups."""
        tasks = [
            DependencyNode(tag="A", name="Task A", chapter=1, section="1.0"),
            DependencyNode(tag="B", name="Task B", chapter=1, section="1.1"),
            DependencyNode(tag="C", name="Task C", chapter=1, section="1.2"),
            DependencyNode(tag="D", name="Task D", chapter=1, section="1.3"),
            DependencyNode(tag="E", name="Task E", chapter=1, section="1.4"),
        ]
        assignments = create_default_assignments(tasks, n_groups=2)

        assert len(assignments) == 2
        assert assignments[0].task_tag == "A"
        assert assignments[0].backup_task == "C"  # 3rd task is backup for 1st
        assert assignments[1].task_tag == "B"
        assert assignments[1].backup_task == "D"  # 4th task is backup for 2nd

    def test_priority_assignment(self):
        """Test that priorities are assigned sequentially."""
        tasks = [
            DependencyNode(tag="A", name="A", chapter=1, section="1.0"),
            DependencyNode(tag="B", name="B", chapter=1, section="1.1"),
            DependencyNode(tag="C", name="C", chapter=1, section="1.2"),
        ]
        assignments = create_default_assignments(tasks, n_groups=3)

        assert assignments[0].priority == 1
        assert assignments[1].priority == 2
        assert assignments[2].priority == 3

    def test_guidance_includes_task_name(self):
        """Test that guidance includes the task name."""
        tasks = [
            DependencyNode(tag="CH4-LIMITS", name="Limits and Colimits", chapter=4, section="4.4"),
        ]
        assignments = create_default_assignments(tasks, n_groups=1)

        assert "Limits and Colimits" in assignments[0].guidance


class TestPlanningPanelGetRecentFailures:
    """Tests for _get_recent_failures method."""

    def test_no_textbook(self):
        """Test with no textbook."""
        panel = PlanningPanel(
            provider=None,
            graph=DependencyGraph(),
            textbook=None,
        )
        failures = panel._get_recent_failures()
        assert failures == []

    def test_with_mock_textbook(self):
        """Test with mock textbook entries."""
        # Create a simple mock textbook
        class MockEntry:
            def __init__(self, title: str, content: str):
                self.title = title
                self.content = content

        class MockTextbook:
            def __init__(self):
                self.entries = [
                    MockEntry("[SUCCESS] CH4-CAT verified", "Category definition works"),
                    MockEntry("[FAILED] CH4-LIMITS", "verification_error: universe mismatch at line 42"),
                    MockEntry("[FAILED] CH4-ADJOINT", "Could not find instance for Group"),
                ]

        panel = PlanningPanel(
            provider=None,
            graph=DependencyGraph(),
            textbook=MockTextbook(),
        )
        failures = panel._get_recent_failures()

        assert len(failures) == 2
        assert "CH4-LIMITS" in failures[0]
        assert "CH4-ADJOINT" in failures[1]


class TestPlanningPanelSystemPrompts:
    """Tests for system prompts."""

    def test_chair_prompt_content(self):
        """Test that chair prompt has expected content."""
        assert "Chair" in PLANNING_CHAIR_SYSTEM_PROMPT
        assert "allocate work" in PLANNING_CHAIR_SYSTEM_PROMPT
        assert "do not write code" in PLANNING_CHAIR_SYSTEM_PROMPT.lower()
        assert "<proposal>" in PLANNING_CHAIR_SYSTEM_PROMPT

    def test_member_prompt_content(self):
        """Test that member prompt has expected content."""
        assert "voting member" in PLANNING_MEMBER_SYSTEM_PROMPT
        assert "APPROVE" in PLANNING_MEMBER_SYSTEM_PROMPT
        assert "REJECT" in PLANNING_MEMBER_SYSTEM_PROMPT
        assert "<vote>" in PLANNING_MEMBER_SYSTEM_PROMPT


class MockProvider:
    """Mock provider for async testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def generate(self, messages: list[dict]) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestPlanningPanelAsync:
    """Async tests for PlanningPanel."""

    @pytest.mark.asyncio
    async def test_run_session_approved(self):
        """Test running a session that gets approved."""
        graph = DependencyGraph()
        graph.add_node(
            DependencyNode(
                tag="CH4-LIMITS",
                name="Limits",
                chapter=4,
                section="4.4",
                status=TaskStatus.AVAILABLE,
            )
        )

        # Mock responses: chair proposal, 3 approve votes
        provider = MockProvider([
            """<proposal>
<rationale>Focus on limits</rationale>
<assignments>
<group id="1" task="CH4-LIMITS" priority="1">
Work on limits.
</group>
</assignments>
</proposal>""",
            "<vote><decision>APPROVE</decision><comment>Good</comment></vote>",
            "<vote><decision>APPROVE</decision><comment>Agree</comment></vote>",
            "<vote><decision>APPROVE</decision><comment>Fine</comment></vote>",
        ])

        panel = PlanningPanel(provider=provider, graph=graph, n_groups=1)
        assignments = await panel.run_session(generation=1)

        assert len(assignments) == 1
        assert assignments[0].task_tag == "CH4-LIMITS"
        # Chair + 3 votes = 4 calls
        assert provider.call_count == 4

    @pytest.mark.asyncio
    async def test_run_session_rejected_then_revised(self):
        """Test session where first proposal is rejected."""
        graph = DependencyGraph()
        graph.add_node(
            DependencyNode(
                tag="A",
                name="Task A",
                chapter=1,
                section="1.0",
                status=TaskStatus.AVAILABLE,
            )
        )
        graph.add_node(
            DependencyNode(
                tag="B",
                name="Task B",
                chapter=1,
                section="1.1",
                status=TaskStatus.AVAILABLE,
            )
        )

        # Mock responses: chair proposal, 2 rejects + 1 approve, revised proposal
        provider = MockProvider([
            # Initial proposal
            """<proposal>
<rationale>Initial</rationale>
<assignments>
<group id="1" task="A" priority="1">Do A.</group>
</assignments>
</proposal>""",
            # Votes (majority reject)
            "<vote><decision>REJECT</decision><comment>Wrong priority</comment></vote>",
            "<vote><decision>REJECT</decision><comment>Disagree</comment></vote>",
            "<vote><decision>APPROVE</decision><comment>OK</comment></vote>",
            # Revised proposal
            """<proposal>
<rationale>Revised based on feedback</rationale>
<assignments>
<group id="1" task="B" priority="1">Do B instead.</group>
</assignments>
</proposal>""",
        ])

        panel = PlanningPanel(provider=provider, graph=graph, n_groups=1)
        assignments = await panel.run_session(generation=1)

        # Should get revised assignment
        assert len(assignments) == 1
        assert assignments[0].task_tag == "B"
        # Chair + 3 votes + revision = 5 calls
        assert provider.call_count == 5

    @pytest.mark.asyncio
    async def test_run_session_no_available_tasks(self):
        """Test session with no available tasks."""
        graph = DependencyGraph()  # Empty

        provider = MockProvider(["Should not be called"])
        panel = PlanningPanel(provider=provider, graph=graph, n_groups=3)
        assignments = await panel.run_session(generation=1)

        assert assignments == []
        assert provider.call_count == 0
