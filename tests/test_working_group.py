"""Tests for the working group module."""

import pytest

from lms.working_group import (
    Role,
    GroupStatus,
    GroupMessage,
    WorkingGroupConfig,
    WorkingGroupState,
    WorkingGroup,
    create_working_group,
    CHAIR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SCRIBE_SYSTEM_PROMPT,
)


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test role enum has expected values."""
        assert Role.CHAIR.value == "chair"
        assert Role.SCRIBE.value == "scribe"
        assert Role.RESEARCHER.value == "researcher"

    def test_role_from_string(self):
        """Test creating role from string value."""
        assert Role("chair") == Role.CHAIR
        assert Role("scribe") == Role.SCRIBE
        assert Role("researcher") == Role.RESEARCHER


class TestGroupStatus:
    """Tests for GroupStatus enum."""

    def test_status_values(self):
        """Test status enum has expected values."""
        assert GroupStatus.FORMING.value == "forming"
        assert GroupStatus.DISCUSSING.value == "discussing"
        assert GroupStatus.DRAFTING.value == "drafting"
        assert GroupStatus.FINALIZING.value == "finalizing"
        assert GroupStatus.COMPLETE.value == "complete"
        assert GroupStatus.FAILED.value == "failed"


class TestGroupMessage:
    """Tests for GroupMessage dataclass."""

    def test_create_message(self):
        """Test basic message creation."""
        msg = GroupMessage(
            sender_id="agent-1",
            role=Role.RESEARCHER,
            content="I propose this code...",
            turn=1,
        )
        assert msg.sender_id == "agent-1"
        assert msg.role == Role.RESEARCHER
        assert msg.content == "I propose this code..."
        assert msg.turn == 1
        assert msg.timestamp > 0

    def test_to_provider_message(self):
        """Test conversion to provider message format."""
        msg = GroupMessage(
            sender_id="agent-1",
            role=Role.CHAIR,
            content="Welcome to the meeting",
            turn=0,
        )
        provider_msg = msg.to_provider_message()
        assert provider_msg["role"] == "user"
        assert "[CHAIR]" in provider_msg["content"]
        assert "agent-1" in provider_msg["content"]
        assert "Welcome to the meeting" in provider_msg["content"]

    def test_to_dict(self):
        """Test serialization to dict."""
        msg = GroupMessage(
            sender_id="agent-1",
            role=Role.SCRIBE,
            content="Compiling artifact...",
            turn=5,
            timestamp=1234567890.0,
        )
        data = msg.to_dict()
        assert data["sender_id"] == "agent-1"
        assert data["role"] == "scribe"
        assert data["content"] == "Compiling artifact..."
        assert data["turn"] == 5
        assert data["timestamp"] == 1234567890.0

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "sender_id": "agent-2",
            "role": "researcher",
            "content": "I disagree because...",
            "turn": 3,
            "timestamp": 1234567890.0,
        }
        msg = GroupMessage.from_dict(data)
        assert msg.sender_id == "agent-2"
        assert msg.role == Role.RESEARCHER
        assert msg.content == "I disagree because..."
        assert msg.turn == 3

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = GroupMessage(
            sender_id="agent-1",
            role=Role.CHAIR,
            content="Let's discuss...",
            turn=2,
        )
        data = original.to_dict()
        restored = GroupMessage.from_dict(data)
        assert restored.sender_id == original.sender_id
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.turn == original.turn


class TestWorkingGroupConfig:
    """Tests for WorkingGroupConfig dataclass."""

    def test_create_config(self):
        """Test basic config creation."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits and Colimits",
            task_content="Define limits...",
            guidance="Use structure not class",
        )
        assert config.group_id == 1
        assert config.task_tag == "CH4-LIMITS"
        assert config.task_name == "Limits and Colimits"
        assert config.max_turns == 5  # Default
        assert Role.CHAIR in config.members_per_role

    def test_config_custom_members(self):
        """Test config with custom member roles."""
        config = WorkingGroupConfig(
            group_id=2,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
            members_per_role={Role.CHAIR: 1, Role.RESEARCHER: 3},
        )
        assert config.members_per_role[Role.CHAIR] == 1
        assert config.members_per_role[Role.RESEARCHER] == 3
        assert Role.SCRIBE not in config.members_per_role

    def test_to_dict(self):
        """Test serialization to dict."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test Task",
            task_content="Content",
            guidance="Guidance",
            max_turns=7,
        )
        data = config.to_dict()
        assert data["group_id"] == 1
        assert data["task_tag"] == "TEST"
        assert data["max_turns"] == 7
        assert "chair" in data["members_per_role"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "group_id": 3,
            "task_tag": "CH5-TOP",
            "task_name": "Topology",
            "task_content": "Define topological space...",
            "guidance": "Follow Foundation patterns",
            "max_turns": 10,
            "members_per_role": {"chair": 1, "researcher": 2},
        }
        config = WorkingGroupConfig.from_dict(data)
        assert config.group_id == 3
        assert config.max_turns == 10
        assert config.members_per_role[Role.CHAIR] == 1
        assert config.members_per_role[Role.RESEARCHER] == 2


class TestWorkingGroupState:
    """Tests for WorkingGroupState dataclass."""

    def test_create_state(self):
        """Test basic state creation."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        state = WorkingGroupState(config=config)
        assert state.config == config
        assert state.messages == []
        assert state.blackboard == ""
        assert state.current_turn == 0
        assert state.status == GroupStatus.FORMING
        assert state.final_artifact is None

    def test_add_message(self):
        """Test adding messages to state."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        state = WorkingGroupState(config=config)
        state.current_turn = 2
        state.add_message("agent-1", Role.CHAIR, "Hello everyone")

        assert len(state.messages) == 1
        assert state.messages[0].sender_id == "agent-1"
        assert state.messages[0].role == Role.CHAIR
        assert state.messages[0].turn == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        state = WorkingGroupState(config=config)
        state.blackboard = "some code"
        state.status = GroupStatus.DISCUSSING
        data = state.to_dict()

        assert "config" in data
        assert data["blackboard"] == "some code"
        assert data["status"] == "discussing"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "config": {
                "group_id": 1,
                "task_tag": "TEST",
                "task_name": "Test Task",
                "task_content": "Content here",
                "guidance": "Guidance here",
                "max_turns": 5,
                "members_per_role": {"chair": 1, "scribe": 1, "researcher": 2},
            },
            "messages": [
                {
                    "sender_id": "agent-1",
                    "role": "chair",
                    "content": "Hello",
                    "turn": 0,
                    "timestamp": 1234567890.0,
                }
            ],
            "blackboard": "def Category",
            "current_turn": 1,
            "status": "discussing",
            "final_artifact": None,
        }
        state = WorkingGroupState.from_dict(data)

        assert state.config.group_id == 1
        assert state.config.task_tag == "TEST"
        assert len(state.messages) == 1
        assert state.messages[0].sender_id == "agent-1"
        assert state.messages[0].role == Role.CHAIR
        assert state.blackboard == "def Category"
        assert state.current_turn == 1
        assert state.status == GroupStatus.DISCUSSING

    def test_state_roundtrip(self):
        """Test WorkingGroupState survives serialization roundtrip."""
        config = WorkingGroupConfig(
            group_id=5,
            task_tag="CH4-YONEDA",
            task_name="Yoneda Lemma",
            task_content="Prove the Yoneda lemma",
            guidance="Use NatTrans from Foundation",
            max_turns=7,
        )
        state = WorkingGroupState(config=config)
        state.add_message("agent-1", Role.CHAIR, "Let's begin")
        state.add_message("agent-2", Role.RESEARCHER, "I propose...")
        state.blackboard = "structure Yoneda"
        state.status = GroupStatus.DRAFTING
        state.current_turn = 3

        # Roundtrip
        data = state.to_dict()
        restored = WorkingGroupState.from_dict(data)

        assert restored.config.group_id == 5
        assert restored.config.task_tag == "CH4-YONEDA"
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "Let's begin"
        assert restored.blackboard == "structure Yoneda"
        assert restored.status == GroupStatus.DRAFTING
        assert restored.current_turn == 3


class TestWorkingGroupParsing:
    """Tests for WorkingGroup parsing methods."""

    @pytest.fixture
    def group(self):
        """Create a group for testing."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits",
            task_content="Define limits...",
            guidance="Use structure syntax",
        )
        return WorkingGroup(config=config, provider=None)

    def test_update_blackboard(self, group):
        """Test extracting code from response."""
        response = """I propose this code:

```lean
structure Limit where
  apex : obj
  cone : Cone
```

This should work with our Foundation."""
        group._update_blackboard(response)
        assert "structure Limit" in group.state.blackboard
        assert "apex : obj" in group.state.blackboard

    def test_update_blackboard_no_code(self, group):
        """Test that no code means no update."""
        group.state.blackboard = "original"
        response = "I agree with the proposal."
        group._update_blackboard(response)
        assert group.state.blackboard == "original"

    def test_has_consensus_true(self, group):
        """Test consensus detection when present."""
        group.state.add_message("chair", Role.CHAIR, "CONSENSUS REACHED. Let's finalize.")
        assert group._has_consensus() is True

    def test_has_consensus_false(self, group):
        """Test consensus detection when not present."""
        group.state.add_message("chair", Role.CHAIR, "We need more discussion.")
        assert group._has_consensus() is False

    def test_has_consensus_case_insensitive(self, group):
        """Test consensus detection is case insensitive."""
        group.state.add_message("chair", Role.CHAIR, "Consensus Reached! Great work.")
        assert group._has_consensus() is True

    def test_has_consensus_empty(self, group):
        """Test consensus with no messages."""
        assert group._has_consensus() is False

    def test_parse_artifact_structured(self, group):
        """Test parsing a well-structured artifact."""
        response = """Here is the final artifact:

<artifact>
type: definition
name: Limit
stacks_tag: CH4-LIMITS
description: Definition of a categorical limit
lean: |
  structure Limit (F : Functor J C) where
    apex : C.obj
notes: |
  The group agreed on using the Cone structure from Foundation.
</artifact>
"""
        artifact = group._parse_artifact(response)
        assert artifact["type"] == "definition"
        assert artifact["name"] == "Limit"
        assert artifact["stacks_tag"] == "CH4-LIMITS"
        assert "Limit" in artifact["lean"]

    def test_parse_artifact_fallback(self, group):
        """Test fallback when no structured artifact found."""
        group.state.blackboard = "structure Limit where\n  apex : obj"
        response = "Here is the code we agreed on."
        artifact = group._parse_artifact(response)

        # Should use blackboard as fallback
        assert artifact["lean"] == group.state.blackboard
        assert artifact["task_tag"] == "CH4-LIMITS"

    def test_parse_artifact_group_info(self, group):
        """Test that artifact includes group info."""
        response = "Some response"
        artifact = group._parse_artifact(response)
        assert artifact["group_id"] == 1
        assert artifact["task_tag"] == "CH4-LIMITS"
        assert "raw" in artifact


class TestWorkingGroupMembers:
    """Tests for member management."""

    def test_member_creation_default(self):
        """Test default member creation."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        group = WorkingGroup(config=config, provider=None)

        # Default: 1 chair, 1 scribe, 1 researcher
        assert len(group.members) == 3
        roles = [role for _, role in group.members]
        assert Role.CHAIR in roles
        assert Role.SCRIBE in roles
        assert Role.RESEARCHER in roles

    def test_member_creation_custom(self):
        """Test custom member creation."""
        config = WorkingGroupConfig(
            group_id=2,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
            members_per_role={Role.CHAIR: 1, Role.RESEARCHER: 3},
        )
        group = WorkingGroup(config=config, provider=None)

        assert len(group.members) == 4
        researcher_count = sum(1 for _, r in group.members if r == Role.RESEARCHER)
        assert researcher_count == 3

    def test_get_member_by_role(self):
        """Test finding member by role."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        group = WorkingGroup(config=config, provider=None)

        chair_id = group._get_member_by_role(Role.CHAIR)
        assert chair_id is not None
        assert "chair" in chair_id

    def test_get_member_by_role_not_found(self):
        """Test finding member when role doesn't exist."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
            members_per_role={Role.RESEARCHER: 2},  # No chair
        )
        group = WorkingGroup(config=config, provider=None)

        chair_id = group._get_member_by_role(Role.CHAIR)
        assert chair_id is None


class TestWorkingGroupContext:
    """Tests for context building."""

    def test_build_context_empty(self):
        """Test building context with no messages."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits",
            task_content="...",
            guidance="...",
        )
        group = WorkingGroup(config=config, provider=None)
        context = group._build_context()

        assert "Working Group 1" in context
        assert "CH4-LIMITS" in context
        assert "Limits" in context

    def test_build_context_with_messages(self):
        """Test building context with messages."""
        config = WorkingGroupConfig(
            group_id=2,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        group = WorkingGroup(config=config, provider=None)
        group.state.add_message("chair-1", Role.CHAIR, "Let's start")
        group.state.current_turn = 1
        group.state.add_message("researcher-1", Role.RESEARCHER, "I propose...")

        context = group._build_context()
        assert "[CHAIR]" in context
        assert "Let's start" in context
        assert "[RESEARCHER]" in context
        assert "I propose..." in context

    def test_build_context_with_blackboard(self):
        """Test building context with blackboard."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
        )
        group = WorkingGroup(config=config, provider=None)
        group.state.blackboard = "structure Foo where\n  x : Nat"

        context = group._build_context()
        assert "Blackboard" in context
        assert "structure Foo" in context
        assert "```lean" in context


class TestWorkingGroupTranscript:
    """Tests for transcript generation."""

    def test_get_transcript(self):
        """Test generating a transcript."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits",
            task_content="...",
            guidance="...",
        )
        group = WorkingGroup(config=config, provider=None)
        group.state.status = GroupStatus.COMPLETE
        group.state.add_message("chair-1", Role.CHAIR, "Welcome")
        group.state.current_turn = 1
        group.state.add_message("researcher-1", Role.RESEARCHER, "I propose X")
        group.state.blackboard = "def X := ..."

        transcript = group.get_transcript()

        assert "Transcript" in transcript
        assert "CH4-LIMITS" in transcript
        assert "complete" in transcript
        assert "Welcome" in transcript
        assert "I propose X" in transcript
        assert "Final Blackboard" in transcript


class TestCreateWorkingGroup:
    """Tests for the factory function."""

    def test_create_working_group(self):
        """Test creating a working group with factory."""
        group = create_working_group(
            group_id=1,
            task_tag="CH4-LIMITS",
            task_name="Limits and Colimits",
            task_content="Define categorical limits...",
            provider=None,
            guidance="Use structure syntax",
            max_turns=7,
        )

        assert group.config.group_id == 1
        assert group.config.task_tag == "CH4-LIMITS"
        assert group.config.max_turns == 7
        assert group.config.guidance == "Use structure syntax"

    def test_create_working_group_defaults(self):
        """Test factory with default values."""
        group = create_working_group(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            provider=None,
        )

        assert group.config.max_turns == 5
        assert "Foundation.lean" in group.config.guidance


class TestSystemPrompts:
    """Tests for system prompts."""

    def test_chair_prompt_content(self):
        """Test chair prompt has expected content."""
        assert "CHAIR" in CHAIR_SYSTEM_PROMPT
        assert "FACILITATE" in CHAIR_SYSTEM_PROMPT
        assert "CONSENSUS REACHED" in CHAIR_SYSTEM_PROMPT
        assert "do not" in CHAIR_SYSTEM_PROMPT.lower() or "not to write code" in CHAIR_SYSTEM_PROMPT.lower()

    def test_researcher_prompt_content(self):
        """Test researcher prompt has expected content."""
        assert "RESEARCHER" in RESEARCHER_SYSTEM_PROMPT
        assert "code" in RESEARCHER_SYSTEM_PROMPT.lower()
        assert "```lean" in RESEARCHER_SYSTEM_PROMPT
        assert "sorry" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_scribe_prompt_content(self):
        """Test scribe prompt has expected content."""
        assert "SCRIBE" in SCRIBE_SYSTEM_PROMPT
        assert "<artifact>" in SCRIBE_SYSTEM_PROMPT
        assert "verification" in SCRIBE_SYSTEM_PROMPT.lower()


class MockProvider:
    """Mock provider for async testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def generate(self, messages: list[dict]) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestWorkingGroupAsync:
    """Async tests for WorkingGroup."""

    @pytest.mark.asyncio
    async def test_run_session_basic(self):
        """Test running a basic session."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test Task",
            task_content="Do the test task",
            guidance="Test guidance",
            max_turns=3,  # Short session
            members_per_role={Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: 1},
        )

        # Mock responses for each turn
        provider = MockProvider([
            # Chair opening
            "Let's discuss the test task. What approaches should we consider?",
            # Researcher turn 1
            """I propose this code:
```lean
def test := 42
```""",
            # Chair summary turn 1
            "CONSENSUS REACHED. We agree on the simple definition.",
            # Scribe finalize
            """<artifact>
type: definition
name: test
stacks_tag: TEST
description: A test definition
lean: |
  def test := 42
notes: |
  Simple consensus reached
</artifact>""",
        ])

        group = WorkingGroup(config=config, provider=provider)
        artifact = await group.run_session()

        assert artifact is not None
        assert artifact["task_tag"] == "TEST"
        assert group.state.status == GroupStatus.COMPLETE
        # Should have called provider multiple times
        assert provider.call_count >= 3

    @pytest.mark.asyncio
    async def test_run_session_updates_blackboard(self):
        """Test that blackboard is updated from researcher code."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
            max_turns=3,
            members_per_role={Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: 1},
        )

        provider = MockProvider([
            "Let's begin.",  # Chair
            """```lean
structure MyStruct where
  field : Nat
```""",  # Researcher with code
            "Good discussion.",  # Chair summary
            "Final artifact",  # Scribe
        ])

        group = WorkingGroup(config=config, provider=provider)
        await group.run_session()

        # Blackboard should have been updated
        assert "MyStruct" in group.state.blackboard

    @pytest.mark.asyncio
    async def test_run_session_early_consensus(self):
        """Test that session ends early when consensus is reached."""
        config = WorkingGroupConfig(
            group_id=1,
            task_tag="TEST",
            task_name="Test",
            task_content="...",
            guidance="...",
            max_turns=10,  # High max, but should exit early
            members_per_role={Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: 1},
        )

        call_count = 0

        async def mock_generate(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Opening statement"
            elif call_count == 2:
                return "```lean\ncode\n```"
            elif call_count == 3:
                return "CONSENSUS REACHED"  # Early consensus
            else:
                return "Final artifact"

        class CountingProvider:
            async def generate(self, messages):
                return await mock_generate(messages)

        group = WorkingGroup(config=config, provider=CountingProvider())
        await group.run_session()

        # Should have exited after consensus, not run all 10 turns
        # Chair opening + 1 round (researcher + chair summary) + scribe = ~4 calls
        assert call_count < 10
