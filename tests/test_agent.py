"""Tests for LMS agents."""

from unittest import mock

import pytest

from lms.agent import Agent, AgentResponse
from lms.artifacts import Artifact, ArtifactType, ArtifactLibrary
from lms.config import ProviderConfig
from lms.lean.mock import MockLeanVerifier
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    name = "mock"

    def __init__(self, config: ProviderConfig, responses: list[str] | None = None):
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0
        self.last_messages: list[Message] = []

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        self.last_messages = messages
        if self.responses:
            content = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
        else:
            content = "mock response"

        usage = TokenUsage(input_tokens=100, output_tokens=50)
        return GenerationResponse(content=content, usage=usage, provider=self.name)


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_create_response(self):
        """AgentResponse holds artifacts and raw text."""
        artifact = Artifact(
            id="test-001",
            type=ArtifactType.LEMMA,
            natural_language="Test",
            created_by="agent-1",
            generation=0,
        )
        response = AgentResponse(
            raw_text="I propose a lemma...",
            proposed_artifacts=[artifact],
            referenced_artifacts=["existing-001"],
        )
        assert response.raw_text == "I propose a lemma..."
        assert len(response.proposed_artifacts) == 1
        assert response.referenced_artifacts == ["existing-001"]


class TestAgent:
    """Tests for Agent class."""

    def test_create_agent(self):
        """Agent can be created with provider and ID."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        agent = Agent(id="agent-1", provider=provider)

        assert agent.id == "agent-1"
        assert agent.provider == provider

    def test_agent_has_generation(self):
        """Agent tracks its current generation."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        agent = Agent(id="agent-1", provider=provider, generation=2)

        assert agent.generation == 2

    @pytest.mark.asyncio
    async def test_agent_propose_parses_artifacts(self):
        """Agent can propose artifacts from LLM response."""
        config = ProviderConfig(api_key="test", model="test")
        # Response with artifact in expected format
        response_text = """
I'll prove a basic lemma about addition.

<artifact>
type: lemma
name: add_comm
description: Addition is commutative
lean: lemma add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response_text])
        agent = Agent(id="agent-1", provider=provider, generation=0)
        library = ArtifactLibrary()

        response = await agent.propose(library)

        assert len(response.proposed_artifacts) == 1
        artifact = response.proposed_artifacts[0]
        assert artifact.type == ArtifactType.LEMMA
        assert artifact.natural_language == "Addition is commutative"

    @pytest.mark.asyncio
    async def test_agent_includes_library_in_context(self):
        """Agent receives library contents in system prompt."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config, responses=["No new artifacts"])
        agent = Agent(id="agent-1", provider=provider, generation=1)

        library = ArtifactLibrary()
        library.add(Artifact(
            id="existing-001",
            type=ArtifactType.LEMMA,
            natural_language="Existing lemma",
            lean_code="lemma existing : True := trivial",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        await agent.propose(library)

        # Check that library contents were passed to the provider
        assert provider.call_count == 1
        # The system prompt should contain the existing artifact

    @pytest.mark.asyncio
    async def test_agent_tracks_references(self):
        """Agent tracks when artifacts reference existing ones."""
        config = ProviderConfig(api_key="test", model="test")
        response_text = """
Building on the existing lemma...

<artifact>
type: theorem
name: main_theorem
description: Main result using existing lemma
lean: theorem main_theorem : True := trivial
references: [existing-001]
</artifact>
"""
        provider = MockProvider(config, responses=[response_text])
        agent = Agent(id="agent-1", provider=provider, generation=1)

        library = ArtifactLibrary()
        library.add(Artifact(
            id="existing-001",
            type=ArtifactType.LEMMA,
            natural_language="Existing lemma",
            created_by="agent-0",
            generation=0,
        ))

        response = await agent.propose(library)

        assert "existing-001" in response.referenced_artifacts

    @pytest.mark.asyncio
    async def test_agent_can_propose_insights(self):
        """Agent can propose non-formal insights."""
        config = ProviderConfig(api_key="test", model="test")
        response_text = """
<artifact>
type: insight
name: induction_approach
description: Consider using strong induction on n
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response_text])
        agent = Agent(id="agent-1", provider=provider, generation=0)

        response = await agent.propose(ArtifactLibrary())

        assert len(response.proposed_artifacts) == 1
        assert response.proposed_artifacts[0].type == ArtifactType.INSIGHT

    @pytest.mark.asyncio
    async def test_agent_assigns_correct_metadata(self):
        """Agent sets created_by and generation on proposed artifacts."""
        config = ProviderConfig(api_key="test", model="test")
        response_text = """
<artifact>
type: lemma
name: test_lemma
description: Test
lean: lemma test : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response_text])
        agent = Agent(id="agent-42", provider=provider, generation=3)

        response = await agent.propose(ArtifactLibrary())

        artifact = response.proposed_artifacts[0]
        assert artifact.created_by == "agent-42"
        assert artifact.generation == 3

    def test_build_library_context_verified_first(self):
        """Verified artifacts should appear first in library context."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        agent = Agent(id="agent-1", provider=provider, generation=0)

        library = ArtifactLibrary()

        # Add unverified artifact first
        library.add(Artifact(
            id="unverified-001",
            type=ArtifactType.LEMMA,
            natural_language="Unverified lemma",
            verified=False,
            created_by="agent-0",
            generation=0,
        ))

        # Add verified artifact second
        library.add(Artifact(
            id="verified-001",
            type=ArtifactType.THEOREM,
            natural_language="Verified theorem",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        # Add another unverified
        library.add(Artifact(
            id="unverified-002",
            type=ArtifactType.INSIGHT,
            natural_language="Unverified insight",
            verified=False,
            created_by="agent-0",
            generation=1,
        ))

        context = agent._build_library_context(library)

        # Verified should appear before unverified in context
        verified_pos = context.find("verified-001")
        unverified_pos = context.find("unverified-001")

        assert verified_pos < unverified_pos, "Verified artifacts should appear first"

    def test_build_library_context_newer_generations_first_within_verified(self):
        """Within verified/unverified groups, newer generations should appear first."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        agent = Agent(id="agent-1", provider=provider, generation=0)

        library = ArtifactLibrary()

        # Add older verified artifact
        library.add(Artifact(
            id="verified-old",
            type=ArtifactType.LEMMA,
            natural_language="Old verified",
            verified=True,
            created_by="agent-0",
            generation=1,
        ))

        # Add newer verified artifact
        library.add(Artifact(
            id="verified-new",
            type=ArtifactType.THEOREM,
            natural_language="New verified",
            verified=True,
            created_by="agent-0",
            generation=5,
        ))

        context = agent._build_library_context(library)

        # Newer should appear before older within same verified status
        new_pos = context.find("verified-new")
        old_pos = context.find("verified-old")

        assert new_pos < old_pos, "Newer generations should appear first"


class TestAgentFoundationContext:
    """Tests for agent receiving foundation context."""

    @pytest.mark.asyncio
    async def test_agent_propose_receives_foundation(self, tmp_path):
        """Agent.propose() can receive foundation context."""
        from lms.foundation import FoundationFile

        config = ProviderConfig(api_key="test", model="test")
        response_text = """
<artifact>
type: definition
name: CFunctor
description: Functor between categories
lean: structure CFunctor (C D : Cat) where
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response_text])
        agent = Agent(id="agent-1", provider=provider, generation=1)

        # Create foundation with a definition
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")
        foundation.add_artifact(Artifact(
            id="def-Cat",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        library = ArtifactLibrary()
        response = await agent.propose(library, foundation=foundation)

        # Agent should have received foundation context in prompt
        last_message = provider.last_messages[0].content
        assert "Foundation" in last_message or "Cat" in last_message or "import" in last_message.lower()

    @pytest.mark.asyncio
    async def test_agent_foundation_context_shows_importable(self, tmp_path):
        """Foundation context tells agent what to import."""
        from lms.foundation import FoundationFile

        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config, responses=["mock response"])
        agent = Agent(id="agent-1", provider=provider, generation=1)

        # Create foundation with definitions
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")
        foundation.add_artifact(Artifact(
            id="def-Cat",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u\n  Hom : Obj → Obj → Type v",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))
        foundation.add_artifact(Artifact(
            id="def-Functor",
            type=ArtifactType.DEFINITION,
            natural_language="Functor",
            lean_code="structure CFunctor (C D : Cat) where\n  obj : C.Obj → D.Obj",
            verified=True,
            created_by="agent-1",
            generation=1,
        ))

        library = ArtifactLibrary()
        await agent.propose(library, foundation=foundation)

        # Check the prompt includes foundation info
        last_message = provider.last_messages[0].content

        # Should show import statement
        assert "import LMS.Foundation" in last_message
        # Should show available definitions
        assert "Cat" in last_message
        assert "CFunctor" in last_message

    @pytest.mark.asyncio
    async def test_agent_empty_foundation_no_import(self, tmp_path):
        """Empty foundation doesn't tell agent to import."""
        from lms.foundation import FoundationFile

        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config, responses=["mock response"])
        agent = Agent(id="agent-1", provider=provider, generation=0)

        # Empty foundation
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        library = ArtifactLibrary()
        await agent.propose(library, foundation=foundation)

        last_message = provider.last_messages[0].content

        # Should indicate empty or starting fresh
        assert "empty" in last_message.lower() or "no definitions" in last_message.lower() or "starting" in last_message.lower()
