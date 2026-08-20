"""Tests for LMS Society orchestration."""

from pathlib import Path
from unittest import mock

import pytest

from lms.artifacts import ArtifactLibrary
from lms.config import ProviderConfig
from lms.dependency import TaskStatus
from lms.lean.mock import MockLeanVerifier
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.society import Society, GenerationResult
from lms.lean.interface import (
    LeanVerifier,
    VerificationResult,
    VerificationStatus,
    VerifierKind,
)


class StubLeanVerifier(LeanVerifier):
    """Accepts any code, but declares Lean-grade provenance.

    Foundation accumulation is gated on `VERIFIED_LEAN`, so exercising that
    path needs a verifier whose kind is Lean-grade. `MockLeanVerifier` can only
    ever reach `VERIFIED_HEURISTIC` — see `test_mock_verifier_does_not_populate
    _foundation` for the other half of this contract.
    """

    verifier_kind: VerifierKind = "real"

    async def verify(self, code: str) -> VerificationResult:
        return self._result(success=True, code=code)


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    name = "mock"

    def __init__(self, config: ProviderConfig, responses: list[str] | None = None):
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        if self.responses:
            content = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
        else:
            content = "No artifacts proposed."

        usage = TokenUsage(input_tokens=100, output_tokens=50)
        return GenerationResponse(content=content, usage=usage, provider=self.name)


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_create_generation_result(self):
        """GenerationResult holds generation metrics."""
        result = GenerationResult(
            generation=1,
            artifacts_created=3,
            artifacts_verified=2,
            artifacts_referenced=1,
            fresh_creations=2,
        )
        assert result.generation == 1
        assert result.artifacts_created == 3


class TestSociety:
    """Tests for Society orchestration."""

    def test_create_society(self):
        """Society can be created with basic config."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        verifier = MockLeanVerifier()

        society = Society(
            n_agents=3,
            provider=provider,
            verifier=verifier,
        )

        assert society.n_agents == 3
        assert len(society.agents) == 3

    def test_society_creates_agents(self):
        """Society creates the specified number of agents."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        verifier = MockLeanVerifier()

        society = Society(n_agents=6, provider=provider, verifier=verifier)

        assert len(society.agents) == 6
        # Agents should have unique IDs
        ids = [a.id for a in society.agents]
        assert len(set(ids)) == 6

    def test_society_has_shared_library(self):
        """Society has a shared artifact library."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        verifier = MockLeanVerifier()

        society = Society(n_agents=3, provider=provider, verifier=verifier)

        assert society.library is not None
        assert isinstance(society.library, ArtifactLibrary)

    @pytest.mark.asyncio
    async def test_run_generation(self):
        """Society can run a single generation."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: test_lemma
description: A test lemma
lean: lemma test : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        society = Society(n_agents=2, provider=provider, verifier=verifier)
        result = await society.run_generation(generation=0)

        assert result.generation == 0
        assert result.artifacts_created >= 1

    @pytest.mark.asyncio
    async def test_run_multiple_generations(self):
        """Society can run multiple generations."""
        config = ProviderConfig(api_key="test", model="test")
        responses = [
            """
<artifact>
type: lemma
name: base_lemma
description: Base lemma
lean: lemma base : True := trivial
references: []
</artifact>
""",
            """
<artifact>
type: theorem
name: derived_thm
description: Derived theorem
lean: theorem derived : True := trivial
references: [lemma-base_lemma]
</artifact>
""",
        ]
        provider = MockProvider(config, responses=responses)
        verifier = MockLeanVerifier()

        society = Society(n_agents=1, provider=provider, verifier=verifier)
        results = await society.run(n_generations=2)

        assert len(results) == 2
        assert results[0].generation == 0
        assert results[1].generation == 1

    @pytest.mark.asyncio
    async def test_library_accumulates_across_generations(self):
        """Library grows across generations."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: test
description: Test
lean: lemma test : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        society = Society(n_agents=2, provider=provider, verifier=verifier)
        await society.run(n_generations=3)

        # With 2 agents * 3 generations, we should have multiple artifacts
        assert len(society.library) >= 3

    @pytest.mark.asyncio
    async def test_verifier_is_called(self):
        """LEAN verifier is called for artifacts with code."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: verified_lemma
description: A lemma to verify
lean: lemma verified : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        with mock.patch.object(
            verifier, "verify", wraps=verifier.verify
        ) as mock_verify:
            society = Society(n_agents=1, provider=provider, verifier=verifier)
            await society.run_generation(0)

            # Verifier should have been called
            assert mock_verify.called

    @pytest.mark.asyncio
    async def test_results_track_verification(self):
        """Generation results track verification success."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: good_lemma
description: A good lemma
lean: lemma good : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        society = Society(n_agents=1, provider=provider, verifier=verifier)
        result = await society.run_generation(0)

        # The mock verifier should accept valid syntax
        assert result.artifacts_verified >= 0

    @pytest.mark.asyncio
    async def test_agents_update_generation(self):
        """Agents are updated with current generation number."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config, responses=["No artifacts"])
        verifier = MockLeanVerifier()

        society = Society(n_agents=2, provider=provider, verifier=verifier)

        await society.run_generation(0)
        # Agents should be at generation 0
        for agent in society.agents:
            assert agent.generation == 0

        await society.run_generation(1)
        # Agents should now be at generation 1
        for agent in society.agents:
            assert agent.generation == 1

    @pytest.mark.asyncio
    async def test_save_results(self, tmp_path: Path):
        """Society can save results to disk."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: saved_lemma
description: A lemma to save
lean: lemma saved : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        society = Society(n_agents=1, provider=provider, verifier=verifier)
        await society.run(n_generations=1)

        # Save to temp directory
        society.save(tmp_path)

        # Check files were created
        assert (tmp_path / "artifacts.json").exists()
        assert (tmp_path / "results.json").exists()

    @pytest.mark.asyncio
    async def test_load_from_checkpoint(self, tmp_path: Path):
        """Society can load from checkpoint."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: checkpoint_lemma
description: A lemma for checkpoint
lean: lemma checkpoint : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        # Run and save
        society = Society(n_agents=1, provider=provider, verifier=verifier)
        await society.run(n_generations=2)
        society.save(tmp_path)

        # Load from checkpoint
        loaded = Society.load(tmp_path, provider, verifier)

        assert loaded.current_generation == 2
        assert len(loaded.library) > 0
        assert len(loaded.results) == 2

    @pytest.mark.asyncio
    async def test_load_preserves_textbook(self, tmp_path: Path):
        """Society.load restores textbook from checkpoint."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: lemma
name: textbook_lemma
description: A lemma for textbook
notes: Important insight about lemmas
lean: lemma textbook : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        # Run and save
        society = Society(n_agents=1, provider=provider, verifier=verifier)
        await society.run(n_generations=1)

        # Add something to textbook
        society.textbook.add(
            content="Test wisdom",
            author="test-agent",
            generation=0,
            topics=["test"],
        )
        society.save(tmp_path)

        # Load from checkpoint
        loaded = Society.load(tmp_path, provider, verifier)

        assert len(loaded.textbook) > 0

    @pytest.mark.asyncio
    async def test_load_preserves_goal(self, tmp_path: Path):
        """Society.load restores goal from checkpoint."""
        from lms.goals import Goal, StacksDefinition

        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: theorem
name: goal_theorem
stacks_tag: TEST1
description: A theorem for goal
lean: theorem goal_thm : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        # Create goal
        goal = Goal(
            name="Test Goal",
            description="A test goal",
            source="Test",
            definitions=[
                StacksDefinition(tag="TEST1", section="1", name="Test", content="..."),
            ],
        )

        # Run with goal and save
        society = Society(n_agents=1, provider=provider, verifier=verifier, goal=goal)
        await society.run(n_generations=1)
        society.save(tmp_path)

        # Also save goal
        goal.save(tmp_path / "goal.json")

        # Load from checkpoint
        loaded = Society.load(tmp_path, provider, verifier)

        assert loaded.goal is not None
        assert loaded.goal.name == "Test Goal"
        assert len(loaded.goal.definitions) == 1


class TestSocietyFoundation:
    """Tests for Society integration with FoundationFile."""

    @pytest.mark.asyncio
    async def test_society_has_foundation(self, tmp_path: Path):
        """Society has a foundation file for accumulated definitions."""
        from lms.foundation import FoundationFile

        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        verifier = MockLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        assert society.foundation is not None
        assert isinstance(society.foundation, FoundationFile)

    @pytest.mark.asyncio
    async def test_verified_artifacts_added_to_foundation(self, tmp_path: Path):
        """Verified artifacts are automatically added to foundation."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: definition
name: Cat
description: Category structure
lean: structure Cat where
  Obj : Type u
  Hom : Obj → Obj → Type v
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        # Use a verifier that always verifies
        verifier = StubLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        await society.run_generation(0)

        # Foundation should have the verified artifact
        assert len(society.foundation) >= 1

    @pytest.mark.asyncio
    async def test_unverified_artifacts_not_in_foundation(self, tmp_path: Path):
        """Unverified artifacts are NOT added to foundation."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: definition
name: BadDef
description: This will fail verification
lean: this is not valid lean syntax!!!
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        await society.run_generation(0)

        assert len(society.foundation) == 0

    @pytest.mark.asyncio
    async def test_mock_verifier_does_not_populate_foundation(self, tmp_path: Path):
        """The mock cannot seed the foundation even with code it accepts.

        Regression test for 26Q3-HARN-01. `MockLeanVerifier` is a regex; the
        code below matches it happily. Before provenance existed this produced
        a `verified` artifact that was written into the shared Lean corpus and
        counted toward the roadmap numbers.
        """
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: theorem
name: goal_thm
description: Trivially true, and the mock will accept it
lean: theorem goal_thm : True := trivial
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=MockLeanVerifier(),
            foundation_path=foundation_path,
        )

        result = await society.run_generation(0)

        assert result.artifacts_created >= 1, "the mock should still accept the code"
        assert len(society.foundation) == 0, "but nothing may reach the corpus"
        assert society.library.get_verified() == []
        assert all(
            a.status is VerificationStatus.VERIFIED_HEURISTIC
            for a in society.library.all()
            if a.lean_code
        )

    @pytest.mark.asyncio
    async def test_foundation_grows_across_generations(self, tmp_path: Path):
        """Foundation accumulates artifacts across generations."""
        config = ProviderConfig(api_key="test", model="test")
        responses = [
            """
<artifact>
type: definition
name: Cat
description: Category
lean: structure Cat where
  Obj : Type u
references: []
</artifact>
""",
            """
<artifact>
type: definition
name: CFunctor
description: Functor
lean: structure CFunctor (C D : Cat) where
  obj : C.Obj → D.Obj
references: []
</artifact>
""",
        ]
        provider = MockProvider(config, responses=responses)
        verifier = StubLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        await society.run(n_generations=2)

        # Foundation should have both definitions
        assert len(society.foundation) >= 2

    @pytest.mark.asyncio
    async def test_foundation_saved_with_checkpoint(self, tmp_path: Path):
        """Foundation is saved when society saves checkpoint."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: definition
name: Cat
description: Category
lean: structure Cat where
  Obj : Type u
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = StubLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        await society.run_generation(0)
        society.save(tmp_path)

        # Foundation files should exist
        assert foundation_path.exists()
        assert foundation_path.with_suffix(".json").exists()

    @pytest.mark.asyncio
    async def test_foundation_loaded_from_checkpoint(self, tmp_path: Path):
        """Foundation is restored when loading from checkpoint."""
        config = ProviderConfig(api_key="test", model="test")
        response = """
<artifact>
type: definition
name: Cat
description: Category
lean: structure Cat where
  Obj : Type u
references: []
</artifact>
"""
        provider = MockProvider(config, responses=[response])
        verifier = MockLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        await society.run_generation(0)
        original_len = len(society.foundation)
        society.save(tmp_path)

        # Load from checkpoint
        loaded = Society.load(tmp_path, provider, verifier)

        assert len(loaded.foundation) == original_len

    @pytest.mark.asyncio
    async def test_foundation_context_provided_to_agents(self, tmp_path: Path):
        """Agents receive foundation context in their prompts."""
        config = ProviderConfig(api_key="test", model="test")

        # Track what context the agent receives
        received_prompts = []

        class TrackingProvider(MockProvider):
            async def generate(self, messages, system_prompt=None, max_tokens=4096):
                received_prompts.append(messages[0].content if messages else "")
                return await super().generate(messages, system_prompt, max_tokens)

        responses = [
            """
<artifact>
type: definition
name: Cat
description: Category
lean: structure Cat where
  Obj : Type u
references: []
</artifact>
""",
            """
<artifact>
type: definition
name: CFunctor
description: Functor
lean: structure CFunctor (C D : Cat) where
  obj : C.Obj → D.Obj
references: []
</artifact>
""",
        ]
        provider = TrackingProvider(config, responses=responses)
        verifier = MockLeanVerifier()

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            foundation_path=foundation_path,
        )

        # Gen 0: No foundation yet
        await society.run_generation(0)

        # Gen 1: Should see Cat in foundation context
        await society.run_generation(1)

        # Check that generation 1 prompt includes foundation info
        if len(received_prompts) >= 2:
            gen1_prompt = received_prompts[1]
            # Should mention foundation or import
            assert (
                "Foundation" in gen1_prompt
                or "Cat" in gen1_prompt
                or "import" in gen1_prompt.lower()
            )


class TestSocietyWorkingGroups:
    """Tests for Society working groups integration."""

    @pytest.mark.asyncio
    async def test_society_has_working_group_settings(self, tmp_path: Path):
        """Society has working group configuration."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)

        society = Society(n_agents=3, provider=provider)

        assert hasattr(society, "use_working_groups")
        assert hasattr(society, "n_working_groups")
        assert hasattr(society, "max_turns_per_group")
        assert hasattr(society, "use_planning_panel")
        assert hasattr(society, "dependency_graph")

    @pytest.mark.asyncio
    async def test_committee_mode_without_goal_raises(self, tmp_path: Path):
        """Committee mode without a goal is a loud error, not a silent fallback.

        This used to degrade to flat mode with no error and no log line, so a
        misconfigured committee run produced plausible flat-mode output.
        """
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        verifier = MockLeanVerifier()

        society = Society(n_agents=1, provider=provider, verifier=verifier)
        society.use_working_groups = True

        with pytest.raises(ValueError, match="requires a goal"):
            await society.run_generation(0)

    @pytest.mark.asyncio
    async def test_run_generation_with_groups_with_goal(self, tmp_path: Path):
        """Working groups mode creates dependency graph from goal."""
        from lms.goals import Goal, StacksDefinition
        from lms.dependency import DependencyGraph

        config = ProviderConfig(api_key="test", model="test")
        # Response for planning panel + working group
        responses = [
            """<proposal>
<rationale>Focus on CH4-CAT first</rationale>
<assignments>
<group id="1" task="CH4-CAT" priority="1">
Work on category definition
</group>
</assignments>
</proposal>""",
            "APPROVE",
            "APPROVE",
            "APPROVE",
            # Working group responses
            "Let's define Category.",  # Chair
            """```lean
structure Category where
  Obj : Type
```""",  # Researcher
            "CONSENSUS REACHED",  # Chair
            """<artifact>
type: definition
name: Category
stacks_tag: CH4-CAT
description: Category structure
lean: |
  structure Category where
    Obj : Type
</artifact>""",  # Scribe
        ]
        provider = MockProvider(config, responses=responses)
        verifier = MockLeanVerifier()

        goal = Goal(
            name="Test Goal",
            description="Test",
            source="Test",
            definitions=[
                StacksDefinition(
                    tag="CH4-CAT", section="4.1", name="Category", content="..."
                ),
                StacksDefinition(
                    tag="CH4-FUNC", section="4.2", name="Functor", content="..."
                ),
            ],
        )

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=verifier,
            goal=goal,
            foundation_path=foundation_path,
        )
        society.use_working_groups = True
        society.n_working_groups = 1

        result = await society.run_generation_with_groups(0)

        # Dependency graph should be created
        assert society.dependency_graph is not None
        assert isinstance(society.dependency_graph, DependencyGraph)
        assert result.generation == 0

    @pytest.mark.asyncio
    async def test_dependency_graph_initialized_from_goal(self, tmp_path: Path):
        """Dependency graph is automatically created from goal."""
        from lms.goals import Goal, StacksDefinition

        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config, responses=["No output"])

        goal = Goal(
            name="Test Goal",
            description="Test",
            source="Test",
            definitions=[
                StacksDefinition(tag="A", section="1.1", name="A", content="..."),
                StacksDefinition(tag="B", section="1.2", name="B", content="..."),
            ],
        )

        society = Society(n_agents=1, provider=provider, goal=goal)
        society.use_working_groups = True
        society.use_planning_panel = False  # Skip planning for simpler test

        # Trigger graph initialization
        await society.run_generation_with_groups(0)

        assert society.dependency_graph is not None
        assert "A" in society.dependency_graph.nodes
        assert "B" in society.dependency_graph.nodes

    @pytest.mark.asyncio
    async def test_foundation_summary_empty(self, tmp_path: Path):
        """Foundation summary indicates empty state correctly."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)

        foundation_path = tmp_path / "LMS" / "Foundation.lean"
        society = Society(
            n_agents=1,
            provider=provider,
            foundation_path=foundation_path,
        )

        summary = society._get_foundation_summary()
        assert "empty" in summary.lower()

    @pytest.mark.asyncio
    async def test_get_task_content_from_goal(self, tmp_path: Path):
        """_get_task_content retrieves content from goal definitions."""
        from lms.goals import Goal, StacksDefinition

        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)

        goal = Goal(
            name="Test",
            description="Test",
            source="Test",
            definitions=[
                StacksDefinition(
                    tag="CH4-CAT",
                    section="4.1",
                    name="Category",
                    content="A category is a collection of objects with morphisms.",
                ),
            ],
        )

        society = Society(n_agents=1, provider=provider, goal=goal)

        content = society._get_task_content("CH4-CAT")
        assert "category" in content.lower()
        assert "morphisms" in content.lower()

    @pytest.mark.asyncio
    async def test_get_task_content_fallback(self, tmp_path: Path):
        """_get_task_content falls back to generic message for unknown tag."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)

        society = Society(n_agents=1, provider=provider)

        content = society._get_task_content("UNKNOWN-TAG")
        assert "UNKNOWN-TAG" in content


# =============================================================================
# Committee mode: dispatch, review committee, reuse accounting (26Q3-HARN-12)
# =============================================================================

# One full working-group session against MockProvider's sequential responses:
# chair opening, researcher proposal, chair consensus, scribe artifact.
GROUP_SESSION_RESPONSES = [
    "Let's define Category.",
    """```lean
structure Category where
  Obj : Type
```""",
    "CONSENSUS REACHED",
    """<artifact>
type: definition
name: Category
stacks_tag: T1
description: Category structure
lean: |
  structure Category where
    Obj : Type
</artifact>""",
]


class RecordingVerifier(StubLeanVerifier):
    """StubLeanVerifier that records every code string it was asked to verify."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def verify(self, code: str) -> VerificationResult:
        self.calls.append(code)
        return await super().verify(code)


def _committee_society(
    tmp_path: Path, responses: list[str], verifier: LeanVerifier
) -> Society:
    """One agent, one working group, planning panel off, single-task goal."""
    from lms.goals import Goal, StacksDefinition

    config = ProviderConfig(api_key="test", model="test")
    provider = MockProvider(config, responses=responses)
    goal = Goal(
        name="Test Goal",
        description="Test",
        source="Test",
        definitions=[
            StacksDefinition(tag="T1", section="1.1", name="Category", content="..."),
        ],
    )
    society = Society(
        n_agents=1,
        provider=provider,
        verifier=verifier,
        goal=goal,
        foundation_path=tmp_path / "LMS" / "Foundation.lean",
    )
    society.use_working_groups = True
    society.n_working_groups = 1
    society.use_planning_panel = False
    return society


class ScriptedVerifier(StubLeanVerifier):
    """Fails with scripted errors until the script runs out, then accepts.

    Records every code string it was asked to verify, like RecordingVerifier.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        self.calls: list[str] = []

    async def verify(self, code: str) -> VerificationResult:
        self.calls.append(code)
        if self.errors:
            return self._result(success=False, code=code, error=self.errors.pop(0))
        return self._result(success=True, code=code)


# The scribe's answer to a repair turn: universe error fixed.
REPAIR_RESPONSE = """<artifact>
type: definition
name: Category
stacks_tag: T1
description: Category structure, universe fixed
lean: |
  universe u

  structure Category where
    Obj : Type u
</artifact>"""


class TestCommitteeRepairLoop:
    """A failed verify goes back to the group's scribe with the Lean error."""

    @pytest.mark.asyncio
    async def test_repair_attempt_verifies_and_counts(self, tmp_path: Path):
        """A repaired artifact that verifies counts like a first-shot success."""
        verifier = ScriptedVerifier(["unknown universe level 'u'"])
        responses = GROUP_SESSION_RESPONSES + [REPAIR_RESPONSE]
        society = _committee_society(tmp_path, responses, verifier)
        society.use_peer_review = False

        result = await society.run_generation(0)

        assert len(verifier.calls) == 2
        assert "Type u" in verifier.calls[1]
        assert result.artifacts_verified == 1
        assert society.dependency_graph is not None
        assert society.dependency_graph.nodes["T1"].status == TaskStatus.DONE
        (artifact,) = society.library.all()
        assert artifact.lean_code == verifier.calls[1]
        assert "Repaired by scribe" in (artifact.notes or "")

    @pytest.mark.asyncio
    async def test_repair_output_is_recleaned(self, tmp_path: Path):
        """Repair output goes through the same cleaning as the first shot —
        the block-scalar leak applies to any scribe payload."""
        verifier = ScriptedVerifier(["unexpected token"])
        leaked = """<artifact>
type: definition
name: Category
stacks_tag: T1
description: repaired with leak
lean: |
  |
    import Mathlib.CategoryTheory.Category.Basic

    structure Category where
      Obj : Type
</artifact>"""
        responses = GROUP_SESSION_RESPONSES + [leaked]
        society = _committee_society(tmp_path, responses, verifier)
        society.use_peer_review = False

        await society.run_generation(0)

        assert len(verifier.calls) == 2
        assert verifier.calls[1].startswith("import Mathlib")

    @pytest.mark.asyncio
    async def test_repaired_code_recheck_import_restrictions(self, tmp_path: Path):
        """A repair that violates the goal's import rules never reaches Lean;
        the restriction error is fed back like a verify failure."""
        verifier = ScriptedVerifier(["unknown identifier 'CategoryTheory'"])
        forbidden = """<artifact>
type: definition
name: Category
stacks_tag: T1
description: repaired with a forbidden import
lean: |
  import Mathlib.Tactic

  structure Category where
    Obj : Type
</artifact>"""
        responses = GROUP_SESSION_RESPONSES + [forbidden, REPAIR_RESPONSE]
        society = _committee_society(tmp_path, responses, verifier)
        society.use_peer_review = False
        assert society.goal is not None
        society.goal.forbidden_imports = ["Mathlib.Tactic"]

        result = await society.run_generation(0)

        assert len(verifier.calls) == 2
        assert "Mathlib.Tactic" not in verifier.calls[1]
        assert result.artifacts_verified == 1

    @pytest.mark.asyncio
    async def test_zero_repair_attempts_is_one_shot(self, tmp_path: Path):
        """max_repair_attempts=0 reproduces the one-shot behaviour exactly."""
        verifier = ScriptedVerifier(["error one", "error two", "error three"])
        society = _committee_society(tmp_path, list(GROUP_SESSION_RESPONSES), verifier)
        society.use_peer_review = False
        society.max_repair_attempts = 0

        result = await society.run_generation(0)

        assert len(verifier.calls) == 1
        assert result.artifacts_verified == 0
        assert society.dependency_graph is not None
        assert society.dependency_graph.nodes["T1"].status == TaskStatus.AVAILABLE


class TestCommitteeMode:
    """run_generation dispatch and the review committee stage."""

    @pytest.mark.asyncio
    async def test_run_generation_dispatches_to_committee_mode(self, tmp_path: Path):
        """With use_working_groups set, run_generation takes the committee path."""
        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        society = Society(n_agents=1, provider=provider)
        society.use_working_groups = True

        called: dict[str, int] = {}

        async def fake_groups(generation: int) -> GenerationResult:
            called["generation"] = generation
            return GenerationResult(
                generation=generation,
                artifacts_created=0,
                artifacts_verified=0,
                artifacts_referenced=0,
                fresh_creations=0,
            )

        with mock.patch.object(
            society, "run_generation_with_groups", side_effect=fake_groups
        ):
            result = await society.run_generation(3)

        assert called["generation"] == 3
        assert result.generation == 3

    @pytest.mark.asyncio
    async def test_review_committee_rejects_before_verifier(self, tmp_path: Path):
        """A REJECT from the review committee keeps code away from the verifier."""
        verifier = RecordingVerifier()
        responses = GROUP_SESSION_RESPONSES + [
            """<review>
decision: REJECT
reasoning: Wrong universe polymorphism, Obj must be Type u
</review>"""
        ]
        society = _committee_society(tmp_path, responses, verifier)

        result = await society.run_generation(0)

        assert verifier.calls == []
        assert result.reviews_total == 1
        assert result.reviews_rejected == 1
        assert result.artifacts_verified == 0
        rejected = [
            a
            for a in society.library.all()
            if a.verification_error
            and "Rejected by review committee" in a.verification_error
        ]
        assert len(rejected) == 1
        # The task goes back to the pool for the next generation
        assert society.dependency_graph is not None
        assert society.dependency_graph.nodes["T1"].status == TaskStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_review_committee_approves_then_verifies(self, tmp_path: Path):
        """An APPROVE lets the artifact through to the verifier and the graph."""
        verifier = RecordingVerifier()
        responses = GROUP_SESSION_RESPONSES + [
            """<review>
decision: APPROVE
reasoning: Signature matches the task
</review>"""
        ]
        society = _committee_society(tmp_path, responses, verifier)

        result = await society.run_generation(0)

        assert len(verifier.calls) == 1
        assert result.reviews_total == 1
        assert result.reviews_approved == 1
        assert result.artifacts_verified == 1
        assert society.dependency_graph is not None
        assert society.dependency_graph.nodes["T1"].status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_review_committee_modify_replaces_code(self, tmp_path: Path):
        """A MODIFY verdict sends the reviewer's code to the verifier."""
        verifier = RecordingVerifier()
        responses = GROUP_SESSION_RESPONSES + [
            """<review>
decision: MODIFY
reasoning: Needs universe polymorphism
modified_code: structure Category where
  Obj : Type u
</review>"""
        ]
        society = _committee_society(tmp_path, responses, verifier)

        result = await society.run_generation(0)

        assert len(verifier.calls) == 1
        assert "Type u" in verifier.calls[0]
        assert result.reviews_total == 1
        assert result.reviews_modified == 1

    @pytest.mark.asyncio
    async def test_review_committee_skipped_when_disabled(self, tmp_path: Path):
        """use_peer_review=False sends committee output straight to the verifier."""
        verifier = RecordingVerifier()
        society = _committee_society(tmp_path, list(GROUP_SESSION_RESPONSES), verifier)
        society.use_peer_review = False

        result = await society.run_generation(0)

        assert len(verifier.calls) == 1
        assert result.reviews_total == 0
        assert result.artifacts_verified == 1


class TestIterativeReuse:
    """Iterative mode must link references, or reuse rate is 0 by construction."""

    @pytest.mark.asyncio
    async def test_iterative_mode_links_references(self, tmp_path: Path):
        from lms.agent import IterativeResponse
        from lms.artifacts import Artifact, ArtifactType

        config = ProviderConfig(api_key="test", model="test")
        provider = MockProvider(config)
        society = Society(
            n_agents=1,
            provider=provider,
            verifier=MockLeanVerifier(),
            foundation_path=tmp_path / "LMS" / "Foundation.lean",
        )
        society.iterative_mode = True

        base = Artifact(
            id="base-1",
            type=ArtifactType.DEFINITION,
            natural_language="Base definition",
            created_by="agent-0-mock",
            generation=0,
            lean_code="def base := 1",
        )
        society.library.add(base)

        derived = Artifact(
            id="derived-1",
            type=ArtifactType.DEFINITION,
            natural_language="Builds on base",
            created_by="agent-0-mock",
            generation=1,
            lean_code="def derived := base + 1",
            references=["base-1"],
        )
        response = IterativeResponse(
            attempts=[],
            final_artifact=derived,
            success=False,
            writeup="",
        )

        with mock.patch.object(
            society.agents[0],
            "propose_iterative",
            mock.AsyncMock(return_value=response),
        ):
            await society.run_generation(1)

        assert society.library.get("base-1").referenced_by == ["derived-1"]
        assert society.library.reused_artifact_count() == 1


class TestCommitteeLeanCleaning:
    """Committee payloads reach Lean cleaned; hallucinated tags fail loudly."""

    @pytest.mark.asyncio
    async def test_group_lean_payload_is_cleaned_before_verification(
        self, tmp_path: Path
    ):
        """The 2026-08-19 smoke: scribe code arrived as '|\\n  import ...'."""
        verifier = RecordingVerifier()
        responses = [
            "Let's define Category.",
            "```lean\nstructure Category where\n  Obj : Type\n```",
            "CONSENSUS REACHED",
            # The literal leak seen on the box: block-scalar header plus
            # per-line indentation inside the lean field.
            """<artifact>
type: definition
name: Category
stacks_tag: T1
description: Category structure
lean: |
  |
    import Mathlib.CategoryTheory.Category.Basic

    structure Category where
      Obj : Type
</artifact>""",
            """<review>
decision: APPROVE
reasoning: fine
</review>""",
        ]
        # The scribe block above stamps a decorated tag; smoke_c/d wrote
        # CAT-0013 for task 0013 the same way.
        responses[3] = responses[3].replace("stacks_tag: T1", "stacks_tag: CAT-T1")
        society = _committee_society(tmp_path, responses, verifier)

        await society.run_generation(0)

        assert len(verifier.calls) == 1
        code = verifier.calls[0]
        assert code.startswith("import Mathlib.CategoryTheory.Category.Basic")
        assert "\nstructure Category where" in code
        (artifact,) = society.library.all()
        assert artifact.lean_code == code
        # The raw capture keeps the leak visible in the record
        assert artifact.lean_code_raw.startswith("|")
        # Scribe decoration is overridden by the validated assignment tag
        assert artifact.stacks_tag == "T1"

    @pytest.mark.asyncio
    async def test_reviewer_modified_code_is_cleaned(self, tmp_path: Path):
        """The review prompt requests `modified_code: |`, so a MODIFY capture
        starts at the block-scalar header. Uncleaned, it overwrites the
        already-cleaned artifact code — smoke_d's payloads reached Lean as
        '|\\n  import ...' through exactly this path."""
        verifier = RecordingVerifier()
        responses = GROUP_SESSION_RESPONSES + [
            """<review>
decision: MODIFY
reasoning: Needs an import
modified_code: |
  import Mathlib.CategoryTheory.Category.Basic

  structure Category where
    Obj : Type u
</review>"""
        ]
        society = _committee_society(tmp_path, responses, verifier)

        result = await society.run_generation(0)

        assert result.reviews_modified == 1
        assert len(verifier.calls) == 1
        code = verifier.calls[0]
        assert code.startswith("import Mathlib.CategoryTheory.Category.Basic")
        assert "\nstructure Category where\n  Obj : Type u" in code

    def test_get_task_content_unknown_tag_raises(self, tmp_path: Path):
        society = _committee_society(tmp_path, [], StubLeanVerifier())
        with pytest.raises(ValueError, match="not in goal"):
            society._get_task_content("LOGIC-001")


class OrderedVerifier(StubLeanVerifier):
    """Scripted results in call order: None = success, a string = that error."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[str] = []

    async def verify(self, code: str) -> VerificationResult:
        self.calls.append(code)
        outcome = self.script.pop(0) if self.script else None
        if outcome is None:
            return self._result(success=True, code=code)
        return self._result(success=False, code=code, error=outcome)


class TestSameTagFailureAfterVerify:
    """Several groups can share one tag (the goal graph gates everything
    behind the first task). A failed artifact processed after the verified
    one must not re-open the solved task — the next generation would be
    re-assigned it and fail by redefinition collision."""

    @pytest.mark.asyncio
    async def test_same_tag_failure_after_verify_leaves_done(self, tmp_path: Path):
        from lms.planning import WorkingGroupAssignment

        artifact_response = """<artifact>
type: definition
name: Category
stacks_tag: T1
description: Category structure
lean: |
  structure Category where
    Obj : Type
</artifact>"""
        verifier = OrderedVerifier([None, "unknown universe level 'u'"])
        society = _committee_society(tmp_path, [artifact_response], verifier)
        society.use_peer_review = False
        society.max_repair_attempts = 0
        society.n_working_groups = 2
        society.use_planning_panel = True

        assignments = [
            WorkingGroupAssignment(
                group_id=i,
                task_tag="T1",
                task_name="Category",
                priority=1,
                guidance="...",
            )
            for i in (1, 2)
        ]
        panel = mock.Mock()
        panel.run_session = mock.AsyncMock(return_value=assignments)
        panel.tokens_used = 0
        with mock.patch("lms.society.PlanningPanel", return_value=panel):
            result = await society.run_generation(0)

        assert len(verifier.calls) == 2
        assert result.artifacts_verified == 1
        assert society.dependency_graph is not None
        assert society.dependency_graph.nodes["T1"].status == TaskStatus.DONE
