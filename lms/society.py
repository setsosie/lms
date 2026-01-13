"""LMS Society - orchestrates agents across generations."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

from lms.agent import Agent, AgentResponse, ReviewResult, IterativeResponse
from lms.artifacts import ArtifactLibrary, ReviewQueue, PendingReview
from lms.foundation import FoundationFile
from lms.lean.interface import LeanVerifier
from lms.providers.base import BaseLLMProvider
from lms.textbook import Textbook
from lms.traces import TraceStore, ConversationLog, ReasoningTrace

if TYPE_CHECKING:
    from lms.goals import Goal


class BudgetExceeded(Exception):
    """Raised when token budget is exceeded."""
    pass


@dataclass
class GenerationResult:
    """Results from running a single generation.

    Attributes:
        generation: Generation number
        artifacts_created: Total artifacts proposed this generation
        artifacts_verified: Artifacts that passed LEAN verification
        artifacts_referenced: Artifacts that reference existing work
        fresh_creations: Artifacts created without references
        tokens_used: Total tokens used this generation
        reviews_total: Total peer reviews performed
        reviews_approved: Artifacts approved by peer review
        reviews_rejected: Artifacts rejected by peer review
        reviews_modified: Artifacts modified during peer review
    """

    generation: int
    artifacts_created: int
    artifacts_verified: int
    artifacts_referenced: int
    fresh_creations: int
    tokens_used: int = 0
    # Peer review stats
    reviews_total: int = 0
    reviews_approved: int = 0
    reviews_rejected: int = 0
    reviews_modified: int = 0


class Society:
    """Orchestrates a society of LLM agents doing mathematics.

    The society maintains a shared artifact library and runs agents
    through multiple generations, tracking knowledge accumulation
    and cultural transmission. Like a correspondence network of
    18th century mathematicians.
    """

    def __init__(
        self,
        n_agents: int,
        provider: BaseLLMProvider | None = None,
        verifier: LeanVerifier | None = None,
        max_tokens: int | None = None,
        providers: list[BaseLLMProvider] | None = None,
        goal: Goal | None = None,
        foundation_path: Path | None = None,
    ) -> None:
        """Initialize the society.

        Args:
            n_agents: Number of agents in the society
            provider: Single LLM provider for all agents (use this OR providers)
            verifier: LEAN verifier for checking proofs
            max_tokens: Maximum tokens to use (None = unlimited)
            providers: List of providers, one per agent (for heterogeneous societies)
            goal: Optional goal to work towards (enables goal-directed mode)
            foundation_path: Path to Foundation.lean for accumulated definitions
        """
        self.n_agents = n_agents
        self.verifier = verifier
        self.library = ArtifactLibrary()
        self.results: list[GenerationResult] = []
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.current_generation = 0
        self.goal = goal
        self.tokens_by_agent: dict[str, int] = {}  # Track per-agent token usage
        self.artifacts_by_agent: dict[str, dict[str, int]] = {}  # Track created/verified/referenced per agent
        self.reviews_by_agent: dict[str, dict[str, int]] = {}  # Track reviews given per agent
        self.use_peer_review: bool = True  # Enable/disable peer review phase
        self.textbook = Textbook()  # Accumulated wisdom
        self.iterative_mode: bool = False  # Enable iterative proposals (5 attempts per agent)
        self.max_attempts: int = 5  # Max attempts per agent in iterative mode
        self.trace_store = TraceStore()  # Full conversation logs and reasoning traces

        # Foundation file for accumulated verified definitions
        if foundation_path:
            self.foundation = FoundationFile(foundation_path)
        else:
            # Default path in lean project
            self.foundation = FoundationFile(Path("lean/LMS/Foundation.lean"))

        # Handle provider(s)
        if providers is not None:
            if len(providers) != n_agents:
                raise ValueError(f"providers list length ({len(providers)}) must match n_agents ({n_agents})")
            self.providers = providers
            self.provider = providers[0]  # Keep for backwards compatibility
        elif provider is not None:
            self.provider = provider
            self.providers = [provider] * n_agents
        else:
            raise ValueError("Must provide either 'provider' or 'providers'")

        # Create agents with their respective providers
        self.agents = [
            Agent(
                id=f"agent-{i}-{self.providers[i].name}",
                provider=self.providers[i],
                generation=0,
            )
            for i in range(n_agents)
        ]

    async def run_generation(self, generation: int) -> GenerationResult:
        """Run a single generation of the society using three phases.

        Phase 1 (PROPOSE): All agents propose artifacts in parallel
        Phase 2 (REVIEW): Each agent reviews another agent's work in parallel
        Phase 3 (VERIFY): Approved artifacts are verified with LEAN in parallel

        Args:
            generation: Current generation number

        Returns:
            GenerationResult with metrics for this generation

        Raises:
            BudgetExceeded: If token budget would be exceeded
        """
        # Check budget before starting
        if self.max_tokens and self.total_tokens_used >= self.max_tokens:
            raise BudgetExceeded(
                f"Token budget exceeded: {self.total_tokens_used}/{self.max_tokens}"
            )

        # Update agent generations
        for agent in self.agents:
            agent.generation = generation

        # Initialize counters
        artifacts_created = 0
        artifacts_verified = 0
        artifacts_referenced = 0
        fresh_creations = 0
        generation_tokens = 0
        reviews_total = 0
        reviews_approved = 0
        reviews_rejected = 0
        reviews_modified = 0

        review_queue = ReviewQueue()

        # ===== ITERATIVE MODE: Each agent gets multiple verification attempts =====
        if self.iterative_mode and self.verifier:
            return await self._run_generation_iterative(
                generation, artifacts_created, artifacts_verified,
                artifacts_referenced, fresh_creations, generation_tokens
            )

        # ===== PHASE 1: PROPOSE (parallel) =====
        propose_tasks = [
            agent.propose(self.library, goal=self.goal, textbook=self.textbook, foundation=self.foundation)
            for agent in self.agents
        ]
        responses: list[AgentResponse] = await asyncio.gather(*propose_tasks)

        # Process propose results and queue for review
        for agent, response in zip(self.agents, responses):
            # Track tokens
            if response.tokens_used:
                tokens = response.tokens_used.total_tokens
                generation_tokens += tokens
                self.total_tokens_used += tokens
                if agent.id not in self.tokens_by_agent:
                    self.tokens_by_agent[agent.id] = 0
                self.tokens_by_agent[agent.id] += tokens

            # Initialize per-agent stats
            if agent.id not in self.artifacts_by_agent:
                self.artifacts_by_agent[agent.id] = {"created": 0, "verified": 0, "referenced": 0}

            # Queue artifacts for review
            for artifact in response.proposed_artifacts:
                artifacts_created += 1
                self.artifacts_by_agent[agent.id]["created"] += 1

                if artifact.references:
                    artifacts_referenced += 1
                    self.artifacts_by_agent[agent.id]["referenced"] += 1
                else:
                    fresh_creations += 1

                # Only queue artifacts with LEAN code for review
                if artifact.lean_code and self.verifier:
                    review_queue.add(artifact)
                else:
                    # No code to verify, add directly to library
                    self.library.add(artifact)

        # ===== PHASE 2: REVIEW (parallel) =====
        if self.use_peer_review and review_queue.pending:
            # Shuffle pending items for random assignment
            random.shuffle(review_queue.pending)

            # Assign reviews - each agent gets items from other agents
            review_tasks = []
            review_assignments: list[tuple[Agent, PendingReview]] = []

            for agent in self.agents:
                # Initialize review stats
                if agent.id not in self.reviews_by_agent:
                    self.reviews_by_agent[agent.id] = {"given": 0, "approved": 0, "rejected": 0, "modified": 0}

                # Get an artifact to review (not own work)
                pending = review_queue.get_for_review(exclude_agent=agent.id)
                if pending:
                    review_assignments.append((agent, pending))
                    review_tasks.append(agent.review(pending))

            # Run reviews in parallel
            if review_tasks:
                review_results: list[ReviewResult] = await asyncio.gather(*review_tasks)

                # Process review results
                for (agent, pending), result in zip(review_assignments, review_results):
                    reviews_total += 1
                    self.reviews_by_agent[agent.id]["given"] += 1

                    # Track tokens from review
                    if result.tokens_used:
                        tokens = result.tokens_used.total_tokens
                        generation_tokens += tokens
                        self.total_tokens_used += tokens
                        self.tokens_by_agent[agent.id] += tokens

                    # Mark as reviewed
                    approved = result.decision in ("APPROVE", "MODIFY")
                    review_queue.mark_reviewed(
                        pending=pending,
                        reviewer=agent.id,
                        approved=approved,
                        notes=result.reasoning,
                        modified_code=result.modified_code,
                        tokens_used=result.tokens_used.total_tokens if result.tokens_used else 0,
                    )

                    if result.decision == "APPROVE":
                        reviews_approved += 1
                        self.reviews_by_agent[agent.id]["approved"] += 1
                    elif result.decision == "MODIFY":
                        reviews_modified += 1
                        self.reviews_by_agent[agent.id]["modified"] += 1
                    else:
                        reviews_rejected += 1
                        self.reviews_by_agent[agent.id]["rejected"] += 1

            # Any remaining pending items (couldn't be assigned) go straight to verify
            for pending in review_queue.pending:
                pending.review_approved = True  # Auto-approve unreviewed items
                review_queue.reviewed.append(pending)
                review_queue.pending = []

        else:
            # No peer review - all items auto-approved
            for pending in review_queue.pending:
                pending.review_approved = True
                review_queue.reviewed.append(pending)
            review_queue.pending = []

        # ===== PHASE 3: VERIFY (parallel) =====
        approved_items = review_queue.get_approved()

        if approved_items and self.verifier:
            # Get code to verify (possibly modified by reviewer)
            # First, check import restrictions if goal has them
            verify_tasks = []
            items_to_verify = []

            for pending in approved_items:
                code = pending.get_code_for_verification()
                if not code:
                    continue

                # Check import restrictions before running expensive verification
                if self.goal and (self.goal.allowed_imports or self.goal.forbidden_imports):
                    valid, error = self.goal.validate_code(code)
                    if not valid:
                        # Reject immediately - forbidden import
                        pending.artifact.verified = False
                        pending.artifact.verification_error = f"Import restriction violated: {error}"
                        self.library.add(pending.artifact)
                        continue

                verify_tasks.append(self.verifier.verify(code))
                items_to_verify.append(pending)

            # Replace approved_items with only those that passed import check
            approved_items = items_to_verify

            # Run verifications in parallel
            verify_results = await asyncio.gather(*verify_tasks)

            # Process verification results
            for pending, result in zip(approved_items, verify_results):
                artifact = pending.artifact

                # If code was modified by reviewer, update the artifact
                if pending.modified_code:
                    artifact.lean_code = pending.modified_code
                    artifact.notes = (artifact.notes or "") + f"\n[Modified by {pending.reviewed_by}]"

                artifact.verified = result.success
                if result.success:
                    artifacts_verified += 1
                    creator_id = artifact.created_by
                    if creator_id in self.artifacts_by_agent:
                        self.artifacts_by_agent[creator_id]["verified"] += 1

                    # Add to foundation for future generations to import
                    try:
                        self.foundation.add_artifact(artifact)
                    except ValueError:
                        pass  # Skip if artifact has issues

                    # Add successful insights to textbook
                    if artifact.notes:
                        topics = [artifact.stacks_tag] if artifact.stacks_tag else []
                        topics.append(artifact.type.value)
                        # Use artifact name as title basis
                        title = f"[VERIFIED] {artifact.natural_language[:60]}"
                        self.textbook.add(
                            content=artifact.notes,  # Full notes
                            author=artifact.created_by,
                            generation=artifact.generation,
                            topics=topics,
                            title=title,
                            entry_type="success",
                        )
                else:
                    artifact.verification_error = result.error
                    # Also add failed attempts to textbook - learning from failures
                    if artifact.notes and result.error:
                        topics = [artifact.stacks_tag] if artifact.stacks_tag else []
                        topics.append("error")
                        # Extract error type for title
                        error_summary = result.error.split('\n')[0][:50] if result.error else "Unknown"
                        title = f"[ERROR] {error_summary}"
                        self.textbook.add(
                            content=f"{artifact.notes}\n\n---\nError: {result.error}",
                            author=artifact.created_by,
                            generation=artifact.generation,
                            topics=topics,
                            title=title,
                            entry_type="error",
                        )

                # Add to library
                self.library.add(artifact)

                # Update reference tracking
                for ref_id in artifact.references:
                    if ref_id in self.library:
                        self.library.add_reference(artifact.id, ref_id)

                # Track goal progress
                if self.goal and artifact.stacks_tag and artifact.verified:
                    self.goal.mark_formalized(artifact.stacks_tag, artifact.id)

        # Add rejected items to library (unverified, with rejection reason)
        for pending in review_queue.get_rejected():
            artifact = pending.artifact
            artifact.verification_error = f"Rejected by {pending.reviewed_by}: {pending.review_notes}"
            self.library.add(artifact)

        self.current_generation = generation + 1

        result = GenerationResult(
            generation=generation,
            artifacts_created=artifacts_created,
            artifacts_verified=artifacts_verified,
            artifacts_referenced=artifacts_referenced,
            fresh_creations=fresh_creations,
            tokens_used=generation_tokens,
            reviews_total=reviews_total,
            reviews_approved=reviews_approved,
            reviews_rejected=reviews_rejected,
            reviews_modified=reviews_modified,
        )
        self.results.append(result)
        return result

    async def _run_generation_iterative(
        self,
        generation: int,
        artifacts_created: int,
        artifacts_verified: int,
        artifacts_referenced: int,
        fresh_creations: int,
        generation_tokens: int,
    ) -> GenerationResult:
        """Run a generation in iterative mode (each agent gets multiple attempts).

        In iterative mode:
        - Each agent calls propose_iterative with the verifier
        - No separate peer review phase (verification happens in the loop)
        - Agents write up their learnings which go to textbook
        """
        # Create verify function for agents
        async def verify_fn(code: str) -> tuple[bool, str | None]:
            # Check import restrictions first
            if self.goal and (self.goal.allowed_imports or self.goal.forbidden_imports):
                valid, error = self.goal.validate_code(code)
                if not valid:
                    return False, f"Import restriction: {error}"
            # Run LEAN verification
            result = await self.verifier.verify(code)
            return result.success, result.error

        # Run all agents in parallel with iterative proposals
        iterative_tasks = [
            agent.propose_iterative(
                library=self.library,
                verify_fn=verify_fn,
                goal=self.goal,
                textbook=self.textbook,
                max_attempts=self.max_attempts,
                foundation=self.foundation,
            )
            for agent in self.agents
        ]
        responses: list[IterativeResponse] = await asyncio.gather(*iterative_tasks)

        # Process results
        for agent, response in zip(self.agents, responses):
            # Track tokens
            generation_tokens += response.total_tokens
            self.total_tokens_used += response.total_tokens
            if agent.id not in self.tokens_by_agent:
                self.tokens_by_agent[agent.id] = 0
            self.tokens_by_agent[agent.id] += response.total_tokens

            # Initialize per-agent stats
            if agent.id not in self.artifacts_by_agent:
                self.artifacts_by_agent[agent.id] = {"created": 0, "verified": 0, "referenced": 0}

            # Count all attempts as "created"
            artifacts_created += len(response.attempts)
            self.artifacts_by_agent[agent.id]["created"] += len(response.attempts)

            # Process final artifact
            if response.final_artifact:
                artifact = response.final_artifact

                if artifact.references:
                    artifacts_referenced += 1
                    self.artifacts_by_agent[agent.id]["referenced"] += 1
                else:
                    fresh_creations += 1

                if artifact.verified:
                    artifacts_verified += 1
                    self.artifacts_by_agent[agent.id]["verified"] += 1

                    # Add to foundation for future generations to import
                    try:
                        self.foundation.add_artifact(artifact)
                    except ValueError:
                        pass  # Skip if artifact has issues

                    # Add successful insights to textbook
                    if artifact.notes:
                        topics = [artifact.stacks_tag] if artifact.stacks_tag else []
                        topics.append(artifact.type.value)
                        title = f"[VERIFIED] {artifact.natural_language[:60]}"
                        self.textbook.add(
                            content=artifact.notes,  # Full notes
                            author=artifact.created_by,
                            generation=artifact.generation,
                            topics=topics,
                            title=title,
                            entry_type="success",
                        )

                    # Track goal progress
                    if self.goal and artifact.stacks_tag:
                        self.goal.mark_formalized(artifact.stacks_tag, artifact.id)

                # Add to library
                self.library.add(artifact)

            # Add writeup to textbook (valuable whether success or failure)
            if response.writeup:
                topics = ["writeup", "iterative"]
                if self.goal:
                    # Add the goal tags that are still pending
                    pending = [d.tag for d in self.goal.definitions if not d.formalized]
                    topics.extend(pending[:2])  # Add up to 2 pending tags

                # Use agent-provided title if available, otherwise generate one
                status = "SUCCESS" if response.success else "FAILED"
                if response.writeup_title:
                    title = f"[{status}] {response.writeup_title}"
                else:
                    # Fallback: extract first sentence or first 80 chars
                    first_line = response.writeup.split('\n')[0].strip()
                    if len(first_line) > 80:
                        first_line = first_line[:77] + "..."
                    title = f"[{status}] {first_line}"

                self.textbook.add(
                    content=response.writeup,  # Full content, no truncation
                    author=agent.id,
                    generation=generation,
                    topics=topics,
                    title=title,
                    entry_type="success" if response.success else "writeup",
                )

        self.current_generation = generation + 1

        result = GenerationResult(
            generation=generation,
            artifacts_created=artifacts_created,
            artifacts_verified=artifacts_verified,
            artifacts_referenced=artifacts_referenced,
            fresh_creations=fresh_creations,
            tokens_used=generation_tokens,
            reviews_total=0,  # No peer review in iterative mode
            reviews_approved=0,
            reviews_rejected=0,
            reviews_modified=0,
        )
        self.results.append(result)
        return result

    async def run(self, n_generations: int) -> list[GenerationResult]:
        """Run the society for multiple generations.

        Args:
            n_generations: Number of generations to run

        Returns:
            List of GenerationResult for each generation
        """
        results = []
        for gen in range(n_generations):
            result = await self.run_generation(gen)
            results.append(result)
        return results

    def save(self, output_dir: Path) -> None:
        """Save society state and results to disk (checkpoint).

        Args:
            output_dir: Directory to save files to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save artifacts (includes correspondence and total_tokens)
        self.library.save(output_dir / "artifacts.json")

        # Save textbook (accumulated wisdom)
        self.textbook.save(output_dir / "textbook.json")

        # Save conversation logs and reasoning traces
        self.trace_store.save(output_dir)

        # Save foundation (accumulated verified definitions)
        # Only save if we have entries and verifier is available
        if len(self.foundation) > 0:
            self.foundation.save()
            # Verify the foundation compiles (if verifier available)
            if self.verifier and self.foundation.path.exists():
                # TODO: Add compilation check in future iteration
                # For now, we trust that individually verified artifacts compile together
                pass

        # Save per-agent stats for accurate resumption
        stats_data = {
            "tokens_by_agent": self.tokens_by_agent,
            "artifacts_by_agent": self.artifacts_by_agent,
            "reviews_by_agent": self.reviews_by_agent,
        }
        (output_dir / "agent_stats.json").write_text(json.dumps(stats_data, indent=2))

        # Save results and checkpoint data
        results_data = {
            "n_agents": self.n_agents,
            "generations": [asdict(r) for r in self.results],
            "checkpoint": {
                "current_generation": self.current_generation,
                "total_tokens_used": self.total_tokens_used,
                "max_tokens": self.max_tokens,
            },
        }
        (output_dir / "results.json").write_text(json.dumps(results_data, indent=2))

    @classmethod
    def load(
        cls,
        output_dir: Path,
        provider: BaseLLMProvider,
        verifier: LeanVerifier,
        goal: "Goal | None" = None,
    ) -> "Society":
        """Load a society from a checkpoint to resume.

        Args:
            output_dir: Directory with saved state
            provider: LLM provider for agents
            verifier: LEAN verifier for checking proofs
            goal: Optional goal (if None, loaded from checkpoint if available)

        Returns:
            Society restored from checkpoint
        """
        # Load results and checkpoint
        results_data = json.loads((output_dir / "results.json").read_text())
        checkpoint = results_data.get("checkpoint", {})

        # Load goal from checkpoint if not provided and file exists
        goal_path = output_dir / "goal.json"
        if goal is None and goal_path.exists():
            from lms.goals import Goal
            goal = Goal.load(goal_path)

        # Create society with same config
        society = cls(
            n_agents=results_data["n_agents"],
            provider=provider,
            verifier=verifier,
            max_tokens=checkpoint.get("max_tokens"),
            goal=goal,
        )

        # Restore state
        society.library = ArtifactLibrary.load(output_dir / "artifacts.json")
        society.results = [
            GenerationResult(**gen_data)
            for gen_data in results_data["generations"]
        ]
        society.current_generation = checkpoint.get("current_generation", 0)
        society.total_tokens_used = checkpoint.get("total_tokens_used", 0)

        # Load textbook if it exists
        textbook_path = output_dir / "textbook.json"
        if textbook_path.exists():
            society.textbook = Textbook.load(textbook_path)

        # Load traces if they exist
        conversations_path = output_dir / "conversations.json"
        if conversations_path.exists():
            society.trace_store = TraceStore.load(output_dir)

        # Load foundation if it exists
        foundation_path = output_dir / "LMS" / "Foundation.lean"
        if foundation_path.with_suffix(".json").exists():
            society.foundation = FoundationFile.load(foundation_path)

        # Load per-agent stats if they exist
        stats_path = output_dir / "agent_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            society.tokens_by_agent = stats.get("tokens_by_agent", {})
            society.artifacts_by_agent = stats.get("artifacts_by_agent", {})
            society.reviews_by_agent = stats.get("reviews_by_agent", {})

        return society

    async def run_from_checkpoint(self, target_generation: int) -> list[GenerationResult]:
        """Continue running from current checkpoint to target generation.

        Args:
            target_generation: Generation to run until

        Returns:
            List of new GenerationResults
        """
        new_results = []
        for gen in range(self.current_generation, target_generation):
            try:
                result = await self.run_generation(gen)
                new_results.append(result)
            except BudgetExceeded:
                print(f"Budget exceeded at generation {gen}")
                break
        return new_results
