"""LMS Society - orchestrates agents across generations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lms.accounting import AttemptRecord, CostLedger, statement_key
from lms.agent import (
    Agent,
    AgentResponse,
    IterativeResponse,
    ReviewResult,
    _clean_lean_code,
)
from lms.artifacts import (
    ArtifactLibrary,
    ReviewQueue,
    PendingReview,
    Artifact,
    ArtifactType,
)
from lms.dependency import DependencyGraph, TaskStatus
from lms.foundation import FoundationFile
from lms.gates import GateOutcome, default_gate_runner
from lms.gates.lean_source import named_declarations
from lms.gates.novelty import apply_novelty_gate, default_novelty_classifier
from lms.lean.interface import (
    LeanVerifier,
    VerificationResult,
    VerificationStatus,
)
from lms.planning import PlanningPanel, create_default_assignments
from lms.providers.base import BaseLLMProvider
from lms.textbook import Textbook
from lms.traces import TraceStore
from lms.working_group import Role, WorkingGroup, WorkingGroupConfig

if TYPE_CHECKING:
    from lms.goals import Goal


#: Shared Lean corpus a Society writes to when no `foundation_path` is given.
DEFAULT_FOUNDATION_PATH = (
    Path(__file__).parent.parent / "lean" / "LMS" / "Foundation.lean"
)

# For `_derive_references`: top-level names a piece of Lean code declares,
# and the comment forms to strip before scanning it for foundation names.
_DECL_NAME_RE = re.compile(
    r"^\s*(?:noncomputable\s+)?(?:private\s+|protected\s+)?"
    r"(?:structure|def|theorem|lemma|class|inductive|abbrev|instance)\s+"
    r"([A-Za-z_][A-Za-z0-9_.']*)",
    re.MULTILINE,
)
_LINE_COMMENT_RE = re.compile(r"--.*$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)


class BudgetExceeded(Exception):
    """Raised when token budget is exceeded."""

    pass


@dataclass
class GenerationResult:
    """Results from running a single generation.

    Attributes:
        generation: Generation number
        artifacts_created: Artifacts that actually entered the library this
            generation. Iterative mode used to count every attempt here,
            inflating the denominator of any per-artifact rate by the retry
            count; attempts are now reported separately
        artifacts_verified: Artifacts that passed LEAN verification
        artifacts_referenced: Artifacts that reference existing work
        fresh_creations: Artifacts created without references
        tokens_used: Total tokens used this generation
        attempts_total: Total verification attempts made this generation
            (iterative mode; 0 in flat mode where there is one proposal pass)
        wall_clock_s: Wall-clock seconds for this generation, including
            foundation persistence
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
    attempts_total: int = 0
    wall_clock_s: float = 0.0
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

    #: Entries rendered into a committee prompt's foundation summary. The
    #: agent-facing context is unbounded on purpose -- an agent needs the whole
    #: API it is being asked to build on -- but a committee prompt carries this
    #: alongside the goal, the roster and the round history, so it needs a
    #: ceiling. Matches the `entries[:10]` the previous renderer used.
    COMMITTEE_SUMMARY_MAX_ENTRIES = 10

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
        # Cost of record: every generation call attributed per statement,
        # including failed attempts. `total_tokens_used` stays as the running
        # aggregate; the ledger is what CVFN reads.
        self.ledger = CostLedger()
        self.current_generation = 0
        self.goal = goal
        self.tokens_by_agent: dict[str, int] = {}  # Track per-agent token usage
        self.artifacts_by_agent: dict[
            str, dict[str, int]
        ] = {}  # Track created/verified/referenced per agent
        self.reviews_by_agent: dict[
            str, dict[str, int]
        ] = {}  # Track reviews given per agent
        self.use_peer_review: bool = True  # Enable/disable peer review phase
        self.textbook = Textbook()  # Accumulated wisdom
        self.iterative_mode: bool = (
            False  # Enable iterative proposals (5 attempts per agent)
        )
        self.max_attempts: int = 5  # Max attempts per agent in iterative mode
        self.trace_store = TraceStore()  # Full conversation logs and reasoning traces
        # T4/T2 machine gates (faithfulness protocol §4), run after
        # verification on every successfully verified artifact.
        self.gate_runner = default_gate_runner(verifier)
        # Gate 4 (novelty). None on mock/projectless runs, where "absent from
        # Mathlib" could not be checked and must not be asserted.
        self.novelty_classifier = default_novelty_classifier(verifier)

        # Working Group settings
        self.use_working_groups: bool = False  # Enable working group mode
        self.n_working_groups: int = 3  # Number of parallel groups
        self.max_turns_per_group: int = 5  # Max turns per group
        # Scribe repair turns after a failed verify. 2, not 5: every error
        # class in committee_real_a was single-retry-fixable, and the group
        # already spent a whole session converging on the approach.
        self.max_repair_attempts: int = 2
        self.use_planning_panel: bool = True  # Use planning panel for allocation
        self.dependency_graph: DependencyGraph | None = None  # Task dependencies

        # Foundation file for accumulated verified definitions
        if foundation_path:
            self.foundation = FoundationFile(foundation_path)
        else:
            # Anchored to the package, not the cwd. As a relative path this
            # resolved to whatever directory the process happened to start in,
            # so a run launched from the repo root wrote into the tracked
            # corpus at lean/LMS/Foundation.lean.
            self.foundation = FoundationFile(DEFAULT_FOUNDATION_PATH)

        # Handle provider(s)
        if providers is not None:
            if len(providers) != n_agents:
                raise ValueError(
                    f"providers list length ({len(providers)}) must match n_agents ({n_agents})"
                )
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
        """Run one generation, then make its verified work importable.

        Dispatches on `use_working_groups`: committee mode (planning panel →
        working groups → review committee → verify) when set, the flat
        propose/review/verify pipeline otherwise. This is the only entry point
        `lms/run.py` calls, so the dispatch has to live here — the flag was
        previously declared and never read, which is how committee mode stayed
        unreachable (26Q3-HARN-12).

        Persisting the foundation is part of finishing a generation, not part
        of checkpointing. `Society.save()` only runs every `checkpoint_interval`
        generations (default 10), so on any shorter run the foundation stayed
        in memory for the whole experiment and every later generation imported
        a `Foundation.lean` that predated the run. Agents cited prior artifacts
        correctly and got an empty module back.

        Args:
            generation: Current generation number

        Returns:
            GenerationResult with metrics for this generation
        """
        gen_start = time.monotonic()
        if self.use_working_groups:
            result = await self.run_generation_with_groups(generation)
        else:
            result = await self._run_generation_impl(generation)

        if result.artifacts_verified > 0:
            await self.persist_foundation()

        result.wall_clock_s = time.monotonic() - gen_start
        return result

    async def persist_foundation(self) -> bool:
        """Write the foundation to disk and recompile it.

        Both halves are required. Writing without rebuilding leaves a stale
        `.olean`, and `import LMS.Foundation` keeps resolving to the previous
        contents -- silently, because an import that succeeds against an old
        module looks exactly like one that succeeds against a current one.

        Returns:
            True if the foundation is on disk and compiled (or if there is
            nothing to persist, or no real project to build against).
        """
        if len(self.foundation) == 0:
            return True

        return await self._write_and_build_foundation()

    async def reset_foundation(self) -> bool:
        """Start a run from an empty foundation, on disk and compiled.

        The foundation cannot live under the experiment directory: `import
        LMS.Foundation` resolves through Lake's `LEAN_PATH`, so the file has to
        sit inside the Lean project. That makes it shared mutable state across
        every run, and it was never reset.

        The consequences were real. The copy committed to the repo is the
        output of a December *mock-verified* run -- its own header reads
        "Verified by: 15x Gemini 3 Flash Preview agents" -- so agents have been
        importing a `Category` that Lean never checked. And each run inherited
        whatever the previous one left behind, which means no run was
        independent of the one before it.

        Writing the empty foundation rather than deleting it keeps `import
        LMS.Foundation` resolving from the first generation; an empty module is
        valid, a missing one is not.

        Returns:
            True if the empty foundation is on disk and compiled.
        """
        return await self._write_and_build_foundation()

    async def _write_and_build_foundation(self) -> bool:
        """Write the foundation and recompile so imports see current contents."""
        self.foundation.save()

        # MockLeanVerifier has no project; nothing to compile against.
        project = getattr(self.verifier, "project", None)
        if project is None:
            return True

        return await project.rebuild_changed_sources()

    async def _run_generation_impl(self, generation: int) -> GenerationResult:
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
                generation,
                artifacts_created,
                artifacts_verified,
                artifacts_referenced,
                fresh_creations,
                generation_tokens,
            )

        # ===== PHASE 1: PROPOSE (parallel) =====
        async def timed_propose(agent: Agent) -> tuple[AgentResponse, float]:
            start = time.monotonic()
            response = await agent.propose(
                self.library,
                goal=self.goal,
                textbook=self.textbook,
                foundation=self.foundation,
            )
            return response, time.monotonic() - start

        propose_tasks = [timed_propose(agent) for agent in self.agents]
        timed_responses: list[tuple[AgentResponse, float]] = await asyncio.gather(
            *propose_tasks
        )

        # Process propose results and queue for review
        for agent, (response, propose_elapsed) in zip(self.agents, timed_responses):
            # Track tokens
            if response.tokens_used:
                tokens = response.tokens_used.total_tokens
                generation_tokens += tokens
                self.total_tokens_used += tokens
                if agent.id not in self.tokens_by_agent:
                    self.tokens_by_agent[agent.id] = 0
                self.tokens_by_agent[agent.id] += tokens

            # Attribute the spend per statement (overhead if nothing parsed)
            self._ledger_propose_response(agent, response, propose_elapsed, generation)

            # Initialize per-agent stats
            if agent.id not in self.artifacts_by_agent:
                self.artifacts_by_agent[agent.id] = {
                    "created": 0,
                    "verified": 0,
                    "referenced": 0,
                }

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
                    self.reviews_by_agent[agent.id] = {
                        "given": 0,
                        "approved": 0,
                        "rejected": 0,
                        "modified": 0,
                    }

                # Get an artifact to review (not own work)
                pending = review_queue.get_for_review(exclude_agent=agent.id)
                if pending:
                    review_assignments.append((agent, pending))
                    review_tasks.append(self._timed_review(agent, pending))

            # Run reviews in parallel
            if review_tasks:
                review_results: list[tuple[ReviewResult, float]] = await asyncio.gather(
                    *review_tasks
                )

                # Process review results
                for (agent, pending), (review_result, review_elapsed) in zip(
                    review_assignments, review_results
                ):
                    reviews_total += 1
                    self.reviews_by_agent[agent.id]["given"] += 1

                    # Track tokens from review
                    if review_result.tokens_used:
                        tokens = review_result.tokens_used.total_tokens
                        generation_tokens += tokens
                        self.total_tokens_used += tokens
                        self.tokens_by_agent[agent.id] += tokens

                    # Review spend belongs to the reviewed statement's cost
                    self.ledger.record(
                        AttemptRecord(
                            statement_key=statement_key(
                                pending.artifact.stacks_tag,
                                pending.artifact.lean_code,
                                pending.artifact.natural_language,
                            ),
                            agent_id=agent.id,
                            generation=generation,
                            prompt_tokens=review_result.tokens_used.input_tokens
                            if review_result.tokens_used
                            else 0,
                            completion_tokens=review_result.tokens_used.output_tokens
                            if review_result.tokens_used
                            else 0,
                            wall_clock_s=review_elapsed,
                            outcome="review",
                        )
                    )

                    # Mark as reviewed
                    approved = review_result.decision in ("APPROVE", "MODIFY")
                    review_queue.mark_reviewed(
                        pending=pending,
                        reviewer=agent.id,
                        approved=approved,
                        notes=review_result.reasoning,
                        modified_code=review_result.modified_code,
                        tokens_used=review_result.tokens_used.total_tokens
                        if review_result.tokens_used
                        else 0,
                    )

                    if review_result.decision == "APPROVE":
                        reviews_approved += 1
                        self.reviews_by_agent[agent.id]["approved"] += 1
                    elif review_result.decision == "MODIFY":
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
                if self.goal and (
                    self.goal.allowed_imports or self.goal.forbidden_imports
                ):
                    valid, error = self.goal.validate_code(code)
                    if not valid:
                        # Reject immediately - forbidden import
                        pending.artifact.status = VerificationStatus.FAILED
                        pending.artifact.verification_error = (
                            f"Import restriction violated: {error}"
                        )
                        self.library.add(pending.artifact)
                        continue

                verify_tasks.append(self._verify_admissible(code))
                items_to_verify.append(pending)

            # Replace approved_items with only those that passed import check
            approved_items = items_to_verify

            # Run verifications in parallel
            verify_results = await asyncio.gather(*verify_tasks)

            # Process verification results
            for pending, verification in zip(approved_items, verify_results):
                artifact = pending.artifact

                # If code was modified by reviewer, update the artifact
                if pending.modified_code:
                    artifact.lean_code = pending.modified_code
                    artifact.notes = (
                        artifact.notes or ""
                    ) + f"\n[Modified by {pending.reviewed_by}]"

                artifact.status = verification.status
                await self._apply_gates(artifact)
                if verification.success:
                    artifacts_verified += 1
                    creator_id = artifact.created_by
                    if creator_id in self.artifacts_by_agent:
                        self.artifacts_by_agent[creator_id]["verified"] += 1

                    # Add to foundation for future generations to import,
                    # unless a gate failed: the next generation imports this
                    # file, so anything the gates rejected must not reach it.
                    if self._blocked_by_gates(artifact):
                        self._note_gate_block(artifact)
                    else:
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
                    artifact.verification_error = verification.error
                    # Also add failed attempts to textbook - learning from failures
                    if artifact.notes and verification.error:
                        topics = [artifact.stacks_tag] if artifact.stacks_tag else []
                        topics.append("error")
                        # Extract error type for title
                        error_summary = verification.error.split("\n")[0][:50]
                        title = f"[ERROR] {error_summary}"
                        self.textbook.add(
                            content=f"{artifact.notes}\n\n---\nError: {verification.error}",
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
            artifact.verification_error = (
                f"Rejected by {pending.reviewed_by}: {pending.review_notes}"
            )
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

    async def _apply_gates(self, artifact: Artifact) -> None:
        """Attach T4/T2 gate verdicts to a successfully verified artifact.

        Gates run post-compile by design (`26Q3-HARN-03`): they judge what the
        verifier accepted, they do not re-verify. Verification status is left
        untouched — "Lean accepted it" and "it passed the gates" are separate
        facts, and collapsing them is how a trivial `example` came to count.
        """
        if not artifact.lean_code:
            return
        if artifact.status not in (
            VerificationStatus.VERIFIED_LEAN,
            VerificationStatus.VERIFIED_HEURISTIC,
        ):
            return
        artifact.gate_results = await self.gate_runner.run(artifact.lean_code)
        await self._apply_novelty_gate(artifact)

    async def _apply_novelty_gate(self, artifact: Artifact) -> None:
        """Stamp the N0/N1 classification onto `artifact` (Gate 4).

        Kept out of `gate_runner` because `NoveltyClassifier.classify` is
        synchronous and talks to Loogle/LeanSearch over the network; running it
        inline would stall every other group's generation for the duration.
        `to_thread` keeps the event loop free.

        A search that raises is a hole in the audit, not a failed artifact:
        the run continues and `novelty_level` stays None, which reads as
        "never classified" rather than as a novelty claim. That distinction is
        the whole point of Gate 4 — `committee_fix_c` shipped 71 artifacts with
        `novelty_level` None and no way to tell that from a checked verdict.
        """
        if self.novelty_classifier is None:
            return
        try:
            await asyncio.to_thread(
                apply_novelty_gate, artifact, self.novelty_classifier
            )
        except Exception as exc:  # noqa: BLE001 - audit hole, not a run-ender
            artifact.novelty_evidence = [f"novelty gate errored: {exc}"[:300]]

    def _blocked_by_gates(self, artifact: Artifact) -> bool:
        """True when some gate positively failed on an otherwise-verified artifact.

        Only `FAILED` blocks. `INCONCLUSIVE` must not: `T2.duplicate` is
        inconclusive by construction whenever no duplicate checker is injected,
        so treating it as blocking would stop every artifact from ever being
        promoted.

        Blocking governs *promotion* — foundation admission and closing a task
        in the dependency graph — never `status`. "Lean accepted it" and "it is
        safe to build on" are separate facts; collapsing them is what let a
        file containing only `-- Your LEAN 4 code here` close the Yoneda
        milestone in `committee_fix_c`.
        """
        return any(r.outcome is GateOutcome.FAILED for r in artifact.gate_results)

    def _note_gate_block(self, artifact: Artifact) -> None:
        """Record on the artifact why it was verified but not promoted.

        `gate_results` already carries the verdicts for `artifacts.json`; this
        puts the reason where a human reading the notes will see it, so a
        verified-but-unpromoted artifact does not look like a bookkeeping bug.
        """
        failed = "; ".join(
            f"{r.gate}: {r.reason}"
            for r in artifact.gate_results
            if r.outcome is GateOutcome.FAILED
        )
        artifact.notes = (
            artifact.notes or ""
        ) + f"\n[Not promoted to foundation — gate failure: {failed}]"

    def _content_violation(self, code: str) -> str | None:
        """Why `code` is not a formalization attempt at all, or None if it is.

        A file of comments compiles with zero errors and zero sorries, so the
        verifier reports success and the artifact is promoted. That is not a
        hypothetical: in `committee_fix_c` three artifacts contained exactly
        the scribe's own prompt scaffold, `-- Your LEAN 4 code here`, and one
        of them closed the Yoneda Lemma milestone.

        Checked before Lean runs, alongside the import restrictions, because a
        submission introducing nothing should never reach the verifier, enter
        the foundation, or consume a Lean invocation.
        """
        if not code or not code.strip():
            return "Empty submission: no Lean code was produced."
        if not named_declarations(code):
            return (
                "Contentless submission: no named declaration. Comments, "
                "imports and `example`s alone are not a formalization — "
                "emit a `theorem`, `def`, `structure` or `class`."
            )
        return None

    async def _verify_admissible(self, code: str) -> VerificationResult:
        """Verify `code`, rejecting inadmissible submissions before Lean runs.

        The single funnel for every verification in the class, so the standard,
        iterative and committee paths cannot drift on what they will accept.
        """
        blocked = self._content_violation(code)
        if blocked is not None:
            return self._rejected(code, blocked)
        if self.verifier is None:
            return self._rejected(code, "No verifier configured")
        return await self.verifier.verify(code)

    def _rejected(self, code: str, error: str) -> VerificationResult:
        """A failure decided before the verifier ran (e.g. import restrictions).

        Still stamped with the configured verifier's provenance so that every
        result in a run reports which machinery the run was using.
        """
        return VerificationResult(
            success=False,
            code=code,
            error=error,
            verifier_kind=self.verifier.verifier_kind if self.verifier else "mock",
            verifier_id=self.verifier.verifier_id if self.verifier else "none",
        )

    def verifier_metadata(self) -> dict[str, Any]:
        """Provenance block recorded with every experiment.

        Without this, a run's artifacts cannot be told apart from any other
        run's after the fact — which is exactly how a mock run came to
        calibrate the three-text roadmap.
        """
        if self.verifier is None:
            return {"kind": None, "id": None, "lean_version": None, "mathlib_rev": None}
        return {
            "kind": self.verifier.verifier_kind,
            "id": self.verifier.verifier_id,
            **self.verifier.toolchain_info(),
        }

    async def _timed_review(
        self, agent: Agent, pending: PendingReview
    ) -> tuple[ReviewResult, float]:
        """Run one review and measure its wall-clock."""
        start = time.monotonic()
        result = await agent.review(pending)
        return result, time.monotonic() - start

    def _ledger_propose_response(
        self,
        agent: Agent,
        response: AgentResponse,
        elapsed: float,
        generation: int,
    ) -> None:
        """Attribute one propose() response's spend across its artifacts.

        Tokens split across the parsed artifacts with the division remainder
        going to the first, so conservation against the society total is
        exact. A response that parses to zero artifacts is pure overhead —
        recorded, never dropped: at low success rates that spend is most of
        the bill.
        """
        usage = response.tokens_used
        prompt = usage.input_tokens if usage else 0
        completion = usage.output_tokens if usage else 0
        artifacts = response.proposed_artifacts
        if not artifacts:
            self.ledger.record_overhead(
                agent_id=agent.id,
                generation=generation,
                prompt_tokens=prompt,
                completion_tokens=completion,
                wall_clock_s=elapsed,
                outcome="no_artifact",
            )
            return
        n = len(artifacts)
        for i, artifact in enumerate(artifacts):
            p, c = prompt // n, completion // n
            if i == 0:
                p += prompt % n
                c += completion % n
            self.ledger.record(
                AttemptRecord(
                    statement_key=statement_key(
                        artifact.stacks_tag,
                        artifact.lean_code,
                        artifact.natural_language,
                    ),
                    agent_id=agent.id,
                    generation=generation,
                    prompt_tokens=p,
                    completion_tokens=c,
                    wall_clock_s=elapsed / n,
                    outcome="proposed",
                )
            )

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
        async def verify_fn(code: str) -> VerificationResult:
            # Check import restrictions first
            if self.goal and (self.goal.allowed_imports or self.goal.forbidden_imports):
                valid, error = self.goal.validate_code(code)
                if not valid:
                    return self._rejected(code, f"Import restriction: {error}")
            # Run LEAN verification
            return await self._verify_admissible(code)

        # Run all agents in parallel with iterative proposals
        iterative_tasks = [
            agent.propose_iterative(
                library=self.library,
                verify_fn=verify_fn,
                goal=self.goal,
                textbook=self.textbook,
                max_attempts=self.max_attempts,
                foundation=self.foundation,
                ledger=self.ledger,
            )
            for agent in self.agents
        ]
        responses: list[IterativeResponse] = await asyncio.gather(*iterative_tasks)
        attempts_total = 0

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
                self.artifacts_by_agent[agent.id] = {
                    "created": 0,
                    "verified": 0,
                    "referenced": 0,
                }

            # Attempts are attempts, not artifacts. Counting every retry as
            # "created" inflated the denominator of any per-artifact rate by
            # the retry count; only what actually enters the library counts.
            attempts_total += len(response.attempts)

            # Process final artifact
            if response.final_artifact:
                artifact = response.final_artifact
                await self._apply_gates(artifact)
                artifacts_created += 1
                self.artifacts_by_agent[agent.id]["created"] += 1

                if artifact.references:
                    artifacts_referenced += 1
                    self.artifacts_by_agent[agent.id]["referenced"] += 1
                else:
                    fresh_creations += 1

                if artifact.verified:
                    artifacts_verified += 1
                    self.artifacts_by_agent[agent.id]["verified"] += 1

                    # Add to foundation for future generations to import,
                    # unless a gate failed (see the standard path).
                    if self._blocked_by_gates(artifact):
                        self._note_gate_block(artifact)
                    else:
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

                # Link references so reuse is measurable. This call existed
                # only in the flat pipeline, so every --iterative run before
                # it reported Reuse Rate 0.0% by construction — including runs
                # where reuse demonstrably happened (shakedown_3x3_d).
                for ref_id in artifact.references:
                    if ref_id in self.library:
                        self.library.add_reference(artifact.id, ref_id)

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
                    first_line = response.writeup.split("\n")[0].strip()
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
            attempts_total=attempts_total,
            reviews_total=0,  # No peer review in iterative mode
            reviews_approved=0,
            reviews_rejected=0,
            reviews_modified=0,
        )
        self.results.append(result)
        return result

    async def run_generation_with_groups(self, generation: int) -> GenerationResult:
        """Run a generation using the Working Group (committee) architecture.

        This method uses:
        1. PlanningPanel to allocate tasks to groups
        2. WorkingGroups for synchronous agent collaboration
        3. A review committee that screens group output before verification
        4. LEAN verification of surviving artifacts
        5. Foundation.lean accumulation of verified code

        Prefer calling `run_generation` with `use_working_groups = True`; that
        wrapper also persists the foundation after a verifying generation.

        Args:
            generation: Current generation number

        Returns:
            GenerationResult with metrics for this generation

        Raises:
            ValueError: If there is no goal and no pre-built dependency graph —
                committee mode has nothing to allocate without one.
        """
        # Check budget before starting
        if self.max_tokens and self.total_tokens_used >= self.max_tokens:
            raise BudgetExceeded(
                f"Token budget exceeded: {self.total_tokens_used}/{self.max_tokens}"
            )

        # Initialize dependency graph from goal if not already done
        if self.dependency_graph is None and self.goal:
            self.dependency_graph = DependencyGraph.from_goal(self.goal)

        if self.dependency_graph is None:
            # This used to silently fall back to flat mode, so a misconfigured
            # committee run produced a plausible-looking flat run with no error
            # and no log line. A committee run without a task graph is a
            # configuration error, not a preference.
            raise ValueError(
                "Committee mode requires a goal: the planning panel allocates "
                "tasks from the goal's dependency graph. Pass a goal (--goal) "
                "or disable use_working_groups."
            )

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

        # Get Foundation summary for context
        foundation_summary = self._get_foundation_summary()

        # ===== PHASE 1: PLANNING =====
        if self.use_planning_panel:
            panel = PlanningPanel(
                provider=self.provider,
                graph=self.dependency_graph,
                textbook=self.textbook,
                foundation_summary=foundation_summary,
                n_groups=self.n_working_groups,
                ledger=self.ledger,
            )
            assignments = await panel.run_session(generation)
            # Panel spend used to be dropped on the floor: not in the ledger,
            # not in the society totals, not counted against the budget.
            generation_tokens += panel.tokens_used
            self.total_tokens_used += panel.tokens_used
        else:
            # Use default assignments without LLM
            available = self.dependency_graph.available_tasks()
            assignments = create_default_assignments(available, self.n_working_groups)

        if not assignments:
            # No tasks available; the planning spend is still real
            result = GenerationResult(
                generation=generation,
                artifacts_created=0,
                artifacts_verified=0,
                artifacts_referenced=0,
                fresh_creations=0,
                tokens_used=generation_tokens,
            )
            self.results.append(result)
            return result

        # Mark assigned tasks as in progress
        for assignment in assignments:
            self.dependency_graph.update_status(
                assignment.task_tag, TaskStatus.IN_PROGRESS
            )

        # ===== PHASE 2: WORKING GROUPS (parallel) =====
        groups = []
        for assignment in assignments:
            task_content = self._get_task_content(assignment.task_tag)
            config = WorkingGroupConfig(
                group_id=assignment.group_id,
                task_tag=assignment.task_tag,
                task_name=assignment.task_name,
                task_content=task_content,
                guidance=assignment.guidance,
                max_turns=self.max_turns_per_group,
                members_per_role=self._committee_members_per_role(),
            )
            group = WorkingGroup(
                config=config,
                provider=self.provider,
                foundation_summary=foundation_summary,
                ledger=self.ledger,
                generation=generation,
            )
            groups.append(group)

        # Run all groups in parallel
        group_results = await asyncio.gather(*[g.run_session() for g in groups])

        # Group-session spend used to be dropped like the panel's; it is the
        # bulk of a committee generation's bill.
        for group in groups:
            generation_tokens += group.tokens_used
            self.total_tokens_used += group.tokens_used

        # ===== PHASE 3: BUILD ARTIFACTS =====
        pending_reviews: list[tuple[PendingReview, WorkingGroup]] = []
        for group_result, group in zip(group_results, groups):
            if not group_result:
                # Group failed to produce artifact
                self.dependency_graph.update_status(
                    group.config.task_tag, TaskStatus.AVAILABLE
                )
                continue

            artifacts_created += 1

            # Extract lean code from group result. The scribe's payload can
            # carry a YAML block-scalar header and per-line indentation (the
            # HARN-02 leak, seen again on the 2026-08-19 smoke: code starting
            # with '|\n  import ...'), so clean before Lean ever sees it and
            # keep the raw capture on the artifact record.
            lean_code_raw = group_result.get("lean", group_result.get("blackboard", ""))
            lean_code = _clean_lean_code(lean_code_raw) or ""
            if not lean_code:
                self.dependency_graph.update_status(
                    group.config.task_tag, TaskStatus.AVAILABLE
                )
                continue

            # Create artifact
            artifact_name = group_result.get("name", f"group_{group.config.group_id}")
            artifact_id = (
                f"{artifact_name}-{hashlib.sha1(lean_code.encode()).hexdigest()[:8]}"
            )
            artifact = Artifact(
                id=artifact_id,
                type=ArtifactType(group_result.get("type", "definition")),
                natural_language=group_result.get(
                    "description", group.config.task_name
                ),
                lean_code=lean_code,
                lean_code_raw=lean_code_raw,
                references=list(group_result.get("references") or []),
                # Always the validated assignment tag. The scribe stamps its
                # own (smoke_c/d wrote "CAT-0013" for task 0013), and goal
                # progress keys on this field — a decorated tag makes
                # mark_formalized silently no-op.
                stacks_tag=group.config.task_tag,
                created_by=f"group-{group.config.group_id}",
                generation=generation,
                notes=group_result.get("notes"),
            )
            pending_reviews.append((PendingReview(artifact=artifact), group))

            # The transcript is the record of the session; it goes to the
            # textbook whatever the review or the verifier later decide.
            self.textbook.add(
                content=group.get_transcript(),
                author=f"group-{group.config.group_id}",
                generation=generation,
                topics=[group.config.task_tag, "transcript"],
                title=f"[GROUP {group.config.group_id}] {group.config.task_name}",
                entry_type="transcript",
            )

        # ===== PHASE 4: REVIEW COMMITTEE =====
        # The stage the pipeline always intended and never had: committee
        # output is reviewed before it reaches the verifier. Reviewers are the
        # society's agents — idle in committee mode — through the same tested
        # `Agent.review` path the flat pipeline uses. MODIFY is honored, so a
        # reviewer can repair code rather than only veto it.
        if self.use_peer_review and pending_reviews and self.agents:
            reviewers = [
                self.agents[i % len(self.agents)] for i in range(len(pending_reviews))
            ]
            review_results: list[tuple[ReviewResult, float]] = await asyncio.gather(
                *[
                    self._timed_review(reviewer, pending)
                    for reviewer, (pending, _) in zip(reviewers, pending_reviews)
                ]
            )

            surviving: list[tuple[PendingReview, WorkingGroup]] = []
            for (pending, group), reviewer, (review, review_elapsed) in zip(
                pending_reviews, reviewers, review_results
            ):
                reviews_total += 1
                if reviewer.id not in self.reviews_by_agent:
                    self.reviews_by_agent[reviewer.id] = {
                        "given": 0,
                        "approved": 0,
                        "rejected": 0,
                        "modified": 0,
                    }
                self.reviews_by_agent[reviewer.id]["given"] += 1

                if review.tokens_used:
                    tokens = review.tokens_used.total_tokens
                    generation_tokens += tokens
                    self.total_tokens_used += tokens
                    self.tokens_by_agent[reviewer.id] = (
                        self.tokens_by_agent.get(reviewer.id, 0) + tokens
                    )

                # Review spend belongs to the reviewed statement's cost
                self.ledger.record(
                    AttemptRecord(
                        statement_key=statement_key(
                            pending.artifact.stacks_tag,
                            pending.artifact.lean_code,
                            pending.artifact.natural_language,
                        ),
                        agent_id=reviewer.id,
                        generation=generation,
                        prompt_tokens=review.tokens_used.input_tokens
                        if review.tokens_used
                        else 0,
                        completion_tokens=review.tokens_used.output_tokens
                        if review.tokens_used
                        else 0,
                        wall_clock_s=review_elapsed,
                        outcome="review",
                    )
                )

                if review.decision == "REJECT":
                    reviews_rejected += 1
                    self.reviews_by_agent[reviewer.id]["rejected"] += 1
                    artifact = pending.artifact
                    artifact.verification_error = (
                        f"Rejected by review committee ({reviewer.id}): "
                        f"{review.reasoning}"
                    )
                    self.library.add(artifact)
                    self.dependency_graph.update_status(
                        group.config.task_tag, TaskStatus.AVAILABLE
                    )
                    continue

                if review.decision == "MODIFY" and review.modified_code:
                    reviews_modified += 1
                    self.reviews_by_agent[reviewer.id]["modified"] += 1
                    pending.artifact.lean_code = review.modified_code
                    pending.artifact.notes = (
                        pending.artifact.notes or ""
                    ) + f"\n[Modified by {reviewer.id}]"
                else:
                    reviews_approved += 1
                    self.reviews_by_agent[reviewer.id]["approved"] += 1

                surviving.append((pending, group))

            pending_reviews = surviving

        # ===== PHASE 5: VERIFICATION =====
        for pending, group in pending_reviews:
            artifact = pending.artifact
            lean_code = artifact.lean_code or ""

            # Verify with LEAN
            if self.verifier:
                # An import violation never reaches Lean, but it enters the
                # same repair loop a compile error does. The previous
                # hard-fail spent zero repair turns on the one error class
                # whose message names the exact fix.
                verify_result: VerificationResult | None = None
                last_error = self._import_violation(lean_code)
                if last_error is None:
                    verify_result = await self._verify_admissible(lean_code)
                    last_error = verify_result.error

                # A failed verify goes back to the group's scribe with the
                # Lean error, up to max_repair_attempts times. One-shot
                # committee groups burned a whole ~30K-token session per
                # failure while the iterative path fixed the same error
                # classes with a single feedback turn (committee_real_a).
                repair_tokens_before = group.tokens_used
                attempts_used = 0
                while (
                    verify_result is None or not verify_result.success
                ) and attempts_used < self.max_repair_attempts:
                    attempts_used += 1
                    repaired = await group.repair(
                        lean_code, last_error or "unknown error"
                    )
                    new_code = _clean_lean_code((repaired or {}).get("lean", "")) or ""
                    # The artifact parser falls back to the blackboard, which
                    # can hand back the code that just failed — stop rather
                    # than re-verify it.
                    if not new_code or new_code == lean_code:
                        break
                    blocked = self._import_violation(new_code)
                    if blocked is not None:
                        # Burns an attempt; the restriction is the error
                        # the next repair turn sees. The offending code is
                        # never adopted and never reaches Lean.
                        last_error = blocked
                        continue
                    lean_code = new_code
                    artifact.lean_code = new_code
                    artifact.notes = (
                        artifact.notes or ""
                    ) + f"\n[Repaired by scribe, attempt {attempts_used}]"
                    verify_result = await self._verify_admissible(lean_code)
                    last_error = verify_result.error

                # Phase 2 summed session spend before repairs existed; the
                # repair delta would otherwise vanish from the totals.
                repair_tokens = group.tokens_used - repair_tokens_before
                generation_tokens += repair_tokens
                self.total_tokens_used += repair_tokens

                # verify_result stays None when every draft was blocked on
                # imports — Lean never ran, and the artifact fails with the
                # restriction message as its error.
                artifact.status = (
                    verify_result.status
                    if verify_result is not None
                    else VerificationStatus.FAILED
                )
                await self._apply_gates(artifact)

                if verify_result is not None and verify_result.success:
                    artifacts_verified += 1

                    # A gate failure blocks promotion but not the count: Lean
                    # did accept it. Leaving the task un-DONE is the point --
                    # `committee_fix_c` closed the Yoneda milestone with a
                    # file containing only the scribe's prompt scaffold, and
                    # the graph then released everything downstream of it.
                    promoted = not self._blocked_by_gates(artifact)
                    if not promoted:
                        self._note_gate_block(artifact)
                    else:
                        # Add to foundation
                        try:
                            self.foundation.add_artifact(artifact)
                        except ValueError:
                            pass

                        # Update dependency graph
                        self.dependency_graph.update_status(
                            group.config.task_tag, TaskStatus.DONE, artifact.id
                        )

                        # Update goal progress
                        if self.goal and artifact.stacks_tag:
                            self.goal.mark_formalized(artifact.stacks_tag, artifact.id)

                    # Add to textbook
                    self.textbook.add(
                        content=f"Verified: {artifact.natural_language}\n\n{artifact.notes or ''}",
                        author=artifact.created_by,
                        generation=generation,
                        topics=[group.config.task_tag, "verified"],
                        title=f"[VERIFIED] {group.config.task_name}",
                        entry_type="success",
                    )
                else:
                    artifact.verification_error = last_error
                    self.dependency_graph.update_status(
                        group.config.task_tag, TaskStatus.AVAILABLE
                    )

                    # Add failure to textbook for learning
                    self.textbook.add(
                        content=f"Failed: {artifact.natural_language}\n\nError: {last_error}",
                        author=artifact.created_by,
                        generation=generation,
                        topics=[group.config.task_tag, "error"],
                        title=f"[FAILED] {group.config.task_name}",
                        entry_type="error",
                    )

            # Add to library
            self.library.add(artifact)

            # References are derived from the code, not trusted to the
            # scribe. committee_yolo_a ran with the citation prompt live for
            # all 100 generations and 0 of 266 artifacts carried one — while
            # ~97 failures were reuse *attempts* and one verified artifact
            # (0014) demonstrably built on another (0013). Self-report only
            # ever adds to what the code already proves.
            derived = self._derive_references(artifact.lean_code or "")
            merged = set(artifact.references) | derived
            merged.discard(artifact.id)
            artifact.references = sorted(merged)

            # Link references so reuse is measurable on committee runs too.
            # This call existed only on the flat and iterative paths, so
            # committee_real_b printed "Ratchet failure detected" on a run
            # whose groups demonstrably built on the foundation.
            for ref_id in artifact.references:
                if ref_id in self.library:
                    self.library.add_reference(artifact.id, ref_id)

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

    def _committee_members_per_role(self) -> dict[Role, int]:
        """Working-group cast scaled by society size.

        committee_6x10 (2026-08-20): `--agents 6` produced the same cast
        and token spend as `--agents 3` — every group seated a fixed
        chair + scribe + one researcher, and society agents only ever
        reviewed. Researchers now scale with the society: floor
        (n_agents / n_groups) per group, never below one, so 6 agents
        over 3 groups seat 2 researchers each and 3 agents reproduce the
        old cast exactly. Chair and scribe stay fixed overhead.
        """
        researchers = max(1, self.n_agents // max(1, self.n_working_groups))
        return {Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: researchers}

    def _import_violation(self, code: str) -> str | None:
        """The goal's import-restriction message for `code`, or None if legal.

        One decision for both the first draft and every repair; the two call
        sites previously duplicated the condition and could drift.
        """
        if self.goal and (self.goal.allowed_imports or self.goal.forbidden_imports):
            valid, error = self.goal.validate_code(code)
            if not valid:
                return f"Import restriction: {error}"
        return None

    def _derive_references(self, lean_code: str) -> set[str]:
        """Artifact ids of foundation entries whose names this code uses.

        Comments are stripped first: "-- unlike Category in Mathlib" is
        prose, not reuse. A name the code *declares itself* is excluded —
        redefining `Category` is what T2.duplicate exists to flag, not a
        citation of the entry it shadows.
        """
        if not self.foundation or not lean_code:
            return set()
        code = _BLOCK_COMMENT_RE.sub("", lean_code)
        code = _LINE_COMMENT_RE.sub("", code)
        own = set(_DECL_NAME_RE.findall(code))
        refs: set[str] = set()
        for entry in self.foundation.entries:
            name = entry.name
            if not name or name in own:
                continue
            pattern = rf"(?<![A-Za-z0-9_']){re.escape(name)}(?![A-Za-z0-9_'])"
            if re.search(pattern, code):
                refs.add(entry.artifact_id)
        return refs

    def _get_foundation_summary(self) -> str:
        """Get a summary of what's in Foundation.lean, for committee prompts.

        This used to be a second, weaker renderer: `- {tag}: {name}` for the
        first ten entries. It never ran -- `FoundationFile.entries` is a list,
        not a dict, and `FoundationEntry` has no `tag`, so both lines raised
        `AttributeError` on any non-empty foundation. Committee mode is
        unreachable today (26Q3-HARN-12), which is the only reason nobody saw
        it.

        Rather than repair the weaker renderer, defer to the one agents get.
        A work committee was being told strictly less than the agent that
        already failed on API shape; two renderers that must stay in sync is
        how that happened.

        Bounded, unlike the agent-facing call. This string is interpolated
        into the chair and planning-panel prompts alongside much else, and the
        old `entries[:10]` cap went away with nothing replacing it. At the
        measured ~118 chars/entry, a multi-thousand-statement foundation would
        push the prompt past the served `max_model_len` -- and the truncation
        would silently remove the section the prompt tells the committee to
        rely on. The count of what was left out is rendered, not hidden.
        """
        if not self.foundation or len(self.foundation) == 0:
            return "Foundation.lean is empty. You must define everything from scratch."

        return self.foundation.get_context_for_agent(
            max_entries=self.COMMITTEE_SUMMARY_MAX_ENTRIES
        )

    def _get_task_content(self, task_tag: str) -> str:
        """Get the full content for a task from the goal."""
        if not self.goal:
            return f"Define {task_tag}"

        for defn in self.goal.definitions:
            if defn.tag == task_tag:
                return defn.content

        # The silent fallback here ("Define <tag>") is how the 2026-08-19
        # smoke turned hallucinated panel tags into contentless tasks. The
        # panel now validates tags against the graph, so reaching this line
        # means a bypass — fail loudly rather than invent a task.
        raise ValueError(
            f"Task tag {task_tag!r} is not in goal '{self.goal.name}'. "
            "Committee assignments must use tags from the goal's task graph."
        )

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

        # Save the cost ledger — per-statement attribution incl. failed
        # attempts. This is what cvfn_report reads.
        self.ledger.save(output_dir / "attempts.json")

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
            GenerationResult(**gen_data) for gen_data in results_data["generations"]
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

        # Restore the cost ledger so a resumed run keeps appending to it
        attempts_path = output_dir / "attempts.json"
        if attempts_path.exists():
            society.ledger = CostLedger.load(attempts_path)

        return society

    async def run_from_checkpoint(
        self, target_generation: int
    ) -> list[GenerationResult]:
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
