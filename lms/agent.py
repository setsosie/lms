"""LMS Agent - wraps an LLM provider to propose mathematical artifacts."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lms.artifacts import Artifact, ArtifactType, ArtifactLibrary, PendingReview
from lms.foundation import FoundationFile
from lms.prompts import get_prompt
from lms.providers.base import BaseLLMProvider, Message, TokenUsage
from lms.textbook import Textbook

if TYPE_CHECKING:
    from lms.goals import Goal


@dataclass
class AttemptResult:
    """Result of a single verification attempt.

    Attributes:
        attempt_num: Which attempt (1-indexed)
        artifact: The artifact that was proposed
        verified: Whether LEAN verification succeeded
        error: Error message if verification failed
        tokens_used: Tokens used for this attempt
    """

    attempt_num: int
    artifact: Artifact
    verified: bool
    error: str | None = None
    tokens_used: int = 0


@dataclass
class IterativeResponse:
    """Response from an agent's iterative propose() call.

    Attributes:
        attempts: List of all attempts made
        final_artifact: The final artifact (verified or best effort)
        success: Whether any attempt succeeded
        writeup: Agent's summary of what they learned
        writeup_title: Optional title for the writeup (for textbook search)
        total_tokens: Total tokens used across all attempts
    """

    attempts: list[AttemptResult]
    final_artifact: Artifact | None
    success: bool
    writeup: str
    writeup_title: str = ""  # Agent-provided title for textbook
    total_tokens: int = 0


@dataclass
class ReviewResult:
    """Result of a peer review.

    Attributes:
        decision: APPROVE, REJECT, or MODIFY
        reasoning: Why the reviewer made this decision
        modified_code: If MODIFY, the fixed code
        tokens_used: Token usage for this review
    """

    decision: str  # "APPROVE", "REJECT", "MODIFY"
    reasoning: str
    modified_code: str | None = None
    tokens_used: TokenUsage | None = None


@dataclass
class AgentResponse:
    """Response from an agent's propose() call.

    Attributes:
        raw_text: The raw text response from the LLM
        proposed_artifacts: Artifacts parsed from the response
        referenced_artifacts: IDs of existing artifacts that were referenced
        tokens_used: Token usage for this response
    """

    raw_text: str
    proposed_artifacts: list[Artifact] = field(default_factory=list)
    referenced_artifacts: list[str] = field(default_factory=list)
    tokens_used: TokenUsage | None = None


class Agent:
    """An agent in the LLM Mathematical Society.

    Agents propose mathematical artifacts by querying an LLM provider
    and parsing the response for structured artifact definitions.
    Like 18th century mathematicians, they leave notes about their reasoning.
    """

    # Regex to parse artifact blocks from LLM response
    # Includes optional fields: stacks_tag, lean, notes
    ARTIFACT_PATTERN = re.compile(
        r"<artifact>\s*"
        r"type:\s*(?P<type>\w+)\s*"
        r"name:\s*(?P<name>\S+)\s*"
        r"(?:stacks_tag:\s*(?P<stacks_tag>\S+)\s*)?"
        r"description:\s*(?P<description>.+?)\s*"
        r"(?:lean:\s*(?P<lean>.+?)\s*)?"
        r"(?:notes:\s*(?P<notes>.+?)\s*)?"
        r"references:\s*\[(?P<references>[^\]]*)\]\s*"
        r"</artifact>",
        re.DOTALL,
    )

    def __init__(
        self,
        id: str,
        provider: BaseLLMProvider,
        generation: int = 0,
    ) -> None:
        """Initialize an agent.

        Args:
            id: Unique identifier for this agent
            provider: LLM provider to use for generating responses
            generation: Current generation number
        """
        self.id = id
        self.provider = provider
        self.generation = generation

    async def propose(
        self,
        library: ArtifactLibrary,
        goal: "Goal | None" = None,
        textbook: Textbook | None = None,
        foundation: FoundationFile | None = None,
    ) -> AgentResponse:
        """Propose new artifacts given the current library and optional goal.

        Args:
            library: Current artifact library
            goal: Optional goal to work towards (uses goal-directed prompts)
            textbook: Optional collective wisdom to search
            foundation: Optional foundation file with verified definitions to import

        Returns:
            AgentResponse with proposed artifacts, references, and token usage
        """
        # Build context from library
        context = self._build_library_context(library)

        # Add foundation context (verified definitions available for import)
        if foundation:
            foundation_context = foundation.get_context_for_agent()
            context = foundation_context + "\n\n" + context

        # Add textbook wisdom if available
        if textbook and len(textbook) > 0:
            # Get relevant wisdom based on goal tags
            tags = [d.tag for d in goal.definitions] if goal else None
            wisdom = textbook.get_for_context(tags=tags, max_tokens=400)
            if wisdom:
                context = wisdom + "\n\n" + context

        # Get appropriate prompts based on whether we have a goal
        if goal is not None:
            system_prompt = get_prompt("agent_system_goal")
            user_prompt = get_prompt("agent_user_goal")
            user_content = user_prompt.content.format(
                goal_context=goal.to_prompt_context(),
                library_context=context,
            )
        else:
            system_prompt = get_prompt("agent_system")
            user_prompt = get_prompt("agent_user")
            user_content = user_prompt.content.format(library_context=context)

        # Query LLM
        messages = [
            Message(
                role="user",
                content=user_content,
            )
        ]

        response = await self.provider.generate(
            messages,
            system_prompt=system_prompt.content,
        )

        # Parse artifacts from response
        proposed, referenced = self._parse_artifacts(response.content, response.usage)

        return AgentResponse(
            raw_text=response.content,
            proposed_artifacts=proposed,
            referenced_artifacts=referenced,
            tokens_used=response.usage,
        )

    async def propose_iterative(
        self,
        library: ArtifactLibrary,
        verify_fn,  # Callable[[str], Awaitable[tuple[bool, str | None]]]
        goal: "Goal | None" = None,
        textbook: Textbook | None = None,
        max_attempts: int = 5,
        foundation: FoundationFile | None = None,
    ) -> IterativeResponse:
        """Propose artifacts with multiple verification attempts.

        The agent gets up to max_attempts tries to produce verified LEAN code.
        After each failed attempt, they see the error and can revise.

        Args:
            library: Current artifact library
            verify_fn: Async function that takes LEAN code and returns (success, error)
            goal: Optional goal to work towards
            textbook: Optional collective wisdom
            max_attempts: Maximum verification attempts (default 5)
            foundation: Optional foundation file with verified definitions to import

        Returns:
            IterativeResponse with all attempts and final writeup
        """
        attempts: list[AttemptResult] = []
        total_tokens = 0
        conversation: list[Message] = []

        # Build initial context
        context = self._build_library_context(library)

        # Add foundation context (verified definitions available for import)
        if foundation:
            foundation_context = foundation.get_context_for_agent()
            context = foundation_context + "\n\n" + context

        if textbook and len(textbook) > 0:
            tags = [d.tag for d in goal.definitions] if goal else None
            wisdom = textbook.get_for_context(tags=tags, max_tokens=400)
            if wisdom:
                context = wisdom + "\n\n" + context

        # Get prompts
        if goal is not None:
            system_prompt = get_prompt("agent_system_goal")
            user_prompt = get_prompt("agent_user_goal")
            initial_content = user_prompt.content.format(
                goal_context=goal.to_prompt_context(),
                library_context=context,
            )
        else:
            system_prompt = get_prompt("agent_system")
            user_prompt = get_prompt("agent_user")
            initial_content = user_prompt.content.format(library_context=context)

        # Add iteration instructions
        initial_content += f"""

═══════════════════════════════════════════════════════════════════════════════
                         ITERATIVE MODE ({max_attempts} attempts)
═══════════════════════════════════════════════════════════════════════════════
You have {max_attempts} attempts to produce VERIFIED LEAN code.
After each attempt, you'll see the verification result.
If it fails, ANALYZE the error and TRY AGAIN with fixes.

On your FINAL attempt (or success), include a writeup for the collective textbook:
<writeup>
<title>A descriptive title that would help future agents find this knowledge
(What would you have wanted to know at the start?)</title>

Full explanation: What you tried, what worked, what didn't, and specific advice.
Include error messages and how you fixed them. Be detailed - this helps future agents!
</writeup>
"""

        conversation.append(Message(role="user", content=initial_content))

        for attempt_num in range(1, max_attempts + 1):
            # Query LLM
            response = await self.provider.generate(
                conversation,
                system_prompt=system_prompt.content,
            )

            tokens_used = response.usage.total_tokens if response.usage else 0
            total_tokens += tokens_used

            # Add assistant response to conversation
            conversation.append(Message(role="assistant", content=response.content))

            # Parse artifact from response
            proposed, _ = self._parse_artifacts(response.content, response.usage)

            if not proposed or not proposed[0].lean_code:
                # No valid artifact, ask to try again
                if attempt_num < max_attempts:
                    conversation.append(Message(
                        role="user",
                        content=f"Attempt {attempt_num}/{max_attempts}: No valid LEAN code found. "
                                f"Please propose an artifact with lean: code block."
                    ))
                continue

            artifact = proposed[0]
            artifact.generation = self.generation
            artifact.created_by = self.id

            # Verify the code
            success, error = await verify_fn(artifact.lean_code)
            artifact.verified = success
            if error:
                artifact.verification_error = error

            attempts.append(AttemptResult(
                attempt_num=attempt_num,
                artifact=artifact,
                verified=success,
                error=error,
                tokens_used=tokens_used,
            ))

            if success:
                # Success! Ask for writeup
                conversation.append(Message(
                    role="user",
                    content=f"✓ Attempt {attempt_num}/{max_attempts}: VERIFIED! "
                            f"Please provide a <writeup> with a <title> summarizing what you learned."
                ))
                writeup_response = await self.provider.generate(
                    conversation,
                    system_prompt=system_prompt.content,
                )
                total_tokens += writeup_response.usage.total_tokens if writeup_response.usage else 0
                writeup, writeup_title = self._extract_writeup(writeup_response.content)

                return IterativeResponse(
                    attempts=attempts,
                    final_artifact=artifact,
                    success=True,
                    writeup=writeup,
                    writeup_title=writeup_title,
                    total_tokens=total_tokens,
                )

            # Failed - add error feedback for next attempt
            if attempt_num < max_attempts:
                conversation.append(Message(
                    role="user",
                    content=f"✗ Attempt {attempt_num}/{max_attempts}: Verification FAILED.\n\n"
                            f"Error:\n```\n{error[:500] if error else 'Unknown error'}\n```\n\n"
                            f"Analyze the error and try again. You have {max_attempts - attempt_num} attempts left."
                ))
            else:
                # Final attempt failed - ask for writeup anyway
                conversation.append(Message(
                    role="user",
                    content=f"✗ Attempt {attempt_num}/{max_attempts}: Verification FAILED.\n\n"
                            f"Error:\n```\n{error[:500] if error else 'Unknown error'}\n```\n\n"
                            f"No more attempts. Please provide a <writeup> with <title> - "
                            f"what you learned and advice for the next generation."
                ))
                writeup_response = await self.provider.generate(
                    conversation,
                    system_prompt=system_prompt.content,
                )
                total_tokens += writeup_response.usage.total_tokens if writeup_response.usage else 0
                writeup, writeup_title = self._extract_writeup(writeup_response.content)

                # Return best attempt (last one)
                return IterativeResponse(
                    attempts=attempts,
                    final_artifact=artifact,
                    success=False,
                    writeup=writeup,
                    writeup_title=writeup_title,
                    total_tokens=total_tokens,
                )

        # No valid attempts at all
        return IterativeResponse(
            attempts=attempts,
            final_artifact=None,
            success=False,
            writeup="No valid LEAN code was produced in any attempt.",
            total_tokens=total_tokens,
        )

    def _extract_writeup(self, text: str) -> tuple[str, str]:
        """Extract writeup and optional title from response text.

        Returns:
            Tuple of (content, title). Title may be empty if not provided.
        """
        match = re.search(r"<writeup>(.*?)</writeup>", text, re.DOTALL)
        if match:
            writeup_text = match.group(1).strip()

            # Try to extract title
            title_match = re.search(r"<title>(.*?)</title>", writeup_text, re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()
                # Remove title from content
                content = re.sub(r"<title>.*?</title>", "", writeup_text, flags=re.DOTALL).strip()
                return content, title

            return writeup_text, ""

        # Fall back to last paragraph if no tags
        paragraphs = text.strip().split("\n\n")
        return (paragraphs[-1] if paragraphs else "No writeup provided.", "")

    def _build_library_context(self, library: ArtifactLibrary) -> str:
        """Build context string from library contents.

        Args:
            library: Artifact library

        Returns:
            Formatted string describing library contents
        """
        if len(library) == 0:
            return "The artifact library is currently empty. You are starting fresh."

        # Sort: verified first (to encourage reuse), then by generation (newest first)
        sorted_artifacts = sorted(
            library.all(),
            key=lambda a: (not a.verified, -a.generation)
        )

        lines = ["Current artifact library:"]
        for artifact in sorted_artifacts:
            status = "[verified]" if artifact.verified else "[unverified]"
            lines.append(f"\n- ID: {artifact.id} {status}")
            lines.append(f"  Type: {artifact.type.value}")
            lines.append(f"  Author: {artifact.created_by} (gen {artifact.generation})")
            lines.append(f"  Description: {artifact.natural_language}")
            if artifact.lean_code:
                lines.append(f"  LEAN: {artifact.lean_code[:100]}...")
            if artifact.references:
                lines.append(f"  References: {artifact.references}")
            # Include notes - the agent's marginalia for successors
            if artifact.notes:
                lines.append(f"  Notes: {artifact.notes[:200]}...")
            # If verification failed, show the error so successors can learn
            if artifact.verification_error:
                lines.append(f"  ⚠ Verification failed: {artifact.verification_error[:100]}...")

        return "\n".join(lines)

    def _parse_artifacts(
        self, text: str, usage: TokenUsage | None = None
    ) -> tuple[list[Artifact], list[str]]:
        """Parse artifact definitions from LLM response.

        Args:
            text: Raw LLM response text
            usage: Token usage for this response (split among artifacts)

        Returns:
            Tuple of (proposed artifacts, referenced artifact IDs)
        """
        proposed = []
        all_references = []

        # Count artifacts first to split token cost
        matches = list(self.ARTIFACT_PATTERN.finditer(text))
        tokens_per_artifact = 0
        if usage and matches:
            tokens_per_artifact = usage.total_tokens // len(matches)

        for match in matches:
            artifact_type = match.group("type").lower()
            name = match.group("name")
            description = match.group("description").strip()
            lean_code = match.group("lean")
            if lean_code:
                lean_code = lean_code.strip()

            # Parse notes - the agent's marginalia
            notes = match.group("notes")
            if notes:
                notes = notes.strip()

            # Parse references
            refs_str = match.group("references").strip()
            references = []
            if refs_str:
                references = [r.strip() for r in refs_str.split(",") if r.strip()]

            # Map type string to enum
            type_map = {
                "definition": ArtifactType.DEFINITION,
                "lemma": ArtifactType.LEMMA,
                "theorem": ArtifactType.THEOREM,
                "insight": ArtifactType.INSIGHT,
                "strategy": ArtifactType.STRATEGY,
            }
            art_type = type_map.get(artifact_type, ArtifactType.INSIGHT)

            # Parse stacks_tag if present
            stacks_tag = match.group("stacks_tag")
            if stacks_tag:
                stacks_tag = stacks_tag.strip()

            # Generate unique ID
            artifact_id = f"{artifact_type}-{name}-{uuid.uuid4().hex[:8]}"

            artifact = Artifact(
                id=artifact_id,
                type=art_type,
                natural_language=description,
                lean_code=lean_code,
                verified=False,  # Will be verified later
                created_by=self.id,
                generation=self.generation,
                references=references,
                notes=notes,
                tokens_used=tokens_per_artifact,
                stacks_tag=stacks_tag,
            )
            proposed.append(artifact)
            all_references.extend(references)

        return proposed, list(set(all_references))

    # Regex to parse review blocks from LLM response
    REVIEW_PATTERN = re.compile(
        r"<review>\s*"
        r"decision:\s*(?P<decision>\w+)\s*"
        r"reasoning:\s*(?P<reasoning>.+?)\s*"
        r"(?:modified_code:\s*(?P<modified_code>.+?)\s*)?"
        r"</review>",
        re.DOTALL,
    )

    async def review(self, pending: PendingReview) -> ReviewResult:
        """Review another agent's artifact before LEAN verification.

        Args:
            pending: The pending review containing the artifact to review

        Returns:
            ReviewResult with decision, reasoning, and optional modified code
        """
        artifact = pending.artifact

        # Get review prompts
        system_prompt = get_prompt("review_system")
        user_prompt = get_prompt("review_user")

        user_content = user_prompt.content.format(
            creator=artifact.created_by,
            artifact_type=artifact.type.value,
            description=artifact.natural_language,
            stacks_tag=artifact.stacks_tag or "None",
            lean_code=artifact.lean_code or "No code provided",
            notes=artifact.notes or "No notes provided",
        )

        # Query LLM
        messages = [Message(role="user", content=user_content)]
        response = await self.provider.generate(
            messages,
            system_prompt=system_prompt.content,
        )

        # Parse review from response
        return self._parse_review(response.content, response.usage)

    def _parse_review(
        self, text: str, usage: TokenUsage | None = None
    ) -> ReviewResult:
        """Parse review decision from LLM response.

        Args:
            text: Raw LLM response text
            usage: Token usage for this response

        Returns:
            ReviewResult with decision and reasoning
        """
        match = self.REVIEW_PATTERN.search(text)

        if match:
            decision = match.group("decision").upper()
            reasoning = match.group("reasoning").strip()
            modified_code = match.group("modified_code")
            if modified_code:
                modified_code = modified_code.strip()

            # Normalize decision
            if decision not in ("APPROVE", "REJECT", "MODIFY"):
                decision = "APPROVE"  # Default to approve if unclear

            return ReviewResult(
                decision=decision,
                reasoning=reasoning,
                modified_code=modified_code if decision == "MODIFY" else None,
                tokens_used=usage,
            )

        # If no match, default to approve (don't block on parsing issues)
        return ReviewResult(
            decision="APPROVE",
            reasoning="Could not parse review response - defaulting to approve",
            tokens_used=usage,
        )
