"""Cultural artifact storage and tracking for LMS."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(Enum):
    """Types of mathematical artifacts."""

    DEFINITION = "definition"
    LEMMA = "lemma"
    THEOREM = "theorem"
    INSIGHT = "insight"
    STRATEGY = "strategy"


@dataclass
class Artifact:
    """A cultural artifact in the mathematical society.

    Artifacts represent knowledge that can be accumulated and reused
    across generations of agents. Like letters between 18th century
    mathematicians, they include both formal content and personal notes.
    """

    id: str
    type: ArtifactType
    natural_language: str
    created_by: str
    generation: int
    lean_code: str | None = None
    verified: bool = False
    references: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)
    # Agent's notes about their reasoning - like Fermat's marginalia
    notes: str | None = None
    # If verification failed, what was the error? (helps successors learn)
    verification_error: str | None = None
    # Token cost to produce this artifact
    tokens_used: int = 0
    # Stacks Project tag this artifact addresses (for goal tracking)
    stacks_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "natural_language": self.natural_language,
            "lean_code": self.lean_code,
            "verified": self.verified,
            "created_by": self.created_by,
            "generation": self.generation,
            "references": self.references,
            "referenced_by": self.referenced_by,
            "notes": self.notes,
            "verification_error": self.verification_error,
            "tokens_used": self.tokens_used,
            "stacks_tag": self.stacks_tag,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Artifact":
        """Create artifact from dictionary."""
        return cls(
            id=d["id"],
            type=ArtifactType(d["type"]),
            natural_language=d["natural_language"],
            lean_code=d.get("lean_code"),
            verified=d.get("verified", False),
            created_by=d["created_by"],
            generation=d["generation"],
            references=d.get("references", []),
            referenced_by=d.get("referenced_by", []),
            notes=d.get("notes"),
            verification_error=d.get("verification_error"),
            tokens_used=d.get("tokens_used", 0),
            stacks_tag=d.get("stacks_tag"),
        )


@dataclass
class PendingReview:
    """An artifact awaiting peer review before LEAN verification.

    This enables collaboration - another agent reviews the code
    before we spend expensive LEAN verification time on it.
    """

    artifact: Artifact
    reviewed_by: str | None = None
    review_approved: bool | None = None
    review_notes: str | None = None
    modified_code: str | None = None  # If reviewer suggests a fix
    review_tokens: int = 0  # Tokens used for review

    def get_code_for_verification(self) -> str | None:
        """Get the code to verify - modified if reviewer changed it."""
        return self.modified_code if self.modified_code else self.artifact.lean_code


class ReviewQueue:
    """Queue of artifacts awaiting peer review.

    Manages the flow: propose → review → verify
    """

    def __init__(self) -> None:
        self.pending: list[PendingReview] = []
        self.reviewed: list[PendingReview] = []
        # Metrics
        self.total_reviews: int = 0
        self.approvals: int = 0
        self.rejections: int = 0
        self.modifications: int = 0

    def add(self, artifact: Artifact) -> None:
        """Add an artifact to the review queue."""
        self.pending.append(PendingReview(artifact=artifact))

    def get_for_review(self, exclude_agent: str) -> PendingReview | None:
        """Get an artifact for review, excluding the creator.

        Args:
            exclude_agent: Agent ID to exclude (can't review own work)

        Returns:
            A PendingReview, or None if nothing available
        """
        for i, pr in enumerate(self.pending):
            if pr.artifact.created_by != exclude_agent:
                return self.pending.pop(i)
        return None

    def mark_reviewed(
        self,
        pending: PendingReview,
        reviewer: str,
        approved: bool,
        notes: str | None = None,
        modified_code: str | None = None,
        tokens_used: int = 0,
    ) -> None:
        """Mark an artifact as reviewed."""
        pending.reviewed_by = reviewer
        pending.review_approved = approved
        pending.review_notes = notes
        pending.modified_code = modified_code
        pending.review_tokens = tokens_used

        self.reviewed.append(pending)
        self.total_reviews += 1

        if approved:
            self.approvals += 1
            if modified_code:
                self.modifications += 1
        else:
            self.rejections += 1

    def get_approved(self) -> list[PendingReview]:
        """Get all approved artifacts ready for LEAN verification."""
        return [pr for pr in self.reviewed if pr.review_approved]

    def get_rejected(self) -> list[PendingReview]:
        """Get all rejected artifacts (won't go to LEAN)."""
        return [pr for pr in self.reviewed if not pr.review_approved]

    def clear(self) -> None:
        """Clear the queue for the next generation."""
        self.pending.clear()
        self.reviewed.clear()


@dataclass
class Correspondence:
    """A letter between agents - informal notes, attempts, and discussions.

    Unlike formal Artifacts, Correspondence captures the messy process
    of mathematical discovery: failed attempts, partial ideas, questions
    for future generations.
    """

    id: str
    author: str
    generation: int
    subject: str  # What were they working on?
    content: str  # Their notes, reasoning, questions
    related_artifacts: list[str] = field(default_factory=list)
    timestamp: str = ""  # ISO format

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "author": self.author,
            "generation": self.generation,
            "subject": self.subject,
            "content": self.content,
            "related_artifacts": self.related_artifacts,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Correspondence":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            author=d["author"],
            generation=d["generation"],
            subject=d["subject"],
            content=d["content"],
            related_artifacts=d.get("related_artifacts", []),
            timestamp=d.get("timestamp", ""),
        )


class ArtifactLibrary:
    """Shared library of cultural artifacts.

    Tracks artifacts created by agents, their relationships (references),
    and provides metrics for measuring cultural accumulation vs. decay.
    Also stores correspondence - the informal notes and discussions.
    """

    def __init__(self) -> None:
        """Initialize an empty artifact library."""
        self.artifacts: dict[str, Artifact] = {}
        self.correspondence: list[Correspondence] = []
        self.total_tokens_used: int = 0

    def __len__(self) -> int:
        """Return number of artifacts in library."""
        return len(self.artifacts)

    def __contains__(self, artifact_id: str) -> bool:
        """Check if artifact exists in library."""
        return artifact_id in self.artifacts

    def add(self, artifact: Artifact) -> None:
        """Add an artifact to the library.

        Args:
            artifact: Artifact to add
        """
        self.artifacts[artifact.id] = artifact
        self.total_tokens_used += artifact.tokens_used

    def add_correspondence(self, letter: Correspondence) -> None:
        """Add correspondence (informal notes) to the library.

        Args:
            letter: Correspondence to add
        """
        self.correspondence.append(letter)

    def get(self, artifact_id: str) -> Artifact | None:
        """Get an artifact by ID.

        Args:
            artifact_id: ID of artifact to retrieve

        Returns:
            The artifact, or None if not found
        """
        return self.artifacts.get(artifact_id)

    def add_reference(self, from_id: str, to_id: str) -> None:
        """Record that one artifact references another.

        Updates both the referencing artifact's `references` list
        and the referenced artifact's `referenced_by` list.

        Args:
            from_id: ID of the artifact making the reference
            to_id: ID of the artifact being referenced
        """
        from_artifact = self.artifacts.get(from_id)
        to_artifact = self.artifacts.get(to_id)

        if from_artifact and to_id not in from_artifact.references:
            from_artifact.references.append(to_id)

        if to_artifact and from_id not in to_artifact.referenced_by:
            to_artifact.referenced_by.append(from_id)

    def get_by_generation(self, generation: int) -> list[Artifact]:
        """Get all artifacts from a specific generation.

        Args:
            generation: Generation number to filter by

        Returns:
            List of artifacts from that generation
        """
        return [a for a in self.artifacts.values() if a.generation == generation]

    def get_verified(self) -> list[Artifact]:
        """Get all verified artifacts.

        Returns:
            List of artifacts that have been verified by LEAN
        """
        return [a for a in self.artifacts.values() if a.verified]

    def all(self) -> list[Artifact]:
        """Get all artifacts as a list.

        Returns:
            List of all artifacts
        """
        return list(self.artifacts.values())

    def reused_artifact_count(self) -> int:
        """Count artifacts that have been referenced by others.

        This measures cultural reuse - artifacts that contribute to
        subsequent work.

        Returns:
            Number of artifacts with at least one reference to them
        """
        return sum(1 for a in self.artifacts.values() if a.referenced_by)

    def fresh_creation_count(self) -> int:
        """Count artifacts created without referencing existing work.

        High fresh creation rates might indicate the 'Tasmania effect' -
        knowledge is being recreated from scratch rather than built upon.

        Returns:
            Number of artifacts with no references to other artifacts
        """
        return sum(1 for a in self.artifacts.values() if not a.references)

    def save(self, path: Path) -> None:
        """Save library to JSON file.

        Args:
            path: Path to save JSON file
        """
        data = {
            "artifacts": [a.to_dict() for a in self.artifacts.values()],
            "correspondence": [c.to_dict() for c in self.correspondence],
            "total_tokens_used": self.total_tokens_used,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ArtifactLibrary":
        """Load library from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded artifact library
        """
        data = json.loads(path.read_text())
        library = cls()
        for artifact_dict in data["artifacts"]:
            artifact = Artifact.from_dict(artifact_dict)
            library.artifacts[artifact.id] = artifact
        for corr_dict in data.get("correspondence", []):
            library.correspondence.append(Correspondence.from_dict(corr_dict))
        library.total_tokens_used = data.get("total_tokens_used", 0)
        return library
