"""LMS: LLM Mathematical Society prototype.

A prototype exploring whether LLM collectives with formal verification
can achieve emergent mathematical reasoning.
"""

__version__ = "0.1.0"

from lms.config import Config, ProviderConfig
from lms.artifacts import Artifact, ArtifactType, ArtifactLibrary
from lms.agent import Agent, AgentResponse
from lms.society import Society, GenerationResult
from lms.voting import VotingSystem, PromptProposal, VoteResult
from lms.metrics import analyze_library, LibraryAnalysis

__all__ = [
    # Config
    "Config",
    "ProviderConfig",
    # Artifacts
    "Artifact",
    "ArtifactType",
    "ArtifactLibrary",
    # Agents
    "Agent",
    "AgentResponse",
    # Society
    "Society",
    "GenerationResult",
    # Voting
    "VotingSystem",
    "PromptProposal",
    "VoteResult",
    # Metrics
    "analyze_library",
    "LibraryAnalysis",
]
