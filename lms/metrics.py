"""Metrics and analysis for LMS experiments.

Provides functions to analyze artifact libraries and generation results
to detect cultural accumulation vs. decay (Tasmania effect).
"""

from dataclasses import dataclass

from lms.artifacts import ArtifactLibrary
from lms.society import GenerationResult


def calculate_reuse_rate(library: ArtifactLibrary) -> float:
    """Calculate the fraction of artifacts that have been reused.

    Reuse rate measures cultural accumulation - how much prior work
    is being built upon.

    Args:
        library: Artifact library to analyze

    Returns:
        Fraction of artifacts that have been referenced (0.0 to 1.0)
    """
    if len(library) == 0:
        return 0.0

    reused = library.reused_artifact_count()
    return reused / len(library)


def calculate_fresh_creation_rate(library: ArtifactLibrary) -> float:
    """Calculate the fraction of artifacts created without references.

    High fresh creation rates may indicate the Tasmania effect -
    knowledge is being recreated rather than built upon.

    Args:
        library: Artifact library to analyze

    Returns:
        Fraction of artifacts with no references (0.0 to 1.0)
    """
    if len(library) == 0:
        return 0.0

    fresh = library.fresh_creation_count()
    return fresh / len(library)


def calculate_verification_rate(results: list[GenerationResult]) -> float:
    """Calculate the overall verification success rate.

    Args:
        results: List of generation results

    Returns:
        Fraction of artifacts that passed verification (0.0 to 1.0)
    """
    if not results:
        return 0.0

    total_created = sum(r.artifacts_created for r in results)
    total_verified = sum(r.artifacts_verified for r in results)

    if total_created == 0:
        return 0.0

    return total_verified / total_created


def calculate_growth_rate(results: list[GenerationResult]) -> float:
    """Calculate the average growth in artifacts per generation.

    Positive growth indicates expanding knowledge base.
    Negative or zero growth may indicate stagnation.

    Args:
        results: List of generation results

    Returns:
        Average change in artifacts created between generations
    """
    if len(results) < 2:
        return 0.0

    deltas = []
    for i in range(1, len(results)):
        delta = results[i].artifacts_created - results[i - 1].artifacts_created
        deltas.append(delta)

    return sum(deltas) / len(deltas)


@dataclass
class LibraryAnalysis:
    """Comprehensive analysis of an artifact library.

    Attributes:
        total_artifacts: Total number of artifacts
        verified_artifacts: Number of verified artifacts
        reuse_rate: Fraction of artifacts that have been reused
        fresh_creation_rate: Fraction created without references
        verification_rate: Fraction that passed verification
        growth_rate: Average artifacts added per generation
        potential_tasmania_effect: True if metrics suggest knowledge decay
    """

    total_artifacts: int
    verified_artifacts: int
    reuse_rate: float
    fresh_creation_rate: float
    verification_rate: float
    growth_rate: float
    potential_tasmania_effect: bool


def analyze_library(
    library: ArtifactLibrary,
    results: list[GenerationResult],
    tasmania_threshold: float = 0.8,
) -> LibraryAnalysis:
    """Perform comprehensive analysis of an artifact library.

    Args:
        library: Artifact library to analyze
        results: Generation results for the experiment
        tasmania_threshold: Fresh creation rate above which to flag
            potential Tasmania effect (default 0.8)

    Returns:
        LibraryAnalysis with all computed metrics
    """
    reuse_rate = calculate_reuse_rate(library)
    fresh_rate = calculate_fresh_creation_rate(library)
    verification_rate = calculate_verification_rate(results)
    growth_rate = calculate_growth_rate(results)

    # Detect potential Tasmania effect:
    # - High fresh creation rate (not building on prior work)
    # - Low reuse rate (prior work not being used)
    potential_tasmania = fresh_rate >= tasmania_threshold and reuse_rate < 0.2

    return LibraryAnalysis(
        total_artifacts=len(library),
        verified_artifacts=len(library.get_verified()),
        reuse_rate=reuse_rate,
        fresh_creation_rate=fresh_rate,
        verification_rate=verification_rate,
        growth_rate=growth_rate,
        potential_tasmania_effect=potential_tasmania,
    )


def print_analysis(analysis: LibraryAnalysis) -> None:
    """Print a human-readable analysis summary.

    Args:
        analysis: LibraryAnalysis to print
    """
    print("\n" + "=" * 50)
    print("LMS Experiment Analysis")
    print("=" * 50)
    print(f"\nTotal Artifacts: {analysis.total_artifacts}")
    print(f"Verified Artifacts: {analysis.verified_artifacts}")
    print(f"\nReuse Rate: {analysis.reuse_rate:.1%}")
    print(f"Fresh Creation Rate: {analysis.fresh_creation_rate:.1%}")
    print(f"Verification Rate: {analysis.verification_rate:.1%}")
    print(f"Growth Rate: {analysis.growth_rate:+.1f} artifacts/generation")

    if analysis.potential_tasmania_effect:
        print("\n[!] WARNING: Potential Tasmania Effect detected!")
        print("    Agents may be creating from scratch instead of")
        print("    building on accumulated knowledge.")
    else:
        print("\n[+] Cultural accumulation appears healthy.")

    print("=" * 50 + "\n")
