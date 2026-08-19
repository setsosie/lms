"""Metrics and analysis for LMS experiments.

Provides functions to analyze artifact libraries and generation results
to detect cultural accumulation vs. ratchet failure.

"Ratchet failure" is named for Tomasello's cultural ratchet (Tomasello,
Kruger & Ratner 1993) -- the mechanism by which a population holds onto
what it has learned so the next generation starts from there rather than
from scratch. When agents keep re-deriving results the library already
contains, the ratchet is not engaging.

This was previously called the "Tasmania effect", after Henrich (2004),
"Demography and Cultural Evolution: How Adaptive Cultural Processes Can
Produce Maladaptive Losses -- The Tasmanian Case". Renamed because that
describes the *loss* of existing technology, which is not what is measured
here -- these agents never accumulated anything to lose. The underlying
demographic claim is also contested (Vaesen et al. 2016, PNAS).
"""

from collections import Counter
from dataclasses import dataclass, field

from lms.artifacts import ArtifactLibrary
from lms.gates import GateOutcome
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

    High fresh creation rates may indicate ratchet failure -
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


def gate_failure_histogram(library: ArtifactLibrary) -> dict[str, int]:
    """Count FAILED gate sub-checks per gate name across the library.

    The *shape* of this distribution is a primary calibration output
    (`calibration-program.md` §2): which gate kills artifacts says more about
    the collective than how many died.
    """
    counts: Counter[str] = Counter()
    for artifact in library.all():
        for result in artifact.gate_results:
            if result.outcome is GateOutcome.FAILED:
                counts[result.gate] += 1
    return dict(counts)


def gate_inconclusive_histogram(library: ArtifactLibrary) -> dict[str, int]:
    """Count INCONCLUSIVE gate sub-checks per gate name across the library.

    These are the artifacts routed to D4 human review — the machine's
    explicit "I don't know", kept separate from failures so neither reads as
    the other.
    """
    counts: Counter[str] = Counter()
    for artifact in library.all():
        for result in artifact.gate_results:
            if result.outcome is GateOutcome.INCONCLUSIVE:
                counts[result.gate] += 1
    return dict(counts)


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
        potential_ratchet_failure: True if the library is large enough to
            reuse and agents are not reusing it
        gate_failure_histogram: FAILED gate sub-checks by gate name
        gate_inconclusive_histogram: INCONCLUSIVE sub-checks by gate name
            (routed to D4 review)
    """

    total_artifacts: int
    verified_artifacts: int
    reuse_rate: float
    fresh_creation_rate: float
    verification_rate: float
    growth_rate: float
    potential_ratchet_failure: bool
    gate_failure_histogram: dict[str, int] = field(default_factory=dict)
    gate_inconclusive_histogram: dict[str, int] = field(default_factory=dict)


# Below these, "nobody reused anything" is a statement about the library being
# empty, not about agent behavior. A one-artifact library offers nothing to
# build on, so a warning there is guaranteed noise -- and a warning that always
# fires is one nobody reads on the run where it means something.
MIN_ARTIFACTS_FOR_RATCHET = 5
MIN_GENERATIONS_FOR_RATCHET = 2


def analyze_library(
    library: ArtifactLibrary,
    results: list[GenerationResult],
    ratchet_threshold: float = 0.8,
) -> LibraryAnalysis:
    """Perform comprehensive analysis of an artifact library.

    Args:
        library: Artifact library to analyze
        results: Generation results for the experiment
        ratchet_threshold: Fresh creation rate above which to flag
            potential ratchet failure (default 0.8)

    Returns:
        LibraryAnalysis with all computed metrics
    """
    reuse_rate = calculate_reuse_rate(library)
    fresh_rate = calculate_fresh_creation_rate(library)
    verification_rate = calculate_verification_rate(results)
    growth_rate = calculate_growth_rate(results)

    # Detect potential ratchet failure:
    # - Enough prior work existed to be worth reusing
    # - High fresh creation rate (not building on prior work)
    # - Low reuse rate (prior work not being used)
    had_something_to_reuse = (
        len(library) >= MIN_ARTIFACTS_FOR_RATCHET
        and len(results) >= MIN_GENERATIONS_FOR_RATCHET
    )
    potential_ratchet_failure = (
        had_something_to_reuse and fresh_rate >= ratchet_threshold and reuse_rate < 0.2
    )

    return LibraryAnalysis(
        total_artifacts=len(library),
        verified_artifacts=len(library.get_verified()),
        reuse_rate=reuse_rate,
        fresh_creation_rate=fresh_rate,
        verification_rate=verification_rate,
        growth_rate=growth_rate,
        potential_ratchet_failure=potential_ratchet_failure,
        gate_failure_histogram=gate_failure_histogram(library),
        gate_inconclusive_histogram=gate_inconclusive_histogram(library),
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

    if analysis.gate_failure_histogram or analysis.gate_inconclusive_histogram:
        print("\nGate results (faithfulness protocol §4):")
        for gate, count in sorted(analysis.gate_failure_histogram.items()):
            print(f"  FAILED       {gate}: {count}")
        for gate, count in sorted(analysis.gate_inconclusive_histogram.items()):
            print(f"  INCONCLUSIVE {gate}: {count} (routed to D4 review)")

    if analysis.potential_ratchet_failure:
        print("\n[!] WARNING: Ratchet failure detected!")
        print("    Agents are creating from scratch instead of")
        print("    building on accumulated knowledge.")
    elif analysis.total_artifacts < MIN_ARTIFACTS_FOR_RATCHET:
        # Saying accumulation "appears healthy" off one or two artifacts would
        # be the same overclaim in the opposite direction.
        print("\n[.] Library too small to judge accumulation.")
    else:
        print("\n[+] Cultural accumulation appears healthy.")

    print("=" * 50 + "\n")
