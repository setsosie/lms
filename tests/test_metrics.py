"""Tests for LMS metrics and analysis."""

import pytest

from lms.artifacts import Artifact, ArtifactType, ArtifactLibrary
from lms.metrics import (
    calculate_reuse_rate,
    calculate_fresh_creation_rate,
    calculate_verification_rate,
    calculate_growth_rate,
    analyze_library,
    LibraryAnalysis,
)
from lms.society import GenerationResult


class TestReuseRate:
    """Tests for reuse rate calculation."""

    def test_reuse_rate_empty_library(self):
        """Empty library has 0 reuse rate."""
        library = ArtifactLibrary()
        assert calculate_reuse_rate(library) == 0.0

    def test_reuse_rate_no_reuse(self):
        """Library with no reuse has 0 rate."""
        library = ArtifactLibrary()
        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        assert calculate_reuse_rate(library) == 0.0

    def test_reuse_rate_full_reuse(self):
        """Library where everything is reused has 1.0 rate."""
        library = ArtifactLibrary()
        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["a1"]))
        library.add_reference("a2", "a1")

        # a1 is referenced, so 1 out of 2 artifacts are reused
        rate = calculate_reuse_rate(library)
        assert rate == 0.5

    def test_reuse_rate_partial(self):
        """Partial reuse gives fractional rate."""
        library = ArtifactLibrary()
        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a3", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["a1"]))
        library.add_reference("a3", "a1")

        # 1 out of 3 is reused
        rate = calculate_reuse_rate(library)
        assert abs(rate - 1/3) < 0.01


class TestFreshCreationRate:
    """Tests for fresh creation rate calculation."""

    def test_fresh_rate_empty(self):
        """Empty library has 0 fresh rate."""
        library = ArtifactLibrary()
        assert calculate_fresh_creation_rate(library) == 0.0

    def test_fresh_rate_all_fresh(self):
        """All artifacts being fresh gives 1.0 rate."""
        library = ArtifactLibrary()
        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        assert calculate_fresh_creation_rate(library) == 1.0

    def test_fresh_rate_none_fresh(self):
        """All artifacts having references gives 0 rate."""
        library = ArtifactLibrary()
        library.add(Artifact(id="seed", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a1", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["seed"]))

        # Only a1 is counted, and it has references. But seed is fresh.
        rate = calculate_fresh_creation_rate(library)
        assert rate == 0.5


class TestVerificationRate:
    """Tests for verification rate calculation."""

    def test_verification_rate_from_results(self):
        """Verification rate calculated from generation results."""
        results = [
            GenerationResult(generation=0, artifacts_created=10, artifacts_verified=5, artifacts_referenced=0, fresh_creations=10),
            GenerationResult(generation=1, artifacts_created=10, artifacts_verified=8, artifacts_referenced=5, fresh_creations=5),
        ]
        rate = calculate_verification_rate(results)
        # 13 verified out of 20 created
        assert abs(rate - 0.65) < 0.01

    def test_verification_rate_empty_results(self):
        """Empty results give 0 rate."""
        assert calculate_verification_rate([]) == 0.0


class TestGrowthRate:
    """Tests for growth rate calculation."""

    def test_growth_rate_from_results(self):
        """Growth rate is artifacts per generation."""
        results = [
            GenerationResult(generation=0, artifacts_created=5, artifacts_verified=0, artifacts_referenced=0, fresh_creations=5),
            GenerationResult(generation=1, artifacts_created=10, artifacts_verified=0, artifacts_referenced=0, fresh_creations=10),
            GenerationResult(generation=2, artifacts_created=15, artifacts_verified=0, artifacts_referenced=0, fresh_creations=15),
        ]
        rate = calculate_growth_rate(results)
        # Average: (10-5 + 15-10) / 2 = 5.0
        assert rate == 5.0

    def test_growth_rate_single_generation(self):
        """Single generation gives 0 growth rate."""
        results = [
            GenerationResult(generation=0, artifacts_created=10, artifacts_verified=0, artifacts_referenced=0, fresh_creations=10),
        ]
        assert calculate_growth_rate(results) == 0.0


class TestLibraryAnalysis:
    """Tests for comprehensive library analysis."""

    def test_analyze_library(self):
        """analyze_library returns comprehensive metrics."""
        library = ArtifactLibrary()
        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0, verified=True))
        library.add(Artifact(id="a2", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["a1"], verified=True))
        library.add(Artifact(id="a3", type=ArtifactType.INSIGHT, natural_language="", created_by="", generation=1))
        library.add_reference("a2", "a1")

        results = [
            GenerationResult(generation=0, artifacts_created=1, artifacts_verified=1, artifacts_referenced=0, fresh_creations=1),
            GenerationResult(generation=1, artifacts_created=2, artifacts_verified=1, artifacts_referenced=1, fresh_creations=1),
        ]

        analysis = analyze_library(library, results)

        assert isinstance(analysis, LibraryAnalysis)
        assert analysis.total_artifacts == 3
        assert analysis.verified_artifacts == 2
        assert analysis.reuse_rate >= 0
        assert analysis.fresh_creation_rate >= 0

    def test_analysis_detects_tasmania_effect(self):
        """Analysis flags potential Tasmania effect."""
        library = ArtifactLibrary()
        # All artifacts are fresh - no reuse
        for i in range(10):
            library.add(Artifact(id=f"a{i}", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=i % 3))

        results = [
            GenerationResult(generation=g, artifacts_created=3, artifacts_verified=0, artifacts_referenced=0, fresh_creations=3)
            for g in range(3)
        ]

        analysis = analyze_library(library, results)

        # With 100% fresh creation and 0 reuse, should flag potential problem
        assert analysis.fresh_creation_rate == 1.0
        assert analysis.reuse_rate == 0.0
        assert analysis.potential_tasmania_effect is True

    def test_analysis_healthy_culture(self):
        """Analysis recognizes healthy cultural accumulation."""
        library = ArtifactLibrary()
        # Foundation
        library.add(Artifact(id="base1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0, verified=True))
        library.add(Artifact(id="base2", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0, verified=True))

        # Build on foundation
        library.add(Artifact(id="derived1", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["base1"], verified=True))
        library.add(Artifact(id="derived2", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["base1", "base2"], verified=True))
        library.add_reference("derived1", "base1")
        library.add_reference("derived2", "base1")
        library.add_reference("derived2", "base2")

        # Further build
        library.add(Artifact(id="advanced", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=2, references=["derived1", "derived2"], verified=True))
        library.add_reference("advanced", "derived1")
        library.add_reference("advanced", "derived2")

        results = [
            GenerationResult(generation=0, artifacts_created=2, artifacts_verified=2, artifacts_referenced=0, fresh_creations=2),
            GenerationResult(generation=1, artifacts_created=2, artifacts_verified=2, artifacts_referenced=2, fresh_creations=0),
            GenerationResult(generation=2, artifacts_created=1, artifacts_verified=1, artifacts_referenced=1, fresh_creations=0),
        ]

        analysis = analyze_library(library, results)

        # Should not flag Tasmania effect
        assert analysis.potential_tasmania_effect is False
        # Should show healthy reuse
        assert analysis.reuse_rate > 0.5
