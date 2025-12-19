"""Tests for cultural artifact storage and tracking."""

import json
from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType, ArtifactLibrary


class TestArtifact:
    """Tests for Artifact dataclass."""

    def test_create_artifact(self):
        """Artifact holds all required fields."""
        artifact = Artifact(
            id="lemma-001",
            type=ArtifactType.LEMMA,
            natural_language="If n is even, then n^2 is even",
            lean_code="lemma even_sq (n : Nat) (h : Even n) : Even (n * n) := ...",
            verified=True,
            created_by="agent-1",
            generation=1,
        )
        assert artifact.id == "lemma-001"
        assert artifact.type == ArtifactType.LEMMA
        assert artifact.verified is True

    def test_artifact_defaults(self):
        """Artifact has sensible defaults for optional fields."""
        artifact = Artifact(
            id="insight-001",
            type=ArtifactType.INSIGHT,
            natural_language="Try induction on n",
            created_by="agent-2",
            generation=0,
        )
        assert artifact.lean_code is None
        assert artifact.verified is False
        assert artifact.references == []
        assert artifact.referenced_by == []

    def test_artifact_to_dict(self):
        """Artifact can be converted to dictionary."""
        artifact = Artifact(
            id="thm-001",
            type=ArtifactType.THEOREM,
            natural_language="Main theorem",
            lean_code="theorem main : ...",
            verified=True,
            created_by="agent-1",
            generation=2,
            references=["lemma-001"],
        )
        d = artifact.to_dict()
        assert d["id"] == "thm-001"
        assert d["type"] == "theorem"
        assert d["references"] == ["lemma-001"]

    def test_artifact_from_dict(self):
        """Artifact can be created from dictionary."""
        d = {
            "id": "strat-001",
            "type": "strategy",
            "natural_language": "Use contradiction",
            "lean_code": None,
            "verified": False,
            "created_by": "agent-3",
            "generation": 1,
            "references": [],
            "referenced_by": [],
        }
        artifact = Artifact.from_dict(d)
        assert artifact.id == "strat-001"
        assert artifact.type == ArtifactType.STRATEGY


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_artifact_types_exist(self):
        """All expected artifact types exist."""
        assert ArtifactType.LEMMA.value == "lemma"
        assert ArtifactType.THEOREM.value == "theorem"
        assert ArtifactType.INSIGHT.value == "insight"
        assert ArtifactType.STRATEGY.value == "strategy"


class TestArtifactLibrary:
    """Tests for ArtifactLibrary."""

    def test_create_empty_library(self):
        """Library starts empty."""
        library = ArtifactLibrary()
        assert len(library) == 0
        assert library.artifacts == {}

    def test_add_artifact(self):
        """Can add artifacts to library."""
        library = ArtifactLibrary()
        artifact = Artifact(
            id="lemma-001",
            type=ArtifactType.LEMMA,
            natural_language="Test lemma",
            created_by="agent-1",
            generation=0,
        )
        library.add(artifact)
        assert len(library) == 1
        assert "lemma-001" in library

    def test_get_artifact(self):
        """Can retrieve artifact by ID."""
        library = ArtifactLibrary()
        artifact = Artifact(
            id="lemma-001",
            type=ArtifactType.LEMMA,
            natural_language="Test lemma",
            created_by="agent-1",
            generation=0,
        )
        library.add(artifact)
        retrieved = library.get("lemma-001")
        assert retrieved == artifact

    def test_get_missing_artifact_returns_none(self):
        """Getting missing artifact returns None."""
        library = ArtifactLibrary()
        assert library.get("nonexistent") is None

    def test_add_reference(self):
        """Adding reference updates both artifacts."""
        library = ArtifactLibrary()

        lemma = Artifact(
            id="lemma-001",
            type=ArtifactType.LEMMA,
            natural_language="Base lemma",
            created_by="agent-1",
            generation=0,
        )
        theorem = Artifact(
            id="thm-001",
            type=ArtifactType.THEOREM,
            natural_language="Main theorem",
            created_by="agent-2",
            generation=1,
            references=["lemma-001"],
        )

        library.add(lemma)
        library.add(theorem)
        library.add_reference("thm-001", "lemma-001")

        assert "lemma-001" in library.get("thm-001").references
        assert "thm-001" in library.get("lemma-001").referenced_by

    def test_get_by_generation(self):
        """Can filter artifacts by generation."""
        library = ArtifactLibrary()

        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a3", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=1))

        gen0 = library.get_by_generation(0)
        gen1 = library.get_by_generation(1)

        assert len(gen0) == 2
        assert len(gen1) == 1

    def test_get_verified(self):
        """Can filter to only verified artifacts."""
        library = ArtifactLibrary()

        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0, verified=True))
        library.add(Artifact(id="a2", type=ArtifactType.INSIGHT, natural_language="", created_by="", generation=0, verified=False))

        verified = library.get_verified()
        assert len(verified) == 1
        assert verified[0].id == "a1"

    def test_save_and_load(self, tmp_path: Path):
        """Library can be saved to and loaded from JSON."""
        library = ArtifactLibrary()
        library.add(Artifact(
            id="lemma-001",
            type=ArtifactType.LEMMA,
            natural_language="Test lemma",
            lean_code="lemma test : True := trivial",
            verified=True,
            created_by="agent-1",
            generation=0,
        ))

        path = tmp_path / "artifacts.json"
        library.save(path)

        loaded = ArtifactLibrary.load(path)
        assert len(loaded) == 1
        assert loaded.get("lemma-001").natural_language == "Test lemma"

    def test_all_artifacts_list(self):
        """Can get all artifacts as a list."""
        library = ArtifactLibrary()
        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=0))

        all_artifacts = library.all()
        assert len(all_artifacts) == 2

    def test_reuse_count(self):
        """Can count how many artifacts have been reused."""
        library = ArtifactLibrary()

        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["a1"]))
        library.add(Artifact(id="a3", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=1))
        library.add_reference("a2", "a1")

        # a1 is referenced, a2 references something, a3 is standalone
        assert library.reused_artifact_count() == 1  # a1 has been reused

    def test_fresh_creation_count(self):
        """Can count artifacts created without references."""
        library = ArtifactLibrary()

        library.add(Artifact(id="a1", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=0))
        library.add(Artifact(id="a2", type=ArtifactType.THEOREM, natural_language="", created_by="", generation=1, references=["a1"]))
        library.add(Artifact(id="a3", type=ArtifactType.LEMMA, natural_language="", created_by="", generation=1))
        library.add_reference("a2", "a1")

        # a1 and a3 have no references (fresh), a2 references a1
        assert library.fresh_creation_count() == 2
