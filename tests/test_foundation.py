"""Tests for foundation file management - accumulated verified Lean code."""

import pytest
from pathlib import Path

from lms.artifacts import Artifact, ArtifactType
from lms.foundation import FoundationFile, FoundationEntry


class TestFoundationEntry:
    """Tests for FoundationEntry dataclass."""

    def test_create_entry(self):
        """Entry holds artifact metadata and code."""
        entry = FoundationEntry(
            artifact_id="definition-Category-abc123",
            name="Category",
            entry_type="structure",
            signature="structure Category (Obj : Type u) : Type (max u (v + 1))",
            lean_code="structure Category (Obj : Type u) : Type (max u (v + 1)) where\n  Hom : Obj -> Obj -> Type v",
            generation=0,
            author="agent-0-anthropic",
        )
        assert entry.artifact_id == "definition-Category-abc123"
        assert entry.name == "Category"
        assert entry.entry_type == "structure"
        assert "Hom" in entry.lean_code

    def test_entry_to_dict(self):
        """Entry can be serialized to dict."""
        entry = FoundationEntry(
            artifact_id="def-cat-123",
            name="Cat",
            entry_type="structure",
            signature="structure Cat",
            lean_code="structure Cat where",
            generation=0,
            author="agent-0",
        )
        d = entry.to_dict()
        assert d["artifact_id"] == "def-cat-123"
        assert d["name"] == "Cat"

    def test_entry_from_dict(self):
        """Entry can be deserialized from dict."""
        d = {
            "artifact_id": "def-cat-123",
            "name": "Cat",
            "entry_type": "structure",
            "signature": "structure Cat",
            "lean_code": "structure Cat where",
            "generation": 0,
            "author": "agent-0",
        }
        entry = FoundationEntry.from_dict(d)
        assert entry.artifact_id == "def-cat-123"
        assert entry.name == "Cat"


class TestFoundationFile:
    """Tests for FoundationFile - manages accumulated verified Lean code."""

    def test_create_empty_foundation(self, tmp_path: Path):
        """Foundation starts empty."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")
        assert len(foundation) == 0
        assert foundation.entries == []

    def test_add_verified_artifact(self, tmp_path: Path):
        """Can add a verified artifact to foundation."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="definition-Category-abc123",
            type=ArtifactType.DEFINITION,
            natural_language="Definition of a category",
            lean_code="""structure Category (Obj : Type u) : Type (max u (v + 1)) where
  Hom : Obj → Obj → Type v
  id : (X : Obj) → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z""",
            verified=True,
            created_by="agent-0-anthropic",
            generation=0,
        )

        foundation.add_artifact(artifact)

        assert len(foundation) == 1
        assert foundation.entries[0].name == "Category"

    def test_reject_unverified_artifact(self, tmp_path: Path):
        """Cannot add unverified artifacts to foundation."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="definition-Bad-xyz",
            type=ArtifactType.DEFINITION,
            natural_language="Bad definition",
            lean_code="structure Bad where",
            verified=False,  # Not verified!
            created_by="agent-1",
            generation=0,
        )

        with pytest.raises(ValueError, match="only verified"):
            foundation.add_artifact(artifact)

    def test_reject_artifact_without_code(self, tmp_path: Path):
        """Cannot add artifacts without Lean code."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="insight-abc",
            type=ArtifactType.INSIGHT,
            natural_language="Some insight",
            lean_code=None,  # No code!
            verified=True,
            created_by="agent-1",
            generation=0,
        )

        with pytest.raises(ValueError, match="no Lean code"):
            foundation.add_artifact(artifact)

    def test_extract_definitions_from_code(self, tmp_path: Path):
        """Extracts structure/def/theorem names from Lean code."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        code = """
structure Category (Obj : Type u) : Type (max u (v + 1)) where
  Hom : Obj → Obj → Type v

def Category.id (C : Category Obj) (X : Obj) : C.Hom X X := sorry

theorem category_id_unique : ∀ C, unique_id C := sorry
"""
        entries = foundation._extract_entries(code, "art-123", 0, "agent-0")

        names = [e.name for e in entries]
        assert "Category" in names
        assert "Category.id" in names
        assert "category_id_unique" in names

    def test_get_import_statement(self, tmp_path: Path):
        """Returns the import statement for agents to use."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        assert foundation.get_import_statement() == "import LMS.Foundation"

    def test_get_available_definitions(self, tmp_path: Path):
        """Returns summary of what's available for import."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="definition-Category-abc123",
            type=ArtifactType.DEFINITION,
            natural_language="Category definition",
            lean_code="""structure Category (Obj : Type u) where
  Hom : Obj → Obj → Type v
  id : (X : Obj) → Hom X X""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact)

        summary = foundation.get_available_definitions()

        assert "Category" in summary
        assert "structure" in summary.lower() or "Structure" in summary

    def test_save_lean_file(self, tmp_path: Path):
        """Foundation saves to an actual .lean file."""
        lean_path = tmp_path / "LMS" / "Foundation.lean"
        foundation = FoundationFile(lean_path)

        artifact = Artifact(
            id="definition-Cat-123",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="""structure Cat where
  Obj : Type u""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact)
        foundation.save()

        assert lean_path.exists()
        content = lean_path.read_text()
        assert "structure Cat" in content
        assert "import" in content  # Should have imports header

    def test_save_metadata(self, tmp_path: Path):
        """Foundation saves metadata JSON alongside Lean file."""
        lean_path = tmp_path / "LMS" / "Foundation.lean"
        foundation = FoundationFile(lean_path)

        artifact = Artifact(
            id="definition-Cat-123",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact)
        foundation.save()

        metadata_path = tmp_path / "LMS" / "Foundation.json"
        assert metadata_path.exists()

    def test_load_from_metadata(self, tmp_path: Path):
        """Foundation can be loaded from saved metadata."""
        lean_path = tmp_path / "LMS" / "Foundation.lean"
        foundation = FoundationFile(lean_path)

        artifact = Artifact(
            id="definition-Cat-123",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact)
        foundation.save()

        # Load from saved
        loaded = FoundationFile.load(lean_path)

        assert len(loaded) == 1
        assert loaded.entries[0].name == "Cat"

    def test_no_duplicate_artifacts(self, tmp_path: Path):
        """Same artifact ID cannot be added twice."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="definition-Cat-123",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.add_artifact(artifact)  # Add same artifact again

        assert len(foundation) == 1  # Should not duplicate

    def test_multiple_artifacts_accumulate(self, tmp_path: Path):
        """Multiple artifacts accumulate in the foundation."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        cat_artifact = Artifact(
            id="definition-Cat-123",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        functor_artifact = Artifact(
            id="definition-Functor-456",
            type=ArtifactType.DEFINITION,
            natural_language="Functor",
            lean_code="structure CFunctor (C D : Cat) where\n  obj : C.Obj → D.Obj",
            verified=True,
            created_by="agent-1",
            generation=1,
        )

        foundation.add_artifact(cat_artifact)
        foundation.add_artifact(functor_artifact)

        assert len(foundation) == 2
        foundation.save()

        content = foundation.path.read_text()
        assert "Cat" in content
        assert "CFunctor" in content


class TestFoundationContext:
    """Tests for foundation providing context to agents."""

    def test_get_context_for_agent(self, tmp_path: Path):
        """Foundation provides context string for agent prompts."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="definition-Category-abc123",
            type=ArtifactType.DEFINITION,
            natural_language="Category with objects, morphisms, identity, composition",
            lean_code="""structure Category (Obj : Type u) where
  Hom : Obj → Obj → Type v
  id : (X : Obj) → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  id_comp : ∀ f, comp (id _) f = f
  comp_id : ∀ f, comp f (id _) = f
  assoc : ∀ f g h, comp (comp f g) h = comp f (comp g h)""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact)

        context = foundation.get_context_for_agent()

        # Should tell agent how to import
        assert "import LMS.Foundation" in context
        # Should list what's available
        assert "Category" in context
        # Should show structure/signature info
        assert "Hom" in context or "structure" in context.lower()

    def test_empty_foundation_context(self, tmp_path: Path):
        """Empty foundation provides helpful context."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        context = foundation.get_context_for_agent()

        assert "empty" in context.lower() or "no definitions" in context.lower()


class TestFoundationGenerations:
    """Tests simulating foundation evolution over multiple generations."""

    def test_generation_0_category_only(self, tmp_path: Path):
        """Generation 0: Agent defines Category from scratch."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Gen 0 agent creates Category
        category_artifact = Artifact(
            id="definition-Category-gen0",
            type=ArtifactType.DEFINITION,
            natural_language="A category with objects, morphisms, identity, and composition",
            lean_code="""/-
  Generation 0: Category Definition
  Author: agent-0-anthropic
-/

universe u v

structure Cat where
  Obj : Type u
  Hom : Obj → Obj → Type v
  id : (X : Obj) → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  id_comp : ∀ {X Y} (f : Hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y} (f : Hom X Y), comp f (id Y) = f
  assoc : ∀ {W X Y Z} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
    comp (comp f g) h = comp f (comp g h)""",
            verified=True,
            created_by="agent-0-anthropic",
            generation=0,
        )

        foundation.add_artifact(category_artifact)
        foundation.save()

        # Check the Lean file content
        content = foundation.path.read_text()
        assert "structure Cat" in content
        assert "Hom" in content
        assert "id_comp" in content

        # Check context for next generation
        context = foundation.get_context_for_agent()
        assert "Cat" in context
        assert "import LMS.Foundation" in context

    def test_generation_1_functor_builds_on_category(self, tmp_path: Path):
        """Generation 1: Agent imports Category, defines Functor."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # First add Gen 0's category
        category_artifact = Artifact(
            id="definition-Category-gen0",
            type=ArtifactType.DEFINITION,
            natural_language="Category definition",
            lean_code="""universe u v

structure Cat where
  Obj : Type u
  Hom : Obj → Obj → Type v
  id : (X : Obj) → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  id_comp : ∀ {X Y} (f : Hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y} (f : Hom X Y), comp f (id Y) = f
  assoc : ∀ {W X Y Z} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
    comp (comp f g) h = comp f (comp g h)""",
            verified=True,
            created_by="agent-0-anthropic",
            generation=0,
        )
        foundation.add_artifact(category_artifact)

        # Gen 1 agent now adds Functor (building on Cat)
        # NOTE: The agent's code should NOT redefine Cat - it imports it!
        functor_artifact = Artifact(
            id="definition-Functor-gen1",
            type=ArtifactType.DEFINITION,
            natural_language="Functor between categories",
            lean_code="""/-
  Generation 1: Functor Definition
  Author: agent-0-anthropic
  Builds on: Cat from Generation 0
-/

-- Cat is already defined in this file, so we use it directly

structure CFunctor (C D : Cat) where
  obj : C.Obj → D.Obj
  map : {X Y : C.Obj} → C.Hom X Y → D.Hom (obj X) (obj Y)
  map_id : ∀ X, map (C.id X) = D.id (obj X)
  map_comp : ∀ {X Y Z} (f : C.Hom X Y) (g : C.Hom Y Z),
    map (C.comp f g) = D.comp (map f) (map g)""",
            verified=True,
            created_by="agent-0-anthropic",
            generation=1,
        )
        foundation.add_artifact(functor_artifact)
        foundation.save()

        # Foundation now has both
        assert len(foundation) == 2

        content = foundation.path.read_text()
        assert "Cat" in content
        assert "CFunctor" in content

        # Context shows both are available
        context = foundation.get_context_for_agent()
        assert "Cat" in context
        assert "CFunctor" in context

    def test_generation_2_natural_transformation(self, tmp_path: Path):
        """Generation 2: Agent defines NatTrans using Cat and CFunctor."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Add Gen 0 and Gen 1 artifacts
        foundation.add_artifact(Artifact(
            id="def-Cat-gen0",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="""universe u v
structure Cat where
  Obj : Type u
  Hom : Obj → Obj → Type v
  id : (X : Obj) → Hom X X
  comp : {X Y Z : Obj} → Hom X Y → Hom Y Z → Hom X Z
  id_comp : ∀ {X Y} (f : Hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y} (f : Hom X Y), comp f (id Y) = f
  assoc : ∀ {W X Y Z} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
    comp (comp f g) h = comp f (comp g h)""",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        foundation.add_artifact(Artifact(
            id="def-Functor-gen1",
            type=ArtifactType.DEFINITION,
            natural_language="Functor",
            lean_code="""structure CFunctor (C D : Cat) where
  obj : C.Obj → D.Obj
  map : {X Y : C.Obj} → C.Hom X Y → D.Hom (obj X) (obj Y)
  map_id : ∀ X, map (C.id X) = D.id (obj X)
  map_comp : ∀ {X Y Z} (f : C.Hom X Y) (g : C.Hom Y Z),
    map (C.comp f g) = D.comp (map f) (map g)""",
            verified=True,
            created_by="agent-1",
            generation=1,
        ))

        # Gen 2: NatTrans
        foundation.add_artifact(Artifact(
            id="def-NatTrans-gen2",
            type=ArtifactType.DEFINITION,
            natural_language="Natural transformation between functors",
            lean_code="""/-
  Generation 2: Natural Transformation
  Builds on: Cat (gen0), CFunctor (gen1)
-/

structure NatTrans {C D : Cat} (F G : CFunctor C D) where
  app : (X : C.Obj) → D.Hom (F.obj X) (G.obj X)
  naturality : ∀ {X Y} (f : C.Hom X Y),
    D.comp (F.map f) (app Y) = D.comp (app X) (G.map f)""",
            verified=True,
            created_by="agent-2",
            generation=2,
        ))

        foundation.save()

        # All three definitions accumulated
        assert len(foundation) == 3

        content = foundation.path.read_text()
        assert "Cat" in content
        assert "CFunctor" in content
        assert "NatTrans" in content
        assert "naturality" in content

    def test_full_yoneda_evolution(self, tmp_path: Path):
        """Full simulation: 4 generations building toward Yoneda."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Gen 0: Category
        foundation.add_artifact(Artifact(
            id="def-Cat-gen0",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u\n  Hom : Obj → Obj → Type v",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        # Gen 1: Functor
        foundation.add_artifact(Artifact(
            id="def-Functor-gen1",
            type=ArtifactType.DEFINITION,
            natural_language="Functor",
            lean_code="structure CFunctor (C D : Cat) where\n  obj : C.Obj → D.Obj",
            verified=True,
            created_by="agent-1",
            generation=1,
        ))

        # Gen 2: NatTrans + Opposite
        foundation.add_artifact(Artifact(
            id="def-NatTrans-gen2",
            type=ArtifactType.DEFINITION,
            natural_language="Natural transformation",
            lean_code="structure NatTrans {C D : Cat} (F G : CFunctor C D) where\n  app : (X : C.Obj) → D.Hom (F.obj X) (G.obj X)",
            verified=True,
            created_by="agent-0",
            generation=2,
        ))

        foundation.add_artifact(Artifact(
            id="def-OpCat-gen2",
            type=ArtifactType.DEFINITION,
            natural_language="Opposite category",
            lean_code="def Cat.op (C : Cat) : Cat where\n  Obj := C.Obj\n  Hom X Y := C.Hom Y X",
            verified=True,
            created_by="agent-1",
            generation=2,
        ))

        # Gen 3: HomFunctor + Yoneda
        foundation.add_artifact(Artifact(
            id="def-HomFunctor-gen3",
            type=ArtifactType.DEFINITION,
            natural_language="Hom functor",
            lean_code="def HomFunctor (C : Cat) (X : C.Obj) : CFunctor C.op TypeCat where",
            verified=True,
            created_by="agent-0",
            generation=3,
        ))

        foundation.add_artifact(Artifact(
            id="thm-Yoneda-gen3",
            type=ArtifactType.THEOREM,
            natural_language="Yoneda lemma",
            lean_code="theorem yoneda {C : Cat} (F : CFunctor C.op TypeCat) (X : C.Obj) :\n  NatTrans (HomFunctor C X) F ≃ F.obj X := sorry",
            verified=True,
            created_by="agent-2",
            generation=3,
        ))

        foundation.save()

        # Check evolution
        assert len(foundation) == 6

        # Verify generations tracked
        gen_counts = {}
        for entry in foundation.entries:
            gen = entry.generation
            gen_counts[gen] = gen_counts.get(gen, 0) + 1

        assert gen_counts[0] == 1  # Cat
        assert gen_counts[1] == 1  # Functor
        assert gen_counts[2] == 2  # NatTrans, OpCat
        assert gen_counts[3] == 2  # HomFunctor, Yoneda

        # Final foundation file has everything
        content = foundation.path.read_text()
        assert "Cat" in content
        assert "CFunctor" in content
        assert "NatTrans" in content
        assert "Cat.op" in content
        assert "HomFunctor" in content
        assert "yoneda" in content

    def test_context_grows_each_generation(self, tmp_path: Path):
        """Agent context grows richer with each generation."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Gen 0: Empty context
        context_0 = foundation.get_context_for_agent()
        assert "empty" in context_0.lower() or "no definitions" in context_0.lower()

        # Add Category
        foundation.add_artifact(Artifact(
            id="def-Cat",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u\n  Hom : Obj → Obj → Type v",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        # Gen 1 context has Category
        context_1 = foundation.get_context_for_agent()
        assert "Cat" in context_1
        assert "Obj" in context_1 or "structure" in context_1.lower()

        # Add Functor
        foundation.add_artifact(Artifact(
            id="def-Functor",
            type=ArtifactType.DEFINITION,
            natural_language="Functor",
            lean_code="structure CFunctor (C D : Cat) where\n  obj : C.Obj → D.Obj",
            verified=True,
            created_by="agent-1",
            generation=1,
        ))

        # Gen 2 context has both
        context_2 = foundation.get_context_for_agent()
        assert "Cat" in context_2
        assert "CFunctor" in context_2
        # Context grew
        assert len(context_2) > len(context_1)


class TestFoundationCodeCleaning:
    """Tests for cleaning artifact code before adding to foundation.

    These tests cover bug fixes for malformed code from LLM outputs.
    """

    def test_strips_yaml_pipe_prefix(self, tmp_path: Path):
        """Foundation strips YAML multiline '|' prefix from code."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Agent output with YAML multiline marker
        artifact = Artifact(
            id="def-test",
            type=ArtifactType.DEFINITION,
            natural_language="Test",
            lean_code="|\n  theorem test : 1 + 1 = 2 := by rfl",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        # Should NOT have the pipe character
        assert "|\n" not in content
        # Should have the actual theorem
        assert "theorem test" in content

    def test_strips_yaml_pipe_with_extra_whitespace(self, tmp_path: Path):
        """Foundation handles various YAML pipe formats."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="def-test",
            type=ArtifactType.DEFINITION,
            natural_language="Test",
            lean_code="|\n\n\n  lemma foo : True := trivial",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()
        assert "|\n" not in content
        assert "lemma foo" in content

    def test_removes_embedded_imports(self, tmp_path: Path):
        """Foundation removes import statements from artifact code."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Agent code that includes its own import
        artifact = Artifact(
            id="def-test",
            type=ArtifactType.DEFINITION,
            natural_language="Test",
            lean_code="""import Mathlib.Data.Nat.Basic

lemma div_self_eq_one (n : ℕ) (h : n ≠ 0) : n / n = 1 :=
  Nat.div_self (Nat.pos_of_ne_zero h)""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        # Count imports - should only have header imports, not embedded ones
        import_lines = [l for l in content.split('\n') if l.strip().startswith('import ')]
        # All imports should be before namespace
        namespace_pos = content.find('namespace LMS.Foundation')
        for line in content.split('\n'):
            if line.strip().startswith('import '):
                assert content.find(line) < namespace_pos, "Import found after namespace"

    def test_removes_multiple_embedded_imports(self, tmp_path: Path):
        """Foundation removes all import statements from artifact code."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="def-test",
            type=ArtifactType.DEFINITION,
            natural_language="Test",
            lean_code="""import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Tactic.Ring

lemma test : True := trivial""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        # The lemma should still be there
        assert "lemma test" in content

        # No imports after the namespace
        namespace_pos = content.find('namespace LMS.Foundation')
        after_namespace = content[namespace_pos:]
        assert 'import ' not in after_namespace

    def test_handles_both_pipe_and_imports(self, tmp_path: Path):
        """Foundation handles code with both YAML pipe and embedded imports."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="def-test",
            type=ArtifactType.DEFINITION,
            natural_language="Test",
            lean_code="""|\n  import Mathlib.Data.Nat.Basic\n  \n  theorem test : 1 = 1 := rfl""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        assert "|\n" not in content
        assert "theorem test" in content
        # Import should be stripped from the artifact section
        namespace_pos = content.find('namespace LMS.Foundation')
        after_namespace = content[namespace_pos:]
        assert 'import Mathlib.Data.Nat.Basic' not in after_namespace

    def test_removes_universe_declarations(self, tmp_path: Path):
        """Foundation removes universe declarations from artifact code."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="def-test",
            type=ArtifactType.DEFINITION,
            natural_language="Test",
            lean_code="""universe u v w

structure Cat where
  Obj : Type u""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        # Header should have universe declarations
        assert "universe u v w" in content

        # But artifact section should not have duplicate
        namespace_pos = content.find('namespace LMS.Foundation')
        after_namespace = content[namespace_pos:]
        # Count universe lines after namespace - should be 0
        universe_lines_after = [l for l in after_namespace.split('\n') if l.strip().startswith('universe ')]
        assert len(universe_lines_after) == 0, f"Found universe declarations after namespace: {universe_lines_after}"

    def test_extracts_indented_definitions(self, tmp_path: Path):
        """Foundation extracts definitions with leading whitespace."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Code with indented definitions (as produced by LLMs in YAML)
        artifact = Artifact(
            id="def-cat-indented",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="""  /-- A category -/
  structure Category where
    Obj : Type u
    Hom : Obj → Obj → Type v

  def Category.id (C : Category) (x : C.Obj) : C.Hom x x := sorry""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)

        # Should have extracted Category and id definitions
        names = [e.name for e in foundation.entries]
        assert "Category" in names, f"Expected Category in {names}"
        assert "Category.id" in names, f"Expected Category.id in {names}"

    def test_ignores_definitions_inside_block_comments(self, tmp_path: Path):
        """Foundation ignores definitions that appear inside /- ... -/ comments."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Code with a structure definition INSIDE a comment block (common LLM pattern)
        artifact = Artifact(
            id="def-with-hint",
            type=ArtifactType.DEFINITION,
            natural_language="NatTrans with hint",
            lean_code="""/-
  HINT: For opposite categories, you may want to define:

  structure Op (α : Type*) where
    unop : α

  This lets you distinguish morphisms in C from Cᵒᵖ.
-/

structure NatTrans where
  app : Type u

def NatTrans.identity : NatTrans := { app := Nat }""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)

        # Should NOT have extracted Op (it's inside a comment)
        # SHOULD have extracted NatTrans and NatTrans.identity
        names = [e.name for e in foundation.entries]
        assert "Op" not in names, f"Op should not be extracted from comment, got {names}"
        assert "NatTrans" in names, f"Expected NatTrans in {names}"
        assert "NatTrans.identity" in names, f"Expected NatTrans.identity in {names}"

    def test_removes_namespace_and_end_statements(self, tmp_path: Path):
        """Foundation removes namespace/end statements that conflict with wrapper."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # Code with namespace that would break the Foundation namespace
        artifact = Artifact(
            id="def-with-namespace",
            type=ArtifactType.DEFINITION,
            natural_language="Category with namespace",
            lean_code="""namespace Category

structure Cat where
  Obj : Type u

theorem cat_id : True := trivial

end Category""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        # namespace/end statements should be stripped
        assert "namespace Category" not in content
        assert "end Category" not in content

        # But the actual definitions should be there
        assert "structure Cat" in content
        assert "theorem cat_id" in content

    def test_first_mover_wins_for_core_concepts(self, tmp_path: Path):
        """Foundation only accepts first structure for core concepts (Category/Cat are same concept)."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # First agent defines Category (bundled style)
        artifact1 = Artifact(
            id="def-cat-agent0",
            type=ArtifactType.DEFINITION,
            natural_language="Category v1",
            lean_code="""structure Category where
  Obj : Type u
  Hom : Obj → Obj → Type v

def Category.helper : Nat := 42""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact1)

        # Second agent defines Cat (DIFFERENT NAME, same concept)
        # This should be REJECTED because "category" concept is already claimed
        artifact2 = Artifact(
            id="def-cat-agent1",
            type=ArtifactType.DEFINITION,
            natural_language="Category v2 (different name)",
            lean_code="""structure Cat (Obj : Type u) where
  Hom : Obj → Obj → Type v

def Cat.other_helper : Nat := 99""",
            verified=True,
            created_by="agent-1",
            generation=0,
        )
        foundation.add_artifact(artifact2)

        names = [e.name for e in foundation.entries]

        # Category should be kept (first-mover)
        assert "Category" in names

        # Cat should be REJECTED even though it's a different name
        # because Category and Cat map to the same "category" concept
        assert "Cat" not in names, f"Cat should be rejected (category concept already claimed)"

        # Non-structure defs (Cat.other_helper) should also be excluded since
        # the parent structure was rejected
        assert "Cat.other_helper" not in names

        # But Category.helper should be there (from accepted artifact)
        assert "Category.helper" in names

    def test_deduplicates_definitions_by_name(self, tmp_path: Path):
        """Foundation skips definitions with names that already exist."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        # First artifact defines Category
        artifact1 = Artifact(
            id="def-cat-gen0",
            type=ArtifactType.DEFINITION,
            natural_language="Category v1",
            lean_code="""structure Category where
  Obj : Type u""",
            verified=True,
            created_by="agent-0",
            generation=0,
        )
        foundation.add_artifact(artifact1)

        # Second artifact ALSO defines Category (agent included all definitions)
        artifact2 = Artifact(
            id="def-functor-gen1",
            type=ArtifactType.DEFINITION,
            natural_language="Functor (includes Category)",
            lean_code="""structure Category where
  Obj : Type u
  Hom : Obj → Obj → Type v

structure Functor (C D : Category) where
  obj : C.Obj → D.Obj""",
            verified=True,
            created_by="agent-1",
            generation=1,
        )
        foundation.add_artifact(artifact2)
        foundation.save()

        content = foundation.path.read_text()

        # Category should only appear once (from gen 0)
        category_count = content.count("structure Category")
        assert category_count == 1, f"Expected 1 Category definition, found {category_count}"

        # But Functor should be there
        assert "structure Functor" in content


class TestFoundationCompilation:
    """Tests for foundation compilation verification.

    These tests ensure that the foundation file compiles correctly
    before being saved, preventing broken foundations.
    """

    def test_foundation_generates_valid_lean_structure(self, tmp_path: Path):
        """Foundation generates syntactically correct Lean structure."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        foundation.add_artifact(Artifact(
            id="def-Cat",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0",
            generation=0,
        ))

        foundation.save()

        content = foundation.path.read_text()

        # Should have proper Lean structure
        assert "namespace LMS.Foundation" in content
        assert "end LMS.Foundation" in content
        assert "import" in content

    def test_foundation_includes_artifact_provenance(self, tmp_path: Path):
        """Foundation includes comments about where each artifact came from."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        foundation.add_artifact(Artifact(
            id="def-Cat-abc123",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0-anthropic",
            generation=0,
        ))

        foundation.save()

        content = foundation.path.read_text()

        # Should show artifact ID and author
        assert "def-Cat-abc123" in content
        assert "agent-0-anthropic" in content
        assert "gen 0" in content

    def test_foundation_avoids_duplicate_definitions(self, tmp_path: Path):
        """Foundation doesn't duplicate artifacts with same ID."""
        foundation = FoundationFile(tmp_path / "LMS" / "Foundation.lean")

        artifact = Artifact(
            id="def-Cat",
            type=ArtifactType.DEFINITION,
            natural_language="Category",
            lean_code="structure Cat where\n  Obj : Type u",
            verified=True,
            created_by="agent-0",
            generation=0,
        )

        # Add same artifact twice
        foundation.add_artifact(artifact)
        foundation.add_artifact(artifact)
        foundation.save()

        content = foundation.path.read_text()

        # Should only have one definition
        assert content.count("structure Cat") == 1
