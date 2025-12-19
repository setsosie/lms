"""Tests for the goals module."""

import json
import tempfile
from pathlib import Path

import pytest

from lms.goals import (
    Goal,
    StacksDefinition,
    get_goal,
    list_goals,
    validate_imports,
    ALLOWED_IMPORTS_FOUNDATION,
    FORBIDDEN_IMPORTS,
)


class TestStacksDefinition:
    """Tests for StacksDefinition dataclass."""

    def test_create_definition(self):
        """Test basic definition creation."""
        defn = StacksDefinition(
            tag="0013",
            section="4.2",
            name="Definition: Category",
            content="A category consists of...",
        )
        assert defn.tag == "0013"
        assert defn.section == "4.2"
        assert defn.name == "Definition: Category"
        assert defn.formalized is False
        assert defn.artifact_ids == []

    def test_definition_defaults(self):
        """Test default values."""
        defn = StacksDefinition(
            tag="test",
            section="1.0",
            name="Test",
            content="...",
        )
        assert defn.formalized is False
        assert defn.artifact_ids == []


class TestGoal:
    """Tests for Goal class."""

    def test_create_goal(self):
        """Test basic goal creation."""
        goal = Goal(
            name="Test Goal",
            description="A test goal",
            source="Test Source",
        )
        assert goal.name == "Test Goal"
        assert goal.description == "A test goal"
        assert goal.definitions == []

    def test_progress_empty(self):
        """Test progress with no definitions."""
        goal = Goal(name="Test", description="...", source="...")
        assert goal.progress() == 0.0

    def test_progress_none_formalized(self):
        """Test progress with no formalized definitions."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="A", section="1", name="A", content="..."),
                StacksDefinition(tag="B", section="2", name="B", content="..."),
            ],
        )
        assert goal.progress() == 0.0

    def test_progress_partial(self):
        """Test progress with some formalized."""
        defn1 = StacksDefinition(tag="A", section="1", name="A", content="...")
        defn1.formalized = True
        defn2 = StacksDefinition(tag="B", section="2", name="B", content="...")
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[defn1, defn2],
        )
        assert goal.progress() == 0.5

    def test_progress_all_formalized(self):
        """Test progress with all formalized."""
        defn1 = StacksDefinition(tag="A", section="1", name="A", content="...")
        defn1.formalized = True
        defn2 = StacksDefinition(tag="B", section="2", name="B", content="...")
        defn2.formalized = True
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[defn1, defn2],
        )
        assert goal.progress() == 1.0

    def test_mark_formalized(self):
        """Test marking a definition as formalized."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="0013", section="1", name="A", content="..."),
                StacksDefinition(tag="0014", section="2", name="B", content="..."),
            ],
        )
        goal.mark_formalized("0013", "artifact-123")

        assert goal.definitions[0].formalized is True
        assert "artifact-123" in goal.definitions[0].artifact_ids
        assert goal.definitions[1].formalized is False

    def test_mark_formalized_strips_quotes(self):
        """Test that mark_formalized strips quotes from tags.

        This is critical because agents often output stacks_tag: "0019"
        with quotes, but goal definitions have bare tags like 0019.
        """
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="0019", section="1", name="Yoneda", content="..."),
            ],
        )

        # Agent outputs with double quotes
        goal.mark_formalized('"0019"', "artifact-yoneda-1")
        assert goal.definitions[0].formalized is True
        assert "artifact-yoneda-1" in goal.definitions[0].artifact_ids

    def test_mark_formalized_strips_single_quotes(self):
        """Test that single quotes are also stripped."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="001P", section="1", name="Yoneda FF", content="..."),
            ],
        )

        # Agent outputs with single quotes
        goal.mark_formalized("'001P'", "artifact-ff-1")
        assert goal.definitions[0].formalized is True

    def test_mark_formalized_nonexistent_tag(self):
        """Test marking a non-existent tag does nothing."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="0013", section="1", name="A", content="..."),
            ],
        )
        goal.mark_formalized("NONEXISTENT", "artifact-123")

        assert goal.definitions[0].formalized is False

    def test_to_prompt_context(self):
        """Test generating prompt context."""
        goal = Goal(
            name="Test Goal",
            description="Test",
            source="Test Source",
            definitions=[
                StacksDefinition(tag="A", section="1", name="Def A", content="Content A"),
            ],
        )
        context = goal.to_prompt_context()

        assert "Test Goal" in context
        assert "Test Source" in context
        assert "[TODO]" in context
        assert "Def A" in context
        assert "Content A" in context

    def test_save_and_load(self):
        """Test saving and loading goals."""
        goal = Goal(
            name="Test",
            description="Desc",
            source="Source",
            definitions=[
                StacksDefinition(tag="A", section="1", name="A", content="..."),
            ],
        )
        goal.mark_formalized("A", "artifact-1")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goal.json"
            goal.save(path)

            loaded = Goal.load(path)
            assert loaded.name == "Test"
            assert loaded.definitions[0].formalized is True
            assert "artifact-1" in loaded.definitions[0].artifact_ids


class TestGoalRegistry:
    """Tests for goal registry functions."""

    def test_list_goals(self):
        """Test listing available goals."""
        goals = list_goals()
        assert isinstance(goals, list)
        assert len(goals) > 0
        assert "stacks-ch4-milestones" in goals

    def test_get_goal_valid(self):
        """Test getting a valid goal."""
        goal = get_goal("stacks-ch4-milestones")
        assert goal.name == "Stacks Project Chapter 4: Milestone Theorems"
        assert len(goal.definitions) > 0

    def test_get_goal_invalid(self):
        """Test getting an invalid goal raises error."""
        with pytest.raises(ValueError) as exc:
            get_goal("nonexistent-goal")
        assert "Unknown goal" in str(exc.value)

    def test_from_scratch_goal_exists(self):
        """Test the from-scratch goal is registered."""
        goals = list_goals()
        assert "stacks-ch4-scratch" in goals

    def test_from_scratch_goal_has_restrictions(self):
        """Test the from-scratch goal has import restrictions."""
        goal = get_goal("stacks-ch4-scratch")
        assert goal.forbidden_imports is not None
        assert "Mathlib.CategoryTheory" in goal.forbidden_imports
        assert goal.allowed_imports is not None
        assert goal.preamble is not None


class TestImportValidation:
    """Tests for import validation functionality."""

    def test_validate_imports_no_restrictions(self):
        """Test validation with no restrictions allows everything."""
        code = """
import Mathlib.CategoryTheory.Category.Basic
import Anything.Else
"""
        valid, error = validate_imports(code)
        assert valid is True
        assert error is None

    def test_validate_imports_forbidden_detected(self):
        """Test forbidden imports are detected."""
        code = """
import Mathlib.CategoryTheory.Yoneda
import Mathlib.Logic.Basic
"""
        valid, error = validate_imports(code, forbidden=["Mathlib.CategoryTheory"])
        assert valid is False
        assert "Forbidden import" in error
        assert "Mathlib.CategoryTheory.Yoneda" in error

    def test_validate_imports_allowed_only(self):
        """Test only allowed imports pass."""
        code = """
import Mathlib.Logic.Basic
import Mathlib.Tactic.Common
"""
        allowed = ["Mathlib.Logic", "Mathlib.Tactic"]
        valid, error = validate_imports(code, allowed=allowed)
        assert valid is True

    def test_validate_imports_not_in_allowed_list(self):
        """Test imports not in allowed list are rejected."""
        code = """
import Mathlib.Logic.Basic
import Mathlib.CategoryTheory.Category.Basic
"""
        allowed = ["Mathlib.Logic", "Mathlib.Tactic"]
        valid, error = validate_imports(code, allowed=allowed)
        assert valid is False
        assert "not in allowed list" in error

    def test_validate_imports_forbidden_takes_precedence(self):
        """Test forbidden imports are caught even if prefix matches allowed."""
        code = """
import Mathlib.CategoryTheory.Functor.Basic
"""
        allowed = ["Mathlib"]  # Everything would be allowed
        forbidden = ["Mathlib.CategoryTheory"]  # But this is forbidden
        valid, error = validate_imports(code, allowed=allowed, forbidden=forbidden)
        assert valid is False
        assert "Forbidden import" in error

    def test_validate_imports_empty_code(self):
        """Test validation on code with no imports."""
        code = """
def foo : Nat := 42
"""
        valid, error = validate_imports(code, forbidden=["Mathlib.CategoryTheory"])
        assert valid is True

    def test_goal_validate_code(self):
        """Test Goal.validate_code method."""
        goal = Goal(
            name="Test",
            description="Test",
            source="Test",
            forbidden_imports=["Mathlib.CategoryTheory"],
        )
        code = "import Mathlib.CategoryTheory.Yoneda"
        valid, error = goal.validate_code(code)
        assert valid is False
        assert "Forbidden" in error

    def test_from_scratch_goal_rejects_category_theory(self):
        """Test from-scratch goal rejects Mathlib.CategoryTheory imports."""
        goal = get_goal("stacks-ch4-scratch")
        code = """
import Mathlib.Tactic.Common
import Mathlib.CategoryTheory.Category.Basic  -- cheating!
"""
        valid, error = goal.validate_code(code)
        assert valid is False
        assert "Mathlib.CategoryTheory" in error

    def test_from_scratch_goal_accepts_valid_imports(self):
        """Test from-scratch goal accepts valid foundation imports."""
        goal = get_goal("stacks-ch4-scratch")
        code = """
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic
import Mathlib.Data.Set.Basic
"""
        valid, error = goal.validate_code(code)
        assert valid is True

    def test_from_scratch_goal_rejects_opposite_import(self):
        """Test from-scratch goal rejects Mathlib.Data.Opposite import."""
        goal = get_goal("stacks-ch4-scratch")
        code = """
import Mathlib.Tactic.Common
import Mathlib.Data.Opposite  -- should define this yourself!
"""
        valid, error = goal.validate_code(code)
        # Data.Opposite is not in allowed list
        assert valid is False

    def test_prompt_context_includes_restrictions(self):
        """Test that prompt context shows import restrictions."""
        goal = get_goal("stacks-ch4-scratch")
        context = goal.to_prompt_context()
        assert "FORBIDDEN IMPORTS" in context
        assert "CategoryTheory" in context
        assert "ALLOWED IMPORTS" in context
