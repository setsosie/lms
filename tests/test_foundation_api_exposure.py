"""Agents must be shown the *shape* of a foundation entry (26Q3-HARN-11).

Found on the box 2026-08-10 in `experiments/shakedown_3x3_d`, the first run to
produce cross-generational reuse. Gen 2 cited `definition-Category-d3a579da`,
the import elaborated, and the attempt then died on API shape:

    error: type expected, got (Category : Type v → Type (max v (u + 1)))

The agent used `Category` as a bare type. The foundation's `Category` is
indexed by an object type. Not Lean 3 syntax, not a missing Mathlib -- the
agent could not see the shape of the thing it was importing.
"""

from __future__ import annotations

from pathlib import Path

from lms.artifacts import Artifact, ArtifactType
from lms.foundation import FoundationFile
from lms.lean.interface import VerificationStatus
from lms.providers.base import ProviderConfig
from lms.society import Society
from tests.test_society import MockProvider

CATEGORY_SRC = (
    "structure Category (obj : Type u) where\n"
    "  Hom : obj → obj → Type v\n"
    "  id : (X : obj) → Hom X X\n"
    "  comp : {X Y Z : obj} → Hom X Y → Hom Y Z → Hom X Z\n"
    "  id_l : True\n"
    "  id_r : True\n"
    "  assoc : True"
)


def _foundation_with(
    path: Path, lean_code: str, name: str = "Category"
) -> FoundationFile:
    foundation = FoundationFile(path)
    foundation.add_artifact(
        Artifact(
            id=f"definition-{name}-d3a579da",
            type=ArtifactType.DEFINITION,
            natural_language=f"the {name} declaration",
            lean_code=lean_code,
            status=VerificationStatus.VERIFIED_LEAN,
            created_by="agent-0",
            generation=0,
        )
    )
    return foundation


def _foundation_with_category(path: Path) -> FoundationFile:
    return _foundation_with(path, CATEGORY_SRC)


# --- What the per-agent context renders -------------------------------------


def test_declaration_header_is_rendered_exactly_once(tmp_path: Path) -> None:
    """`signature` already carries `<type> <name>`; prepending them repeats it."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    assert "structure Categorystructure Category" not in context
    assert context.count("structure Category") == 1


def test_declaration_header_reproduces_the_source_line(tmp_path: Path) -> None:
    """The parameter is why the agent failed -- it must survive verbatim."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    assert "structure Category (obj : Type u) where" in context


def test_structure_fields_carry_their_types(tmp_path: Path) -> None:
    """A bare `Hom` does not tell an agent how to apply it."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    assert "Hom : obj → obj → Type v" in context
    assert "id : (X : obj) → Hom X X" in context


def test_no_structure_field_is_silently_dropped(tmp_path: Path) -> None:
    """`field_match[:5]` amputates the 6th field with no marker."""
    context = _foundation_with_category(
        tmp_path / "Foundation.lean"
    ).get_context_for_agent()

    for field in ("Hom", "id", "comp", "id_l", "id_r", "assoc"):
        assert field in context, f"field {field!r} never reached the agent"


# --- What committee mode renders (strictly weaker today) --------------------


def _society_with_category(tmp_path: Path) -> Society:
    society = Society(
        n_agents=1,
        provider=MockProvider(ProviderConfig(api_key="test", model="test")),
        foundation_path=tmp_path / "LMS" / "Foundation.lean",
    )
    society.foundation = _foundation_with_category(tmp_path / "LMS" / "Foundation.lean")
    return society


def test_committee_summary_survives_a_nonempty_foundation(tmp_path: Path) -> None:
    """`entries.values()` on a list, `entry.tag` on a tag-less dataclass."""
    summary = _society_with_category(tmp_path)._get_foundation_summary()

    assert "Category" in summary


def test_committee_summary_shows_shape_not_just_names(tmp_path: Path) -> None:
    """A work committee is told less than the agent that already failed."""
    summary = _society_with_category(tmp_path)._get_foundation_summary()

    assert "(obj : Type u)" in summary


# --- Comments are not the declaration ---------------------------------------
#
# `_extract_entries` blanks block comments before matching but slices the
# *original* source, and DEFINITION_PATTERN's leading `^\s*` reaches back across
# the blanked region -- so `lean_code` begins at the doc comment, not at the
# declaration. 425 of 870 corpus artifacts carry one.


DOC_COMMENTED_CATEGORY = f"/-- A category, in the usual sense. -/\n{CATEGORY_SRC}"


def test_doc_comment_does_not_become_the_header(tmp_path: Path) -> None:
    """The regression: agents were shown `/-- A category ... -/` and nothing else."""
    context = _foundation_with(
        tmp_path / "Foundation.lean", DOC_COMMENTED_CATEGORY
    ).get_context_for_agent()

    assert "structure Category (obj : Type u) where" in context


def test_doc_commented_declaration_still_shows_its_fields(tmp_path: Path) -> None:
    """`field_lines()` broke on the same input: comment as header, then break."""
    context = _foundation_with(
        tmp_path / "Foundation.lean", DOC_COMMENTED_CATEGORY
    ).get_context_for_agent()

    assert "Hom : obj → obj → Type v" in context
    assert "assoc : True" in context


def test_line_comment_does_not_become_the_header(tmp_path: Path) -> None:
    context = _foundation_with(
        tmp_path / "Foundation.lean", f"-- the objects are a parameter\n{CATEGORY_SRC}"
    ).get_context_for_agent()

    assert "structure Category (obj : Type u) where" in context


# --- Declarations that are not `structure` ----------------------------------


def test_class_fields_are_rendered(tmp_path: Path) -> None:
    """Lean 4 algebra is written with `class`; the render gate said `structure`."""
    src = "class Mon (a : Type u) where\n  mul : a → a → a\n  one : a"
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="Mon"
    ).get_context_for_agent()

    assert "mul : a → a → a" in context
    assert "one : a" in context


def test_inductive_constructors_are_rendered(tmp_path: Path) -> None:
    src = "inductive Tree (a : Type u) where\n  | leaf : Tree a\n  | node : Tree a → Tree a"
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="Tree"
    ).get_context_for_agent()

    assert "| leaf : Tree a" in context
    assert "| node : Tree a → Tree a" in context


# --- A theorem's statement is its API ---------------------------------------


WRAPPED_THEOREM = (
    "theorem yoneda_computation {C : Type u} (F : C → Type v)\n"
    "    (X : C) :\n"
    "    F X = F X := by\n"
    "  exact rfl_marker_do_not_render"
)


def test_theorem_statement_survives_a_line_break(tmp_path: Path) -> None:
    """Rendering line 1 only leaves the agent half the binders and no conclusion."""
    context = _foundation_with(
        tmp_path / "Foundation.lean", WRAPPED_THEOREM, name="yoneda"
    ).get_context_for_agent()

    assert "(X : C) :" in context
    assert "F X = F X" in context


def test_proof_body_is_not_rendered(tmp_path: Path) -> None:
    """The statement is the API; the proof is noise that grows without bound."""
    context = _foundation_with(
        tmp_path / "Foundation.lean", WRAPPED_THEOREM, name="yoneda"
    ).get_context_for_agent()

    assert "rfl_marker_do_not_render" not in context


def test_proof_body_on_the_header_line_is_not_rendered(tmp_path: Path) -> None:
    """`:= by` on line 1 means no continuation line carries `:=` to stop at."""
    src = (
        "theorem add_comm2 (a b : Nat) : a + b = b + a := by\n"
        "  induction a with\n"
        "  | zero => simp [tactic_marker_do_not_render]"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="addcomm"
    ).get_context_for_agent()

    assert "tactic_marker_do_not_render" not in context


def test_named_argument_assign_does_not_cut_the_statement(tmp_path: Path) -> None:
    """Lean 4 named args put `:=` *inside* the statement; a blind find() amputates it."""
    src = (
        "theorem pullback_commutes {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :\n"
        "    pullback.fst (f := f) (g := g) ≫ f = pullback.snd ≫ g :=\n"
        "  pullback.condition"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="pullback"
    ).get_context_for_agent()

    assert "pullback.fst (f := f) (g := g) ≫ f = pullback.snd ≫ g" in context
    assert "pullback.condition" not in context


def test_instance_with_a_where_body_renders_all_its_fields(tmp_path: Path) -> None:
    """`instance … where` is a body-is-API declaration, not a signature."""
    src = (
        "instance TypeCat : Category.{u} (Type u) where\n"
        "  Hom X Y := X → Y\n"
        "  id X := fun x => x\n"
        "  comp f g := g ∘ f"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="TypeCat"
    ).get_context_for_agent()

    assert "Hom X Y := X → Y" in context
    assert "id X := fun x => x" in context
    assert "comp f g := g ∘ f" in context


def test_foreign_declarations_do_not_render_as_fields(tmp_path: Path) -> None:
    """`_extract_entries` slices to the next match, and it does not match these."""
    src = (
        "structure IsProduct (C : Type u) where\n"
        "  fst : C\n"
        "  section\n"
        "  variable {x y : C}\n"
        "  example : True := trivial"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="IsProduct"
    ).get_context_for_agent()

    assert "fst : C" in context
    assert "variable" not in context
    assert "example" not in context


def test_where_body_keeps_its_fields_but_not_their_proofs(tmp_path: Path) -> None:
    """`def … where` fields are API; the tactic blocks under them are not."""
    src = (
        "def nat_trans_vcomp (α : F ⟶ G) (β : G ⟶ H) : F ⟶ H where\n"
        "  app := fun X => α.app X ≫ β.app X\n"
        "  naturality := by\n"
        "    intros X Y f\n"
        "    simp only [tactic_marker_do_not_render]"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="vcomp"
    ).get_context_for_agent()

    assert "app := fun X => α.app X ≫ β.app X" in context
    assert "naturality" in context
    assert "tactic_marker_do_not_render" not in context


def test_wrapped_field_type_is_not_mistaken_for_a_proof(tmp_path: Path) -> None:
    """The deeper-indent rule must not eat a type that simply wrapped."""
    src = (
        "structure IsPullback (C : Type u) where\n"
        "  univ : ∀ {S : C} (a : S ⟶ X),\n"
        "      ∃! (l : S ⟶ W), l ≫ f = a"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="IsPullback"
    ).get_context_for_agent()

    assert "∃! (l : S ⟶ W), l ≫ f = a" in context


def test_omitted_entries_are_counted_not_dropped(tmp_path: Path) -> None:
    """A silent elision is the bug class this card exists to remove."""
    foundation = _foundation_with(tmp_path / "Foundation.lean", CATEGORY_SRC)
    foundation.add_artifact(
        Artifact(
            id="definition-notes-0000beef",
            type=ArtifactType.DEFINITION,
            natural_language="a comment-only artifact",
            lean_code="/- roadmap notes, no declaration here -/",
            status=VerificationStatus.VERIFIED_LEAN,
            created_by="agent-0",
            generation=0,
        )
    )

    context = foundation.get_context_for_agent()

    assert "1" in context and "omitted" in context.lower()
