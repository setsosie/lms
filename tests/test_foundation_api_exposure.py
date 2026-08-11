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
from lms.foundation import FoundationEntry, FoundationFile
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


def test_where_on_a_wrapped_header_still_means_body_is_api(tmp_path: Path) -> None:
    """`where` lands on line 2 when the binders wrap; line 1 alone misses it."""
    src = (
        "theorem iso_is_equivalence {C : Type u} [Category C] :\n"
        "    Equivalence (fun X Y : C => Nonempty (X ≅ Y)) where\n"
        "  refl := fun X => ⟨Iso.refl X⟩\n"
        "  symm := fun {X Y} h => h.map Iso.symm\n"
        "  trans := fun {X Y Z} h₁ h₂ => h₁.map2 h₂"
    )
    context = _foundation_with(
        tmp_path / "Foundation.lean", src, name="isoequiv"
    ).get_context_for_agent()

    assert "symm := fun {X Y} h => h.map Iso.symm" in context
    assert "trans := fun {X Y Z} h₁ h₂ => h₁.map2 h₂" in context


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


# --- Review findings, 2026-08-11 (AC-15..AC-24) ------------------------------
#
# Ten findings from the high-effort pre-merge review. Every one of these
# renders *wrong Lean* into an agent prompt, which is the failure this card
# exists to remove -- an agent shown a fragment it cannot elaborate reimplements
# instead of reusing, exactly as in `shakedown_3x3_d`.


def _entry(lean_code: str, entry_type: str = "structure") -> FoundationEntry:
    """A bare entry, bypassing extraction, to pin one renderer at a time."""
    return FoundationEntry(
        artifact_id="definition-X-000000000000",
        name="X",
        entry_type=entry_type,
        signature="",
        lean_code=lean_code,
        generation=0,
        author="agent-0",
    )


def test_notation_family_is_not_rendered_as_fields() -> None:
    """AC-15: `infixr:80` / `scoped notation` / `#check` are commands, not fields.

    `FOREIGN_TOKENS` matched whole tokens, so a precedence suffix or a
    `scoped`/`local` modifier walked straight past it -- 131 of 4,612 corpus
    entries rendered a Lean command inside their API block, most often on
    `structure Category` itself.
    """
    fields = _entry(
        "structure Category (obj : Type u) where\n"
        "  Hom : obj → obj → Type v\n"
        "  id : (X : obj) → Hom X X\n"
        '  infixr:80 " ⟶ " => Category.Hom\n'
        '  scoped notation "𝟙" => Category.id\n'
        "  #check @Category.id\n"
    ).field_lines()

    assert fields == ["Hom : obj → obj → Type v", "id : (X : obj) → Hom X X"]


def test_def_where_body_never_renders_unbalanced_fragments() -> None:
    """AC-16: a wrapped field value must not reach an agent half-open.

    `_continuation_lines` fell back to the raw source line whenever its trim
    came back empty, so `{ app := by` -- an unclosed brace and a bare `by` --
    rendered as API on 24 corpus entries. An agent that copies that gets
    `unexpected token`; one that reads `left_inv` as a typed field gets
    `unknown identifier`.
    """
    fields = _entry(
        "def yonedaEquiv {C : Type u} (x : C) : Equiv A B where\n"
        "  toFun := fun η => η.app (op x) (C.id x)\n"
        "  invFun := fun a =>\n"
        "    { app := by\n"
        "        simp }\n",
        entry_type="def",
    ).field_lines()

    for field in fields:
        assert field.count("(") == field.count(")"), field
        assert field.count("{") == field.count("}"), field
    assert not any("by" == f.split()[-1] for f in fields if f.split())


def test_cut_into_a_bracket_group_skips_to_its_close() -> None:
    """AC-16 (cont.): a lone `)` is not a field.

    Cutting `naturality {y z} f := funext (λ g => by` drops the tail, but the
    `(` it opened closes on a later line at the *field's own* indent -- so the
    proof-indent test could not see it and the closer rendered as a bare `)`.
    The first fix here counted brackets in the kept head, which is balanced by
    construction; the group can just as easily sit in the dropped tail.
    """
    fields = _entry(
        "def yoneda_inv (a : F.obj (Op.mk x)) : NatTrans (hom_functor C x) F where\n"
        "    app y f := F.map f a\n"
        "    naturality {y z} f := funext (λ g => by\n"
        "      dsimp [TypeCat, hom_functor]\n"
        "      rfl\n"
        "    )\n",
        entry_type="def",
    ).field_lines()

    assert ")" not in fields
    assert fields == ["app y f := F.map f a", "naturality {y z} f"]


def test_instance_where_still_renders_every_field() -> None:
    """AC-17: the `instance ... where` exception the card added must survive."""
    entry = _entry(
        "instance TypeCat : Category (Type u) where\n"
        "  Hom := fun a b => a → b\n"
        "  id := fun _ x => x\n",
        entry_type="instance",
    )

    assert entry.body_is_api() is True
    assert entry.field_lines() == ["Hom := fun a b => a → b", "id := fun _ x => x"]


def test_rendered_fields_have_balanced_brackets() -> None:
    """AC-18: an unbalanced fragment as API is worse than a shorter one."""
    fields = _entry(
        "structure Wrapper where\n  f : (a : A) → B a\n  g : List (Nat × ⟨A⟩)\n"
    ).field_lines()

    for field in fields:
        assert field.count("(") == field.count(")")
        assert field.count("⟨") == field.count("⟩")


def test_letI_in_type_does_not_truncate_return_type() -> None:
    """AC-19: `letI := …;` in the *type* is a binder, not the body.

    Six shipped `Compat.lean` entries rendered as `… : letI`, dropping the real
    return type. An agent shown no return type cannot apply the declaration.
    """
    stmt = _entry(
        "def functorToMathlib {C : Category} {D : Category} (F : Functor C D) :\n"
        "    letI := toMathlib C; letI := toMathlib D;\n"
        "    CategoryTheory.Functor C.Obj D.Obj := by\n"
        "  exact F.toMathlib\n",
        entry_type="def",
    ).statement_lines()

    joined = " ".join(stmt)
    assert "CategoryTheory.Functor C.Obj D.Obj" in joined
    assert joined.strip() != "letI"
    assert "exact F.toMathlib" not in joined


def test_trailing_comment_bracket_does_not_leak_proof_body() -> None:
    """AC-20: an unclosed bracket inside a `--` comment is not real depth.

    One `(` in a trailing comment kept depth at 1 forever, so the top-level
    `:=` was never found and the whole tactic script rendered as the API.
    """
    stmt = _entry(
        "theorem yoneda_lemma (F : C -> Type v) (X : C) : -- see (Stacks 001A\n"
        "    F X = F X := by\n"
        "  simp only [Functor.map_id]\n"
        "  exact rfl\n",
        entry_type="theorem",
    ).statement_lines()

    joined = "\n".join(stmt)
    assert "F X = F X" in joined
    assert "simp only" not in joined
    assert "exact rfl" not in joined


def test_where_with_trailing_comment_still_renders_all_fields() -> None:
    """AC-21: `endswith("where")` missed a header carrying a `--` comment."""
    entry = _entry(
        "instance Foo : Bar where -- the fields\n  a := 1\n  b := 2\n",
        entry_type="instance",
    )

    assert entry.body_is_api() is True
    assert entry.field_lines() == ["a := 1", "b := 2"]


def test_declaration_header_has_no_dangling_assign() -> None:
    """AC-22: 18 corpus headers ended in a bare `:=`, which is not valid Lean."""
    header = _entry(
        "lemma mem_span_singleton (a : R) : a ∈ Ideal.span {a} :=\n"
        "  Ideal.subset_span rfl\n",
        entry_type="lemma",
    ).declaration_header()

    assert header == "lemma mem_span_singleton (a : R) : a ∈ Ideal.span {a}"


def test_inline_proof_term_is_not_part_of_the_declaration() -> None:
    """AC-22 (cont.): `:= I.neg_mem ha` on the header line is the proof."""
    header = _entry(
        "lemma ideal_neg_mem (ha : a ∈ I) : -a ∈ I := I.neg_mem ha\n",
        entry_type="lemma",
    ).declaration_header()

    assert header == "lemma ideal_neg_mem (ha : a ∈ I) : -a ∈ I"


def test_equation_style_def_body_is_not_rendered() -> None:
    """AC-23: a pattern-matching def has no top-level `:=` to stop at."""
    stmt = _entry(
        "def f : Nat -> Nat\n  | 0 => 1\n  | n+1 => f n + secret_body_marker\n",
        entry_type="def",
    ).statement_lines()

    assert stmt == []


def test_code_fallback_entry_still_names_itself(tmp_path: Path) -> None:
    """AC-24: a `code` entry matched neither branch and lost its own name.

    Without the name the agent cannot reference the artifact at all, and if it
    recreates that name `add_artifact` silently drops the new verified work.
    """
    foundation = FoundationFile(tmp_path / "Foundation.lean")
    foundation.add_artifact(
        Artifact(
            id="definition-openonly-0000cafe",
            type=ArtifactType.DEFINITION,
            natural_language="an artifact with no declaration",
            lean_code="open CategoryTheory\nexample : True := by trivial\n",
            status=VerificationStatus.VERIFIED_LEAN,
            created_by="agent-0",
            generation=0,
        )
    )

    entry = foundation.entries[0]
    assert entry.entry_type == "code"
    assert entry.name in foundation.get_context_for_agent()


def test_available_definitions_shows_the_wrapped_statement(tmp_path: Path) -> None:
    """AC-25: the two renderers claimed they could not disagree. They did.

    `get_available_definitions` printed only the first physical line, so on the
    10.6% of entries whose header wraps it showed a dangling binder list -- no
    better than the `signature[:80]` cut it replaced.
    """
    src = (
        "theorem yoneda_computation {C : Type u} (F : C → Type v)\n"
        "    (X : C) : F X = F X := by\n"
        "  rfl\n"
    )
    summary = _foundation_with(
        tmp_path / "Foundation.lean", src, name="yoneda_computation"
    ).get_available_definitions()

    assert "F X = F X" in summary


def test_committee_summary_is_bounded(tmp_path: Path) -> None:
    """AC-26: the `entries[:10]` cap went away with nothing replacing it.

    Unbounded, the Foundation Summary alone outgrows the served
    `max_model_len` on the corpora this program is aiming at.
    """
    foundation = FoundationFile(tmp_path / "Foundation.lean")
    for i in range(25):
        foundation.add_artifact(
            Artifact(
                id=f"definition-d{i}-{i:012d}",
                type=ArtifactType.DEFINITION,
                natural_language=f"definition {i}",
                lean_code=f"def d{i} : Nat := {i}",
                status=VerificationStatus.VERIFIED_LEAN,
                created_by="agent-0",
                generation=0,
            )
        )

    bounded = foundation.get_context_for_agent(max_entries=10)

    assert "d24" not in bounded
    assert "more" in bounded.lower()
