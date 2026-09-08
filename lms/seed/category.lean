/-
Generation-0 axiom layer (26Q3-HARN-23). Hand-written and human-reviewed;
NOT agent output. Inlined into `LMS.Foundation` by `reset_foundation`, so
everything here lives in `namespace LMS.Foundation` and reaches agents through
`import LMS.Foundation` + `open LMS.Foundation`.

Deliberately shaped like Mathlib's `CategoryTheory` API -- a `class`, with
`⟶`, `𝟙` and `≫`. Every failure in `committee_fix_c` generations 5-9 was the
model reaching for exactly this idiom against a foundation that was a plain
parameterized `structure`: `invalid binder annotation, type is not a class`,
`type expected, got (Category : Type ?u.…)`. Fighting that prior costs tokens
every generation and it never won. Matching it makes the model's strongest
instinct correct by construction.

Notation is `scoped`: it activates on `open LMS.Foundation` (which agents are
already told to write) and stays out of the way of any Mathlib module that
binds the same symbols.

`comp` is DIAGRAMMATIC, matching Mathlib's `≫`: `f ≫ g` is "f, then g".
-/

universe u v

/-- A category: objects, morphisms, identities, composition, and the laws. -/
class Category (C : Type u) where
  /-- Morphisms from `X` to `Y`, written `X ⟶ Y`. -/
  Hom : C → C → Type v
  /-- The identity morphism on `X`, written `𝟙 X`. -/
  id : (X : C) → Hom X X
  /-- Diagrammatic composition: `f ≫ g` is "`f`, then `g`". -/
  comp : {X Y Z : C} → Hom X Y → Hom Y Z → Hom X Z
  id_comp : ∀ {X Y : C} (f : Hom X Y), comp (id X) f = f
  comp_id : ∀ {X Y : C} (f : Hom X Y), comp f (id Y) = f
  assoc : ∀ {W X Y Z : C} (f : Hom W X) (g : Hom X Y) (h : Hom Y Z),
    comp (comp f g) h = comp f (comp g h)

@[inherit_doc] scoped infixr:10 " ⟶ " => Category.Hom
@[inherit_doc] scoped notation "𝟙" => Category.id
@[inherit_doc] scoped infixr:80 " ≫ " => Category.comp

/-- Types and functions form a category, with `≫` as diagrammatic composition.

A worked model, not decoration: it witnesses that the axioms above are
satisfiable, and it is the example an agent can pattern-match against when
building its own instance. -/
instance typeCategory : Category (Type u) where
  Hom X Y := X → Y
  id _ x := x
  comp f g x := g (f x)
  id_comp _ := rfl
  comp_id _ := rfl
  assoc _ _ _ := rfl

/-- The one-object, one-morphism category: the degenerate model. -/
instance punitCategory : Category PUnit.{u + 1} where
  Hom _ _ := PUnit
  id _ := PUnit.unit
  comp _ _ := PUnit.unit
  id_comp _ := rfl
  comp_id _ := rfl
  assoc _ _ _ := rfl
