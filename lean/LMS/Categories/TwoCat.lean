import Mathlib.CategoryTheory.Bicategory.Basic
import Mathlib.CategoryTheory.Bicategory.Strict.Basic

/-!
# 2-Categories and (2,1)-Categories

Formalizations corresponding to Stacks Project Sections 27-28.

Section 27 defines strict 2-categories, which correspond to Mathlib's
`Bicategory.Strict`. Section 28 defines (2,1)-categories as 2-categories
where all 2-morphisms are isomorphisms.

## Main definitions

* `TwoOneCategory` — a bicategory where every 2-morphism is an isomorphism
-/

open CategoryTheory

universe w v u

namespace LMS.Categories

/-- A (2,1)-category is a bicategory where every 2-morphism is an isomorphism.

This corresponds to Definition 003P in the Stacks Project (Section 28):
"A (strict) (2,1)-category is a 2-category in which all 2-morphisms are isomorphisms."

Mathlib's `Bicategory` is the general (weak) 2-category; the strict case is
`Bicategory.Strict`. This class adds the condition that all 2-morphisms are invertible,
which is orthogonal to strictness. -/
class TwoOneCategory (B : Type u) [Bicategory.{w, v} B] : Prop where
  /-- Every 2-morphism in a (2,1)-category is an isomorphism. -/
  isIso_of_two_morphism : ∀ {a b : B} {f g : a ⟶ b} (α : f ⟶ g), IsIso α

namespace TwoOneCategory

variable {B : Type u} [Bicategory.{w, v} B] [TwoOneCategory B]

/-- In a (2,1)-category, every 2-morphism is automatically an isomorphism. -/
instance isIso_two_morphism {a b : B} {f g : a ⟶ b} (α : f ⟶ g) : IsIso α :=
  isIso_of_two_morphism α

/-- In a (2,1)-category, any two 2-isomorphic 1-morphisms have an explicit isomorphism. -/
noncomputable def twoIso {a b : B} {f g : a ⟶ b} (α : f ⟶ g) : f ≅ g :=
  asIso α

end TwoOneCategory

end LMS.Categories
