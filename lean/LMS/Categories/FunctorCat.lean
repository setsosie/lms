import LMS.Categories.NatTrans
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic

/-!
# LMS.Categories.FunctorCat — The Functor Category

Defines the functor category `FunCat C D`, whose objects are functors `C ⥤ D`
and whose morphisms are natural transformations. Composition is vertical composition
of natural transformations.

## Conventions
- Universe polymorphism: `FunCat C D : Category.{max u₁ u₂ v₁ v₂, max u₁ v₂}`
- `@[simp]` lemmas for identity and composition applied at a component
- Namespace: `LMS.Categories`
-/

namespace LMS.Categories

universe u₁ v₁ u₂ v₂

/-- The functor category `[C, D]`: objects are functors `C ⥤ D`, morphisms are
    natural transformations, composition is vertical composition. -/
def FunCat (C : Category.{u₁, v₁}) (D : Category.{u₂, v₂}) :
    Category.{max u₁ u₂ v₁ v₂, max u₁ v₂} where
  Obj := Functor C D
  Hom := fun F G => NatTrans F G
  id := fun F => NatTrans.id F
  comp := fun α β => NatTrans.vcomp α β
  id_comp := fun α => NatTrans.id_vcomp α
  comp_id := fun α => NatTrans.vcomp_id α
  assoc := fun α β γ => NatTrans.vcomp_assoc α β γ

variable {C : Category.{u₁, v₁}} {D : Category.{u₂, v₂}}

/-! ## Simp lemmas for FunCat -/

/-- The identity morphism in the functor category applied at an object. -/
@[simp]
theorem FunCat.id_app (F : Functor C D) (x : C.Obj) :
    ((FunCat C D).id F).app x = D.id (F.obj x) := rfl

/-- Composition in the functor category applied at an object. -/
@[simp]
theorem FunCat.comp_app {F G H : Functor C D}
    (α : NatTrans F G) (β : NatTrans G H) (x : C.Obj) :
    ((FunCat C D).comp α β).app x = D.comp (α.app x) (β.app x) := rfl

end LMS.Categories
