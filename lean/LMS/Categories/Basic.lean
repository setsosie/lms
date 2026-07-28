import Mathlib.Tactic.Common
import Mathlib.Logic.Basic

/-!
# LMS.Categories.Basic — Core Category Definition

## Conventions for all LMS.Categories.* files:
- Universe polymorphism: Category.{u, v} with Obj : Type u, Hom : ... → Type v
- Naming: snake_case for lemmas/theorems, CamelCase for structures
- All core definitions should have @[simp] lemmas
- Namespace: LMS.Categories
- Imports: Only Mathlib.Tactic.Common and Mathlib.Logic.Basic (no Mathlib category theory)
-/

namespace LMS.Categories

universe u v w

/-- A category with objects in `Type u` and morphisms in `Type v`. -/
structure Category where
  Obj : Type u
  Hom : Obj → Obj → Type v
  id : (x : Obj) → Hom x x
  comp : {x y z : Obj} → Hom x y → Hom y z → Hom x z
  id_comp : ∀ {x y : Obj} (f : Hom x y), comp (id x) f = f
  comp_id : ∀ {x y : Obj} (f : Hom x y), comp f (id y) = f
  assoc : ∀ {w x y z : Obj} (f : Hom w x) (g : Hom x y) (h : Hom y z),
          comp (comp f g) h = comp f (comp g h)

variable {C : Category.{u, v}}

@[simp]
theorem id_comp (f : C.Hom x y) : C.comp (C.id x) f = f := C.id_comp f

@[simp]
theorem comp_id (f : C.Hom x y) : C.comp f (C.id y) = f := C.comp_id f

@[simp]
theorem assoc (f : C.Hom w x) (g : C.Hom x y) (h : C.Hom y z) :
    C.comp (C.comp f g) h = C.comp f (C.comp g h) := C.assoc f g h

end LMS.Categories
