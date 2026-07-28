import LMS.Categories.Basic
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic

/-!
# LMS.Categories.Functor — Functors between categories

Defines the `Functor` structure for morphisms between categories,
along with identity and composition functors and their laws.

## Conventions
- Universe polymorphism: `Functor` maps `Category.{u₁, v₁}` to `Category.{u₂, v₂}`
- `@[simp]` lemmas for `map_id` and `map_comp`
- Namespace: `LMS.Categories`
-/

namespace LMS.Categories

universe u₁ v₁ u₂ v₂ u₃ v₃ u₄ v₄

/-- A functor from category `C` to category `D`, mapping objects and morphisms
    while preserving identity and composition. -/
structure Functor (C : Category.{u₁, v₁}) (D : Category.{u₂, v₂}) where
  /-- The action of the functor on objects. -/
  obj : C.Obj → D.Obj
  /-- The action of the functor on morphisms. -/
  map : {x y : C.Obj} → C.Hom x y → D.Hom (obj x) (obj y)
  /-- The functor preserves identity morphisms. -/
  map_id : ∀ (x : C.Obj), map (C.id x) = D.id (obj x)
  /-- The functor preserves composition. -/
  map_comp : ∀ {x y z : C.Obj} (f : C.Hom x y) (g : C.Hom y z),
             map (C.comp f g) = D.comp (map f) (map g)

variable {C : Category.{u₁, v₁}} {D : Category.{u₂, v₂}} {E : Category.{u₃, v₃}}

@[simp]
theorem Functor.map_id_simp (F : Functor C D) (x : C.Obj) :
    F.map (C.id x) = D.id (F.obj x) := F.map_id x

@[simp]
theorem Functor.map_comp_simp (F : Functor C D) {x y z : C.Obj}
    (f : C.Hom x y) (g : C.Hom y z) :
    F.map (C.comp f g) = D.comp (F.map f) (F.map g) := F.map_comp f g

/-- The identity functor on a category. -/
def Functor.id (C : Category.{u₁, v₁}) : Functor C C where
  obj := fun x => x
  map := fun f => f
  map_id := fun _ => rfl
  map_comp := fun _ _ => rfl

@[simp]
theorem Functor.id_obj (x : C.Obj) : (Functor.id C).obj x = x := rfl

@[simp]
theorem Functor.id_map {x y : C.Obj} (f : C.Hom x y) : (Functor.id C).map f = f := rfl

/-- Composition of functors. Given `F : C ⥤ D` and `G : D ⥤ E`, produce `G ∘ F : C ⥤ E`. -/
def Functor.comp (F : Functor C D) (G : Functor D E) : Functor C E where
  obj := fun x => G.obj (F.obj x)
  map := fun f => G.map (F.map f)
  map_id := fun x => by simp
  map_comp := fun f g => by simp

@[simp]
theorem Functor.comp_obj (F : Functor C D) (G : Functor D E) (x : C.Obj) :
    (Functor.comp F G).obj x = G.obj (F.obj x) := rfl

@[simp]
theorem Functor.comp_map (F : Functor C D) (G : Functor D E) {x y : C.Obj} (f : C.Hom x y) :
    (Functor.comp F G).map f = G.map (F.map f) := rfl

/-- Functor composition is associative. -/
theorem Functor.comp_assoc {B : Category.{u₄, v₄}}
    (F : Functor C D) (G : Functor D E) (H : Functor E B) :
    Functor.comp (Functor.comp F G) H = Functor.comp F (Functor.comp G H) := by
  rfl

/-- The identity functor is a left unit for composition. -/
theorem Functor.id_comp_eq (F : Functor C D) :
    Functor.comp (Functor.id C) F = F := by
  cases F; rfl

/-- The identity functor is a right unit for composition. -/
theorem Functor.comp_id_eq (F : Functor C D) :
    Functor.comp F (Functor.id D) = F := by
  cases F; rfl

end LMS.Categories
