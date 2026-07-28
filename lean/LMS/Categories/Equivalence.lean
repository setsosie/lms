import LMS.Categories.FunctorCat
import LMS.Categories.Morphisms
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic

/-!
# LMS.Categories.Equivalence — Natural Isomorphisms and Equivalences of Categories

Defines natural isomorphisms (as isomorphisms in the functor category),
functor properties (full, faithful, essentially surjective), and
equivalences of categories.

## Conventions
- `NatIso` is an abbreviation for `Iso (FunCat C D) F G`
- Universe polymorphism throughout
- `@[simp]` on key lemmas
- Namespace: `LMS.Categories`
-/

namespace LMS.Categories

universe u₁ v₁ u₂ v₂

variable {C : Category.{u₁, v₁}} {D : Category.{u₂, v₂}}

/-! ## Natural Isomorphisms -/

/-- A natural isomorphism between functors `F` and `G` is an isomorphism in the
    functor category `[C, D]`. -/
abbrev NatIso (C : Category.{u₁, v₁}) (D : Category.{u₂, v₂})
    (F G : Functor C D) : Type (max u₁ v₂) :=
  Iso (FunCat C D) F G

/-- Construct a natural isomorphism from componentwise isomorphisms and naturality. -/
def NatIso.mk {F G : Functor C D}
    (app : (x : C.Obj) → D.Hom (F.obj x) (G.obj x))
    (inv : (x : C.Obj) → D.Hom (G.obj x) (F.obj x))
    (naturality : ∀ {x y : C.Obj} (f : C.Hom x y),
      D.comp (F.map f) (app y) = D.comp (app x) (G.map f))
    (hom_inv : ∀ (x : C.Obj), D.comp (app x) (inv x) = D.id (F.obj x))
    (inv_hom : ∀ (x : C.Obj), D.comp (inv x) (app x) = D.id (G.obj x)) :
    NatIso C D F G where
  hom := { app := app, naturality := naturality }
  inv := {
    app := inv
    naturality := fun {x y} f => by
      -- Goal: D.comp (G.map f) (inv y) = D.comp (inv x) (F.map f)
      calc D.comp (G.map f) (inv y)
          = D.comp (D.comp (D.id (G.obj x)) (G.map f)) (inv y) := by
              rw [id_comp]
        _ = D.comp (D.comp (D.comp (inv x) (app x)) (G.map f)) (inv y) := by
              rw [inv_hom x]
        _ = D.comp (D.comp (inv x) (D.comp (app x) (G.map f))) (inv y) := by
              rw [assoc (inv x) (app x) (G.map f)]
        _ = D.comp (D.comp (inv x) (D.comp (F.map f) (app y))) (inv y) := by
              rw [naturality f]
        _ = D.comp (inv x) (D.comp (D.comp (F.map f) (app y)) (inv y)) := by
              rw [assoc]
        _ = D.comp (inv x) (D.comp (F.map f) (D.comp (app y) (inv y))) := by
              rw [assoc (F.map f) (app y) (inv y)]
        _ = D.comp (inv x) (D.comp (F.map f) (D.id (F.obj y))) := by
              rw [hom_inv y]
        _ = D.comp (inv x) (F.map f) := by
              rw [comp_id]
  }
  hom_inv := by
    show NatTrans.vcomp _ _ = NatTrans.id F
    exact NatTrans.ext (funext fun x => hom_inv x)
  inv_hom := by
    show NatTrans.vcomp _ _ = NatTrans.id G
    exact NatTrans.ext (funext fun x => inv_hom x)

/-! ## NatIso operations -/

/-- The identity natural isomorphism on a functor. -/
def NatIso.refl (F : Functor C D) : NatIso C D F F :=
  Iso.refl (FunCat C D) F

/-- Reverse a natural isomorphism. -/
def NatIso.symm {F G : Functor C D} (η : NatIso C D F G) : NatIso C D G F :=
  Iso.symm η

/-- Compose natural isomorphisms. -/
def NatIso.trans {F G H : Functor C D}
    (η : NatIso C D F G) (θ : NatIso C D G H) : NatIso C D F H :=
  Iso.trans η θ

/-! ## Functor Properties -/

/-- A functor `F` is full if every morphism `F.obj x → F.obj y` in `D`
    is the image of some morphism `x → y` in `C`. -/
structure Full (F : Functor C D) : Prop where
  /-- Every morphism between objects in the image is hit. -/
  preimage : ∀ {x y : C.Obj} (g : D.Hom (F.obj x) (F.obj y)), ∃ f : C.Hom x y, F.map f = g

/-- A functor `F` is faithful if it is injective on hom sets:
    `F.map f = F.map g → f = g`. -/
structure Faithful (F : Functor C D) : Prop where
  /-- The map action is injective on each hom set. -/
  map_injective : ∀ {x y : C.Obj} {f g : C.Hom x y}, F.map f = F.map g → f = g

/-- A functor is fully faithful if it is bijective on hom sets. -/
structure FullyFaithful (F : Functor C D) : Prop where
  /-- The functor is full. -/
  full : Full F
  /-- The functor is faithful. -/
  faithful : Faithful F

/-- A functor `F` is essentially surjective if every object of `D`
    is isomorphic to some `F.obj x`. -/
structure EssSurj (F : Functor C D) : Prop where
  /-- Every object in `D` is isomorphic to one in the image of `F`. -/
  obj_preimage : ∀ (d : D.Obj), ∃ (c : C.Obj), Nonempty (Iso D (F.obj c) d)

/-! ## Equivalence of Categories -/

/-- An equivalence of categories `C ≃ D` consists of a functor `F : C ⥤ D`,
    a quasi-inverse `G : D ⥤ C`, and natural isomorphisms between the
    composites and the identity functors. -/
structure Equivalence (C : Category.{u₁, v₁}) (D : Category.{u₂, v₂}) where
  /-- The forward functor. -/
  forward : Functor C D
  /-- The quasi-inverse functor. -/
  inverse : Functor D C
  /-- The unit: `id_C ≅ G ∘ F`. -/
  unit : Iso (FunCat C C) (Functor.id C) (Functor.comp forward inverse)
  /-- The counit: `F ∘ G ≅ id_D`. -/
  counit : Iso (FunCat D D) (Functor.comp inverse forward) (Functor.id D)

/-! ## Equivalence is reflexive -/

/-- Every category is equivalent to itself via the identity functor. -/
def Equivalence.refl (C : Category.{u₁, v₁}) : Equivalence C C where
  forward := Functor.id C
  inverse := Functor.id C
  unit := by
    rw [Functor.id_comp_eq]
    exact Iso.refl (FunCat C C) (Functor.id C)
  counit := by
    rw [Functor.id_comp_eq]
    exact Iso.refl (FunCat C C) (Functor.id C)

/-! ## Equivalence is symmetric -/

/-- If `C ≃ D` then `D ≃ C`. -/
def Equivalence.symm {C : Category.{u₁, v₁}} {D : Category.{u₂, v₂}}
    (e : Equivalence C D) : Equivalence D C where
  forward := e.inverse
  inverse := e.forward
  unit := Iso.symm e.counit
  counit := Iso.symm e.unit

end LMS.Categories
