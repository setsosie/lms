import LMS.Categories.Basic
import LMS.Categories.Functor
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic

/-!
# LMS.Categories.NatTrans — Natural Transformations

Defines natural transformations between functors, vertical composition,
identity natural transformations, whiskering (left and right), horizontal
composition (Godement product), and the interchange law.

## Conventions
- `app` for the component field (Mathlib convention)
- `@[ext]` on the NatTrans structure
- `@[simp]` on key lemmas
- Namespace: `LMS.Categories`
-/

namespace LMS.Categories

universe u₁ v₁ u₂ v₂ u₃ v₃

variable {C : Category.{u₁, v₁}} {D : Category.{u₂, v₂}} {E : Category.{u₃, v₃}}

/-- A natural transformation between functors `F` and `G : C → D`.
    Components `app x : D.Hom (F.obj x) (G.obj x)` satisfy the naturality square. -/
@[ext]
structure NatTrans (F G : Functor C D) where
  /-- The component of the natural transformation at object `x`. -/
  app : (x : C.Obj) → D.Hom (F.obj x) (G.obj x)
  /-- The naturality condition: `F.map f ≫ app y = app x ≫ G.map f`. -/
  naturality : ∀ {x y : C.Obj} (f : C.Hom x y),
               D.comp (F.map f) (app y) = D.comp (app x) (G.map f)

/-! ## Identity natural transformation -/

/-- The identity natural transformation on a functor `F`. -/
def NatTrans.id (F : Functor C D) : NatTrans F F where
  app := fun x => D.id (F.obj x)
  naturality := fun f => by simp [comp_id, id_comp]

@[simp]
theorem NatTrans.id_app (F : Functor C D) (x : C.Obj) :
    (NatTrans.id F).app x = D.id (F.obj x) := rfl

/-! ## Vertical composition -/

/-- Vertical composition of natural transformations: given `α : F ⟶ G` and `β : G ⟶ H`,
    produce `β ∘ᵥ α : F ⟶ H`. -/
def NatTrans.vcomp {F G H : Functor C D} (α : NatTrans F G) (β : NatTrans G H) :
    NatTrans F H where
  app := fun x => D.comp (α.app x) (β.app x)
  naturality := fun {x y} f => by
    -- Goal: D.comp (F.map f) (D.comp (α.app y) (β.app y))
    --     = D.comp (D.comp (α.app x) (β.app x)) (H.map f)
    rw [← assoc, α.naturality f, assoc, β.naturality f, ← assoc]

@[simp]
theorem NatTrans.vcomp_app {F G H : Functor C D}
    (α : NatTrans F G) (β : NatTrans G H) (x : C.Obj) :
    (NatTrans.vcomp α β).app x = D.comp (α.app x) (β.app x) := rfl

/-! ## Vertical composition laws -/

/-- The identity natural transformation is a left unit for vertical composition. -/
@[simp]
theorem NatTrans.id_vcomp {F G : Functor C D} (α : NatTrans F G) :
    NatTrans.vcomp (NatTrans.id F) α = α := by
  ext x; simp [id_comp]

/-- The identity natural transformation is a right unit for vertical composition. -/
@[simp]
theorem NatTrans.vcomp_id {F G : Functor C D} (α : NatTrans F G) :
    NatTrans.vcomp α (NatTrans.id G) = α := by
  ext x; simp [comp_id]

/-- Vertical composition is associative. -/
theorem NatTrans.vcomp_assoc {F G H K : Functor C D}
    (α : NatTrans F G) (β : NatTrans G H) (γ : NatTrans H K) :
    NatTrans.vcomp (NatTrans.vcomp α β) γ = NatTrans.vcomp α (NatTrans.vcomp β γ) := by
  ext x; simp [assoc]

/-! ## Whiskering -/

/-- Left whiskering: given a functor `F : C ⥤ D` and a natural transformation `α : G ⟶ H`
    where `G H : D ⥤ E`, produce `F ◁ α : F ⋙ G ⟶ F ⋙ H`. -/
def NatTrans.whiskerLeft (F : Functor C D) {G H : Functor D E} (α : NatTrans G H) :
    NatTrans (Functor.comp F G) (Functor.comp F H) where
  app := fun x => α.app (F.obj x)
  naturality := fun {x y} f => by
    show E.comp (G.map (F.map f)) (α.app (F.obj y)) =
         E.comp (α.app (F.obj x)) (H.map (F.map f))
    exact α.naturality (F.map f)

@[simp]
theorem NatTrans.whiskerLeft_app (F : Functor C D) {G H : Functor D E}
    (α : NatTrans G H) (x : C.Obj) :
    (NatTrans.whiskerLeft F α).app x = α.app (F.obj x) := rfl

/-- Right whiskering: given a natural transformation `α : F ⟶ G` where `F G : C ⥤ D`
    and a functor `H : D ⥤ E`, produce `α ▷ H : F ⋙ H ⟶ G ⋙ H`. -/
def NatTrans.whiskerRight {F G : Functor C D} (α : NatTrans F G) (H : Functor D E) :
    NatTrans (Functor.comp F H) (Functor.comp G H) where
  app := fun x => H.map (α.app x)
  naturality := fun {x y} f => by
    show E.comp (H.map (F.map f)) (H.map (α.app y)) =
         E.comp (H.map (α.app x)) (H.map (G.map f))
    rw [← H.map_comp, ← H.map_comp, α.naturality f]

@[simp]
theorem NatTrans.whiskerRight_app {F G : Functor C D}
    (α : NatTrans F G) (H : Functor D E) (x : C.Obj) :
    (NatTrans.whiskerRight α H).app x = H.map (α.app x) := rfl

/-! ## Horizontal composition (Godement product) -/

/-- Horizontal composition (Godement product) of natural transformations.
    Given `α : F₁ ⟶ G₁` and `β : F₂ ⟶ G₂`, produce `α ⊗ β : F₁ ⋙ F₂ ⟶ G₁ ⋙ G₂`.
    Defined as `(α ▷ F₂) ∘ᵥ (G₁ ◁ β)`. -/
def NatTrans.hcomp {F₁ G₁ : Functor C D} {F₂ G₂ : Functor D E}
    (α : NatTrans F₁ G₁) (β : NatTrans F₂ G₂) :
    NatTrans (Functor.comp F₁ F₂) (Functor.comp G₁ G₂) :=
  NatTrans.vcomp (NatTrans.whiskerRight α F₂) (NatTrans.whiskerLeft G₁ β)

@[simp]
theorem NatTrans.hcomp_app {F₁ G₁ : Functor C D} {F₂ G₂ : Functor D E}
    (α : NatTrans F₁ G₁) (β : NatTrans F₂ G₂) (x : C.Obj) :
    (NatTrans.hcomp α β).app x = E.comp (F₂.map (α.app x)) (β.app (G₁.obj x)) := rfl

/-! ## Interchange law -/

/-- The interchange law: `(α' ∘ᵥ α) ⊗ (β' ∘ᵥ β) = (α' ⊗ β') ∘ᵥ (α ⊗ β)`. -/
theorem NatTrans.interchange
    {F₁ G₁ H₁ : Functor C D} {F₂ G₂ H₂ : Functor D E}
    (α : NatTrans F₁ G₁) (α' : NatTrans G₁ H₁)
    (β : NatTrans F₂ G₂) (β' : NatTrans G₂ H₂) :
    NatTrans.hcomp (NatTrans.vcomp α α') (NatTrans.vcomp β β') =
    NatTrans.vcomp (NatTrans.hcomp α β) (NatTrans.hcomp α' β') := by
  ext x
  simp only [hcomp_app, vcomp_app]
  -- LHS: E.comp (F₂.map (D.comp (α.app x) (α'.app x)))
  --            (E.comp (β.app (H₁.obj x)) (β'.app (H₁.obj x)))
  -- RHS: E.comp (E.comp (F₂.map (α.app x)) (β.app (G₁.obj x)))
  --            (E.comp (G₂.map (α'.app x)) (β'.app (H₁.obj x)))
  rw [Functor.map_comp_simp]
  -- LHS: E.comp (E.comp (F₂.map (α.app x)) (F₂.map (α'.app x)))
  --            (E.comp (β.app (H₁.obj x)) (β'.app (H₁.obj x)))
  rw [assoc (F₂.map (α.app x)) (F₂.map (α'.app x)) _]
  -- LHS: E.comp (F₂.map (α.app x))
  --            (E.comp (F₂.map (α'.app x)) (E.comp (β.app (H₁.obj x)) (β'.app (H₁.obj x))))
  rw [← assoc (F₂.map (α'.app x)) (β.app (H₁.obj x)) (β'.app (H₁.obj x))]
  -- LHS: E.comp (F₂.map (α.app x))
  --            (E.comp (E.comp (F₂.map (α'.app x)) (β.app (H₁.obj x))) (β'.app (H₁.obj x)))
  rw [β.naturality (α'.app x)]
  -- Now middle term uses β.naturality: F₂.map (α'.app x) ≫ β.app (H₁.obj x)
  --   becomes β.app (G₁.obj x) ≫ G₂.map (α'.app x)
  rw [assoc (β.app (G₁.obj x)) (G₂.map (α'.app x)) (β'.app (H₁.obj x))]
  rw [← assoc (F₂.map (α.app x)) (β.app (G₁.obj x)) _]

end LMS.Categories
