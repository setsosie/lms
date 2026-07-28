import Mathlib.CategoryTheory.Category.Basic
import Mathlib.CategoryTheory.Functor.Basic
import Mathlib.CategoryTheory.NatTrans
import Mathlib.CategoryTheory.Iso
import Mathlib.CategoryTheory.NatIso
import Mathlib.CategoryTheory.EqToHom

/-!
# IsoComma Category (2-Fibre Product of Categories)

Given functors `L : A ⥤ C` and `R : B ⥤ C`, the **IsoComma category** `IsoComma L R`
(also called the 2-fibre product) has:
- Objects: triples `(a, b, f)` where `a : A`, `b : B`, and `f : L.obj a ≅ R.obj b`
- Morphisms: pairs `(α, β)` making the obvious square commute

This is the category-theoretic construction from Stacks Project, Section 29
(Example 04LL / Definition 003O). Unlike the ordinary comma category where
`hom : L.obj a ⟶ R.obj b` is just a morphism, here it must be an isomorphism.

## Main definitions

* `IsoComma L R` — the 2-fibre product category
* `IsoComma.fst` — first projection functor to `A`
* `IsoComma.snd` — second projection functor to `B`
* `IsoComma.natIso` — the natural isomorphism `fst ⋙ L ≅ snd ⋙ R`
* `IsoComma.lift` — the universal lift from a 2-commutative square
* `IsoComma.isoMk` — construct isomorphisms in `IsoComma L R`
-/

open CategoryTheory

universe v₁ v₂ v₃ v₄ u₁ u₂ u₃ u₄

namespace LMS.Categories

variable {A : Type u₁} [Category.{v₁} A]
variable {B : Type u₂} [Category.{v₂} B]
variable {C : Type u₃} [Category.{v₃} C]

/-- `IsoComma L R` is the 2-fibre product of functors `L : A ⥤ C` and `R : B ⥤ C`.
Objects are triples `(a, b, f)` where `a : A`, `b : B`, and `f : L.obj a ≅ R.obj b`
is an isomorphism in `C`. -/
structure IsoComma (L : A ⥤ C) (R : B ⥤ C) where
  /-- The left component in `A`. -/
  left : A
  /-- The right component in `B`. -/
  right : B
  /-- The isomorphism `L.obj left ≅ R.obj right` in `C`. -/
  hom : L.obj left ≅ R.obj right

namespace IsoComma

variable {L : A ⥤ C} {R : B ⥤ C}

/-- A morphism in the IsoComma category from `X` to `Y` is a pair of morphisms
`left : X.left ⟶ Y.left` and `right : X.right ⟶ Y.right` such that the diagram
```
  L(X.left) --X.hom.hom--> R(X.right)
     |                         |
  L(left)                   R(right)
     |                         |
     v                         v
  L(Y.left) --Y.hom.hom--> R(Y.right)
```
commutes. -/
@[ext]
structure Hom (X Y : IsoComma L R) where
  /-- The left component of the morphism. -/
  left : X.left ⟶ Y.left
  /-- The right component of the morphism. -/
  right : X.right ⟶ Y.right
  /-- The commutativity condition. -/
  w : L.map left ≫ Y.hom.hom = X.hom.hom ≫ R.map right := by aesop_cat

attribute [reassoc (attr := simp)] Hom.w

@[simps]
def Hom.id (X : IsoComma L R) : Hom X X where
  left := 𝟙 _
  right := 𝟙 _

@[simps]
def Hom.comp {X Y Z : IsoComma L R} (f : Hom X Y) (g : Hom Y Z) : Hom X Z where
  left := f.left ≫ g.left
  right := f.right ≫ g.right
  w := by simp [reassoc_of% f.w]

instance : Category (IsoComma L R) where
  Hom := Hom
  id := Hom.id
  comp := Hom.comp
  id_comp _ := Hom.ext (Category.id_comp _) (Category.id_comp _)
  comp_id _ := Hom.ext (Category.comp_id _) (Category.comp_id _)
  assoc _ _ _ := Hom.ext (Category.assoc _ _ _) (Category.assoc _ _ _)

@[simp] theorem hom_left_id (X : IsoComma L R) : (𝟙 X : Hom X X).left = 𝟙 X.left := rfl
@[simp] theorem hom_right_id (X : IsoComma L R) : (𝟙 X : Hom X X).right = 𝟙 X.right := rfl
@[simp] theorem hom_left_comp {X Y Z : IsoComma L R} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).left = f.left ≫ g.left := rfl
@[simp] theorem hom_right_comp {X Y Z : IsoComma L R} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).right = f.right ≫ g.right := rfl

/-- Construct an isomorphism in `IsoComma L R` from isomorphisms on both components,
provided the commutativity square holds. -/
@[simps]
def isoMk {X Y : IsoComma L R} (l : X.left ≅ Y.left) (r : X.right ≅ Y.right)
    (w : L.map l.hom ≫ Y.hom.hom = X.hom.hom ≫ R.map r.hom := by aesop_cat) :
    X ≅ Y where
  hom := ⟨l.hom, r.hom, w⟩
  inv := ⟨l.inv, r.inv, by
    have h : X.hom.hom = L.map l.hom ≫ Y.hom.hom ≫ R.map r.inv := by
      rw [← Category.assoc, w, Category.assoc, ← R.map_comp, r.hom_inv_id, R.map_id,
          Category.comp_id]
    rw [h, ← Category.assoc, ← L.map_comp, l.inv_hom_id, L.map_id, Category.id_comp]⟩
  hom_inv_id := Hom.ext l.hom_inv_id r.hom_inv_id
  inv_hom_id := Hom.ext l.inv_hom_id r.inv_hom_id

/-- The first projection functor from `IsoComma L R` to `A`. -/
@[simps]
def fst : IsoComma L R ⥤ A where
  obj X := X.left
  map f := f.left

/-- The second projection functor from `IsoComma L R` to `B`. -/
@[simps]
def snd : IsoComma L R ⥤ B where
  obj X := X.right
  map f := f.right

/-- The canonical natural isomorphism `fst ⋙ L ≅ snd ⋙ R`, witnessing the
2-commutativity of the diagram. -/
def natIso : (fst : IsoComma L R ⥤ A) ⋙ L ≅ (snd : IsoComma L R ⥤ B) ⋙ R :=
  NatIso.ofComponents (fun X => X.hom) (fun f => f.w)

@[simp]
theorem natIso_app (X : IsoComma L R) : (natIso (L := L) (R := R)).app X = X.hom := rfl

section Lift

variable {W : Type u₄} [Category.{v₄} W]

/-- The universal lift: given functors `a : W ⥤ A`, `b : W ⥤ B` and a natural isomorphism
`t : a ⋙ L ≅ b ⋙ R`, we get a functor `W ⥤ IsoComma L R`.

This is the universal property of the 2-fibre product: any 2-commutative square factors
through `IsoComma L R`. -/
@[simps]
def lift (a : W ⥤ A) (b : W ⥤ B) (t : a ⋙ L ≅ b ⋙ R) : W ⥤ IsoComma L R where
  obj w := ⟨a.obj w, b.obj w, t.app w⟩
  map f := ⟨a.map f, b.map f, by
    have := t.hom.naturality f
    simp [Functor.comp_map] at this
    exact this⟩

@[simp]
theorem lift_fst (a : W ⥤ A) (b : W ⥤ B) (t : a ⋙ L ≅ b ⋙ R) :
    lift a b t ⋙ fst = a :=
  Functor.ext (fun _ => rfl)

@[simp]
theorem lift_snd (a : W ⥤ A) (b : W ⥤ B) (t : a ⋙ L ≅ b ⋙ R) :
    lift a b t ⋙ snd = b :=
  Functor.ext (fun _ => rfl)

end Lift

end IsoComma

end LMS.Categories
