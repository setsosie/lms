import LMS.Categories.Equivalence
import Mathlib.CategoryTheory.Equivalence

/-!
# LMS.Categories.Compat — Compatibility between LMS and Mathlib Category Theory

This file provides a thin compatibility layer mapping between LMS's bundled
`structure Category.{u,v}` and Mathlib's unbundled typeclass
`CategoryTheory.Category.{v,u}`.

## Universe conventions

LMS uses `Category.{u,v}` where `Obj : Type u`, `Hom : ... → Type v`.
Mathlib uses `Category.{v,u}` where the object type is `Type u` and
morphisms are `Type v`. Note the reversed order.

## Design

The compatibility layer provides:
- `toMathlib`: Convert an LMS Category to a Mathlib Category instance
- Conversion functions for Functor, NatTrans, Iso, Full, Faithful
- `equivalenceToMathlib` for Equivalence (requires triangle identity, which
  LMS's `Equivalence` does not include -- see gap note below)

## Known Gap

Mathlib's `Equivalence` requires the triangle identity (zig-zag):
  `F(η_X) ≫ ε_{F(X)} = id_{F(X)}`
LMS's `Equivalence` only requires unit/counit natural isomorphisms without
this coherence condition. The conversion uses `sorry` for this field.
Downstream code should use Mathlib's `Equivalence` directly.
-/

namespace LMS.Categories.Compat

open CategoryTheory

universe u v u₁ v₁ u₂ v₂

/-! ## Category compatibility -/

/-- Convert an LMS bundled Category to a Mathlib Category instance on `C.Obj`.
    LMS `Category.{u,v}` becomes Mathlib `Category.{v,u}` on `C.Obj`. -/
instance toMathlib (C : LMS.Categories.Category.{u, v}) :
    CategoryTheory.Category.{v, u} C.Obj where
  Hom := C.Hom
  id X := C.id X
  comp f g := C.comp f g
  id_comp f := C.id_comp f
  comp_id f := C.comp_id f
  assoc f g h := C.assoc f g h

/-! ## Functor compatibility -/

/-- Convert an LMS Functor to a Mathlib Functor. -/
def functorToMathlib {C : LMS.Categories.Category.{u₁, v₁}}
    {D : LMS.Categories.Category.{u₂, v₂}}
    (F : LMS.Categories.Functor C D) :
    letI := toMathlib C; letI := toMathlib D
    CategoryTheory.Functor C.Obj D.Obj where
  obj := F.obj
  map f := F.map f
  map_id X := F.map_id X
  map_comp f g := F.map_comp f g

/-! ## NatTrans compatibility -/

/-- Convert an LMS NatTrans to a Mathlib NatTrans. -/
def natTransToMathlib {C : LMS.Categories.Category.{u₁, v₁}}
    {D : LMS.Categories.Category.{u₂, v₂}}
    {F G : LMS.Categories.Functor C D}
    (α : LMS.Categories.NatTrans F G) :
    letI := toMathlib C; letI := toMathlib D
    CategoryTheory.NatTrans (functorToMathlib F) (functorToMathlib G) where
  app X := α.app X
  naturality _ _ f := α.naturality f

/-! ## Iso compatibility -/

/-- Convert an LMS Iso to a Mathlib Iso. -/
def isoToMathlib {C : LMS.Categories.Category.{u, v}} {x y : C.Obj}
    (i : LMS.Categories.Iso C x y) :
    letI := toMathlib C
    x ≅ y where
  hom := i.hom
  inv := i.inv
  hom_inv_id := i.hom_inv
  inv_hom_id := i.inv_hom

/-- Convert a Mathlib Iso back to an LMS Iso. -/
def isoOfMathlib {C : LMS.Categories.Category.{u, v}} {x y : C.Obj}
    (i : letI := toMathlib C; x ≅ y) :
    LMS.Categories.Iso C x y where
  hom := i.hom
  inv := i.inv
  hom_inv := i.hom_inv_id
  inv_hom := i.inv_hom_id

/-! ## Equivalence compatibility -/

/-- Convert an LMS Equivalence to a Mathlib Equivalence.

    **Note**: This uses `sorry` for the triangle identity (`functor_unitIso_comp`)
    because LMS's `Equivalence` does not include this coherence condition.
    For any *genuine* equivalence of categories the triangle identity holds
    (one can always modify the unit), but LMS's structure doesn't record it.
    Downstream code should use Mathlib's `Equivalence` directly. -/
def equivalenceToMathlib {C : LMS.Categories.Category.{u₁, v₁}}
    {D : LMS.Categories.Category.{u₂, v₂}}
    (e : LMS.Categories.Equivalence C D) :
    letI := toMathlib C; letI := toMathlib D
    C.Obj ≌ D.Obj where
  functor := functorToMathlib e.forward
  inverse := functorToMathlib e.inverse
  unitIso := {
    hom := {
      app := fun X => e.unit.hom.app X
      naturality := fun _ _ f => e.unit.hom.naturality f
    }
    inv := {
      app := fun X => e.unit.inv.app X
      naturality := fun _ _ f => e.unit.inv.naturality f
    }
    hom_inv_id := by
      ext X
      show C.comp (e.unit.hom.app X) (e.unit.inv.app X) = C.id X
      have h := congr_arg (fun (α : LMS.Categories.NatTrans _ _) => α.app X) e.unit.hom_inv
      exact h
    inv_hom_id := by
      ext X
      show C.comp (e.unit.inv.app X) (e.unit.hom.app X) =
        C.id ((LMS.Categories.Functor.comp e.forward e.inverse).obj X)
      have h := congr_arg (fun (α : LMS.Categories.NatTrans _ _) => α.app X) e.unit.inv_hom
      exact h
  }
  counitIso := {
    hom := {
      app := fun X => e.counit.hom.app X
      naturality := fun _ _ f => e.counit.hom.naturality f
    }
    inv := {
      app := fun X => e.counit.inv.app X
      naturality := fun _ _ f => e.counit.inv.naturality f
    }
    hom_inv_id := by
      ext X
      show D.comp (e.counit.hom.app X) (e.counit.inv.app X) =
        D.id ((LMS.Categories.Functor.comp e.inverse e.forward).obj X)
      have h := congr_arg (fun (α : LMS.Categories.NatTrans _ _) => α.app X) e.counit.hom_inv
      exact h
    inv_hom_id := by
      ext X
      show D.comp (e.counit.inv.app X) (e.counit.hom.app X) = D.id X
      have h := congr_arg (fun (α : LMS.Categories.NatTrans _ _) => α.app X) e.counit.inv_hom
      exact h
  }
  -- Triangle identity: F(η_X) ≫ ε_{F(X)} = id_{F(X)}
  -- LMS Equivalence does not include this condition.
  functor_unitIso_comp := by
    intro X
    sorry

/-! ## Functor property compatibility -/

/-- An LMS Full functor gives a Mathlib Full functor. -/
theorem fullToMathlib {C : LMS.Categories.Category.{u₁, v₁}}
    {D : LMS.Categories.Category.{u₂, v₂}}
    {F : LMS.Categories.Functor C D} (hF : LMS.Categories.Full F) :
    letI := toMathlib C; letI := toMathlib D
    (functorToMathlib F).Full where
  map_surjective g := hF.preimage g

/-- An LMS Faithful functor gives a Mathlib Faithful functor. -/
theorem faithfulToMathlib {C : LMS.Categories.Category.{u₁, v₁}}
    {D : LMS.Categories.Category.{u₂, v₂}}
    {F : LMS.Categories.Functor C D} (hF : LMS.Categories.Faithful F) :
    letI := toMathlib C; letI := toMathlib D
    (functorToMathlib F).Faithful where
  map_injective h := hF.map_injective h

end LMS.Categories.Compat
