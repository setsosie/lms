import LMS.Categories.Basic
import Mathlib.Algebra.Group.Defs

/-!
# LMS.Categories.Morphisms — Morphism Classes

Definitions and basic properties of special morphisms in a category:
monomorphisms, epimorphisms, isomorphisms, split monos, and split epis.

All definitions are from scratch (no Mathlib category theory imports).
-/

namespace LMS.Categories

universe u v

variable {C : Category.{u, v}}

/-! ## Monomorphisms and Epimorphisms -/

/-- A morphism `f : x ⟶ y` is a monomorphism if it is left-cancellable. -/
structure Mono {x y : C.Obj} (f : C.Hom x y) : Prop where
  cancel_left : ∀ {w : C.Obj} (g h : C.Hom w x), C.comp g f = C.comp h f → g = h

/-- A morphism `f : x ⟶ y` is an epimorphism if it is right-cancellable. -/
structure Epi {x y : C.Obj} (f : C.Hom x y) : Prop where
  cancel_right : ∀ {z : C.Obj} (g h : C.Hom y z), C.comp f g = C.comp f h → g = h

/-! ## Isomorphisms -/

/-- An isomorphism between objects `x` and `y`, with explicit forward and inverse maps. -/
structure Iso (C : Category.{u, v}) (x y : C.Obj) where
  hom : C.Hom x y
  inv : C.Hom y x
  hom_inv : C.comp hom inv = C.id x
  inv_hom : C.comp inv hom = C.id y

@[simp]
theorem Iso.hom_inv_simp (i : Iso C x y) : C.comp i.hom i.inv = C.id x := i.hom_inv

@[simp]
theorem Iso.inv_hom_simp (i : Iso C x y) : C.comp i.inv i.hom = C.id y := i.inv_hom

/-! ## Iso operations: refl, symm, trans -/

/-- The identity isomorphism. -/
def Iso.refl (C : Category.{u, v}) (x : C.Obj) : Iso C x x where
  hom := C.id x
  inv := C.id x
  hom_inv := by simp
  inv_hom := by simp

/-- Reverse an isomorphism. -/
def Iso.symm (i : Iso C x y) : Iso C y x where
  hom := i.inv
  inv := i.hom
  hom_inv := i.inv_hom
  inv_hom := i.hom_inv

/-- Compose two isomorphisms. -/
def Iso.trans (i : Iso C x y) (j : Iso C y z) : Iso C x z where
  hom := C.comp i.hom j.hom
  inv := C.comp j.inv i.inv
  hom_inv := by
    rw [assoc, ← assoc j.hom j.inv i.inv, Iso.hom_inv_simp, id_comp, Iso.hom_inv_simp]
  inv_hom := by
    rw [assoc, ← assoc i.inv i.hom j.hom, Iso.inv_hom_simp, id_comp, Iso.inv_hom_simp]

/-! ## Split Monomorphisms and Epimorphisms -/

/-- A split monomorphism: `f` has a left inverse (retraction). -/
structure SplitMono {x y : C.Obj} (f : C.Hom x y) where
  retraction : C.Hom y x
  retraction_comp : C.comp f retraction = C.id x

/-- A split epimorphism: `f` has a right inverse (section). -/
structure SplitEpi {x y : C.Obj} (f : C.Hom x y) where
  section_ : C.Hom y x
  comp_section : C.comp section_ f = C.id y

/-! ## Composition lemmas -/

/-- The composition of two monomorphisms is a monomorphism. -/
theorem mono_comp {x y z : C.Obj} {f : C.Hom x y} {g : C.Hom y z}
    (hf : Mono f) (hg : Mono g) : Mono (C.comp f g) where
  cancel_left := fun a b h => by
    have h1 : C.comp (C.comp a f) g = C.comp (C.comp b f) g := by
      rwa [assoc, assoc]
    have h2 : C.comp a f = C.comp b f := hg.cancel_left _ _ h1
    exact hf.cancel_left _ _ h2

/-- The composition of two epimorphisms is an epimorphism. -/
theorem epi_comp {x y z : C.Obj} {f : C.Hom x y} {g : C.Hom y z}
    (hf : Epi f) (hg : Epi g) : Epi (C.comp f g) where
  cancel_right := fun a b h => by
    have h1 : C.comp f (C.comp g a) = C.comp f (C.comp g b) := by
      rwa [← assoc, ← assoc]
    have h2 : C.comp g a = C.comp g b := hf.cancel_right _ _ h1
    exact hg.cancel_right _ _ h2

/-! ## Isomorphisms imply mono and epi -/

/-- The forward map of an isomorphism is a monomorphism. -/
theorem iso_implies_mono (i : Iso C x y) : Mono i.hom where
  cancel_left := fun g h hgh => by
    have := congr_arg (C.comp · i.inv) hgh
    simp only [assoc, Iso.hom_inv_simp, comp_id] at this
    exact this

/-- The forward map of an isomorphism is an epimorphism. -/
theorem iso_implies_epi (i : Iso C x y) : Epi i.hom where
  cancel_right := fun g h hgh => by
    have := congr_arg (C.comp i.inv) hgh
    simp only [← assoc, Iso.inv_hom_simp, id_comp] at this
    exact this

/-! ## Cancellation restated -/

/-- Left cancellation for monomorphisms. -/
theorem left_cancel_mono {x y : C.Obj} {f : C.Hom x y} (hf : Mono f)
    {w : C.Obj} {g h : C.Hom w x} (H : C.comp g f = C.comp h f) : g = h :=
  hf.cancel_left g h H

/-- Right cancellation for epimorphisms. -/
theorem right_cancel_epi {x y : C.Obj} {f : C.Hom x y} (hf : Epi f)
    {z : C.Obj} {g h : C.Hom y z} (H : C.comp f g = C.comp f h) : g = h :=
  hf.cancel_right g h H

/-! ## Split morphisms imply mono/epi -/

/-- A split monomorphism is a monomorphism. -/
theorem split_mono_implies_mono {x y : C.Obj} {f : C.Hom x y}
    (s : SplitMono f) : Mono f where
  cancel_left := fun g h hgh => by
    have := congr_arg (C.comp · s.retraction) hgh
    simp only [assoc, s.retraction_comp, comp_id] at this
    exact this

/-- A split epimorphism is an epimorphism. -/
theorem split_epi_implies_epi {x y : C.Obj} {f : C.Hom x y}
    (s : SplitEpi f) : Epi f where
  cancel_right := fun g h hgh => by
    have := congr_arg (C.comp s.section_) hgh
    simp only [← assoc, s.comp_section, id_comp] at this
    exact this

/-! ## Endomorphism monoid -/

/-- The endomorphism type of an object `x`. -/
def End (C : Category.{u, v}) (x : C.Obj) := C.Hom x x

/-- The identity endomorphism. -/
def End.id (C : Category.{u, v}) (x : C.Obj) : End C x := C.id x

/-- Composition of endomorphisms. -/
def End.comp {C : Category.{u, v}} {x : C.Obj} (f g : End C x) : End C x := C.comp f g

/-- Endomorphisms form a monoid under composition. -/
instance endMonoid (C : Category.{u, v}) (x : C.Obj) : Monoid (End C x) where
  mul f g := End.comp f g
  one := End.id C x
  mul_assoc f g h := assoc f g h
  one_mul f := id_comp f
  mul_one f := comp_id f

end LMS.Categories
