/-
Copyright (c) 2026 LMS Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.

# Novel localization results (Stacks Project Sec 25-26)

## Stacks Project Coverage

- `sameDenom_eq_iff_exists_postcomp_W`: Tag 04VB (line 4303), (1) ↔ (2).
- `sameDenom_eq_iff_exists_postcomp`: Tag 04VB (line 4303), (1) ↔ (3).
- `exists_lift_commSq`: Tag 04VD (line 4463), diagram lifting.
- `isIso_map_iff_inSaturation`: Tag 05Q2 (line 4795), saturation characterization.

## Already in Mathlib

- `Localization.exists_leftFraction₂` / `₃`: common denominator (Tag 04VA).
- `LeftFraction.map_eq_iff` via `LeftFractionRel`.
- `map_eq_iff_postcomp` / `map_eq_iff_precomp`.
- Sec 26 interchange: `NatTrans.hcomp` and `Bicategory`.
-/

import Mathlib.CategoryTheory.Localization.CalculusOfFractions
import Mathlib.CategoryTheory.Localization.CalculusOfFractions.Fractions

namespace LMS.Categories.Localization

open CategoryTheory Category MorphismProperty

universe v u v' u'

variable {C : Type u} [Category.{v} C] {D : Type u'} [Category.{v'} D]
variable {W : MorphismProperty C}

/-! ### Same-denominator equality (Stacks Tag 04VB) -/

section SameDenominator

variable [W.HasLeftCalculusOfFractions]
variable (L : C ⥤ D) [L.IsLocalization W]

/-- Stacks Tag 04VB (1) ↔ (2). Two left fractions with the same denominator are equal
in the localized category iff their numerators can be equalized by postcomposing with
a morphism in `W`. -/
theorem sameDenom_eq_iff_exists_postcomp_W {X Y Y' : C}
    (f g : X ⟶ Y') (s : Y ⟶ Y') (hs : W s) :
    (LeftFraction.mk f s hs).map L (Localization.inverts L W) =
      (LeftFraction.mk g s hs).map L (Localization.inverts L W) ↔
    ∃ (Z : C) (t : Y' ⟶ Z), W t ∧ f ≫ t = g ≫ t := by
  constructor
  · intro h
    have hiso : IsIso (L.map s) := Localization.inverts L W _ hs
    -- Extract: φ.map ≫ L.map φ.s = L.map φ.f
    have eq1 : (LeftFraction.mk f s hs).map L (Localization.inverts L W) ≫ L.map s =
        L.map f := LeftFraction.map_comp_map_s _ _ _
    have eq2 : (LeftFraction.mk g s hs).map L (Localization.inverts L W) ≫ L.map s =
        L.map g := LeftFraction.map_comp_map_s _ _ _
    have eq : L.map f = L.map g := by
      rw [← eq1, ← eq2, h]
    rw [MorphismProperty.map_eq_iff_postcomp L W] at eq
    obtain ⟨Z, t, ht, hft⟩ := eq
    exact ⟨Z, t, ht, hft⟩
  · rintro ⟨Z, t, ht, hft⟩
    rw [LeftFraction.map_eq_iff L W]
    exact ⟨Z, t, t, rfl, hft, W.comp_mem _ _ hs ht⟩

/-- Stacks Tag 04VB (1) ↔ (3). Two left fractions with the same denominator are equal
in the localized category iff their numerators can be equalized by postcomposing with
some `a` with `s ≫ a ∈ W`. -/
theorem sameDenom_eq_iff_exists_postcomp {X Y Y' : C}
    (f g : X ⟶ Y') (s : Y ⟶ Y') (hs : W s) :
    (LeftFraction.mk f s hs).map L (Localization.inverts L W) =
      (LeftFraction.mk g s hs).map L (Localization.inverts L W) ↔
    ∃ (Z : C) (a : Y' ⟶ Z), f ≫ a = g ≫ a ∧ W (s ≫ a) := by
  rw [sameDenom_eq_iff_exists_postcomp_W L f g s hs]
  constructor
  · rintro ⟨Z, t, ht, hft⟩
    exact ⟨Z, t, hft, W.comp_mem _ _ hs ht⟩
  · rintro ⟨Z, a, hfa, hsa⟩
    -- We have f ≫ a = g ≫ a and W (s ≫ a). We need ∃ t, W t ∧ f ≫ t = g ≫ t.
    -- Since L.map s is iso (from W s) and L.map (s ≫ a) is iso (from W (s ≫ a)),
    -- L.map a = inv(L.map s) ≫ L.map(s ≫ a) is iso, hence mono.
    have : IsIso (L.map s) := Localization.inverts L W _ hs
    have : IsIso (L.map (s ≫ a)) := Localization.inverts L W _ hsa
    have : IsIso (L.map a) := by
      rw [L.map_comp] at *
      exact IsIso.of_isIso_comp_left (L.map s) (L.map a)
    have : L.map f = L.map g := by
      rw [← cancel_mono (L.map a), ← L.map_comp, ← L.map_comp, hfa]
    rw [MorphismProperty.map_eq_iff_postcomp L W] at this
    obtain ⟨Z', t, ht, hft⟩ := this
    exact ⟨Z', t, ht, hft⟩

end SameDenominator

/-! ### Diagram lifting (Stacks Tag 04VD) -/

section DiagramLifting

variable [W.HasLeftCalculusOfFractions]
variable (L : C ⥤ D) [L.IsLocalization W]

/-- Stacks Tag 04VD. Given `f`, `f'` in `C` and a commutative square in the localized
category, the square lifts to `C` with denominators in `W`. -/
theorem exists_lift_commSq {X Y X' Y' : C}
    (f : X ⟶ Y) (f' : X' ⟶ Y')
    (a : L.obj X ⟶ L.obj X') (b : L.obj Y ⟶ L.obj Y')
    (sq : a ≫ L.map f' = L.map f ≫ b) :
    ∃ (X'' Y'' : C) (g : X ⟶ X'') (s : X' ⟶ X'') (_ : W s)
      (f'' : X'' ⟶ Y'') (h : Y ⟶ Y'') (t : Y' ⟶ Y'') (_ : W t),
      a = (LeftFraction.mk g s ‹W s›).map L (Localization.inverts L W) ∧
      b = (LeftFraction.mk h t ‹W t›).map L (Localization.inverts L W) ∧
      s ≫ f'' = f' ≫ t ∧
      g ≫ f'' = f ≫ h := by
  -- Step 1: Write a as a left fraction
  obtain ⟨φ_a, hφ_a⟩ := Localization.exists_leftFraction L W a
  -- Step 2: Ore condition on (φ_a.s, f') to get commutative square
  have hψ_raw := RightFraction.exists_leftFraction
    (W := W) (RightFraction.mk φ_a.s φ_a.hs f')
  obtain ⟨ψ, hψ⟩ := hψ_raw
  -- hψ : f' ≫ ψ.s = φ_a.s ≫ ψ.f (after simp on the RightFraction fields)
  change f' ≫ ψ.s = φ_a.s ≫ ψ.f at hψ
  have hs_iso : IsIso (L.map φ_a.s) := Localization.inverts L W _ φ_a.hs
  have ht_iso : IsIso (L.map ψ.s) := Localization.inverts L W _ ψ.hs
  -- Step 3: Compute L.map f ≫ (b ≫ L.map ψ.s) = L.map (φ_a.f ≫ ψ.f)
  have hmap := LeftFraction.map_comp_map_s φ_a L (Localization.inverts L W)
  have key : L.map f ≫ (b ≫ L.map ψ.s) = L.map (φ_a.f ≫ ψ.f) := by
    calc L.map f ≫ (b ≫ L.map ψ.s)
        = (L.map f ≫ b) ≫ L.map ψ.s := by rw [assoc]
      _ = (a ≫ L.map f') ≫ L.map ψ.s := by rw [← sq]
      _ = a ≫ (L.map f' ≫ L.map ψ.s) := by rw [assoc]
      _ = a ≫ L.map (f' ≫ ψ.s) := by rw [L.map_comp]
      _ = a ≫ L.map (φ_a.s ≫ ψ.f) := by rw [hψ]
      _ = a ≫ (L.map φ_a.s ≫ L.map ψ.f) := by rw [L.map_comp]
      _ = (a ≫ L.map φ_a.s) ≫ L.map ψ.f := by rw [assoc]
      _ = (φ_a.map L (Localization.inverts L W) ≫ L.map φ_a.s) ≫ L.map ψ.f := by rw [hφ_a]
      _ = L.map φ_a.f ≫ L.map ψ.f := by rw [hmap]
      _ = L.map (φ_a.f ≫ ψ.f) := by rw [L.map_comp]
  -- Step 4: Express c = b ≫ L.map ψ.s as a left fraction
  obtain ⟨φ_c, hφ_c⟩ := Localization.exists_leftFraction L W (b ≫ L.map ψ.s)
  -- Step 5: From key, derive L.map (f ≫ φ_c.f) = L.map (φ_a.f ≫ ψ.f ≫ φ_c.s)
  have key2 : L.map (f ≫ φ_c.f) = L.map (φ_a.f ≫ ψ.f ≫ φ_c.s) := by
    have hmaps := LeftFraction.map_comp_map_s φ_c L (Localization.inverts L W)
    calc L.map (f ≫ φ_c.f)
        = L.map f ≫ L.map φ_c.f := by rw [L.map_comp]
      _ = L.map f ≫ (φ_c.map L (Localization.inverts L W) ≫ L.map φ_c.s) := by rw [hmaps]
      _ = (L.map f ≫ φ_c.map L (Localization.inverts L W)) ≫ L.map φ_c.s := by rw [assoc]
      _ = (L.map f ≫ (b ≫ L.map ψ.s)) ≫ L.map φ_c.s := by rw [hφ_c]
      _ = (L.map f ≫ b ≫ L.map ψ.s) ≫ L.map φ_c.s := by rw [assoc]
      _ = L.map (φ_a.f ≫ ψ.f) ≫ L.map φ_c.s := by rw [key]
      _ = L.map (φ_a.f ≫ ψ.f ≫ φ_c.s) := by rw [← L.map_comp, assoc]
  rw [MorphismProperty.map_eq_iff_postcomp L W] at key2
  obtain ⟨Z'', w, hw, key3⟩ := key2
  -- key3 : (f ≫ φ_c.f) ≫ w = (φ_a.f ≫ ψ.f ≫ φ_c.s) ≫ w
  -- Step 6: Define the output data
  refine ⟨φ_a.Y', Z'', φ_a.f, φ_a.s, φ_a.hs, ψ.f ≫ φ_c.s ≫ w,
    φ_c.f ≫ w, ψ.s ≫ φ_c.s ≫ w,
    W.comp_mem _ _ ψ.hs (W.comp_mem _ _ φ_c.hs hw),
    hφ_a, ?_, ?_, ?_⟩
  -- Goal 1: b = fraction(φ_c.f ≫ w, ψ.s ≫ φ_c.s ≫ w)
  · have ht' : W (ψ.s ≫ φ_c.s ≫ w) := W.comp_mem _ _ ψ.hs (W.comp_mem _ _ φ_c.hs hw)
    have : IsIso (L.map (ψ.s ≫ φ_c.s ≫ w)) := Localization.inverts L W _ ht'
    rw [← cancel_mono (L.map (ψ.s ≫ φ_c.s ≫ w))]
    rw [LeftFraction.map_comp_map_s]
    simp only [L.map_comp]
    -- Goal: b ≫ L.map ψ.s ≫ L.map φ_c.s ≫ L.map w = L.map φ_c.f ≫ L.map w
    rw [← assoc, ← assoc]
    congr 1
    rw [assoc]
    -- Goal: b ≫ L.map ψ.s ≫ L.map φ_c.s = L.map φ_c.f
    calc b ≫ L.map ψ.s ≫ L.map φ_c.s
        = (b ≫ L.map ψ.s) ≫ L.map φ_c.s := by rw [assoc]
      _ = φ_c.map L (Localization.inverts L W) ≫ L.map φ_c.s := by rw [hφ_c]
      _ = L.map φ_c.f := LeftFraction.map_comp_map_s _ _ _
  -- Goal 2: φ_a.s ≫ ψ.f ≫ φ_c.s ≫ w = f' ≫ ψ.s ≫ φ_c.s ≫ w
  · calc φ_a.s ≫ ψ.f ≫ φ_c.s ≫ w
        = (φ_a.s ≫ ψ.f) ≫ φ_c.s ≫ w := by rw [assoc]
      _ = (f' ≫ ψ.s) ≫ φ_c.s ≫ w := by rw [hψ.symm]
      _ = f' ≫ ψ.s ≫ φ_c.s ≫ w := by rw [assoc]
  -- Goal 3: φ_a.f ≫ ψ.f ≫ φ_c.s ≫ w = f ≫ φ_c.f ≫ w
  · simp only [← assoc] at key3 ⊢
    exact key3.symm

end DiagramLifting

/-! ### Saturation characterization (Stacks Tag 05Q2) -/

section Saturation

/-- A morphism `f` is in the saturation of `W` if there exist `g`, `h` with
`g ≫ f ∈ W` and `f ≫ h ∈ W`. -/
def InSaturation {X Y : C} (f : X ⟶ Y) : Prop :=
  ∃ (Z₁ Z₂ : C) (g : Z₁ ⟶ X) (h : Y ⟶ Z₂), W (g ≫ f) ∧ W (f ≫ h)

/-- Any morphism in `W` is in the saturation. -/
theorem W_subset_saturation {X Y : C} (f : X ⟶ Y) (hf : W f) :
    InSaturation (W := W) f :=
  ⟨X, Y, 𝟙 X, 𝟙 Y, by simpa using hf, by simpa using hf⟩

variable [W.HasLeftCalculusOfFractions]
variable [W.HasRightCalculusOfFractions]
variable (L : C ⥤ D) [L.IsLocalization W]

omit [W.HasLeftCalculusOfFractions] [W.HasRightCalculusOfFractions] in
/-- If `f` is in the saturation of `W`, then `L.map f` is an isomorphism. -/
theorem isIso_map_of_inSaturation {X Y : C} (f : X ⟶ Y)
    (hf : InSaturation (W := W) f) : IsIso (L.map f) := by
  obtain ⟨Z₁, Z₂, g, h, hgf, hfh⟩ := hf
  have hgf_iso : IsIso (L.map (g ≫ f)) := Localization.inverts L W _ hgf
  have hfh_iso : IsIso (L.map (f ≫ h)) := Localization.inverts L W _ hfh
  rw [L.map_comp] at hgf_iso
  rw [L.map_comp] at hfh_iso
  -- From gf iso: f is split epi (section = inv(gf) ≫ g, and section ≫ f = 𝟙)
  have hse : IsSplitEpi (L.map f) :=
    IsSplitEpi.mk' ⟨inv (L.map g ≫ L.map f) ≫ L.map g, by simp [assoc]⟩
  -- From fh iso: f is split mono (retraction = h ≫ inv(fh), and f ≫ retraction = 𝟙)
  have hsm : IsSplitMono (L.map f) :=
    IsSplitMono.mk' ⟨L.map h ≫ inv (L.map f ≫ L.map h), by simp [← assoc]⟩
  exact isIso_of_mono_of_isSplitEpi (L.map f)

/-- Auxiliary: from `L.map f` iso, derive `f ≫ h ∈ W` for some `h` using left fractions. -/
private theorem exists_postcomp_in_W_of_isIso_map {X Y : C} (f : X ⟶ Y)
    (hf : IsIso (L.map f)) :
    ∃ (Z : C) (h : Y ⟶ Z), W (f ≫ h) := by
  obtain ⟨φ, hφ⟩ := Localization.exists_leftFraction L W (inv (L.map f))
  have : IsIso (L.map φ.s) := Localization.inverts L W _ φ.hs
  have eq1 : L.map f ≫ φ.map L (Localization.inverts L W) = 𝟙 _ := by
    rw [← hφ, IsIso.hom_inv_id]
  have eq : L.map (f ≫ φ.f) = L.map φ.s := by
    rw [L.map_comp]
    have := LeftFraction.map_comp_map_s φ L (Localization.inverts L W)
    -- this : φ.map ≫ L.map φ.s = L.map φ.f
    calc L.map f ≫ L.map φ.f
        = L.map f ≫ (φ.map L (Localization.inverts L W) ≫ L.map φ.s) := by rw [this]
      _ = (L.map f ≫ φ.map L (Localization.inverts L W)) ≫ L.map φ.s := by rw [assoc]
      _ = 𝟙 _ ≫ L.map φ.s := by rw [eq1]
      _ = L.map φ.s := by rw [id_comp]
  rw [MorphismProperty.map_eq_iff_postcomp L W] at eq
  obtain ⟨Z, u, hu, fac⟩ := eq
  exact ⟨Z, φ.f ≫ u, by rw [← assoc, fac]; exact W.comp_mem _ _ φ.hs hu⟩

omit [W.HasLeftCalculusOfFractions] in
/-- Auxiliary: from `L.map f` iso, derive `g ≫ f ∈ W` for some `g` using right fractions. -/
private theorem exists_precomp_in_W_of_isIso_map {X Y : C} (f : X ⟶ Y)
    (hf : IsIso (L.map f)) :
    ∃ (Z : C) (g : Z ⟶ X), W (g ≫ f) := by
  obtain ⟨φ, hφ⟩ := Localization.exists_rightFraction L W (inv (L.map f))
  have : IsIso (L.map φ.s) := Localization.inverts L W _ φ.hs
  have eq1 : φ.map L (Localization.inverts L W) ≫ L.map f = 𝟙 _ := by
    rw [← hφ, IsIso.inv_hom_id]
  have eq : L.map (φ.f ≫ f) = L.map φ.s := by
    rw [L.map_comp]
    have := RightFraction.map_s_comp_map φ L (Localization.inverts L W)
    -- this : L.map φ.s ≫ φ.map = L.map φ.f
    calc L.map φ.f ≫ L.map f
        = (L.map φ.s ≫ φ.map L (Localization.inverts L W)) ≫ L.map f := by rw [this]
      _ = L.map φ.s ≫ (φ.map L (Localization.inverts L W) ≫ L.map f) := by rw [assoc]
      _ = L.map φ.s ≫ 𝟙 _ := by rw [eq1]
      _ = L.map φ.s := by rw [comp_id]
  rw [MorphismProperty.map_eq_iff_precomp L W] at eq
  obtain ⟨Z, u, hu, fac⟩ := eq
  exact ⟨Z, u ≫ φ.f, by rw [assoc, fac]; exact W.comp_mem _ _ hu φ.hs⟩

/-- Stacks Tag 05Q2. If `L.map f` is an isomorphism, then `f` is in the saturation. -/
theorem inSaturation_of_isIso_map {X Y : C} (f : X ⟶ Y)
    (hf : IsIso (L.map f)) : InSaturation (W := W) f := by
  obtain ⟨Z₁, h, hfh⟩ := exists_postcomp_in_W_of_isIso_map (W := W) L f hf
  obtain ⟨Z₂, g, hgf⟩ := exists_precomp_in_W_of_isIso_map (W := W) L f hf
  exact ⟨Z₂, Z₁, g, h, hgf, hfh⟩

/-- Stacks Tag 05Q2. `L.map f` is invertible iff `f` is in the saturation of `W`. -/
theorem isIso_map_iff_inSaturation {X Y : C} (f : X ⟶ Y) :
    IsIso (L.map f) ↔ InSaturation (W := W) f :=
  ⟨inSaturation_of_isIso_map L f, isIso_map_of_inSaturation L f⟩

end Saturation

end LMS.Categories.Localization
