import LMS.Categories.IsoComma

/-!
# 2-Fibre Product Properties

Further properties of the IsoComma construction (2-fibre product of categories),
corresponding to Stacks Project Section 29.

## Main definitions

* `IsoComma.mapFunctor` — Functoriality (Stacks Lemma 003T)
* `IsoComma.assocForward` / `IsoComma.assocBackward` — Associativity functors
* `IsoComma.assocEquiv` — Associativity equivalence (Stacks Lemma 003V)
-/

open CategoryTheory

universe v₁ v₂ v₃ v₄ v₅ v₆ u₁ u₂ u₃ u₄ u₅ u₆

namespace LMS.Categories.IsoComma

/-! ### Functoriality (Stacks Lemma 003T)

Given a 2-commutative diagram with functors `LA : A₁ ⥤ A₂`, `LB : B₁ ⥤ B₂`, `M : T₁ ⥤ T₂`
and natural isomorphisms `α : LA ⋙ F ≅ H ⋙ M` and `β : LB ⋙ G ≅ I ⋙ M`, we get
a functor `IsoComma H I ⥤ IsoComma F G`.

On objects: `(X, Y, φ) ↦ (LA(X), LB(Y), α_X ≫ M(φ) ≫ β⁻¹_Y)`.
On morphisms: `(a, b) ↦ (LA(a), LB(b))`.
-/
section Functoriality

variable {A₁ : Type u₁} [Category.{v₁} A₁] {B₁ : Type u₂} [Category.{v₂} B₁]
  {T₁ : Type u₃} [Category.{v₃} T₁]
  {A₂ : Type u₄} [Category.{v₄} A₂] {B₂ : Type u₅} [Category.{v₅} B₂]
  {T₂ : Type u₆} [Category.{v₆} T₂]

/-- Functoriality of the 2-fibre product (Stacks 003T / lemma-functoriality-2-fibre-product).

Given a 2-commutative diagram, we get a functor between IsoComma categories.
The iso on objects is `α_X ≫ M(φ) ≫ β⁻¹_Y : F(LA(X)) ≅ G(LB(Y))`. -/
def mapFunctor
    {H : A₁ ⥤ T₁} {I : B₁ ⥤ T₁}
    {F : A₂ ⥤ T₂} {G : B₂ ⥤ T₂}
    (LA : A₁ ⥤ A₂) (LB : B₁ ⥤ B₂) (M : T₁ ⥤ T₂)
    (α : LA ⋙ F ≅ H ⋙ M) (β : LB ⋙ G ≅ I ⋙ M) :
    IsoComma H I ⥤ IsoComma F G where
  obj p :=
    { left := LA.obj p.left
      right := LB.obj p.right
      hom := (α.app p.left) ≪≫ M.mapIso p.hom ≪≫ (β.app p.right).symm }
  map {p q} f :=
    { left := LA.map f.left
      right := LB.map f.right
      w := by
        show F.map (LA.map f.left) ≫
          (α.hom.app q.left ≫ M.map q.hom.hom ≫ β.inv.app q.right) =
          (α.hom.app p.left ≫ M.map p.hom.hom ≫ β.inv.app p.right) ≫
          G.map (LB.map f.right)
        have hα := α.hom.naturality f.left
        have hβ := β.inv.naturality f.right
        simp only [Functor.comp_map] at hα hβ
        rw [← Category.assoc (F.map _), hα, Category.assoc,
            ← Category.assoc (M.map _) (M.map _), ← M.map_comp, f.w,
            M.map_comp, Category.assoc, hβ,
            ← Category.assoc, ← Category.assoc, ← Category.assoc] }
  map_id _ := by apply Hom.ext <;> simp
  map_comp _ _ := by apply Hom.ext <;> simp

end Functoriality

/-! ### Associativity (Stacks Lemma 003V)

Given a diagram `A → B ← C → D ← E`, there is a canonical equivalence
`(A ×_B C) ×_D E ≌ A ×_B (C ×_D E)`.

The forward functor sends `((a, c, φ), e, ψ)` to `(a, (c, e, ψ), φ)`.
The inverse sends `(a, (c, e, ψ), φ)` to `((a, c, φ), e, ψ)`.
Both round-trips are definitional identities.
-/
section Assoc

variable {A' : Type u₁} [Category.{v₁} A']
  {B' : Type u₂} [Category.{v₂} B']
  {C' : Type u₃} [Category.{v₃} C']
  {D' : Type u₄} [Category.{v₄} D']
  {E' : Type u₅} [Category.{v₅} E']

/-- The forward associativity functor
`(A ×_B C) ×_D E ⥤ A ×_B (C ×_D E)`.
Sends `((a, c, φ), e, ψ)` to `(a, (c, e, ψ), φ)`. -/
def assocForward (F_AB : A' ⥤ B') (F_CB : C' ⥤ B') (F_CD : C' ⥤ D') (F_ED : E' ⥤ D') :
    IsoComma ((snd : IsoComma F_AB F_CB ⥤ C') ⋙ F_CD) F_ED ⥤
    IsoComma F_AB ((fst : IsoComma F_CD F_ED ⥤ C') ⋙ F_CB) where
  obj p := ⟨p.left.left, ⟨p.left.right, p.right, p.hom⟩, p.left.hom⟩
  map f := ⟨f.left.left, ⟨f.left.right, f.right, f.w⟩, f.left.w⟩

/-- The backward associativity functor
`A ×_B (C ×_D E) ⥤ (A ×_B C) ×_D E`.
Sends `(a, (c, e, ψ), φ)` to `((a, c, φ), e, ψ)`. -/
def assocBackward (F_AB : A' ⥤ B') (F_CB : C' ⥤ B') (F_CD : C' ⥤ D') (F_ED : E' ⥤ D') :
    IsoComma F_AB ((fst : IsoComma F_CD F_ED ⥤ C') ⋙ F_CB) ⥤
    IsoComma ((snd : IsoComma F_AB F_CB ⥤ C') ⋙ F_CD) F_ED where
  obj q := ⟨⟨q.left, q.right.left, q.hom⟩, q.right.right, q.right.hom⟩
  map g := ⟨⟨g.left, g.right.left, g.w⟩, g.right.right, g.right.w⟩

/-- Associativity of 2-fibre products (Stacks Lemma 003V).

`(A ×_B C) ×_D E ≌ A ×_B (C ×_D E)`.

Both round-trip compositions are definitional identities, so the unit and counit
natural isomorphisms are both `Iso.refl`. -/
def assocEquiv (F_AB : A' ⥤ B') (F_CB : C' ⥤ B') (F_CD : C' ⥤ D') (F_ED : E' ⥤ D') :
    IsoComma ((snd : IsoComma F_AB F_CB ⥤ C') ⋙ F_CD) F_ED ≌
    IsoComma F_AB ((fst : IsoComma F_CD F_ED ⥤ C') ⋙ F_CB) where
  functor := assocForward F_AB F_CB F_CD F_ED
  inverse := assocBackward F_AB F_CB F_CD F_ED
  unitIso := Iso.refl _
  counitIso := Iso.refl _

end Assoc

end LMS.Categories.IsoComma
