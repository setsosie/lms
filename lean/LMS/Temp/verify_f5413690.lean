import Mathlib.CategoryTheory.Adjunction.Limits
  
  namespace LMS.RAPL
  
  open CategoryTheory
  open CategoryTheory.Limits
  
  universe v₁ u₁ v₂ u₂
  
  variable {C : Type u₁} [Category.{v₁} C]
  variable {D : Type u₂} [Category.{v₂} D]
  
  variable (F : C ⥤ D) (G : D ⥤ C)
  
  /--
  If `F ⊣ G`, then `G` preserves all limits.
  
  This is already in Mathlib as `preservesLimits_of_adjunction`.
  -/
  theorem rightAdjoint_preservesLimits (adj : F ⊣ G) :
      PreservesLimits G :=
    CategoryTheory.Limits.preservesLimits_of_adjunction adj
  
  end LMS.RAPL