import Mathlib.CategoryTheory.Adjunction.Basic
import Mathlib.CategoryTheory.Adjunction.Limits
import Mathlib.CategoryTheory.Limits.Preserves.Basic
import Mathlib.CategoryTheory.Limits.Preserves.Finite

/-!
# LMS.Bridge.FunctorialProperties — Mathlib coverage verification for categories.tex Sec 23-24

This file machine-verifies that Mathlib covers the definitions and lemmas
in Sections 23-24 of the Stacks Project `categories.tex`. Downstream code
should import Mathlib directly.

## Stacks-to-Mathlib mapping

| Stacks Section | Topic | Mathlib Module |
|----------------|-------|----------------|
| Sec 23 | Exact functors | `Limits.Preserves.*` |
| Sec 24 | Adjoint functors | `Adjunction.*` |

## Terminology note

The Stacks Project uses "exact functor" for what Mathlib calls a functor
that preserves (finite) limits and/or colimits. In Mathlib:
- "Left exact" = `PreservesFiniteLimits`
- "Right exact" = `PreservesFiniteColimits`
- "Exact" = both
-/

namespace LMS.Bridge.FunctorialProperties

open CategoryTheory CategoryTheory.Limits

variable {C D : Type*} [Category C] [Category D]

/-! ## Sec 23: Exact functors (line 3272) -/

-- Definition 23.1: A functor that preserves limits
variable (F : Functor C D)
#check @PreservesLimitsOfSize C _ D _ F
#check @PreservesColimitsOfSize C _ D _ F

-- Finite versions (left/right exact)
#check @PreservesFiniteLimits C _ D _ F
#check @PreservesFiniteColimits C _ D _ F

-- Preserving specific shape limits
#check @PreservesLimit
#check @PreservesColimit
#check @PreservesLimitsOfShape
#check @PreservesColimitsOfShape

-- Reflects limits (detecting limits, mentioned in Stacks)
#check @ReflectsLimit
#check @ReflectsColimit
#check @ReflectsLimitsOfSize

/-! ## Sec 24: Adjoint functors (line 3340) -/

-- Definition 24.1: Adjunction
variable (G : Functor D C)
#check @Adjunction C _ D _ F G -- `F ⊣ G`

-- Adjunction via hom-set bijection
#check @Adjunction.mkOfHomEquiv

-- Adjunction via unit/counit
#check @Adjunction.mk'

-- Unit and counit of an adjunction
variable (adj : F ⊣ G)
#check adj.unit -- η : 𝟭 C ⟶ F ⋙ G
#check adj.counit -- ε : G ⋙ F ⟶ 𝟭 D

-- Key property: left adjoints preserve colimits (Lemma 24.4 in Stacks)
#check @Adjunction.leftAdjoint_preservesColimits C _ D _ F G adj

-- Key property: right adjoints preserve limits (Lemma 24.5 in Stacks)
#check @Adjunction.rightAdjoint_preservesLimits C _ D _ F G adj

-- Adjunction hom-set bijection
#check @Adjunction.homEquiv C _ D _ F G adj

end LMS.Bridge.FunctorialProperties
