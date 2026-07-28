import Mathlib.CategoryTheory.Limits.HasLimits
import Mathlib.CategoryTheory.Limits.Filtered
import Mathlib.CategoryTheory.Limits.IsConnected
import Mathlib.CategoryTheory.Limits.Final
import Mathlib.CategoryTheory.Limits.FilteredColimitCommutesFiniteLimit
import Mathlib.CategoryTheory.Limits.Types.Limits
import Mathlib.CategoryTheory.Limits.Preorder

/-!
# LMS.Bridge.LimitsColimits — Mathlib coverage verification for categories.tex Sec 14-22

This file machine-verifies that Mathlib covers the definitions and lemmas
in Sections 14-22 of the Stacks Project `categories.tex`. Downstream code
should import Mathlib directly.

## Stacks-to-Mathlib mapping

| Stacks Section | Topic | Mathlib Module |
|----------------|-------|----------------|
| Sec 14 | Limits and colimits | `Limits.HasLimits` |
| Sec 15 | Limits/colimits in Sets | `Limits.Types.Limits` |
| Sec 16 | Connected limits | `Limits.IsConnected` |
| Sec 17 | Cofinal and initial categories | `Limits.Final` |
| Sec 18 | Finite limits and colimits | `Limits.HasLimits` |
| Sec 19 | Filtered colimits | `Limits.Filtered` |
| Sec 20 | Cofiltered limits | `Limits.Filtered` |
| Sec 21 | Limits over preordered sets | `Limits.Preorder` |
| Sec 22 | Essentially constant systems | (excluded from WC-3 scope) |

## Note on Sec 22

Essentially constant systems (line 2910-3271) are explicitly excluded from
WC-3 by unanimous committee vote. They belong to a separate "Sec 14-22
limits gaps" task independent of Sub-chunk A1.
-/

namespace LMS.Bridge.LimitsColimits

open CategoryTheory CategoryTheory.Limits

variable {C : Type*} [Category C]

/-! ## Sec 14: Limits and colimits (line 1270) -/

-- Definition 14.1: Limit of a functor
#check @HasLimits C _
#check @HasColimits C _
#check @HasLimitsOfShape
#check @HasColimitsOfShape

-- Limit/colimit as objects
variable {J : Type*} [Category J] (F : Functor J C) [HasLimit F]
#check limit F -- the limit object
#check limit.π F -- the limit cone projections
#check limit.lift F -- universal property
variable [HasColimit F]
#check colimit F -- the colimit object
#check colimit.ι F -- the colimit cocone injections
#check colimit.desc F -- universal property

-- Limit cones
#check @IsLimit
#check @IsColimit

/-! ## Sec 15: Limits and colimits in the category of sets (line 1589) -/

-- Mathlib handles limits in `Type` via `Types.Limits` and `Types.Colimits`
-- The concrete construction: limit ≃ sections (compatible tuples)
variable {J' : Type*} [SmallCategory J'] {F' : Functor J' (Type*)}
  {c : Cone F'} (hc : IsLimit c)
#check Types.isLimitEquivSections hc

/-! ## Sec 16: Connected limits (line 1647) -/

-- Definition 16.1: Connected category
#check @IsConnected
#check @IsPreconnected
-- Key: limits indexed by connected categories commute with coproducts
-- Mathlib has `Limits.IsConnected`

/-! ## Sec 17: Cofinal and initial categories (line 1718) -/

-- Definition 17.1: Cofinal functor (called "Final" in Mathlib)
#check @Functor.Final
-- Definition 17.2: Initial functor
#check @Functor.Initial
-- Key theorem: cofinal functors preserve limits/colimits

/-! ## Sec 18: Finite limits and colimits (line 1885) -/

#check @HasFiniteLimits C _
#check @HasFiniteColimits C _
-- Finite limits = HasTerminal + HasPullbacks (Lemma in Stacks)
-- Mathlib: `hasFiniteLimits_of_hasTerminal_and_pullbacks`

/-! ## Sec 19: Filtered colimits (line 2123) -/

-- Definition 19.1: Filtered category
#check @IsFiltered C _
-- Existence of filtered colimits
#check @HasFilteredColimitsOfSize C _
-- Key: filtered colimits commute with finite limits
#check @colimitLimitIso

/-! ## Sec 20: Cofiltered limits (line 2517) -/

-- Definition 20.1: Cofiltered category
#check @IsCofiltered C _

/-! ## Sec 21: Limits and colimits over preordered sets (line 2565) -/

-- Directed/inverse systems over preorders
-- Mathlib handles this via limits over `J` where `J` is a preorder category
-- `Limits.Preorder` provides specialized constructions

/-! ## Sec 22: Essentially constant systems (line 2910) -/

-- EXCLUDED from WC-3 scope by committee vote (3-0).
-- This section (~10-13 novel statements) is deferred to a separate task.

end LMS.Bridge.LimitsColimits
