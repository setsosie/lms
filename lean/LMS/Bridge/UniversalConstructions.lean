import Mathlib.CategoryTheory.Limits.Shapes.BinaryProducts
import Mathlib.CategoryTheory.Limits.Shapes.Pullback.HasPullback
import Mathlib.CategoryTheory.Limits.Shapes.Equalizers
import Mathlib.CategoryTheory.Limits.Shapes.Terminal
import Mathlib.CategoryTheory.Yoneda
import Mathlib.CategoryTheory.RepresentedBy

/-!
# LMS.Bridge.UniversalConstructions — Mathlib coverage verification for categories.tex Sec 4-12

This file machine-verifies that Mathlib covers the definitions and lemmas
in Sections 4-12 of the Stacks Project `categories.tex`. It is NOT a wrapper:
downstream code should import Mathlib directly.

## Stacks-to-Mathlib mapping

| Stacks Section | Topic | Mathlib Module |
|----------------|-------|----------------|
| Sec 4 | Products of pairs | `Limits.Shapes.BinaryProducts` |
| Sec 5 | Coproducts of pairs | `Limits.Shapes.BinaryProducts` |
| Sec 6 | Fibre products (pullbacks) | `Limits.Shapes.Pullback.HasPullback` |
| Sec 7 | Examples of fibre products | (various `Shapes.*`) |
| Sec 8 | Fibre products & representability | `Yoneda`, `RepresentedBy` |
| Sec 9 | Pushouts | `Limits.Shapes.Pullback.HasPullback` |
| Sec 10 | Equalizers | `Limits.Shapes.Equalizers` |
| Sec 11 | Coequalizers | `Limits.Shapes.Equalizers` |
| Sec 12 | Initial and final objects | `Limits.Shapes.Terminal` |
-/

namespace LMS.Bridge.UniversalConstructions

open CategoryTheory CategoryTheory.Limits

variable {C : Type*} [Category C]

/-! ## Sec 4: Products of pairs (line 598) -/

-- Definition 4.1: Product of a pair of objects
#check @Limits.prod C _ -- `prod : C → C → C` (when `HasBinaryProducts`)
#check @HasBinaryProducts C _
#check @prod.fst C _ -- first projection
#check @prod.snd C _ -- second projection
#check @prod.lift C _ -- universal property: lift to product

/-! ## Sec 5: Coproducts of pairs (line 655) -/

-- Definition 5.1: Coproduct of a pair of objects
#check @coprod C _ -- `coprod : C → C → C`
#check @HasBinaryCoproducts C _
#check @coprod.inl C _ -- first injection
#check @coprod.inr C _ -- second injection
#check @coprod.desc C _ -- universal property: desc from coproduct

/-! ## Sec 6: Fibre products / Pullbacks (line 712) -/

-- Definition 6.1: Fibre product (= pullback)
#check @pullback C _ -- `pullback f g : C`
#check @HasPullbacks C _
#check @pullback.fst C _ -- first projection
#check @pullback.snd C _ -- second projection
#check @pullback.condition C _ -- `f ∘ fst = g ∘ snd`
#check @pullback.lift C _ -- universal property

/-! ## Sec 8: Fibre products and representability (line 928) -/

-- Representability
#check @Functor.IsRepresentable
#check @yoneda C _
#check @coyoneda C _

/-! ## Sec 9: Pushouts (line 1071) -/

-- Definition 9.1: Pushout
#check @pushout C _ -- `pushout f g : C`
#check @HasPushouts C _
#check @pushout.inl C _ -- left inclusion
#check @pushout.inr C _ -- right inclusion
#check @pushout.condition C _
#check @pushout.desc C _

/-! ## Sec 10: Equalizers (line 1131) -/

-- Definition 10.1: Equalizer
#check @equalizer C _ -- `equalizer f g : C`
#check @HasEqualizers C _
#check @equalizer.ι C _ -- inclusion
#check @equalizer.lift C _ -- universal property

/-! ## Sec 11: Coequalizers (line 1151) -/

-- Definition 11.1: Coequalizer
#check @coequalizer C _ -- `coequalizer f g : C`
#check @HasCoequalizers C _
#check @coequalizer.π C _ -- projection
#check @coequalizer.desc C _ -- universal property

/-! ## Sec 12: Initial and final objects (line 1172) -/

-- Definition 12.1: Final (terminal) object
#check @HasTerminal C _
#check @terminal C _
#check @IsTerminal C _
-- Definition 12.2: Initial object
#check @HasInitial C _
#check @initial C _
#check @IsInitial C _

/-! ## Sec 13: Monomorphisms and Epimorphisms (line 1197) -/

-- Covered by Mathlib's core category theory
#check @Mono C _ -- left-cancellable morphism
#check @Epi C _ -- right-cancellable morphism
#check @SplitMono C _ -- has retraction
#check @SplitEpi C _ -- has section
-- Also covered in LMS.Categories.Morphisms (WC-1)

end LMS.Bridge.UniversalConstructions
