/-
LMS (LLM Mathematical Society) - Commutative Algebra Formalization
Source: Stacks Project Chapter 10 (https://stacks.math.columbia.edu/tag/00AO)

Artifacts generated via multi-agent simulation (5 generations, 2 agents)
Verification: All proofs checked by LEAN 4 oracle
Coverage: Milestones 1-3 (Foundations, Localization, Nakayama)
-/
import Mathlib.Algebra.Ring.Defs
import Mathlib.Algebra.Ring.Basic
import Mathlib.RingTheory.Ideal.Basic
import Mathlib.RingTheory.Ideal.Span
import Mathlib.RingTheory.Ideal.Quotient.Basic
import Mathlib.RingTheory.Ideal.Quotient.Operations
import Mathlib.RingTheory.Ideal.Prime
import Mathlib.Algebra.Module.Basic
import Mathlib.RingTheory.Localization.AtPrime.Basic
import Mathlib.RingTheory.Nakayama

namespace LMS

/-!
# Generation 0: Foundation Artifacts
Fresh creations to bootstrap the knowledge base
-/

section Gen0

variable {R : Type*} [CommRing R] (I : Ideal R)

/-- Artifact: lemma-ideal_closure_ops (Agent-0, Gen0)
    Ideals are closed under negation -/
lemma ideal_neg_mem {a : R} (ha : a ∈ I) : -a ∈ I := I.neg_mem ha

/-- Ideals are closed under subtraction -/
lemma ideal_sub_mem {a b : R} (ha : a ∈ I) (hb : b ∈ I) : a - b ∈ I := I.sub_mem ha hb

/-- Zero is always in an ideal -/
lemma ideal_zero_mem : (0 : R) ∈ I := I.zero_mem

/-- Artifact: lemma-span_membership (Agent-1, Gen0)
    Element is in the ideal it generates -/
lemma mem_span_singleton (a : R) : a ∈ Ideal.span {a} :=
  Ideal.subset_span (Set.mem_singleton a)

/-- Multiples are in the principal ideal -/
lemma mul_mem_span_singleton (a r : R) : r * a ∈ Ideal.span {a} :=
  Ideal.mul_mem_left _ r (mem_span_singleton a)

/-- Characterization: x ∈ (a) iff a | x -/
lemma mem_span_singleton_iff (a x : R) : x ∈ Ideal.span {a} ↔ a ∣ x :=
  Ideal.mem_span_singleton

end Gen0

/-!
# Generation 1: Building on Gen0
References: span_membership, ideal_closure_ops
-/

section Gen1

variable {R : Type*} [CommRing R]

/-- Artifact: theorem-span_containment (Agent-0, Gen1)
    References: [lemma-span_membership]
    Principal ideal containment characterized by divisibility -/
theorem span_singleton_le_iff (a b : R) :
    Ideal.span {a} ≤ Ideal.span {b} ↔ b ∣ a := by
  rw [Ideal.span_le]
  simp only [Set.singleton_subset_iff, SetLike.mem_coe]
  rw [Ideal.mem_span_singleton]

/-- Principal ideal equality via mutual divisibility -/
lemma span_singleton_eq_iff (a b : R) :
    Ideal.span {a} = Ideal.span {b} ↔ (a ∣ b ∧ b ∣ a) := by
  constructor
  · intro h
    constructor
    · have hb : b ∈ Ideal.span {a} := h.symm ▸ Ideal.subset_span (Set.mem_singleton b)
      exact Ideal.mem_span_singleton.mp hb
    · have ha : a ∈ Ideal.span {b} := h ▸ Ideal.subset_span (Set.mem_singleton a)
      exact Ideal.mem_span_singleton.mp ha
  · intro ⟨hab, hba⟩
    apply le_antisymm
    · rw [span_singleton_le_iff]; exact hba
    · rw [span_singleton_le_iff]; exact hab

variable (I : Ideal R)

/-- Artifact: lemma-quotient_basics (Agent-1, Gen1)
    References: [lemma-ideal_closure_ops]
    Elements map to same quotient iff difference in ideal -/
lemma quotient_eq_iff (a b : R) :
    Ideal.Quotient.mk I a = Ideal.Quotient.mk I b ↔ a - b ∈ I := by
  rw [Ideal.Quotient.eq]

/-- Zero in quotient iff element in ideal -/
lemma quotient_eq_zero_iff (a : R) :
    Ideal.Quotient.mk I a = 0 ↔ a ∈ I :=
  Ideal.Quotient.eq_zero_iff_mem

/-- Quotient map is surjective -/
lemma quotient_mk_surjective : Function.Surjective (Ideal.Quotient.mk I) :=
  Ideal.Quotient.mk_surjective

end Gen1

/-!
# Generation 2: Deeper Theorems
References: Multiple Gen0 and Gen1 artifacts combined
-/

section Gen2

variable {R S : Type*} [CommRing R] [CommRing S]

/-- Artifact: theorem-first_isomorphism (Agent-0, Gen2)
    References: [lemma-quotient_basics, theorem-span_containment]
    Kernel membership characterization -/
lemma ker_is_ideal (f : R →+* S) : ∀ a ∈ RingHom.ker f, ∀ r : R, r * a ∈ RingHom.ker f := by
  intro a ha r
  simp only [RingHom.mem_ker] at ha ⊢
  rw [map_mul, ha, mul_zero]

/-- Element in kernel iff maps to zero -/
lemma mem_ker_iff (f : R →+* S) (a : R) : a ∈ RingHom.ker f ↔ f a = 0 :=
  RingHom.mem_ker

/-- First Isomorphism Theorem: R/ker(f) ≅ im(f) for surjective f -/
theorem first_isomorphism (f : R →+* S) (hf : Function.Surjective f) :
    Nonempty (R ⧸ RingHom.ker f ≃+* S) :=
  ⟨RingHom.quotientKerEquivOfSurjective hf⟩

variable (P : Ideal R) [hP : P.IsPrime]

/-- Artifact: theorem-prime_ideals (Agent-1, Gen2)
    References: [lemma-ideal_closure_ops, lemma-quotient_basics]
    Prime ideal: if ab ∈ P then a ∈ P or b ∈ P -/
lemma prime_mem_or_mem {a b : R} (hab : a * b ∈ P) : a ∈ P ∨ b ∈ P :=
  hP.mem_or_mem hab

/-- Complement of prime ideal is multiplicatively closed -/
lemma prime_complement_mul_closed {a b : R} (ha : a ∉ P) (hb : b ∉ P) : a * b ∉ P := by
  intro hab
  cases prime_mem_or_mem P hab with
  | inl h => exact ha h
  | inr h => exact hb h

/-- R/P is an integral domain when P is prime -/
theorem quotient_prime_is_domain : IsDomain (R ⧸ P) :=
  Ideal.Quotient.isDomain P

end Gen2

/-!
# Generation 3: Localization (Stacks 00CM)
References: prime_ideals, localization structures
-/

section Gen3

variable {R : Type*} [CommRing R] (P : Ideal R) [hP : P.IsPrime]

/-- Artifact: lemma-localization_basics (Agent-0, Gen3)
    References: [theorem-prime_ideals]
    The prime complement as a submonoid -/
lemma primeCompl_is_submonoid : P.primeCompl.carrier = {x | x ∉ P} := rfl

/-- Elements outside P become units after localization -/
lemma not_mem_becomes_unit (r : R) (hr : r ∈ P.primeCompl) :
    IsUnit (algebraMap R (Localization.AtPrime P) r) :=
  IsLocalization.map_units (Localization.AtPrime P) ⟨r, hr⟩

/-- The canonical map from R to localization at P -/
def toLocalization : R →+* Localization.AtPrime P :=
  algebraMap R (Localization.AtPrime P)

/-- Artifact: lemma-local_ring_structure (Agent-1, Gen3)
    References: [lemma-localization_basics, theorem-prime_ideals]
    The localization at a prime ideal is a local ring -/
lemma locAtPrimeIsLocal : IsLocalRing (Localization.AtPrime P) :=
  inferInstance

/-- The maximal ideal of R_P is the image of P -/
def maxIdealAtPrime : Ideal (Localization.AtPrime P) :=
  IsLocalRing.maximalIdeal (Localization.AtPrime P)

/-- Image of P = maximal ideal (fundamental connection) -/
lemma map_P_eq_maximal :
    Ideal.map (algebraMap R (Localization.AtPrime P)) P = maxIdealAtPrime P :=
  Localization.AtPrime.map_eq_maximalIdeal

/-- The residue field R_P / m_P -/
abbrev residueFieldAtPrime := (Localization.AtPrime P) ⧸ maxIdealAtPrime P

end Gen3

/-!
# Generation 4: Nakayama's Lemma (Stacks 07RC)
References: local_ring_structure, quotient_basics
-/

section Gen4

variable {R : Type*} [CommRing R] {M : Type*} [AddCommGroup M] [Module R M]

/-- Artifact: theorem-nakayama (Agent-0, Gen4)
    References: [lemma-local_ring_structure, lemma-quotient_basics]
    Nakayama's Lemma: If N is f.g., N ⊆ IN, and I ⊆ Jacobson(0), then N = 0 -/
theorem nakayama (I : Ideal R) (N : Submodule R M)
    (hN : N.FG) (hIN : N ≤ I • N) (hIjac : I ≤ (⊥ : Ideal R).jacobson) : N = ⊥ :=
  Submodule.eq_bot_of_le_smul_of_le_jacobson_bot I N hN hIN hIjac

/-- Artifact: theorem-nakayama_corollaries (Agent-1, Gen4)
    References: [theorem-nakayama]
    Contrapositive: nonzero f.g. module can't be contained in IN -/
theorem nakayama_nonzero (I : Ideal R) (N : Submodule R M)
    (hN : N.FG) (hIjac : I ≤ (⊥ : Ideal R).jacobson) (hne : N ≠ ⊥) :
    ¬(N ≤ I • N) := fun h => hne (nakayama I N hN h hIjac)

/-- Elements outside IN exist if N ≠ 0 -/
theorem nakayama_exists_outside (I : Ideal R) (N : Submodule R M)
    (hN : N.FG) (hIjac : I ≤ (⊥ : Ideal R).jacobson) (hne : N ≠ ⊥) :
    ∃ m ∈ N, m ∉ I • N := by
  by_contra h
  push_neg at h
  exact nakayama_nonzero I N hN hIjac hne (fun m hm => h m hm)

end Gen4

end LMS
