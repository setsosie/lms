# D4 review — ANT candidate-arc statement lists

**What this is**: the ~41 statements in `data/ant_arcs/{core,ramification}_arc.json`,
laid out for your review before the N1-density measurement runs on the box.
**Your role** (per `specs/ant_shakedown.md` §2.3): check that each drafted
statement says what Neukirch says — book against Lean/informal — and correct
the references. You know this book; expect ~30–60 min for a first pass.

**How to review**: skim each block. Three checks, in priority order:

1. **Reference** — is the §/number right? All numbers were drafted from memory
   (section-level: high confidence; x.y numbers: **low confidence, confirm every
   one against the book**).
2. **Faithfulness** — does the informal statement match Neukirch's? Where his
   version carries a hypothesis the draft dropped (separability, conductor
   condition, measure normalization), flag it.
3. **Arc shape** — is each list a fair contiguous sample of its arc, or did the
   selection skip statements you'd consider load-bearing?

Lean drafts below are **trimmed to the mathematical content** (binders and
typeclass stacks elided; full signatures in the JSON). None has been elaborated
against Mathlib (local olean cache is stale — see `data/ant_arcs/README.md`);
`SCHEMATIC` marks drafts with known placeholder names. For D4 purposes the
informal statement is the object of review; the Lean is best-effort.

## Open questions for you

1. **Ch. III §2 inclusion**: the program phrase "Ch. I ramification theory …
   different/discriminant" mixes Ch. I §8–§10 with Ch. III §2 (where the
   different actually lives in the book). The ramification arc includes 6
   statements from III §2 (ram-16 … ram-21). Keep them, or hold the arc strictly
   to Ch. I? (If cut, the arc is 15 statements and probably less N1-dense —
   which changes what the measurement can find.)
2. **Neukirch's constant vs Minkowski bound** (core-15/17): drafts use his
   (2/π)^s √|d_K| constant rather than Mathlib's sharper Minkowski bound. Keep
   the book's constant for faithfulness?
3. **Cyclotomic convention** (ram-14): confirm the book's edge-case convention
   (n ≢ 2 mod 4) is the one you want carried.
4. **Localization (§11)**: deliberately excluded from the ramification arc.
   Right call, or does the arc need it for contiguity?

---

## Core arc — Ch. I §2–§7 (20 statements)

**core-01 · integral elements form a ring** — Ch I §2
Sum/product of elements integral over A is integral; the integral closure is a subring.
`IsIntegral A x → IsIntegral A y → IsIntegral A (x + y)` · N0-anchor

**core-02 · transitivity of integrality** — Ch I §2
C integral over B, B integral over A ⇒ C integral over A.
`Algebra.IsIntegral A B → IsIntegral B x → IsIntegral A x`

**core-03 · factorial ⇒ integrally closed** — Ch I §2
Every UFD is integrally closed in its fraction field.
`UniqueFactorizationMonoid A → IsIntegrallyClosed A`

**core-04 · minimal polynomial has integral coefficients** — Ch I §2
A integrally closed, x integral over A ⇒ minpoly of x over K = Frac(A) has coefficients in A.
`IsIntegral A x → minpoly K x ∈ Polynomial.lifts (algebraMap A K)`

**core-05 · trace form nondegenerate** — Ch I §2
L/K finite separable ⇒ (x,y) ↦ Tr(xy) nondegenerate.
`(Algebra.traceForm K L).Nondegenerate`

**core-06 · discriminant of a basis ≠ 0** — Ch I §2 (2.8?)
L/K separable, (αᵢ) a K-basis ⇒ d(α₁…αₙ) = det(Tr(αᵢαⱼ)) ≠ 0.
`Algebra.discr K b ≠ 0`

**core-07 · integral basis exists** — Ch I §2 (2.10?)
O_K is a free ℤ-module of rank [K:ℚ].
`Module.Free ℤ (𝓞 K) ∧ finrank ℤ (𝓞 K) = finrank ℚ K`

**core-08 · O_K is Dedekind** — Ch I §3 (3.1?)
Noetherian, integrally closed, dim 1.
`IsDedekindDomain (𝓞 K)`

**core-09 · unique ideal factorization** — Ch I §3 (3.3?)
Every ideal ≠ (0),(1) factors uniquely into primes.
`UniqueFactorizationMonoid (Ideal R)` for Dedekind R

**core-10 · fractional ideals form a group** — Ch I §3
Nonzero fractional ideals of a Dedekind domain: I · I⁻¹ = (1).
`I ≠ 0 → I * I⁻¹ = 1`

**core-11 · ideal norm multiplicative** — Ch I §6 (placement to confirm)
𝔑(ab) = 𝔑(a)𝔑(b) for 𝔑(a) = [O_K : a].
`Ideal.absNorm (I * J) = absNorm I * absNorm J`

**core-12 · complete lattice ⇔ cocompact** — Ch I §4
Γ complete in V ⇔ V/Γ compact.
`span ℝ L = ⊤ ↔ CompactSpace (V ⧸ L)`

**core-13 · Minkowski lattice point theorem** — Ch I §4 (4.4?)
X convex, symmetric, vol(X) > 2ⁿ covol(Γ) ⇒ X contains a nonzero lattice point. (Drafted for Γ = ℤⁿ.)
`Convex S → symm S → 2^n < vol S → ∃ x : ℤⁿ, x ≠ 0 ∧ x ∈ S`

**core-14 · ideal lattice covolume** — Ch I §5 · SCHEMATIC
j(a) ⊂ K_ℝ is a complete lattice of covolume √|d_K| · 𝔑(a). Check: measure normalization is the book's canonical one.

**core-15 · element of small norm** — Ch I §5–§6
Every nonzero ideal contains a ≠ 0 with |N(a)| ≤ (2/π)^s √|d_K| 𝔑(a). (Book's constant, not the sharper Minkowski bound — Q2 above.)

**core-16 · class group finite** — Ch I §6 (6.3?)
Cl_K is finite.
`Finite (ClassGroup (𝓞 K))`

**core-17 · every class contains a bounded ideal** — Ch I §6 · SCHEMATIC-leaning
Each class contains integral a with 𝔑(a) ≤ (2/π)^s √|d_K|.

**core-18 · μ(K) finite cyclic** — Ch I §7 (7.1?)
Roots of unity in K form a finite cyclic group.
`Finite (torsion K) ∧ IsCyclic (torsion K)`

**core-19 · unit log-lattice complete in H** — Ch I §7 · SCHEMATIC
λ(O_Kˣ) is a complete lattice in the trace-zero hyperplane; ker λ = μ(K).

**core-20 · Dirichlet unit theorem** — Ch I §7 (7.4?)
O_Kˣ ≅ μ(K) × ℤ^{r+s−1}.
`(𝓞 K)ˣ ≃* torsion K × ℤ^{r+s−1}`

## Ramification arc — Ch. I §8–§10 + Ch. III §2 (21 statements)

**ram-01 · integral closure is Dedekind** — Ch I §8 (8.1?)
A Dedekind, L/K finite separable ⇒ B = integral closure of A in L is Dedekind.
`IsDedekindDomain (integralClosure A L)` · N0-anchor

**ram-02 · fundamental identity** — Ch I §8 (8.2?)
Σ eᵢfᵢ = [L:K].
`∑ P | p, e(P) * f(P) = finrank K L`

**ram-03 · Dedekind–Kummer** — Ch I §8 (8.3?) · SCHEMATIC
Factorization of p in A[θ] mirrors factorization of minpoly(θ) mod p (p prime to conductor). Check: the conductor hypothesis is dropped in the draft.

**ram-04 · finitely many ramified primes** — Ch I §8
Only finitely many p ramify in K (those dividing d_K).

**ram-05 · Galois acts transitively on P | p** — Ch I §9 (9.1?)
`∃ σ ∈ Gal(L/K), σP = Q`

**ram-06 · n = efg in Galois extensions** — Ch I §9
e, f constant over p; e·f·g = n.

**ram-07 · |G_P| = e·f** — Ch I §9 · SCHEMATIC
Decomposition group order e·f, index g.

**ram-08 · decomposition field** — Ch I §9 · SCHEMATIC
In Z_P = L^{G_P}: e(P_Z|p) = f(P_Z|p) = 1; P unique over P_Z.

**ram-09 · G_P ↠ Gal(κ(P)/κ(p))** — Ch I §9 · SCHEMATIC
Residue extension normal; reduction map surjective.

**ram-10 · |I_P| = e** — Ch I §9 · SCHEMATIC
Inertia group order e; G_P/I_P ≅ Gal(κ(P)/κ(p)).

**ram-11 · unramified ⇔ I_P trivial** — Ch I §9
`e = 1 ↔ ∀ σ ∈ G_P, ∀ x, σx ≡ x mod P`

**ram-12 · O_{ℚ(ζₙ)} = ℤ[ζₙ]** — Ch I §10 · SCHEMATIC

**ram-13 · [ℚ(ζₙ):ℚ] = φ(n)** — Ch I §10
`finrank ℚ (CyclotomicField n ℚ) = totient n`

**ram-14 · p ramifies in ℚ(ζₙ) ⇔ p | n** — Ch I §10 (convention: Q3 above)

**ram-15 · cyclotomic splitting law** — Ch I §10 · SCHEMATIC
p ∤ n splits into φ(n)/f primes, f = ord of p in (ℤ/n)ˣ.

**ram-16 · different well-defined** — Ch III §2
Inverse different is a fractional ideal ⊇ B; 𝔡 = its inverse is a nonzero integral ideal.
`differentIdeal A B ≠ ⊥`

**ram-17 · P ramified ⇔ P | 𝔡** — Ch III §2 (2.6?)
Dedekind's theorem on the different.

**ram-18 · v_P(𝔡) = e−1 iff tame** — Ch III §2 (2.6?) · SCHEMATIC
Tame: exactly e−1; wild: ≥ e. Check: draft states only the tame equality.

**ram-19 · disc = N(different)** — Ch III §2 · SCHEMATIC
𝔡isc_{B/A} = N_{L/K}(𝔡_{B/A}).

**ram-20 · monogenic different** — Ch III §2
B = A[θ] ⇒ 𝔡 = (f′(θ)).

**ram-21 · p ramified ⇔ p | d** — Ch III §2 (2.12?)
Stated over ℚ: p ramifies in L ⇔ p | d_L.

---

*Drafted 2026-08-19 from memory of the book; every x.y reference above carries
an implicit "(to confirm)". Corrections go into the JSON files — `book_ref` and
`informal` fields — before the classifier runs on the box.*
