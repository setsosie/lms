"""Goal system for directed mathematical formalization.

Goals provide agents with specific targets from mathematical texts like
The Stacks Project. This enables focused work and progress tracking.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# ALLOWED IMPORTS FOR FROM-SCRATCH CATEGORY THEORY
# =============================================================================
# These provide foundations without spoiling the category theory itself.
# Agents importing Mathlib.CategoryTheory.* are "cheating" - we want them
# to BUILD the concepts, not just restate existing Mathlib lemmas.
# =============================================================================

ALLOWED_IMPORTS_FOUNDATION = [
    # CORE LOGIC (for proofs and propositions)
    "Mathlib.Logic.Basic",           # ∧, ∨, ¬, ↔, etc.
    "Mathlib.Logic.Function.Basic",  # Function.Injective, Surjective, comp, id

    # EQUIVALENCES (for defining isomorphisms later)
    "Mathlib.Logic.Equiv.Defs",      # ≃ type equivalences

    # BASIC DATA TYPES (for examples: Set, Grp, etc.)
    "Mathlib.Data.Set.Basic",        # Set α, membership, subset
    "Mathlib.Data.Set.Function",     # Set functions, image, preimage
    "Mathlib.Data.Prod.Basic",       # α × β product types
    "Mathlib.Data.Sum.Basic",        # α ⊕ β sum types
    # NOTE: Mathlib.Data.Opposite is NOT allowed - define it yourself!

    # ALGEBRA (if you want concrete categories like Grp, Ring, Mon)
    "Mathlib.Algebra.Group.Defs",    # Group, Monoid definitions
    "Mathlib.Algebra.Ring.Defs",     # Ring, Semiring definitions
    "Mathlib.Algebra.Group.Hom.Defs", # MonoidHom, GroupHom

    # USEFUL TACTICS
    "Mathlib.Tactic.Common",         # simp, ring, aesop, etc.
    "Mathlib.Tactic",                # All tactics
]

# Imports that are FORBIDDEN (indicate "cheating")
FORBIDDEN_IMPORTS = [
    "Mathlib.CategoryTheory",  # All of category theory - defeats the purpose!
]


def validate_imports(code: str, allowed: list[str] | None = None, forbidden: list[str] | None = None) -> tuple[bool, str | None]:
    """Validate that code only uses allowed imports.

    Args:
        code: Lean source code
        allowed: List of allowed import prefixes (None = allow all)
        forbidden: List of forbidden import prefixes (checked first)

    Returns:
        (valid, error_message) - True if valid, error message if not
    """
    import re

    # Extract all imports
    imports = re.findall(r"^\s*import\s+([A-Za-z][A-Za-z0-9_.]*)", code, re.MULTILINE)

    for imp in imports:
        # Check forbidden first
        if forbidden:
            for f in forbidden:
                if imp.startswith(f):
                    return False, f"Forbidden import: {imp} (matches {f})"

        # Check allowed (if specified)
        if allowed is not None:
            is_allowed = any(imp.startswith(a) or imp == a for a in allowed)
            if not is_allowed:
                return False, f"Import not in allowed list: {imp}"

    return True, None


@dataclass
class StacksDefinition:
    """A definition or lemma from The Stacks Project."""

    tag: str  # e.g., "0013"
    section: str  # e.g., "4.2"
    name: str  # e.g., "Definition of a category"
    content: str  # The actual mathematical content
    formalized: bool = False  # Has this been formalized?
    artifact_ids: list[str] = field(default_factory=list)  # Artifacts addressing this


@dataclass
class Goal:
    """A formalization goal from a mathematical text."""

    name: str
    description: str
    source: str  # e.g., "Stacks Project Chapter 4"
    definitions: list[StacksDefinition] = field(default_factory=list)
    allowed_imports: list[str] | None = None  # None = allow all
    forbidden_imports: list[str] | None = None  # Imports that are forbidden
    preamble: str | None = None  # Required code preamble (imports, etc.)

    def progress(self) -> float:
        """Return progress as a fraction (0.0 to 1.0)."""
        if not self.definitions:
            return 0.0
        formalized = sum(1 for d in self.definitions if d.formalized)
        return formalized / len(self.definitions)

    def validate_code(self, code: str) -> tuple[bool, str | None]:
        """Validate that code respects import restrictions.

        Args:
            code: Lean source code

        Returns:
            (valid, error_message) - True if valid, error message if not
        """
        return validate_imports(code, self.allowed_imports, self.forbidden_imports)

    def to_prompt_context(self) -> str:
        """Generate context string for agent prompts."""
        lines = [
            f"# Current Goal: {self.name}",
            f"Source: {self.source}",
            f"Description: {self.description}",
            f"Progress: {self.progress():.0%}",
        ]

        # Add import restrictions if present
        if self.forbidden_imports:
            lines.append("")
            lines.append("## ⚠️  FORBIDDEN IMPORTS (do NOT use):")
            for imp in self.forbidden_imports:
                lines.append(f"  - {imp}.*")

        if self.allowed_imports:
            lines.append("")
            lines.append("## ✓ ALLOWED IMPORTS:")
            for imp in self.allowed_imports:
                lines.append(f"  - {imp}")

        if self.preamble:
            lines.append("")
            lines.append("## Required Preamble (include at top of your code):")
            lines.append("```lean")
            lines.append(self.preamble)
            lines.append("```")

        lines.append("")
        lines.append("## Definitions to Formalize:")

        for defn in self.definitions:
            status = "[DONE]" if defn.formalized else "[TODO]"
            lines.append(f"\n### {status} {defn.section}: {defn.name} (Tag {defn.tag})")
            lines.append(defn.content)

        return "\n".join(lines)

    def mark_formalized(self, tag: str, artifact_id: str) -> None:
        """Mark a definition as formalized by an artifact."""
        # Normalize tag by stripping quotes (agents may output "0019" vs 0019)
        normalized_tag = tag.strip('"\'')
        for defn in self.definitions:
            if defn.tag == normalized_tag:
                defn.formalized = True
                defn.artifact_ids.append(artifact_id)
                return

    def save(self, path: Path) -> None:
        """Save goal state to JSON."""
        data = {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "definitions": [
                {
                    "tag": d.tag,
                    "section": d.section,
                    "name": d.name,
                    "content": d.content,
                    "formalized": d.formalized,
                    "artifact_ids": d.artifact_ids,
                }
                for d in self.definitions
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Goal":
        """Load goal state from JSON."""
        data = json.loads(path.read_text())
        return cls(
            name=data["name"],
            description=data["description"],
            source=data["source"],
            definitions=[
                StacksDefinition(**d) for d in data["definitions"]
            ],
        )


# =============================================================================
# Stacks Project Chapter 4: Categories - Phase 1 (Sections 4.1-4.6)
# =============================================================================
#
# EXPERIMENTAL DESIGN:
# This goal serves as a BENCHMARK to test the collective brain hypothesis.
# We provide mathematical definitions and milestone targets.
# Agents must figure out how to formalize them in Lean autonomously.
# We do NOT provide Lean/Mathlib hints - that's what we're testing!
#
# =============================================================================

STACKS_CHAPTER_4_PHASE_1 = Goal(
    name="Stacks Project Chapter 4: Categories (Phase 1)",
    description=(
        "Formalize basic category theory from The Stacks Project. "
        "You have definitions and milestone targets. Figure out how to express them in LEAN 4."
    ),
    source="The Stacks Project, Chapter 4: Categories (https://stacks.math.columbia.edu/tag/0011)",
    definitions=[
        # === DEFINITIONS (the mathematical content to formalize) ===
        StacksDefinition(
            tag="0013",
            section="4.2",
            name="Definition: Category",
            content="""A category C consists of:
1. A collection of objects Ob(C)
2. For each pair x, y ∈ Ob(C), a set of morphisms Mor(x, y)
3. For each object x, an identity morphism id_x ∈ Mor(x, x)
4. Composition: Mor(x, y) × Mor(y, z) → Mor(x, z), written (f, g) ↦ g ∘ f

Axioms:
- Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
- Identity: f ∘ id_x = f = id_y ∘ f for f : x → y""",
        ),
        StacksDefinition(
            tag="0014",
            section="4.2",
            name="Definition: Functor",
            content="""A functor F : C → D between categories consists of:
1. A map F : Ob(C) → Ob(D)
2. For each x, y ∈ Ob(C), a map F : Mor(x, y) → Mor(F(x), F(y))

Axioms:
- F(id_x) = id_{F(x)}
- F(g ∘ f) = F(g) ∘ F(f)""",
        ),
        StacksDefinition(
            tag="0015",
            section="4.2",
            name="Definition: Natural Transformation",
            content="""A natural transformation t : F → G between functors F, G : C → D consists of:
For each x ∈ Ob(C), a morphism t_x : F(x) → G(x)

Naturality condition: For f : x → y, we have G(f) ∘ t_x = t_y ∘ F(f)

    F(x) --F(f)--> F(y)
      |             |
     t_x           t_y
      v             v
    G(x) --G(f)--> G(y)""",
        ),
        StacksDefinition(
            tag="0017",
            section="4.3",
            name="Definition: Opposite Category",
            content="""For a category C, the opposite category C^op has:
- Same objects: Ob(C^op) = Ob(C)
- Reversed morphisms: Mor_{C^op}(x, y) = Mor_C(y, x)
- Reversed composition: g ∘^{op} f = f ∘ g""",
        ),
        StacksDefinition(
            tag="001D",
            section="4.4",
            name="Definition: Product",
            content="""A product of objects x, y in category C is an object x × y with
projections p : x × y → x and q : x × y → y satisfying:

Universal property: For any z with f : z → x and g : z → y,
there exists UNIQUE h : z → x × y such that p ∘ h = f and q ∘ h = g.""",
        ),
        StacksDefinition(
            tag="001K",
            section="4.6",
            name="Definition: Fibre Product (Pullback)",
            content="""Given f : x → z and g : y → z, the fibre product x ×_z y is an object with
p : x ×_z y → x and q : x ×_z y → y such that f ∘ p = g ∘ q, satisfying:

Universal property: For any w with a : w → x, b : w → y where f ∘ a = g ∘ b,
there exists UNIQUE h : w → x ×_z y making everything commute.""",
        ),

        # === MILESTONE TARGETS (theorems to prove - can they reach these?) ===
        StacksDefinition(
            tag="0019",
            section="4.3",
            name="MILESTONE: Yoneda Lemma",
            content="""THEOREM (Yoneda): Let C be a category, F : C^op → Set a functor, x ∈ Ob(C).
There is a natural bijection:

    Nat(h_x, F) ≅ F(x)

where h_x = Mor(−, x) is the representable functor.

The bijection sends t ↦ t_x(id_x).

COROLLARY: The Yoneda embedding C → Fun(C^op, Set) is fully faithful.""",
        ),
        StacksDefinition(
            tag="001P",
            section="4.3",
            name="MILESTONE: Yoneda Embedding Fully Faithful",
            content="""LEMMA 4.3.5: The Yoneda embedding h : C → Fun(C^op, Set) is fully faithful.

That is, for any morphism of functors s : h_x → h_y, there is a UNIQUE
morphism φ : x → y such that h(φ) = s.

This means: Mor_C(x, y) ≅ Nat(h_x, h_y)""",
        ),
        StacksDefinition(
            tag="001Z",
            section="4.6",
            name="MILESTONE: Pullback Stability",
            content="""LEMMA 4.6.6: Representability is stable under base change.

If f : x → y is representable and y' → y is any morphism, then the
pullback x' = x ×_y y' → y' is also representable.""",
        ),
    ],
)


# =============================================================================
# Stacks Project Chapter 4: Categories - Milestone-Driven (Emergent Dependencies)
# =============================================================================
#
# EXPERIMENTAL DESIGN:
# We give agents ONLY the milestone theorems they must prove.
# They must DISCOVER what definitions, lemmas, and intermediate results they need.
# This tests true emergent mathematical reasoning - can a collective figure out
# the dependency structure of mathematics?
#
# We do NOT tell them:
# - What definitions to formalize first
# - What intermediate lemmas are needed
# - The dependency order
#
# They must figure it out themselves through trial, error, and collaboration.
#
# =============================================================================

STACKS_CHAPTER_4_MILESTONES = Goal(
    name="Stacks Project Chapter 4: Milestone Theorems",
    description=(
        "Prove the key theorems of category theory. You must discover what definitions "
        "and lemmas you need along the way. Build the dependency structure yourself."
    ),
    source="The Stacks Project, Chapter 4: Categories (https://stacks.math.columbia.edu/tag/0011)",
    definitions=[
        # === TIER 1: FOUNDATIONAL MILESTONES ===
        # These are achievable early but require setting up basic definitions
        StacksDefinition(
            tag="M001",
            section="4.2",
            name="MILESTONE: Functor Composition is Associative",
            content="""THEOREM: Functor composition is associative.

Given functors F : A → B, G : B → C, H : C → D, prove:
    H ∘ (G ∘ F) = (H ∘ G) ∘ F

You will need to establish what a functor is and how composition works.
Prove this in LEAN 4 using Mathlib's category theory library.""",
        ),
        StacksDefinition(
            tag="M002",
            section="4.2",
            name="MILESTONE: Natural Transformation Vertical Composition",
            content="""THEOREM: Natural transformations compose vertically.

Given functors F, G, H : C → D and natural transformations α : F → G and β : G → H,
there is a natural transformation β ∘ α : F → H defined by (β ∘ α)_x = β_x ∘ α_x.

Prove this is indeed a natural transformation (naturality squares commute).
You'll need to establish functors and natural transformations first.""",
        ),
        StacksDefinition(
            tag="M003",
            section="4.2",
            name="MILESTONE: Isomorphisms are Equivalence Relation",
            content="""THEOREM: Isomorphism of objects is an equivalence relation.

For objects in a category, prove:
1. Reflexivity: X ≅ X (identity is an isomorphism)
2. Symmetry: X ≅ Y implies Y ≅ X (inverse of isomorphism is isomorphism)
3. Transitivity: X ≅ Y and Y ≅ Z implies X ≅ Z (composition of isomorphisms)

Establish what you need about morphisms and isomorphisms.""",
        ),

        # === TIER 2: YONEDA AND REPRESENTABILITY ===
        # The crown jewels of basic category theory
        StacksDefinition(
            tag="0019",
            section="4.3",
            name="MILESTONE: Yoneda Lemma",
            content="""THEOREM (Yoneda Lemma): For a category C, functor F : C^op → Set, and object x:

    Nat(Hom(−, x), F) ≅ F(x)

The bijection sends a natural transformation t to t_x(id_x).

This is one of the most important theorems in category theory.
You'll need to build up: categories, functors, natural transformations,
opposite categories, Hom functors, and the notion of representability.

Prove this in LEAN 4. The journey matters as much as the destination.""",
        ),
        StacksDefinition(
            tag="001P",
            section="4.3",
            name="MILESTONE: Yoneda Embedding is Fully Faithful",
            content="""THEOREM: The Yoneda embedding h : C → Fun(C^op, Set) is fully faithful.

This means: Mor_C(x, y) ≅ Nat(h_x, h_y)

In other words, the Yoneda embedding is injective on morphisms and every
natural transformation between representable functors comes from a morphism.

This is a corollary of the Yoneda Lemma - but you must prove it.""",
        ),

        # === TIER 3: UNIVERSAL CONSTRUCTIONS ===
        # Products, limits, and their properties
        StacksDefinition(
            tag="M004",
            section="4.4",
            name="MILESTONE: Products are Unique up to Isomorphism",
            content="""THEOREM: If products exist, they are unique up to unique isomorphism.

If (P, p₁, p₂) and (P', p₁', p₂') are both products of X and Y, then
there exists a UNIQUE isomorphism φ : P → P' compatible with projections.

You'll need to establish products and their universal property.
The uniqueness follows from the universal property - prove it.""",
        ),
        StacksDefinition(
            tag="M005",
            section="4.4",
            name="MILESTONE: Pullbacks are Limits",
            content="""THEOREM: A pullback is a limit of a cospan diagram.

Given f : X → Z and g : Y → Z, the pullback X ×_Z Y is the limit of the diagram:
    X → Z ← Y

Prove that the pullback (with its universal property) satisfies the
definition of a limit over this diagram shape.

You'll need to establish: limits, pullbacks, and diagram categories.""",
        ),
        StacksDefinition(
            tag="002E",
            section="4.4",
            name="MILESTONE: Limits from Products and Equalizers",
            content="""THEOREM: A category has all (small) limits if and only if it has:
1. All (small) products
2. All equalizers

Moreover, any limit can be CONSTRUCTED as an equalizer of a pair of
morphisms between products.

This is a fundamental theorem about the structure of limits.
Prove both directions and give the construction.""",
        ),

        # === TIER 4: ADJUNCTIONS ===
        # The deepest structural theorems
        StacksDefinition(
            tag="M006",
            section="4.5",
            name="MILESTONE: Adjunctions via Unit and Counit",
            content="""THEOREM: An adjunction F ⊣ G is equivalently given by:

1. Natural bijection: Hom(F(x), y) ≅ Hom(x, G(y))

OR

2. Unit η : id → GF and counit ε : FG → id satisfying triangle identities:
   (εF) ∘ (Fη) = id_F  and  (Gε) ∘ (ηG) = id_G

Prove these definitions are equivalent. You'll need to construct
each from the other and verify the required properties.""",
        ),
        StacksDefinition(
            tag="0024",
            section="4.5",
            name="MILESTONE: Right Adjoints Preserve Limits (RAPL)",
            content="""THEOREM: Right adjoints preserve limits.

If G : D → C is a right adjoint (has a left adjoint F), then G preserves
all limits that exist in D.

Concretely: if lim F_i exists in D, then G(lim F_i) ≅ lim G(F_i) in C.

Dually: Left adjoints preserve colimits.

This is one of the most useful theorems in category theory.
You'll need adjunctions and limits established first.""",
        ),

        # === TIER 5: ADVANCED MILESTONES ===
        StacksDefinition(
            tag="M007",
            section="4.5",
            name="MILESTONE: Equivalences Reflect Isomorphisms",
            content="""THEOREM: An equivalence of categories F : C → D reflects isomorphisms.

If F(f) is an isomorphism in D, then f was already an isomorphism in C.

More generally, equivalences preserve and reflect all categorical properties.
You'll need to establish equivalences of categories.""",
        ),
        StacksDefinition(
            tag="001M",
            section="4.6",
            name="MILESTONE: Pullback Pasting Lemma",
            content="""THEOREM (Pullback Pasting): Given a commutative diagram:
    A → B → C
    ↓   ↓   ↓
    D → E → F

1. If both squares are pullbacks, then the outer rectangle is a pullback.
2. If the outer rectangle and right square are pullbacks, the left square is a pullback.

This is essential for working with fiber products in algebraic geometry.
Prove both directions.""",
        ),
        StacksDefinition(
            tag="M008",
            section="4.5",
            name="MILESTONE: Monads from Adjunctions",
            content="""THEOREM: Every adjunction gives rise to a monad.

Given an adjunction F ⊣ G with F : C → D and G : D → C, define T = GF : C → C.
Then (T, η, μ) is a monad where:
- η : id_C → T is the unit of the adjunction
- μ : T² → T is given by μ = G(ε_F) where ε is the counit

Verify the monad laws (associativity and unit laws).
You'll need adjunctions and their unit/counit established.""",
        ),
    ],
)


# =============================================================================
# FROM-SCRATCH Category Theory Goal
# =============================================================================
#
# This is the HARD MODE goal. Agents must build category theory from scratch,
# not just import it from Mathlib.CategoryTheory. This tests true collective
# mathematical reasoning - can they discover the structure themselves?
#
# =============================================================================

# Standard preamble for from-scratch category theory
FROM_SCRATCH_PREAMBLE = """/-
  FROM-SCRATCH CATEGORY THEORY

  You must DEFINE categories, functors, natural transformations yourself.
  Do NOT import Mathlib.CategoryTheory.* - that's cheating!
  Do NOT import Mathlib.Data.Opposite - define opposite types yourself!

  Build the mathematics from foundations.
-/

-- Allowed foundation imports (logic and basic tactics only)
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Data.Set.Basic

-- Universe levels for categories
universe u v w

/-
  HINT: For opposite categories, you may want to define:

  structure Op (α : Type*) where
    unop : α

  This lets you distinguish morphisms in C from morphisms in Cᵒᵖ.
-/
"""

STACKS_CHAPTER_4_FROM_SCRATCH = Goal(
    name="Stacks Project Chapter 4: Categories (FROM SCRATCH)",
    description=(
        "Build basic category theory FROM SCRATCH. You must DEFINE what a category is, "
        "what a functor is, what a natural transformation is. Do NOT import "
        "Mathlib.CategoryTheory - that defeats the purpose! Use only foundational imports."
    ),
    source="The Stacks Project, Chapter 4: Categories (https://stacks.math.columbia.edu/tag/0011)",
    allowed_imports=ALLOWED_IMPORTS_FOUNDATION,
    forbidden_imports=FORBIDDEN_IMPORTS,
    preamble=FROM_SCRATCH_PREAMBLE,
    definitions=[
        # === TIER 1: CORE DEFINITIONS (must build these yourself!) ===
        StacksDefinition(
            tag="DEF-CAT",
            section="4.2",
            name="DEFINE: Category",
            content="""You must DEFINE a category structure in Lean 4.

A category C consists of:
1. A type of objects Ob(C)
2. For each pair x, y, a type of morphisms Hom(x, y)
3. Identity morphism: id : Hom(x, x) for each object x
4. Composition: comp : Hom(x, y) → Hom(y, z) → Hom(x, z)

Axioms (must be proven as part of the structure):
- id_comp: comp id f = f
- comp_id: comp f id = f
- assoc: comp (comp f g) h = comp f (comp g h)

Create a Lean 4 structure called `Category` or `Cat` that captures this.
Do NOT use Mathlib.CategoryTheory.Category.Basic!""",
        ),
        StacksDefinition(
            tag="DEF-FUNCTOR",
            section="4.2",
            name="DEFINE: Functor",
            content="""You must DEFINE a functor structure in Lean 4.

A functor F : C → D between categories consists of:
1. An object map: F_obj : Ob(C) → Ob(D)
2. A morphism map: F_map : Hom_C(x, y) → Hom_D(F_obj x, F_obj y)

Axioms:
- F_map(id_x) = id_{F_obj(x)}
- F_map(g ∘ f) = F_map(g) ∘ F_map(f)

Create a Lean 4 structure that depends on your Category definition.
Do NOT use Mathlib.CategoryTheory.Functor.Basic!""",
        ),
        StacksDefinition(
            tag="DEF-NATTRANS",
            section="4.2",
            name="DEFINE: Natural Transformation",
            content="""You must DEFINE a natural transformation structure in Lean 4.

A natural transformation η : F → G between functors F, G : C → D consists of:
For each object x in C, a morphism η_x : F(x) → G(x)

Naturality axiom:
For any morphism f : x → y in C, we have:
  G(f) ∘ η_x = η_y ∘ F(f)

This says the naturality square commutes:
    F(x) --F(f)--> F(y)
      |             |
     η_x           η_y
      ↓             ↓
    G(x) --G(f)--> G(y)

Create a Lean 4 structure depending on your Functor definition.
Do NOT use Mathlib.CategoryTheory.NatTrans!""",
        ),

        # === TIER 2: BASIC CONSTRUCTIONS ===
        StacksDefinition(
            tag="DEF-OPPOSITE",
            section="4.3",
            name="DEFINE: Opposite Category",
            content="""Define the opposite category C^op.

Given your Category C, define C^op where:
- Objects are the same: Ob(C^op) = Ob(C)
- Morphisms are reversed: Hom_{C^op}(x, y) = Hom_C(y, x)
- Composition is reversed: g ∘^{op} f = f ∘ g

Prove this is indeed a category (satisfies your Category axioms).""",
        ),
        StacksDefinition(
            tag="DEF-HOMFUNCTOR",
            section="4.3",
            name="DEFINE: Hom Functor",
            content="""Define the Hom functor h_x = Hom(-, x) : C^op → Type.

For a fixed object x in C:
- On objects: h_x(y) = Hom(y, x)
- On morphisms: given f : y' → y in C (equivalently, f : y → y' in C^op),
  define h_x(f) : Hom(y, x) → Hom(y', x) by precomposition: g ↦ g ∘ f

Prove this is a functor from C^op to Type.""",
        ),

        # === TIER 3: THE MAIN EVENT ===
        StacksDefinition(
            tag="LEM-YONEDA",
            section="4.3",
            name="MILESTONE: Yoneda Lemma (FROM SCRATCH)",
            content="""PROVE the Yoneda Lemma using YOUR OWN definitions.

THEOREM: For any functor F : C^op → Type and object x in C,
there is a natural bijection:

    Nat(h_x, F) ≅ F(x)

where h_x = Hom(-, x) is the representable functor.

The bijection is:
- Forward: Given natural transformation η : h_x → F, return η_x(id_x)
- Backward: Given element a ∈ F(x), define η where η_y(f) = F(f)(a)

Prove:
1. The backward map gives a natural transformation
2. These maps are mutually inverse

This is the crown jewel of basic category theory. You must build up to it
using YOUR OWN Category, Functor, and NatTrans definitions!""",
        ),
        StacksDefinition(
            tag="LEM-YONEDA-FF",
            section="4.3",
            name="MILESTONE: Yoneda Embedding is Fully Faithful",
            content="""PROVE the Yoneda embedding is fully faithful.

The Yoneda embedding y : C → Fun(C^op, Type) sends:
- Objects: x ↦ h_x = Hom(-, x)
- Morphisms: f : x → y  ↦  natural transformation h_x → h_y

THEOREM: The Yoneda embedding is fully faithful, meaning:
    Hom_C(x, y) ≅ Nat(h_x, h_y)

This is a corollary of the Yoneda Lemma (take F = h_y).
Prove it using your earlier definitions and lemmas.""",
        ),

        # === TIER 4: CONCRETE EXAMPLE ===
        StacksDefinition(
            tag="EX-SET-CAT",
            section="4.2",
            name="EXAMPLE: Category of Sets",
            content="""Define the category Set using your Category structure.

- Objects: Type u (or Set α for some α)
- Morphisms: functions f : A → B
- Identity: the identity function
- Composition: function composition

Prove this satisfies your Category axioms.
This is a sanity check that your definitions work on concrete examples.""",
        ),
    ],
)


# =============================================================================
# COMPREHENSIVE STACKS PROJECT GOAL: CHAPTERS 4-8
# =============================================================================
#
# The path to Stacks:
#   Ch 4 (Categories) → Ch 5 (Topology) → Ch 6 (Sheaves on Spaces)
#     → Ch 7 (Sites and Sheaves) → Ch 8 (Stacks)
#
# This goal tracks progress across all chapters needed to understand Stacks.
# Each chapter builds on the previous, with explicit dependencies.
#
# =============================================================================

STACKS_PATH_PREAMBLE = """/-
  STACKS PROJECT: Path to Stacks (Chapters 4-8)

  Build the foundations needed for the theory of Stacks:
  - Categories and Functors (Ch 4)
  - Topological Spaces (Ch 5)
  - Sheaves on Topological Spaces (Ch 6)
  - Sites and Sheaves (Ch 7)
  - Stacks (Ch 8)

  You may import foundational Mathlib for logic, sets, and algebra.
  Do NOT import Mathlib.CategoryTheory, Mathlib.Topology.Sheaves, etc.
  The goal is to BUILD these concepts from scratch.
-/
"""

# Allowed imports for the full Stacks path
ALLOWED_IMPORTS_STACKS = [
    # Core logic
    "Mathlib.Logic.Basic",
    "Mathlib.Logic.Function.Basic",
    "Mathlib.Logic.Equiv.Defs",

    # Basic data types
    "Mathlib.Data.Set.Basic",
    "Mathlib.Data.Set.Function",
    "Mathlib.Data.Set.Lattice",
    "Mathlib.Data.Prod.Basic",
    "Mathlib.Data.Sum.Basic",
    "Mathlib.Data.Sigma.Basic",

    # Order theory (for lattices, needed for topology)
    "Mathlib.Order.Basic",
    "Mathlib.Order.Lattice",
    "Mathlib.Order.CompleteLattice",
    "Mathlib.Order.GaloisConnection",

    # Algebra (for algebraic examples)
    "Mathlib.Algebra.Group.Defs",
    "Mathlib.Algebra.Ring.Defs",
    "Mathlib.Algebra.Group.Hom.Defs",
    "Mathlib.Algebra.Module.Basic",

    # Tactics
    "Mathlib.Tactic.Common",
    "Mathlib.Tactic",
]

FORBIDDEN_IMPORTS_STACKS = [
    "Mathlib.CategoryTheory",      # Build categories yourself
    "Mathlib.Topology.Basic",      # Build topology yourself
    "Mathlib.Topology.Sheaves",    # Build sheaves yourself
    "Mathlib.Geometry.RingedSpace", # Build ringed spaces yourself
]

STACKS_FULL_PATH = Goal(
    name="Stacks Project: Path to Stacks (Chapters 4-8)",
    description="""Build the mathematical foundations needed to understand Stacks.

This is a comprehensive goal spanning 5 chapters of the Stacks Project:
- Chapter 4: Categories (foundation for everything)
- Chapter 5: Topology (spaces where sheaves live)
- Chapter 6: Sheaves on Spaces (functions that glue)
- Chapter 7: Sites and Sheaves (generalized topology)
- Chapter 8: Stacks (the goal!)

Each chapter builds on the previous. Complete them in order.
The collective must discover how to formalize each concept from scratch.""",
    source="The Stacks Project, Chapters 4-8 (https://stacks.math.columbia.edu)",
    allowed_imports=ALLOWED_IMPORTS_STACKS,
    forbidden_imports=FORBIDDEN_IMPORTS_STACKS,
    preamble=STACKS_PATH_PREAMBLE,
    definitions=[
        # =====================================================================
        # CHAPTER 4: CATEGORIES (Foundation - TIER 1)
        # =====================================================================
        StacksDefinition(
            tag="CH4-CAT",
            section="4.2",
            name="Category",
            content="""DEFINE: Category

A category C consists of:
- A collection of objects Ob(C)
- For each pair (X, Y), a set of morphisms Hom(X, Y)
- For each object X, an identity morphism id_X
- Composition of morphisms: f ∘ g when target(g) = source(f)

Satisfying:
- Left identity: id ∘ f = f
- Right identity: f ∘ id = f
- Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)

This is the foundation for everything else.""",
        ),
        StacksDefinition(
            tag="CH4-FUNCTOR",
            section="4.2",
            name="Functor",
            content="""DEFINE: Functor

A functor F: C → D between categories consists of:
- A map on objects: F(X) for each X in Ob(C)
- A map on morphisms: F(f): F(X) → F(Y) for each f: X → Y

Satisfying:
- Preserves identity: F(id_X) = id_{F(X)}
- Preserves composition: F(f ∘ g) = F(f) ∘ F(g)

Functors are "structure-preserving maps" between categories.""",
        ),
        StacksDefinition(
            tag="CH4-NATTRANS",
            section="4.2",
            name="Natural Transformation",
            content="""DEFINE: Natural Transformation

A natural transformation η: F → G between functors F, G: C → D consists of:
- For each object X in C, a morphism η_X: F(X) → G(X)

Satisfying naturality: For every f: X → Y in C,
  η_Y ∘ F(f) = G(f) ∘ η_X

This makes the "naturality square" commute.""",
        ),
        StacksDefinition(
            tag="CH4-OPPOSITE",
            section="4.2",
            name="Opposite Category",
            content="""DEFINE: Opposite Category

For a category C, the opposite category C^op has:
- Same objects as C
- Morphisms reversed: Hom_{C^op}(X, Y) = Hom_C(Y, X)
- Composition reversed: f ∘^op g = g ∘ f

This is crucial for contravariant functors and presheaves.""",
        ),
        StacksDefinition(
            tag="CH4-YONEDA",
            section="4.3",
            name="Yoneda Lemma",
            content="""THEOREM: Yoneda Lemma

For any functor F: C^op → Set and object X in C:
  Nat(Hom(-, X), F) ≃ F(X)

The bijection is given by:
- Forward: η ↦ η_X(id_X)
- Backward: a ↦ (η where η_Y(f) = F(f)(a))

This is "the most important theorem in category theory".""",
        ),
        StacksDefinition(
            tag="CH4-LIMITS",
            section="4.4",
            name="Limits and Colimits",
            content="""DEFINE: Limits and Colimits

A limit of a diagram D: J → C is an object L with:
- Projections π_j: L → D(j) for each j in J
- Universal property: any cone factors uniquely through L

Dually, a colimit has injections and the dual universal property.

Key examples:
- Product (limit of discrete diagram)
- Equalizer (limit of parallel arrows)
- Pullback (limit of cospan)""",
        ),
        StacksDefinition(
            tag="CH4-ADJOINT",
            section="4.5",
            name="Adjoint Functors",
            content="""DEFINE: Adjoint Functors

Functors F: C → D and G: D → C are adjoint (F ⊣ G) if:
  Hom_D(F(X), Y) ≃ Hom_C(X, G(Y))

naturally in X and Y.

Equivalently, there exist natural transformations:
- Unit: η: Id_C → G ∘ F
- Counit: ε: F ∘ G → Id_D

satisfying the triangle identities.""",
        ),

        # =====================================================================
        # CHAPTER 5: TOPOLOGY (TIER 2 - needs Ch 4)
        # =====================================================================
        StacksDefinition(
            tag="CH5-TOPSPACE",
            section="5.2",
            name="Topological Space",
            content="""DEFINE: Topological Space

A topological space (X, τ) consists of:
- A set X (the underlying set)
- A collection τ of subsets of X (the open sets)

Satisfying:
- ∅ and X are in τ
- τ is closed under arbitrary unions
- τ is closed under finite intersections

Elements of τ are called "open sets".""",
        ),
        StacksDefinition(
            tag="CH5-CONTINUOUS",
            section="5.2",
            name="Continuous Map",
            content="""DEFINE: Continuous Map

A function f: X → Y between topological spaces is continuous if:
  For every open set V ⊆ Y, the preimage f⁻¹(V) is open in X.

Equivalently:
- Preimages of closed sets are closed
- For every x and neighborhood V of f(x), there exists neighborhood U of x with f(U) ⊆ V

Continuous maps are morphisms in the category Top.""",
        ),
        StacksDefinition(
            tag="CH5-BASIS",
            section="5.3",
            name="Basis for a Topology",
            content="""DEFINE: Basis for a Topology

A basis B for a topology on X is a collection of subsets such that:
1. B covers X: ∪B = X
2. For B₁, B₂ in B and x in B₁ ∩ B₂, there exists B₃ in B with x ∈ B₃ ⊆ B₁ ∩ B₂

The topology generated by B consists of all unions of basis elements.

This is essential for defining topologies on schemes.""",
        ),
        StacksDefinition(
            tag="CH5-OPEN-COVER",
            section="5.4",
            name="Open Cover",
            content="""DEFINE: Open Cover

An open cover of a topological space X is a collection {U_i} of open sets such that:
  X = ∪_i U_i

A refinement of a cover {U_i} is a cover {V_j} such that each V_j is contained in some U_i.

Open covers are central to the definition of sheaves.""",
        ),

        # =====================================================================
        # CHAPTER 6: SHEAVES ON SPACES (TIER 3 - needs Ch 4, 5)
        # =====================================================================
        StacksDefinition(
            tag="CH6-PRESHEAF",
            section="6.2",
            name="Presheaf",
            content="""DEFINE: Presheaf

A presheaf F on a topological space X (with values in Sets) is:
- A functor F: Open(X)^op → Set

Concretely:
- For each open U, a set F(U) (sections over U)
- For each inclusion V ⊆ U, a restriction map ρ_{U,V}: F(U) → F(V)
- ρ_{U,U} = id and ρ_{V,W} ∘ ρ_{U,V} = ρ_{U,W}

This is just a contravariant functor from the category of open sets.""",
        ),
        StacksDefinition(
            tag="CH6-SHEAF",
            section="6.3",
            name="Sheaf",
            content="""DEFINE: Sheaf

A presheaf F is a sheaf if for every open cover {U_i} of U:

1. (Locality) If s, t ∈ F(U) satisfy s|_{U_i} = t|_{U_i} for all i, then s = t.

2. (Gluing) If s_i ∈ F(U_i) satisfy s_i|_{U_i ∩ U_j} = s_j|_{U_i ∩ U_j} for all i,j,
   then there exists s ∈ F(U) with s|_{U_i} = s_i.

Sheaves are presheaves where local data glues uniquely to global data.""",
        ),
        StacksDefinition(
            tag="CH6-SHEAFIFY",
            section="6.4",
            name="Sheafification",
            content="""DEFINE: Sheafification

For any presheaf F, there exists a sheaf F^+ and a morphism θ: F → F^+ such that:
- (Universal property) Any morphism F → G to a sheaf factors uniquely through F^+

Construction:
- F^+(U) consists of compatible families of germs
- F^+ is the "best approximation" of F by a sheaf

Sheafification is left adjoint to the inclusion Sh(X) → PSh(X).""",
        ),
        StacksDefinition(
            tag="CH6-STALK",
            section="6.5",
            name="Stalk of a Sheaf",
            content="""DEFINE: Stalk

The stalk of a presheaf F at a point x is:
  F_x = colim_{x ∈ U} F(U)

Elements of F_x are called germs at x.

Two sections s ∈ F(U) and t ∈ F(V) define the same germ if
they agree on some neighborhood of x.

Stalks detect whether a presheaf is a sheaf.""",
        ),
        StacksDefinition(
            tag="CH6-MORPHISM",
            section="6.6",
            name="Morphism of Sheaves",
            content="""DEFINE: Morphism of Sheaves

A morphism φ: F → G of sheaves on X is a natural transformation:
- For each open U, a map φ_U: F(U) → G(U)
- Compatible with restrictions: φ_V ∘ ρ^F_{U,V} = ρ^G_{U,V} ∘ φ_U

The category Sh(X) of sheaves on X is a full subcategory of PSh(X).

Sh(X) has limits, colimits, and is an abelian category (for abelian sheaves).""",
        ),

        # =====================================================================
        # CHAPTER 7: SITES AND SHEAVES (TIER 4 - needs Ch 4, 6)
        # =====================================================================
        StacksDefinition(
            tag="CH7-SIEVE",
            section="7.2",
            name="Sieve",
            content="""DEFINE: Sieve

A sieve S on an object X in a category C is a collection of morphisms to X such that:
- If f: Y → X is in S and g: Z → Y is any morphism, then f ∘ g is in S.

Equivalently, a sieve is a subfunctor of the representable functor Hom(-, X).

Sieves generalize "collections of open subsets" to arbitrary categories.""",
        ),
        StacksDefinition(
            tag="CH7-GROTOP",
            section="7.3",
            name="Grothendieck Topology",
            content="""DEFINE: Grothendieck Topology

A Grothendieck topology J on a category C assigns to each object X a collection J(X) of sieves (called covering sieves) such that:

1. (Maximal) The maximal sieve (all morphisms to X) is in J(X)
2. (Stability) If S ∈ J(X) and f: Y → X, then f*S ∈ J(Y)
3. (Transitivity) If S ∈ J(X) and for each f: Y → X in S we have T_f ∈ J(Y),
   then the sieve generated by all compositions is in J(X)

A category with a Grothendieck topology is called a site.""",
        ),
        StacksDefinition(
            tag="CH7-SITE",
            section="7.4",
            name="Site",
            content="""DEFINE: Site

A site is a pair (C, J) where:
- C is a category
- J is a Grothendieck topology on C

Examples:
- (Top, open covers) - the classical site
- (Sch/S, étale covers) - the small étale site
- (Sch, fppf covers) - the big fppf site

Sites provide the abstract setting for sheaf theory.""",
        ),
        StacksDefinition(
            tag="CH7-SHEAF-SITE",
            section="7.5",
            name="Sheaf on a Site",
            content="""DEFINE: Sheaf on a Site

A presheaf F: C^op → Set on a site (C, J) is a sheaf if:
For every covering sieve S ∈ J(X), the natural map
  F(X) → lim_{(f: Y → X) ∈ S} F(Y)
is an isomorphism.

Equivalently, for every covering family {f_i: U_i → X}:
- F(X) → ∏_i F(U_i) is injective (locality)
- The equalizer condition holds (gluing)

This generalizes sheaves on spaces to sheaves on sites.""",
        ),
        StacksDefinition(
            tag="CH7-TOPOS",
            section="7.6",
            name="Topos",
            content="""DEFINE: (Grothendieck) Topos

A Grothendieck topos is a category equivalent to Sh(C, J) for some site (C, J).

Key properties of a topos E:
- Has all small limits and colimits
- Has exponentials (E is cartesian closed)
- Has a subobject classifier Ω

Topoi are "generalized spaces" - they behave like categories of sheaves.""",
        ),

        # =====================================================================
        # CHAPTER 8: STACKS (TIER 5 - THE GOAL!)
        # =====================================================================
        StacksDefinition(
            tag="CH8-FIBCAT",
            section="8.2",
            name="Fibered Category",
            content="""DEFINE: Fibered Category

A fibered category over C is a functor p: F → C such that:
For every morphism f: X → Y in C and object ξ in F over Y,
there exists a cartesian morphism φ: f*ξ → ξ lifting f.

A morphism φ in F is cartesian if it satisfies the universal property:
any morphism to ξ factors uniquely through φ.

Fibered categories are "categories varying over C".""",
        ),
        StacksDefinition(
            tag="CH8-CFG",
            section="8.3",
            name="Category Fibered in Groupoids",
            content="""DEFINE: Category Fibered in Groupoids (CFG)

A category fibered in groupoids over C is a fibered category p: F → C where:
- All fibers F_X are groupoids (all morphisms are isomorphisms)
- All morphisms in F are cartesian

CFGs are the natural setting for moduli problems:
the fiber F_X represents "objects over X" with only isomorphisms between them.""",
        ),
        StacksDefinition(
            tag="CH8-DESCENT",
            section="8.4",
            name="Descent Data",
            content="""DEFINE: Descent Data

For a covering {U_i → X} and a CFG F, descent data consists of:
- Objects ξ_i in F(U_i) for each i
- Isomorphisms φ_{ij}: ξ_i|_{U_{ij}} → ξ_j|_{U_{ij}} over U_i ×_X U_j

Satisfying the cocycle condition:
  φ_{jk} ∘ φ_{ij} = φ_{ik} over U_i ×_X U_j ×_X U_k

Descent data describes how local objects glue together.""",
        ),
        StacksDefinition(
            tag="CH8-STACK",
            section="8.5",
            name="Stack",
            content="""DEFINE: Stack

A stack over a site (C, J) is a category fibered in groupoids F → C such that:

1. (Isomorphisms form a sheaf) For any X and ξ, η in F(X), the presheaf
   Isom(ξ, η): U ↦ Isom_{F(U)}(ξ|_U, η|_U) is a sheaf.

2. (Descent is effective) Every descent datum is effective:
   it comes from an actual object in F(X).

STACKS ARE SHEAVES OF GROUPOIDS with the correct notion of "local" and "gluing".""",
        ),
        StacksDefinition(
            tag="CH8-STACKIFY",
            section="8.6",
            name="Stackification",
            content="""THEOREM: Stackification

For any CFG F over a site (C, J), there exists a stack F^st and a morphism F → F^st such that:
- (Universal property) Any morphism F → G to a stack factors uniquely through F^st

This is the "sheafification for stacks" - it forces descent to be effective.

The stackification of a presheaf of groupoids is often called the "associated stack".""",
        ),
        StacksDefinition(
            tag="CH8-MORPHISM",
            section="8.7",
            name="Morphism of Stacks",
            content="""DEFINE: 1-Morphism and 2-Morphism of Stacks

A 1-morphism F → G of stacks is a functor over C preserving cartesian morphisms.

A 2-morphism α: f → g between 1-morphisms is a natural transformation over C.

Stacks over C form a 2-category:
- Objects: Stacks
- 1-morphisms: Functors over C
- 2-morphisms: Natural transformations

This 2-categorical structure is essential for moduli theory.""",
        ),
        StacksDefinition(
            tag="CH8-ALGEBRAIC",
            section="8.8",
            name="Algebraic Stack",
            content="""DEFINE: Algebraic Stack (Artin Stack)

An algebraic stack is a stack X over (Sch, fppf) such that:
1. The diagonal Δ: X → X × X is representable and separated
2. There exists a smooth surjective morphism U → X from a scheme U

The scheme U is called an atlas for X.

Algebraic stacks include:
- All schemes (trivially)
- Quotient stacks [X/G]
- Moduli stacks (M_g, Bun_G, etc.)

THIS IS THE MAIN GOAL: Understand what an algebraic stack is!""",
        ),
    ],
)

# Create a variant with Ch4 core definitions pre-marked as complete
# (Category, Functor, NatTrans, Opposite, Yoneda are in Foundation.lean)
def _create_ch4_complete_goal() -> Goal:
    """Create stacks-full-path with Ch4 core definitions pre-marked as formalized."""
    import copy
    goal = copy.deepcopy(STACKS_FULL_PATH)
    goal.name = "Stacks Project: Path to Stacks (Ch4 Core Complete)"
    goal.description = """Continue building toward Stacks from where Chapter 4 left off.

The Foundation.lean already contains:
- Category, Functor, NatTrans, Opposite Category (CH4-CAT, CH4-FUNCTOR, CH4-NATTRANS, CH4-OPPOSITE)
- Yoneda Lemma and Yoneda Embedding (CH4-YONEDA)

Still TODO from Chapter 4:
- Limits and Colimits (CH4-LIMITS)
- Adjoint Functors (CH4-ADJOINT)

Then proceed through:
- Chapter 5: Topology
- Chapter 6: Sheaves on Spaces
- Chapter 7: Sites and Sheaves
- Chapter 8: Stacks"""

    # Mark the 5 core Ch4 definitions as formalized
    ch4_done_tags = {"CH4-CAT", "CH4-FUNCTOR", "CH4-NATTRANS", "CH4-OPPOSITE", "CH4-YONEDA"}
    for defn in goal.definitions:
        if defn.tag in ch4_done_tags:
            defn.formalized = True
            defn.artifact_ids = ["foundation-yoneda-complete"]

    return goal


STACKS_FULL_PATH_CH4_DONE = _create_ch4_complete_goal()


# Registry of available goals
GOALS_REGISTRY = {
    "stacks-ch4-phase1": STACKS_CHAPTER_4_PHASE_1,
    "stacks-ch4-milestones": STACKS_CHAPTER_4_MILESTONES,
    "stacks-ch4-scratch": STACKS_CHAPTER_4_FROM_SCRATCH,
    "stacks-full-path": STACKS_FULL_PATH,
    "stacks-ch4-done": STACKS_FULL_PATH_CH4_DONE,
}


def get_goal(name: str) -> Goal:
    """Get a goal by name from the registry."""
    if name not in GOALS_REGISTRY:
        available = ", ".join(GOALS_REGISTRY.keys())
        raise ValueError(f"Unknown goal '{name}'. Available: {available}")
    return GOALS_REGISTRY[name]


def list_goals() -> list[str]:
    """List available goal names."""
    return list(GOALS_REGISTRY.keys())
