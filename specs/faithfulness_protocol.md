# Faithfulness-Checking Protocol
## Statement Fidelity, Novelty Certification, and the Definition Trust Surface

> Drafted June 9, 2026. Motivated by the verification gap that gated the program:
> Lean verifies that proofs follow from statements, but nothing verifies that
> statements correspond to the mathematics they claim to formalize.
> Case studies: `~/code/condensed-sheaf-cohomology` and `~/code/arithmetic-linguistics`,
> both of which produced novelty claims with no verification oracle of any kind.

---

## 1. The Problem

A society of agents can produce Lean code that compiles and still be worthless:

- `theorem stacks_04VB : True := trivial` compiles.
- A definition with a sign error compiles, and every theorem about it verifies —
  about the wrong object.
- A theorem whose hypotheses are unsatisfiable verifies vacuously.
- A theorem advertised as "new condensed mathematics" can verify while its proof
  secretly factors through classical results, making the novelty claim false even
  though the proof is sound.

The last failure mode is not hypothetical. In `condensed-sheaf-cohomology`, the
flagship goal — "a genuine condensed-mathematical theorem" — deflated on inspection:
the R¹lim = 0 result was classical Mittag-Leffler and the solidification step reduced
to bounded linear algebra. No proof was wrong; the *claim about what was proved* was.
In `arithmetic-linguistics`, six claimed-novel theorems exist only as expert
conjectures, unwritten and unverifiable. Neither project had an oracle, so the claims
are stranded: checking them now costs as much as producing them did.

**The lesson**: verification debt compounds. Faithfulness must be checked at
production time, by machine wherever possible, or the output is not knowledge —
it is a pile of plausible text.

## 2. The Trust Surface Is Definitions, Not Theorems

A theorem stated entirely in Mathlib vocabulary cannot meaningfully misstate:
its constituent concepts were vetted by the Mathlib community, and a human can
check the statement against the textbook in seconds. Misstatement risk
concentrates in **novel definitions**, whose meaning everything downstream
inherits through the type checker.

Projected scale for the three-text program:

| Layer | Count | Verification mode |
|-------|-------|-------------------|
| Novel theorems | ~11,600–14,000 | Lean (proofs) + cheap statement checks (§4) |
| Novel definitions | ~300–500 (mostly in shared kernel) | Full protocol (§3) + human sign-off |
| Mathlib vocabulary | reused | already trusted |

This reduction is why building on Mathlib (WC-3 unanimous decision) was the right
call, and why the shared kernel must be built carefully: it is where nearly all
of the program's trust assumptions live.

## 3. Per-Definition Requirements (D-checks)

Every novel `def`/`structure`/`class` must ship, **in the same WC cycle**, with:

### D1. Instantiation test (machine-checked)
At least one nontrivial concrete instance with expected properties proved.
A spectral sequence definition ships with an actual spectral sequence (e.g., the
degenerate one and one non-degenerate one); a fibred category ships with a concrete
fibration. A definition with contradictory or vacuous content cannot be instantiated.
This is the formal analogue of a unit test. Lives in `lean/LMS/Tests/` alongside the
definition's module.

### D2. Mathlib bridge lemma (machine-checked, when overlap exists)
Where the novel notion restricts to an existing Mathlib notion, prove the agreement:
profinite group cohomology must agree with `groupCohomology` on finite groups;
linearly topologized completions must agree with Mathlib's I-adic completion.
A misstatement that survives a bridge proof is exotic. If no overlap exists,
record that fact explicitly in the definition's docstring (`/-- No Mathlib bridge:
... -/`) so the absence is a reviewed decision, not an oversight.

### D3. Negative example (machine-checked)
Exhibit one object that **fails** the definition, with the failure proved. Catches
the dual failure mode of D1: a definition so weak everything satisfies it. (D1 catches
"nothing satisfies it"; D3 catches "everything does.")

### D4. Source anchor + human sign-off (human, queued)
Docstring must cite the informal source (Stacks tag, ART chapter/section, CNF
theorem number) and quote or closely paraphrase the informal definition. A human
reviews the definition layer — and **only** the definition layer — comparing Lean
text against the cited source. At ~300–500 definitions for the whole program this
is days of human work, not months, and it is the only place human mathematical
judgment is structurally required.

**Gate**: a WC cycle that introduces a definition without D1–D3 does not merge.
D4 sign-off may lag (definitions enter a review queue, tracked in
`specs/definition_review_queue.md`), but no novelty claim (§6) may cite an
unsigned definition.

## 4. Per-Theorem Requirements (T-checks)

Cheap checks applied to every novel theorem; all machine-checkable, suitable for
a skeptic agent or CI pass rather than a human:

### T1. Vocabulary audit
Every constant appearing in the *statement* is either Mathlib or a D4-queued novel
definition. No statement-level constants invented ad hoc inside a proof file.

### T2. Non-vacuity witness
For `theorem foo (h₁ : P x) (h₂ : Q x) : R x`, provide
`example : ∃ x, P x ∧ Q x` (or reuse a D1 instance). Unsatisfiable-hypothesis
vacuity is the most common silent misstatement; this kills it mechanically.
For hypothesis *relevance* (a stronger check, applied selectively), use
`lean_minimal_hypotheses` to detect hypotheses the proof never uses — an unused
load-bearing-looking hypothesis is a misstatement smell worth flagging.

### T3. Source anchor
Statement docstring cites its Stacks tag / ART / CNF location. Statements with no
informal anchor are automatically candidate **novelty claims** and route to §6 —
they do not get to sit quietly in the tree.

### T4. Axiom/sorry audit
`lean_verify` on every public theorem: no `sorry`, no nonstandard axioms in the
dependency cone. (Already practice since WC-3; now a formal gate. The one tracked
sorry in `Compat.lean` stays on an explicit exception list until discharged.)

## 5. Dual Formalization for High-Risk Statements

For the shared kernel and for any statement flagged high-risk (novel definition
dependencies, subtle quantifier structure, or any novelty claim):

1. Two agents independently formalize the same informal statement, **blind** to
   each other, from the same source citation.
2. A third agent attempts to prove the two statements equivalent
   (`theorem agree : Stmt_A ↔ Stmt_B`).
3. **Machine-checked agreement** between independent formalizations is strong
   evidence both match the source — this is the Lean-grade version of the
   "independent convergence is real signal" lesson from the planning committees.
4. **Divergence** is the valuable output: it localizes exactly where the informal
   statement is ambiguous, and routes that statement to the human queue with the
   two candidate readings attached. Human review becomes adjudication between two
   precise options instead of open-ended auditing.

Cost: roughly 2× statement-writing cost (statements are cheap relative to proofs)
plus one usually-easy equivalence proof. Apply to ~100% of kernel definitions,
~100% of novelty claims, and a 5–10% random sample of routine statements as a
drift detector.

## 6. Novelty Certification

The program's real question: can the society produce something **new**? A novelty
claim is a strictly harder artifact than a verified theorem, because Lean checks
the proof but not the claim "this was not already known." Both prior projects
failed exactly here. Protocol:

### 6.1 The novelty ladder

| Level | What it is | Example |
|-------|-----------|---------|
| N0 | Re-proof of an existing Mathlib result | calibration only, never claimed |
| N1 | Formalization of a known textbook statement absent from Mathlib | the bulk of the program (~11.6K stmts) — novel *formalization*, not novel math |
| N2 | Connective lemma stated in no single source text | the p-adic bridge: ART's tilting = CNF's field of norms, made precise |
| N3 | Theorem absent from the source texts and (so far as searchable) the literature | the genuine prize |
| N4 | New definition that earns sustained reuse across WCs | new *concept* — the strongest possible outcome |

### 6.2 Certification requirements by level

**N1** (default): T1–T4 only. The source anchor *is* the faithfulness certificate.

**N2 and above**, additionally:
1. **Search evidence**: documented negative results from `lean_local_search`,
   `lean_leansearch`, `lean_loogle`, plus a literature search note. Absence of
   evidence is weak, but undocumented search is no evidence at all.
2. **Dual formalization** (§5), mandatory.
3. **Dependency-cone audit** — the check that would have caught the
   condensed-sheaf-cohomology deflation mechanically: inspect the proof's constant
   dependency closure and verify the proof **actually uses** the machinery the
   novelty claim invokes. A theorem advertised as "about spectral sequences of
   profinite cohomology" whose cone contains no profinite definitions has a false
   novelty claim regardless of proof validity. In Lean this is a mechanical walk
   of the elaborated proof term's constants — a skeptic agent runs it, no judgment
   required for the negative case.
4. **Skeptic pass**: one agent prompted solely to *deflate* the claim — find the
   Mathlib lemma it restates, the classical theorem it specializes, the trivial
   reformulation. Default verdict is "deflated" unless the skeptic affirmatively
   fails.
5. **Human sign-off**, always, with the search evidence and skeptic report attached.

**N4** has one requirement no agent can fake and no single cycle can produce:
**reuse**. A new definition is certified novel-and-good only after independent
WCs adopt it without prompting. This is the Henrich mechanism made into a
certificate — the collective doesn't just produce knowledge, it filters it — and
it is measured by the same lemma-reuse metric already tracked for phase-transition
detection. One metric, two readings: emergence and trustworthiness.

### 6.3 Folklore capture: Lean as the citation of last resort

A recurring failure in the prior-project audits was the **uncitable folklore
problem**: facts that specialists know, that nobody has stated in a citable
form (e.g., "first-order cohomology of a free monoid is abelianization-blind" —
attributed to Leech–Nico 1969–75, but as folklore, not as a quotable theorem).
Folklore is where informal mathematics is weakest: claims circulate with no
artifact to check against, and attributions themselves go unverified.

Formalization dissolves this. An N1 statement whose source anchor is "folklore"
gets, for the first time, a **citable artifact**: the Lean statement is the
precise claim, the proof term is the evidence, and `lean_verify` is the referee.
This inverts the usual valuation — N1 work on folklore is *more* valuable than
N1 work on well-cited theorems, because for cited theorems the formalization
adds redundancy, while for folklore it adds the only checkable reference that
exists. Tag such statements `source: folklore` in the docstring (with the
best-effort attribution) so they can be surfaced as a deliverable in their own
right: a folklore registry is a publishable contribution independent of any
novelty claims.

### 6.4 Where novelty is expected to surface

Not uniformly. Watch the seams:
- **The p-adic bridge** (ART tilting ↔ CNF field of norms): two texts describing
  one construction in different languages. Forcing them into one formal API is
  precisely where unstated connective lemmas (N2) live.
- **Kernel generality decisions**: when the kernel must state spectral-sequence
  machinery at the generality all three texts need *simultaneously*, the right
  general statement is often in no text.
- **Failed proofs of textbook statements**: when an agent cannot prove a source
  statement and the dual formalization confirms the statement is faithful, the
  obstruction itself is interesting — either the text has a gap (documented,
  publishable as an erratum-grade finding) or a missing lemma (N2 candidate).

## 7. Retroactive Application: the Stranded Claims

The two prior projects hold candidate novel mathematics with zero verification.
Route the *formalizable* claims through this pipeline as the protocol's first
live test:

| Claim | Source | Status | Disposition |
|-------|--------|--------|-------------|
| Bridge 2 Foundation theorem (condensed structure on Σ̂*-invariants) | arithmetic-linguistics, `notes/literature_review.md` | unwritten; expert-flagged "novel but possibly easy" | **Good first N2/N3 target** — small, profinite + condensed, sits inside kernel Track A/B vocabulary |
| Bridge 3 Theorem 1 (Berkovich/tropical WFST) | arithmetic-linguistics | unwritten conjecture | Defer — vocabulary (Berkovich spaces) far outside current kernel |
| §6.30 solid tensor product computations | condensed-sheaf-cohomology, `docs/main.md` | agent-written proof, never read by anyone | Formalize or discard; an unread agent proof has **zero** evidentiary value under this protocol |
| Mittag-Leffler / R¹lim claims | condensed-sheaf-cohomology | "proved" + numerically spot-checked | N0/N1 at best (classical); formalize only if needed as kernel infrastructure |
| Empirical claims (Fisher–H¹ correlation r=0.59, P1 falsification, Δ(L) null) | condensed-sheaf-cohomology | tests pass; not Lean-formalizable | Out of scope for this protocol — these need replication, not formalization. The *definitions* underlying them (K_compat invariance) could be formalized if the project revives |

The Bridge 2 Foundation theorem is the recommended acid test: an expert conjectured
it is true, novel, and small. If the society formalizes it under full N2/N3
certification, the program has its first defensible "something new." If the skeptic
pass deflates it to a known condensed-math fact, that is *also* a success — the
protocol caught in one WC cycle what the prior projects could not catch at all.

## 8. Cost and Integration

Per-WC overhead estimate (on the ~400–500K token / ~50 stmt cycle baseline):

| Check | Overhead | Notes |
|-------|----------|-------|
| D1–D3 (definitions) | ~10–20% of cycle | only cycles introducing definitions; front-loaded into kernel phase |
| T1–T4 (all theorems) | ~5% | skeptic-agent CI pass, mostly mechanical |
| Dual formalization | ~2× statement cost on selected stmts | statements ≪ proofs; net ~5–10% of cycle |
| Novelty certification | ~50–100K tokens per claim | rare events; cost is the point |
| Human review | minutes per definition, ~zero per N1 theorem | the entire human budget concentrates on §3 D4 + §6 sign-offs |

Net: roughly **15–25% token overhead**, concentrated in the kernel phase —
exactly where the planning synthesis already said to spend carefully. In exchange,
every N1 statement carries a machine-checked faithfulness case, and any novelty
claim that survives §6 is defensible to an external mathematician without asking
them to trust a single agent utterance.

### Role changes to the WC structure
- Add a standing **Skeptic** role (per existing Role enum pattern): runs T-checks,
  dependency-cone audits, and deflation passes. Never writes proofs it audits.
- The Planning Panel assigns dual-formalization pairs blind (agents must not share
  context for the statement-writing step).
- `status.json` gains per-statement fields: `d_checks`, `t_checks`,
  `novelty_level`, `review_status` — so faithfulness state is remotely monitorable
  alongside progress.

## 9. What This Protocol Does Not Solve

A coherent-but-wrong definitional ecosystem: novel definitions that are mutually
consistent, instantiable, bridge-compatible where bridges exist, and still subtly
mean the wrong thing in the unbridged region. Defenses are D2 (shrinks the
unbridged region), D4 (human eyes on every definition), and §6.1 N4 reuse
filtering (wrong definitions tend to be unusable downstream) — mitigation, not
elimination. This residual risk is small, lives in a known place (the definition
layer), and is the honest price of scale.
