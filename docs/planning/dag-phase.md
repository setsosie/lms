# The DAG phase — statement-first committee mode

> Written 2026-09-05, the day after the reframe in `decisions/0001` and `0002`.
> Evidence is against `main` @ `85ccf9e` and the Stacks clone under
> `references/stacks-project/` (untracked). One page on what the phase is, why
> the current graph cannot host it, and how it decomposes into cards.

## Why

Two collectives have formalized mathematics at scale with Lean in the last
year, and both worked over an explicit **statement DAG**:

- **The Stacks Project** is written as one. Every tagged statement cites the
  tags it is built from (`\ref{...}` in `categories.tex` etc.), the
  presentation order is a topological sort, and the WC-3 corpus was built by
  following that order. The three-text committees (2026-02) called the Stacks
  dependency graph the thing that made the work decomposable.
- **Prove2Me** (arXiv 2608.28433) made it the platform's data model, and it is
  what turned Anthropic's failed Claude-Code harness into the FLT result with
  the model held fixed: a *theorem card* per statement (natural-language
  description, preamble, formal Lean statement ending in `:= by sorry`),
  immutable once submitted, extended by agents through *proof sketches* that
  import open children, closed bottom-up (a child's local proof auto-resolves
  its parent), with search-before-submit over the cards' descriptions.

LMS has a `DependencyGraph`, a planning panel that allocates from it, and
working groups that consume it. What it does not have is a DAG.

### The graph is a chain

`DependencyGraph.from_goal` infers edges from section order: every node
requires **all** earlier nodes in its chapter, and the first node of a chapter
requires the last node of the previous one (`lms/dependency.py`,
`_infer_dependencies`). That is a total order. Measured on `main`:

| Goal | Nodes | Available at start | Blocked |
|---|---:|---:|---:|
| `stacks-kernel-track-b` | 40 | 1 | 39 |
| `stacks-kernel-track-a` | 20 | 1 | 19 |
| `stacks-ch4-phase1` | 9 | 1 | 8 |
| every other registered goal | — | 1 | rest |

Exactly one task is AVAILABLE at any moment, for the whole run. Consequences
already visible in the run logs, none of them previously attributed to this:

- **No committee run has ever had two distinct tasks to hand out.** Either one
  group forms, or the chair puts several groups on the same tag — the case
  `update_status`'s docstring records from `committee_real_b` ("several groups
  can work one tag in a single generation"). `PlanningPanel._top_up` can never
  top up: the spare list is empty by construction.
- **The population axis is crippled before it is measured.** `--agents 9` over
  three configured groups seats three researchers per group (HARN-18), but
  only one statement is ever in play. Phase C's 1-vs-9 comparison, on this
  graph, compares one small group with one large group on the same statement.
- **"N agents converge on one tag"** (`stacks-restart-notes.md`, the first
  question a Track B run was meant to answer) is not agent behaviour. It is
  the graph.

### The real edges exist and are cheap

The Stacks source carries them. Resolving `\ref{label}` through the clone's
`tags/tags` file (the extractor already loads it) over the two kernel tracks:

| Track | Statement-body refs | Proof refs | Edges kept (union, forward dropped) | Roots (available at start) | Forward refs dropped |
|---|---:|---:|---:|---:|---:|
| B (40 stmts) | 13 | 25 | 33 | **18** | 2 (`02XN→02XW`, `026D→026E`) |
| A (20 stmts) | 1 | 4 | 5 | **16** | 0 |

Roots go from 1 to 18 and 16. Forward references ("see Lemma below") point
later in presentation order and are not prerequisites; dropping them keeps the
graph acyclic by construction. Statement-body references are definitional
prerequisites (02XP cannot be stated without 003Y); proof references are what
a proof sketch would import (02XO's proof cites 02XK and 02XL). Both are edges:
the second kind is exactly the topological ordering a swarm needs so that the
foundation holds the pieces when a proof wants them, and a weak model that
cannot close 02XK blocking everything downstream of it is the ratchet-failure
dynamic the project is built to observe, not a defect to route around.

The ANT candidate arcs (`data/ant_arcs/*.json`, branch `data/ant-arc-statements`)
are already theorem cards in Prove2Me's sense — each entry has `informal`,
`book_ref`, and a `lean_statement` ending in `:= sorry` — but they carry no
dependency edges and nothing turns them into a `Goal`. Phase C has no slice
file yet; the DAG phase supplies the shape it should take.

## What the phase is

Committee mode gains a statement layer in front of proving. Per node:

| Prove2Me | LMS |
|---|---|
| Theorem card: description + preamble + `:= by sorry` statement | `DependencyNode.statement` (field from HARN-25) populated for goal nodes, not only leaves; `StacksDefinition.lean_statement` when curated |
| Immutable once submitted | `set_statement` refuses to overwrite; statements persist in `dependency_graph.json` across generations and resumes |
| Proof submitted *for* a statement | **Statement pinning**: the harness splices the card's header over the agent's declaration of the same name before Lean runs, so statement drift becomes a compile error the repair loop can act on, not a silent redefinition |
| Proof sketch imports open children | HARN-25: `sorry`-bodied named children become AVAILABLE leaves `<parent>/<child>` |
| Child proof auto-resolves parent | HARN-26: a verified artifact pinned to a leaf marks it DONE; when all leaves close, the parent is re-verified with its children stripped and promoted |
| DAG edges | Three sources: **source-anchored** (Stacks `\ref`), **curated** (ANT `requires`, the D4 reviewer's job), **agent-extended** (sketch leaves, and later agent-authored cards) |
| Search before submit | Deferred. At 20–40 statements the graph *is* the index and the prompt shows every statement; a name collision with the graph or foundation is rejected at pinning time. An NL search over the corpus is a scale feature (HARN-30, uncarded) |

Nothing here changes what counts as verified. A sketch is `SKETCH`, never
`VERIFIED_LEAN`; the gates (T2/T4, novelty, axioms) run on closed statements
exactly as today; the foundation only ever receives sorry-free code.

## Cards

| Card | Pts | What it lands | Depends on |
|---|---:|---|---|
| `26Q3-HARN-27` source-anchored dependency edges | 3 | `requires` per goal definition from Stacks `\ref`; `from_goal` uses explicit edges when present, infers otherwise; both kernel goal files regenerated; cycle check | none |
| `26Q3-HARN-28` theorem cards and statement pinning | 5 | `lean_statement` on goal definitions; `DependencyNode.statement` for goal nodes; immutability; header splice before verify; DONE only for the pinned declaration; ANT arc → goal converter | HARN-25 (`statement` field, `allow_sorry`) |
| `26Q3-HARN-26` closing leaves, auto-resolution | 5 | Part 2 of HARN-25: leaf DONE on a pinned child proof; parent re-verified with children stripped and promoted; rollback if the parent fails | HARN-25, HARN-28 (pinning) |
| `26Q3-HARN-29` agent-authored cards for uncurated nodes | — | a statement session whose scribe emits a card, verified to elaborate with `allow_sorry=True`, then frozen. Uncarded until 28 is green | HARN-28 |
| `26Q3-HARN-30` search-before-propose | — | NL index over graph + foundation + library. Uncarded; scale feature | HARN-29 |

Order: **27 → 28 → 26**. HARN-25 (red tests committed, green not started) sits
between 27 and 28 and does not depend on either.

## Where it sits against Phase C

ADR 0001 freezes the harness at the 2026-09-07 state of `main` and varies the
model. The DAG phase *is* a harness variant — the second level of the factor
the Prove2Me paper leaves to future work. Two ways to place it, one decision
for the user:

1. **Baseline unchanged, DAG phase measured after.** Phase C runs on the chain
   graph as frozen; HARN-27/28/26 land during Sprint 4 and run as a third
   configuration on the same slice and budget afterwards. Clean factor
   design; the baseline's 9-agent arm measures one group on one statement.
2. **HARN-27 goes in before the freeze; 28 and 26 after.** Three points, data
   plus one `from_goal` branch, no prompt or verifier change. The baseline
   then has a real DAG and the 1-vs-9 comparison measures parallel groups on
   distinct statements. The DAG *phase* (cards, pinning, auto-resolve) is
   still the post-freeze variant.

Recommendation: **2**. A population-size experiment on a graph that admits one
task at a time is not a population-size experiment, and fixing the graph is a
defect repair, not a harness feature. It needs to be merged by 2026-09-07,
which means it goes ahead of #54–#57 in the queue or it does not happen.

Whichever is chosen, the ANT slice for Phase C needs a `Goal` file with
`requires` edges. The converter is in HARN-28; curating the edges is a
D4-reviewer task on the winning arc (about 20 statements, an hour).

## Not in the DAG phase

- Any change to the iterative or flat paths.
- The genome arm, the three-text program, Prove2Me as a client.
- Reuse and duplication *metrics* — they exist (HARN-16); the DAG phase is what
  gives them something to measure.
