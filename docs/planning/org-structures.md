# Organizational structures — what to compare, and in what order

> Written 2026-10-30, against `main` @ `0c3ad12`. Sources fetched the same day
> into `references/prove2me/` (see `FETCHED.txt` there): the Prove2Me paper
> (arXiv 2608.28433 v1 and v2), its agent workspace docs (commit `d3062c1`),
> Anthropic's FLT page and companion report, and two Xena posts. Structure
> evidence comes from the *Language Models as Agents* draft, Chapters 11–13 and
> 17 (`~/code/llm-agents/chapters/`). Every quotation below was checked against
> those files with whitespace normalized.

## Why

The question is which organization lets *this* collection — Qwen3.6-27B on the
box, Lean as the verifier — produce the most verified-faithful-novel
statements per token. ADR 0003's Phase C already contains three of the arms
needed to start answering it: S1 (one loop), I9 (nine isolated loops, pooled)
and C9 (the committee). This page names the rest, what each needs built, and
the order to run them. None of it enters Sprint 4 or 5. Each stage starts
only after the previous stage's number exists.

## What Prove2Me is, from its own sources

| Aspect | What the sources say |
|---|---|
| Data model | A theorem card holds DESCRIPTION, PREAMBLE, and a FORMAL STATEMENT "required to terminate in a := by sorry placeholder" (§3.1). Statements and sketches "cannot be edited once submitted" (§4.2); the docs allow description and tag edits, with history. A sketch imports open children; "the parent theorem auto-resolves" (§4.2). |
| Who decides the work | A mission "is created by a captain, the human contributor who proposes it", who confirms each statement; a moderator reviews it before it goes live (§3.3). The paper rejects an "orchestrating agent" (§4.1). |
| How agents choose | Through the docs' reading order: milestones first, then the graph, then search. There is no claim or lock endpoint, and credit goes to "the earliest `ACCEPTED` submission". Agents can therefore duplicate work. Per-mission discussion threads; no direct messages. |
| Verifier | Each card is compiled on the server "against only the statements of its children". A "Proved" mark on the platform is therefore not an end-to-end check; the FLT team rebuilt the whole tree off-platform. |
| Openness | Called open source (§1), but the GitHub org holds a workspace, an export and a feedback repo, and **no server code or self-hosting instructions**. Private missions still share the public name space ("Name occupancy is the only thing observable from outside"). |
| What was measured | Four missions, 17K–151K lines, 3–9 agents counting subagents, $200–600 *imputed* at about $200 per month per human contributor. "Table 1 reports case studies, not a controlled experiment." No failure, rejection, faithfulness-audit or reuse numbers. |
| Open questions it names | "How best to coordinate the decentralized, asynchronous agents on Prove2Me remains an open question" (§4.5). Holding the model fixed and varying the harness is left "to future work" (§5). |

**FLT, from Anthropic's page and report.**
- **The failed attempts:** "A number of Claude's initial attempts failed: … they quickly lost track of the project's state and stopped collaborating effectively." The sources describe neither the harness nor the model of those attempts.
- **The successful run** used "Prove2Me and a Claude Code-based multi-agent harness". No source says the model was held fixed.
- **Duplication:** "Each card was proved in isolation. Roughly two in five theorem statements inside the proof files repeat, word for word, a statement already made in another proof file." That is the pile, measured at FLT scale.
- *Inference from the report's excerpts:* the agents also coordinated outside the card store (a "shared plan", "the team's chat"). So the FLT run is not evidence for coordinator-free operation on the DAG alone.
- **The sources are not independent.** Prove2Me's corresponding author is also the Anthropic researcher who ran the FLT effort.

### Can Prove2Me "work better" here?

Not as a hosted arm. Verification runs on Prove2Me's servers, the name space is
shared, and nothing isolates a population, so ADR 0002's objection now rests on
the platform's own documentation. What *can* be tested is Prove2Me's
**organization**, re-implemented locally on lms's own DAG. The DAG-phase cards
(`26Q3-HARN-27/28/25/26`) already borrow its data model. What Prove2Me adds on
top is organizational:
- no LLM planner;
- an agent-side reading order (milestones, then open leaves);
- no claiming, with the first accepted proof winning;
- a discussion channel;
- one compile per card.

That is arm P below. FLT's two-in-five duplication makes "claim versus no claim" a variant worth measuring rather than assuming.

## Corrections to claims already in lms docs

| Claim as written | What the sources say | Sites |
|---|---|---|
| The model-fixed comparison is left to future work in Prove2Me **§7** | The sentence is in **§5 Case study**; §7 does not repeat it | ADR 0001 l.20–21; `calibration-program.md` l.102 |
| The first harness was "a Claude Code multi-agent loop"; "the same model succeeded" on Prove2Me | The failed attempts' harness and model are not described. "Claude Code-based multi-agent harness" describes the **successful** run, used together with Prove2Me. No source says the model was held fixed | README l.11–14; `CLAUDE.md` l.11–13; ADR 0001 l.17–20; `dag-phase.md` l.19–20 |
| Announced 2026-09-03 | The Anthropic page is dated Sep 4 | README, `CLAUDE.md`, ADR 0001, ADR 0002 |

ADR 0001's decision survives the correction: Prove2Me §5 itself calls model
and harness confounded. One supporting sentence in its Context is wrong.
**Applied 2026-10-30:** README, `CLAUDE.md`, `dag-phase.md` and
`calibration-program.md` are corrected. ADR 0001 ages in place, so it carries
a dated correction note at its foot rather than a rewrite. ADR 0002 and the
Sprint 3 archive keep their 2026-09-03 date as historical record.

## Candidate arms

Every arm runs at N=9 and 2M tokens on the same slice. The book's evidence is
tagged by the comparison behind it: **[MS]** a compute-matched single agent,
**[NULL]** an independent-loops null, **[UNM]** unmatched or the paper's own
baseline.

| Arm | Organization | In lms today | Book mechanism | Book evidence |
|---|---|---|---|---|
| **S1** | one loop at the population's tokens | yes (flat, 1 agent) | `prop:11:decomposition`: with no hazard growth, any split loses | single agents "can match or outperform" multi-agent systems once compute is normalized [MS] (ch11) |
| **I9** | nine isolated loops, pooled | ADR 0003, `26Q4-HARN-03` | `def:12:null`: the coupling gain is the swarm minus this | — (it *is* the null) |
| **K9** | best-of-k per statement: k independent attempts at each available node; Lean selects | no (the pass@k harness in `LESSONS_FROM_SONNET_SIMULATION.md` §4c) | with a sound verifier success is 1−(1−p)^k (`prop:17:ceiling`); coverage 1−(1−θ)^N falls with correlation (`rem:13:coverage`) | "Majority Voting alone accounts for most of the performance gains" attributed to debate (ch11); an all-wrong mass caps every policy (ch13) |
| **W9** | one long-lived coordinator; fresh one-shot workers given exact signatures; no panel, no review | no (the Sonnet-simulation shape) | `prop:11:decomposition`: splitting pays only if hazard growth beats handoff failure | "models below a capability threshold exhibit a reversal where rigid structure outperforms autonomy" (ch12) |
| **C9** | the committee: planning panel, working groups, review | yes | `prop:11:contamination`, `prop:12:capacity` | collaboration tax shows a "monotonic decrease with capability" (ch11) |
| **P9** | Prove2Me-style: no coordinator; agents pick nodes by a fixed reading order; the foundation as medium; verified statements written back; bounded reads | no; needs the DAG phase plus a swarm-mode card | `def:12:swarm`; `prop:12:drift`: a verifier on the medium turns copying into selection, and Lean is that verifier at v≈1 | removing the shared-context admission verifier drops accuracy 60.1→55.2, an ablation, not a null (ch12); "full-solution interaction is a weak default" (ch12); "gated isolation" against early convergence (ch12) |
| **H9** | heterogeneous pool: two open models at matched calls | serving only (`--mixed` exists for APIs) | effective size ≤ 1/ρ; lowering ρ is "the only lever" (ch12) | heterogeneity gains at matched agent calls (ch12); "Self-MoA outperforms standard MoA" (ch13); temperature spread inside one model is not the lever (ch12) |
| islands | separate foundations, periodic migration of verified entries | no; blocked by the fixed `Foundation.lean` path | `prop:12:consensus`; sparse topologies | "moderately sparse topologies … typically achieve optimal task performance" [UNM] (ch12) |

**Not proposed: debate, voting, markets, majority-reward cohort training.**
Their value in the book comes from supplying a verifier, and Lean already is
one. `prop:13:cohort-reward` shows a majority reward "sharpens the cohort's
majority and cannot teach it".

## What every arm holds fixed

- **Fixed:** model, slice (a goal with HARN-27 edges, so more than one node is available), seed layer, gates, total tokens, Lean verifier. **Lean calls are reported per arm**: the null "is run at the swarm's budget" with "the verifier's calls per arm matched or reported" (ch12, `def:12:null`).
- **Varies:** organization only, at N=9. A size sweep comes later, with a null at each size.
- **Retention:** the foundation is carried in every arm except a declared R-variant (ADR 0003's R9).
- **Read from `26Q4-HARN-03`:**
  - distinct verified-faithful-novel statements and CVFN;
  - DAG coverage;
  - duplication (FLT's two in five is the reference figure);
  - cross-agent reuse: imports of a statement another agent verified;
  - statements no arm solves;
  - correlation of per-node solves across agents;
  - wall-clock.
- **"Works better" means** beating both S1 and I9 on paired per-statement outcomes, beyond the run-to-run noise from `26Q4-EVO-02 --repeats`. Losing to I9 is a result: the book says "a swarm arm that loses to the null at every N would be a finding and not a defect of the study" (ch12, `rem:12:programme`).
- **Runs are sequential** (ADR 0003): `Foundation.lean` is one fixed path.

## Order and decision rules

| Stage | Arms | Needs | Added tokens |
|---|---|---|---|
| 0 | S1, C9, I9, R9 | ADR 0003, `26Q4-HARN-03`, `26Q4-HARN-04` | (Phase C) |
| 1 | K9, W9 | `26Q4-HARN-05`, `26Q4-HARN-06` | +4M |
| 2 | P9, with variants: claim vs no claim, discussion on vs off | DAG phase (`26Q3-HARN-27`, `-28`, `-25`, `-26`), then `26Q4-HARN-07` | +2M per variant |
| 3 | H9; islands | `26Q4-HARN-08` and a second served model; islands need per-island Lean modules (uncarded) | +2M each |

After stage 0:
- **C9 ≤ I9** (the committee buys nothing over isolation): run stages 1 and 2 to ask whether *any* organization does, with P9 and K9 first.
- **C9 > I9**: ablate the committee before adding structures. Turn the planning panel off (deterministic DAG allocation, `use_planning_panel=False`), then turn review off, to locate the gain.
- **Every arm ~0 at Gate C**: no organizational comparison is possible at this capability. Stop, as `calibration-program.md` §7 already says.

## Cards (written 2026-10-30)

| Card | Arm | Pts | Stage |
|---|---|---:|---|
| `26Q4-HARN-05` best-of-k mode | K9 | 3 | 1 |
| `26Q4-HARN-06` coordinator–worker mode | W9 | 5 | 1 |
| `26Q4-HARN-07` swarm mode (2 parts) | P9 | 5 + 3 | 2 |
| `26Q4-HARN-08` local provider pool and solve correlation | H9 | 3 | 3 |

Islands stay uncarded: they need per-island Lean modules, because
`Foundation.lean` is one fixed path.

## Not in this page

A hosted Prove2Me client (ADR 0002 deferred it, and its docs confirm why);
the evolution arm (`26Q4-EVO-*`); training anything.
