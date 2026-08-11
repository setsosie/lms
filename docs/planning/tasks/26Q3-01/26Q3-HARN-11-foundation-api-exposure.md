### 26Q3-HARN-11: Expose the full API of foundation entries to agents

**User Story**: As an agent building on a prior generation's verified work, I
want to see the full declaration shape of every foundation entry — parameters
and typed fields, not a name — so that I can apply it correctly instead of
guessing and failing to elaborate.

| Field | Value |
|-------|-------|
| **Story Points** | 3 |
| **Priority** | HIGH |
| **Status** | 🟡 IN REVIEW |
| **Branch** | `26Q3-HARN-11-foundation-api-exposure` |
| **Dependencies** | 26Q3-HARN-09, 26Q3-HARN-10 (both merged) |
| **PR Size Target** | <500 lines (max 1000) |
| **Parts** | single PR |

---

#### Context

> Investigated 2026-08-10 against `experiments/shakedown_3x3_d` (3 agents ×
> 3 generations, `stacks-ch4-phase1`, `--iterative`, run on the H100 box with
> #27 merged). All line numbers are from `main` @ `a93c262`.

**What the run showed.** This was the first run in the project's history to
produce cross-generational reuse. All 4 artifacts carried both
`import LMS.Foundation` and `open LMS.Foundation` (HARN-10 confirmed). Gen 2,
agent-1, tag 0017 cited `refs=['definition-Category-d3a579da']`, and the import
elaborated — Lean resolved `Category` and printed its type. The attempt then
failed on **API shape**:

```
verify_3bfa623a.lean:6:32: error: type expected, got
  (Category : Type v → Type (max v (u + 1)))
```

The agent used `Category` as a bare type. The foundation's `Category` is indexed
by an object type (`structure Category (obj : Type u)`). This is not a Lean 3
syntax problem and not a missing-Mathlib problem — the agent did not know the
shape of the thing it was importing.

**Why it did not know.** Agents receive `FoundationFile.get_context_for_agent()`
(`lms/agent.py:216` and `lms/agent.py:300`). Re-verified against `main`
@ `a93c262` by rendering a real `Category` entry — the card's first draft
mis-attributed this to `get_available_definitions()` (`lms/foundation.py:388`),
which **no production caller reaches**; only tests call it. The live renderer is
`get_context_for_agent()` (`lms/foundation.py:460-486`), and what it actually
emits is:

```
  structure Categorystructure Category(obj : Type u) where
    fields: Hom, id, comp, assoc, id_l, ...
```

Three distinct defects, all in that four-line block:

- `lms/foundation.py:470` — `f"{entry.entry_type} {entry.name}{entry.signature}"`
  **duplicates the header**, because `signature` is already
  `f"{entry_type} {name}{rest}"` (`lms/foundation.py:315`). The agent is shown a
  declaration that is not valid Lean and does not appear in `Foundation.lean`.
- `lms/foundation.py:313` — `rest = match.group(3).strip()` eats the separating
  space, so the parameter binds to the name: `Category(obj : Type u)`.
- `lms/foundation.py:482` — `field_match[:5]` drops the 6th field onward behind
  a bare `...`, and fields are rendered **name-only**. An agent sees `Hom`, never
  `Hom : obj → obj → Type v`, so it cannot tell arity or argument order.

So agent-1 was shown a garbled header and six bare field names, and had to guess
how `Category` is applied.

**There are two summaries, and the weaker one feeds committee mode.**

| Path | Summary used | Content |
|---|---|---|
| iterative / standard (`lms/agent.py:216,300`) | `FoundationFile.get_context_for_agent()` (`lms/foundation.py:418`) | duplicated header + first 5 field *names* |
| committee mode (`lms/society.py:771`) | `Society._get_foundation_summary()` (`lms/society.py:955-966`) | **raises `AttributeError`** on any non-empty foundation |

`_get_foundation_summary()` is worse than under-informative — it is broken.
`lms/society.py:960` calls `self.foundation.entries.values()`, but
`FoundationFile.entries` is a `list` (`lms/foundation.py:153`); `:963` then reads
`entry.tag`, a field `FoundationEntry` does not have (`lms/foundation.py:38-44`).
Both raise. This is invisible today only because committee mode is unreachable
(`26Q3-HARN-12`) and the sole test (`tests/test_society.py:837`) exercises the
empty branch. `entries[:10]` is a third silent truncation waiting behind them.

`_get_foundation_summary()`'s output is passed as `foundation_summary` into
`PlanningPanel` (`lms/society.py:779`) and `WorkingGroup`
(`lms/working_group.py:231`, used at `:297`). So the moment `26Q3-HARN-12` makes
committee mode reachable, planning and work committees crash on the first
non-empty foundation. **Both summaries are in scope for this card**; fixing only
one leaves HARN-12 inheriting both this crash and the API-shape blindness that
caused the failure documented above.

**Current State**:
- `FoundationEntry` already stores the full `lean_code` of each definition
  (`lms/foundation.py:333`), so parameters and typed fields are available to
  both summaries; they are simply not rendered.
- `DEFINITION_PATTERN` (`lms/foundation.py:107-112`) captures `[^\n]*` — the
  first line only. That is fine: `lean_code` carries the body, so the fix does
  **not** require touching the pattern (see Decision Gates).

**Investigation**:
```bash
uv run python -c "
from pathlib import Path; import tempfile
from lms.foundation import FoundationFile
f = FoundationFile(Path(tempfile.mkdtemp())/'Foundation.lean')
f.entries = f._extract_entries('structure Category (obj : Type u) where\n  Hom : obj -> obj -> Type v\n', 'a', 0, 'ag')
print(f.get_context_for_agent())"
# ->   structure Categorystructure Category(obj : Type u) where
```

---

#### Acceptance Criteria

> Each criterion must be verifiable with a single command returning exit code 0.
> All of AC-1..AC-6 are covered by `uv run pytest tests/test_foundation_api_exposure.py`.

- [x] **AC-1** `get_context_for_agent()` renders each declaration header exactly
      once — no `structure Categorystructure Category`.
- [x] **AC-2** The rendered header reproduces the source declaration line
      verbatim, space intact: `structure Category (obj : Type u) where`.
- [x] **AC-3** Structure fields are rendered with their types
      (`Hom : obj → obj → Type v`), not as bare names.
- [x] **AC-4** No structure field is dropped silently; a 6-field structure shows
      all 6 (any elision is explicit and counted).
- [x] **AC-5** `Society._get_foundation_summary()` returns a string instead of
      raising on a non-empty foundation.
- [x] **AC-6** That committee summary carries the declaration shape
      (`(obj : Type u)`), not names alone.

Added 2026-08-10 after the first `/pre-merge` review found a regression in the
first implementation (see *Outcome*):

- [x] **AC-7** A doc-commented declaration renders its **declaration**, not its
      comment. `/-- A category. -/\nstructure Category …` must render
      `structure Category (obj : Type u) where`, with its fields.
- [x] **AC-8** `class` fields and `inductive` constructors render, not just
      `structure` fields.
- [x] **AC-9** A theorem statement that wraps across lines renders to its
      conclusion; the proof body does not render.
- [x] **AC-10** No renderer in `lms/foundation.py` retains a silent character
      cap (`signature[:80]` is gone).

Added after the second `/pre-merge` round:

- [x] **AC-11** A `:=` inside Lean 4 named arguments does not cut the statement
      (`pullback.fst (f := f) (g := g) …` renders whole).
- [x] **AC-12** `instance … where` renders every field, not one bare name.
- [x] **AC-13** A proof body never renders, including `:= by` on the
      declaration line and tactic blocks under a `where` field.
- [x] **AC-14** Foreign declarations (`@[ext] theorem`, `section`, `variable`,
      `example`) do not render as fields of the preceding entry.

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/foundation.py` | MODIFY | Add `FoundationEntry.declaration_header()` / `.field_lines()` reading `lean_code`; render from them instead of `signature` (`:470`, `:474-485`) |
| `lms/society.py` | MODIFY | `_get_foundation_summary()` (`:955-966`) defers to `get_context_for_agent()` instead of raising |
| `tests/test_foundation_api_exposure.py` | CREATE | AC-1..AC-6 |
| `docs/planning/tasks/26Q3-01/26Q3-HARN-11-foundation-api-exposure.md` | MODIFY | This card |
| `scripts/verify/26Q3-01/verify_26Q3-HARN-11.sh` | MODIFY | Verification |

> This table is a fence `/pre-merge` checks, not a suggestion: a changed file not
> named here (tests and this task's own card/verify script excepted) is a
> blocking finding. Name every file you expect to touch.

---

#### Implementation Notes

**Render from `lean_code`, not from `signature`.** `signature` is a derived,
lossy string that already caused the duplication bug; `lean_code` is what was
actually verified and what is actually written to `Foundation.lean`, so it cannot
drift from the file. The declaration header is `lean_code`'s first line; the
fields are its indented continuation lines.

**Do not introduce:**
- A new silent cap. Replacing `[:5]` / `[:10]` / `[:80]` with a bigger silent
  number is the same bug. If anything is elided, print how many were elided.
- A change to `DEFINITION_PATTERN` — see Decision Gates.
- A change to what `save()` writes. This card touches renderers only.

**Token budget / growth story.** The foundation grows one entry per verified
artifact. Full-body rendering of every entry is unbounded; header + typed field
lines is bounded by the declaration's own width and is what an agent needs to
*apply* an entry. Theorem entries have no fields, so they cost one line. If a
cap is needed later, it belongs on entry count with an explicit count, not on
characters mid-token.

**Regression found at the gate (2026-08-10).** The first implementation read
"the first non-blank line of `lean_code`" as the declaration. `lean_code` does
not start there. `_strip_block_comments` blanks `/-- … -/` to spaces before
matching, `DEFINITION_PATTERN`'s leading `\s*` reaches back across the blanked
region, and `_extract_entries` slices the **original** source from that offset —
so a doc-commented declaration carries its comment as line 1. A doc-commented
`Category` rendered as `/-- A category. -/` and nothing else: no declaration, no
parameter, no fields, and `field_lines()` returned `[]` because it consumed the
comment as the header and broke on the unindented `structure` below.

That was *worse than base*, which at least emitted a garbled header containing
the parameter. 425 of 870 corpus artifacts carry a doc comment; 85% of extracted
entries rendered one as their header. Now 0 of 4474.

The fix is in the renderers (`FoundationEntry._code()` strips leading comments),
**not** at the slice boundary — correcting `_extract_entries` would change what
`save()` writes into `Foundation.lean`, which the gate below rules out. The cost
is that `lean_code` still starts at a comment for every other consumer.

**Corpus health, 4474 entries** (measured each round, not asserted):

| Defect | Before | Round 1 | Round 2 |
|---|---|---|---|
| Comment rendered as header | 85% | 0 | 0 |
| Foreign declarations as fields | 4.2% | 4.2% | 0 |
| Unbalanced-bracket renders | 31 | 31 | 2 |
| Leaked tactic lines | — | 522 | 1 |

The 2 residual unbalanced renders (0.04%) come from a `by` opening inside a
type-level application; left as known residue rather than chased further.

**Growth story, measured.** A real 140-entry foundation
(`experiments/stacks_ch4_phase1`) renders 15,927 chars ≈ 4K tokens of
agent-facing context — all declarations and typed fields, zero comment lines.
Against a 131,072-token model context that is ~3%.

**Outcome (2026-08-10).** All four gates below held; none fired.
`_extract_entries` and `DEFINITION_PATTERN` are untouched, so `entry.signature`
and `foundation.json` round-trip unchanged and `Foundation.lean`'s on-disk format
is identical. `signature` is now a persisted-metadata field with no renderer
reading it — worth deleting once nothing else depends on it, but not here.
`_get_foundation_summary()` was collapsed onto `get_context_for_agent()` rather
than repaired: two renderers that must stay in sync is how the committee path
drifted into raising `AttributeError` unnoticed in the first place.

---

#### Decision Gates

- If rendering typed fields blows the agent's context budget, stop and surface
  the token math — do not substitute a different silent cap.
- **Do not touch `DEFINITION_PATTERN`.** `lean_code` already carries the body, so
  the fix does not need it. Changing it to capture multi-line bodies would change
  what `_extract_entries` writes into `Foundation.lean` — the accumulated corpus,
  written verbatim by `save()` (`lms/foundation.py:517`). A regression there
  invalidates the run.
- Fixing `rest = match.group(3).strip()` (`lms/foundation.py:313`) changes
  `entry.signature`, which is persisted in `foundation.json`
  (`FoundationEntry.to_dict`). If existing metadata must round-trip, prefer
  fixing the renderer over the extractor and surface the choice.
- If `_get_foundation_summary()` turns out to need `Society` state this card does
  not own, stop — `26Q3-HARN-12` owns committee wiring.
- If the change exceeds the PR Size Target → stop and split.

---

#### Out of Scope

- `26Q3-HARN-12` owns making committee mode reachable (CLI flag, review stage,
  mode reconciliation) — do not implement any of that here. This card changes
  only *what the summaries emit*, in both code paths.
- `26Q3-HARN-08` owns Lean 3 syntax and `import Mathlib` prompting. Note that
  `shakedown_3x3_d` produced **no** evidence for HARN-08 (zero Lean 3 syntax,
  zero missing-Mathlib errors in 4/4 artifacts) — do not fold it in here.
- `26Q3-HARN-03` owns the T2/T4 gates.
- No changes to `Foundation.lean`'s on-disk format or header.

---

#### Verification Script

See `scripts/verify/26Q3-01/verify_26Q3-HARN-11.sh`.

Behaviour goes in pytest, not in the bash script — see the standing lesson that
verify scripts hold no assertions. The script must **fail at the merge base**;
a script that passes without the change gates nothing.

---

#### Outcome Demo

**Where**: mahpiya (4×H100)
**Run**:
```bash
uv run lms run --goal stacks-ch4-phase1 --agents 3 --generations 3 --iterative --output experiments/shakedown_3x3_e
```
Then, over the run's artifacts:
```bash
grep -l "open LMS.Foundation" experiments/shakedown_3x3_e/artifacts/*.lean | xargs grep -c "\.Hom\|\.comp\|Category " 
```
**Expect**: a gen ≥1 artifact that references a prior definition **and**
elaborates — i.e. it names the foundation's actual fields (`Hom`, `comp`, …)
rather than treating the structure as a bare type. Success is not the
verification rate.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `scripts/verify/26Q3-01/verify_26Q3-HARN-11.sh` exits 0 (and exits 1 at merge base `a93c262`)
- [x] `uv run ruff format`, `uv run ruff check` clean on touched files; `mypy` 12 → 11 errors (all remaining pre-existing)
- [ ] PR opened with <500 lines changed (target) / <1000 (max)
- [x] Tests included with implementation
- [ ] Outcome Demo run by a human validator (or the card explicitly says `N/A` and why)
