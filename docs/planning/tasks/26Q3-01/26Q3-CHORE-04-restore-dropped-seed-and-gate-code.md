### 26Q3-CHORE-04: Restore the code the #53 and #56 squashes dropped

> **Found work**, filed 2026-09-08 while merging `main` into #59 and #60.
> `main` has not been green since #53 landed on 2026-08-21: `pytest` cannot
> even finish collection, 123 of 793 tests fail, and `ruff` reports five
> undefined names. Two separate squash-merges each dropped code their source
> branch had. Nothing caught it because the repository had no CI until #60
> (`26Q3-INFRA-03`).

**User Story**: As anyone about to run the harness or judge a model on its
output, I want `main` to import and run, so that a red suite means my change
broke something rather than being the permanent background state.

| Field | Value |
|-------|-------|
| **Story Points** | 1 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-CHORE-04-restore-dropped-seed-and-gate-code` |
| **Dependencies** | none — **must merge before #60**, whose CI cannot go green until this lands |
| **PR Size Target** | <150 lines |

---

#### Context

Two independent drops, both confirmed by diffing `main` against the source
branch that was squashed:

**`lms/society.py` (from #53, `85ccf9e`)** kept one of the three gate import
lines its branch `26Q3-HARN-19-contentless-artifacts-verify` had. Four names
survive in the body with no binding: `default_novelty_classifier`,
`apply_novelty_gate`, `GateOutcome`, `named_declarations`. The first is called
in `Society.__init__`, so **every** construction of a `Society` raises
`NameError` — one root cause accounts for all 114 non-collection failures.

**`lms/foundation.py` (from #56, `4344eb3`)** lost `strip_header_universes()`
and `FoundationFile.set_seed()`, both intact on `26Q3-HARN-23-seed-generation-0`
(`5997ce5`). `tests/test_seed.py` imports the first, so the module fails at
*collection*, which aborts the whole pytest run before any test executes —
that is why the 114 failures were invisible behind a single error. The second
is called from `lms/society.py:327` and from `lms/foundation.py:1191`'s own
load path, so a seeded run crashes and a resumed one cannot restore its seed.
Seeded is the default. `seed_source` / `seed_entries` are still initialised in
`__init__` and read in six places, so the seed layer is not merely broken but
silently inert wherever it does not raise.

#### Change

Restore all of it verbatim from the source branches — no redesign, no
opportunistic cleanup. The three import lines in `lms/society.py` are copied
from `26Q3-HARN-19`; the two functions in `lms/foundation.py` are copied from
`5997ce5` and reinserted at their original anchors. `lms.gates.novelty` is
imported from the submodule rather than the package, as
`lms/gates/__init__.py` documents: re-exporting it there is a circular import.

#### Acceptance criteria

- [ ] `ruff check` reports no undefined names (F821) in the two files.
- [ ] `tests/test_seed.py` collects and passes (31 tests) — collection is the
      regression that hid everything else.
- [ ] Full suite: 815 passed, 0 failed on a machine **with** a Lean toolchain.
      Without one, the 9 toolchain-dependent tests in `test_lean_real.py` and
      `test_verify_namespace.py` still fail; #60 turns those into skips, and
      the two PRs together are green on any machine.
- [ ] A `FoundationFile` given the shipped seed installs names (`set_seed`
      returns a non-empty list) rather than returning silently empty.
