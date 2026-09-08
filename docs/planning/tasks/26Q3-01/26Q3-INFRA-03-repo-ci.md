### 26Q3-INFRA-03: Run the test suite in CI, on a machine that is not the developer's

> **Found work**, filed 2026-09-08 while preparing the #55–#57 merge sequence.
> `gh run list` returns empty: no workflow has ever executed on this repository.
> The Lean workflow that exists is at `lean/.github/workflows/lean_action_ci.yml`,
> where GitHub cannot see it — Actions only reads `.github/workflows/` at the
> repository root. 746 tests and five open PRs have had zero automated gating.

**User Story**: As a reviewer merging harness fixes I cannot run on the box, I
want the suite to run on a clean machine for every pull request, so that
"the tests pass" is evidence rather than an assertion about one laptop.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | HIGH |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-INFRA-03-repo-ci` |
| **Dependencies** | none (unblocks the #55–#57 merge sequence) |
| **PR Size Target** | <300 lines |

---

#### Context

Three run-ending defects reached a cluster run before anyone noticed. Nothing
in the repository was positioned to catch them: there is no root
`.github/` directory, so the `lean-action` workflow `lake new` scaffolded
inside `lean/` has never fired, and the Python suite has only ever run by hand.

Actions is free and unmetered for public repositories on standard runners, so
the only cost here is the one already being paid in cluster time.

Two defects block a green first run, both of the same kind — the suite has
never been executed anywhere but one developer's machine:

1. **Lean guards hardcode a developer's home.** `test_lean_real.py:11` and
   `test_verify_namespace.py:16` gate on

   ```python
   shutil.which("lean") is None and not shutil.which("/home/<developer>/.elan/bin/lean")
   ```

   The fallback is inert anywhere else. Worse, it tests for the *binary* rather
   than a usable toolchain: `elan` installs a `lean` shim that exists before
   `elan default stable` has been run, so the guard reports Lean present and ten
   tests hard-fail with `no default toolchain configured` instead of skipping.
   This reproduces on this workstation today, and is a live hazard for the box
   currently being stood up at the new provider.

2. **`test_from_env_uses_default_models` fails on any clean checkout.** It
   asserts `config.anthropic.model == "claude-opus-4-5-20251101"`; the code
   default is `claude-sonnet-4-5-20250514`, and only a developer's private
   `.env` makes the assertion hold. Already noted in
   `scripts/verify/26Q3-01/verify_26Q3-INFRA-01.sh`, and settled by the
   in-flight config-hermeticity PR. Deselected here with a pointer, not fixed —
   fixing it twice would collide with that PR.

3. **`LeanProject.build()` raises when `lake` is absent.** Surfaced by the
   first CI run this card produced. `build()` calls
   `asyncio.create_subprocess_exec("lake", ...)` unguarded, so on a machine with
   no toolchain it raises `FileNotFoundError` instead of returning a bool.
   `test_build_returns_false_on_missing_project` was written to catch exactly
   this — its comment reads "Either way, it shouldn't crash" — but it could only
   ever observe the failure on a machine without Lean, so it passed for the life
   of the repository. The same box-provisioning window that defect 1 describes
   reaches this one too.

---

#### Acceptance Criteria

- [ ] `.github/workflows/tests.yml` at the repository root runs the suite on
      `ubuntu-latest` for every pull request and every push to `main`, using
      `uv sync --frozen` so a red run means the code changed, not a dependency
- [ ] `.github/workflows/lean.yml` builds the Lean corpus via `lean-action`
      with `lake-package-directory: lean`, scoped to `lean/**` changes so a
      Python-only PR is not made to wait on a Mathlib cache fetch
- [ ] `lean/.github/workflows/lean_action_ci.yml` is deleted — it is superseded,
      and leaving a workflow where GitHub cannot see it invites the same mistake
- [ ] A single `lean_available()` helper replaces both hardcoded-home guards,
      resolves `~` at runtime, and probes `lean --version` so an unconfigured
      `elan` shim counts as absent
- [ ] The suite is green on a checkout with no `.env` and no Lean toolchain,
      with the Lean-dependent modules reported as skipped rather than passed
- [ ] The workflow fails if the suite writes into the tracked Lean corpus,
      making the `conftest.py` guard against issue #19 a checked property
- [ ] The `test_from_env_uses_default_models` deselect carries a comment naming
      what removes it
- [ ] `LeanProject.build()` returns `False` when `lake` is not on `PATH` rather
      than raising, with a regression test that forces the error so the contract
      stays checkable on machines that *do* have a toolchain

---

#### Out of Scope

- Making any check **required** for merge — branch-protection changes are a
  repository-admin action and are the user's call once the workflows have a
  track record.
- Fixing the Anthropic default model string; the in-flight config-hermeticity
  PR owns that.
- Coverage reporting, lint, or type checking. Get one honest gate first.
