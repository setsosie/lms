### 26Q3-INFRA-04: Run the real Lean verifier in CI, without pulling Mathlib

> **Found work**, filed 2026-09-08 while measuring the cost of the CI gate.
> `RealLeanVerifier` — the project's oracle, and the thing `verifier.kind ==
> "real"` asserts about every CVFN number — has **zero** automated coverage.
> Its 16 tests skip in CI (no toolchain) and fail on a developer box with an
> `elan` shim but no configured default. They have never run anywhere.

**User Story**: As someone who will read a CVFN number off this harness, I want
the verifier that produces it exercised by CI, so that "Lean accepted it" is a
checked property rather than an untested code path.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | MEDIUM |
| **Status** | 🔲 PENDING |
| **Branch** | `26Q3-INFRA-04-lean-verifier-in-ci` |
| **Dependencies** | builds on the CI PR, which owns `tests.yml` and `tests/_lean_env.py` |
| **PR Size Target** | <150 lines |

---

#### Context

All 16 skipped tests live in exactly two modules (`test_lean_real.py`,
`test_verify_namespace.py`); the other 803 tests skip nothing. Every one of
them constructs `RealLeanVerifier()` **with no project**, which per
`lms/lean/real.py` runs bare `lean` rather than `lake env lean`, on core-only
code — `Nat`, `True`, `trivial`, `rfl`, a bare `structure`. The namespace
suite says so in its own docstring: "against the real compiler (core-only, no
Mathlib)."

So the toolchain is the only requirement. No Mathlib cache, no `lake build`,
no corpus. `lean.yml` still owns building the corpus and is unchanged.

#### Measured cost

On a developer box, against a real toolchain:

| Run | Result | Time |
|-----|--------|------|
| Full suite, no toolchain (today's CI) | 808 passed, 16 skipped | 6.3s |
| Full suite, real toolchain, cold | 824 passed | 92.3s |
| Full suite, real toolchain, warm | 824 passed | 41.6s |
| The two modules alone, warm | 21 passed | 6.1s |

The cost is one fixed charge, not per-test: `test_verify_valid_theorem` takes
**24.8s** as the first Lean invocation (loading `Init`'s oleans), and the
other 15 invocations run ~0.4s each. That 24.8s charge is a WSL filesystem
artifact and did not reproduce on a runner — see the CI figures below, which
supersede this estimate.

#### Measured in CI (first run)

The job costs **34s** cold: ~10s to acquire the toolchain (elan-init, then a
lazy download on first `lean --version`), 6.1s to run the tests — matching the
6.1s measured locally — and the rest runner setup. The 24.8s cold-start charge
seen on a developer box did not reproduce; it is a WSL filesystem artifact, not
inherent to Lean.

Because the job is parallel to `pytest`, the workflow's critical path goes from
`lint` + `pytest` (15 + 19 = 34s) to `lint` + `lean-tests` (15 + 34 = 49s):
**+15s wall clock**, ~34s of extra runner time.

The toolchain is deliberately **not** cached. Saving the 632 MiB cache cost
9.3s against a ~10s fetch, so it buys nothing, and the repo's cache already
holds two 2.31 GiB library caches that a marginal entry could evict.

#### Change

A `lean-tests` job in `tests.yml`, parallel to `pytest` and gated on `lint`,
that installs elan pinned to `lean/lean-toolchain` and runs those two modules.
Running it beside `pytest` rather than inside it keeps the fast job everyone
waits on fast: the workflow's critical path grows by 15s rather than by the
job's full 34s.

The job asserts `lean --version` succeeds **and** `lean_available()` is True
before running anything. Without that, a missing or unconfigured toolchain
would make all 16 tests skip and the job would still report green — which is
the precise failure this card exists to remove.

#### Acceptance criteria

- [ ] A `lean-tests` job runs the two Lean modules with a real toolchain,
      gated on `lint` and parallel to `pytest`
- [ ] The toolchain is pinned to `lean/lean-toolchain` and installed via elan
      alone — no Mathlib cache, no `lake build`, and not itself cached
- [ ] The job fails, rather than skipping green, when the toolchain is absent
      or is an unconfigured `elan` shim (both guards verified against a real
      unconfigured shim)
- [ ] 16 tests that have never executed anywhere now execute on every PR
- [ ] `lean.yml` is untouched; it keeps sole ownership of building the corpus

#### Out of Scope

- Running the *whole* suite under a toolchain. Only these two modules are
  guarded, and the other 803 tests gain nothing from Lean being present.
- Mathlib, in any form. The moment a test needs it, it belongs in `lean.yml`
  with the cache, not here.
