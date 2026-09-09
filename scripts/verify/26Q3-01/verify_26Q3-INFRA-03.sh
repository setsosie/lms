#!/usr/bin/env bash
# Verification script for 26Q3-INFRA-03: run the suite in CI, on a machine that
# is not the developer's.
#
# Discrimination note: every check in sections 1-3 fails at the merge base,
# where there was no root `.github/` directory at all (`gh run list` returned
# empty — no workflow had ever executed on this repository) and the only
# workflow lived at `lean/.github/workflows/lean_action_ci.yml`, where GitHub
# does not look. Section 4 fails at the merge base because both Lean guards
# hardcoded `/home/<developer>/.elan/bin/lean`; section 5 fails there because
# that guard tested for the binary rather than a usable toolchain, so on a
# machine with an unconfigured elan shim the Lean modules errored instead of
# skipping.
set -uo pipefail

fail=0
section() { printf '\n== %s ==\n' "$1"; }
check() {
  if eval "$2" >/dev/null 2>&1; then
    printf '  ok   %s\n' "$1"
  else
    printf '  FAIL %s\n' "$1"
    fail=1
  fi
}

section "1. Workflows are where GitHub can see them"
check "root .github/workflows/tests.yml exists" \
  "test -f .github/workflows/tests.yml"
check "root .github/workflows/lean.yml exists" \
  "test -f .github/workflows/lean.yml"
check "the unreachable lean/.github/ workflow is gone" \
  "! test -e lean/.github/workflows/lean_action_ci.yml"
check "no workflow of ours is left nested under lean/ (.lake vendors deps' own)" \
  "! find lean -path '*/.github/workflows/*' -name '*.yml' -not -path '*/.lake/*' | grep -q ."

section "2. The Python job gates every pull request"
check "tests.yml triggers on pull_request" \
  "grep -q '^  pull_request:' .github/workflows/tests.yml"
check "dependencies come from the lockfile, not a fresh resolve" \
  "grep -q 'uv sync --frozen' .github/workflows/tests.yml"
check "it actually runs pytest" \
  "grep -q 'uv run pytest' .github/workflows/tests.yml"
check "no deselect remains (the config suite is hermetic on a clean checkout)" \
  "! grep -q 'deselect' .github/workflows/tests.yml"
check "the tracked Lean corpus is asserted clean after the suite" \
  "grep -q 'git status --porcelain lean/' .github/workflows/tests.yml"

section "3. The Lean job builds the corpus without blocking Python PRs"
check "lean.yml uses lean-action" \
  "grep -q 'leanprover/lean-action' .github/workflows/lean.yml"
check "it points at the lake package under lean/" \
  "grep -q 'lake-package-directory: lean' .github/workflows/lean.yml"
check "it is scoped to lean/ changes" \
  "grep -q \"lean/\\*\\*\" .github/workflows/lean.yml"

section "4. No machine-specific paths survive in the suite"
check "no developer home is hardcoded anywhere in tests/" \
  "! grep -rn '/home/stsosie' tests/ --include=*.py"
check "the shared helper exists" \
  "test -f tests/_lean_env.py"
check "test_lean_real.py uses it" \
  "grep -q 'from tests._lean_env import' tests/test_lean_real.py"
check "test_verify_namespace.py uses it" \
  "grep -q 'from tests._lean_env import' tests/test_verify_namespace.py"

section "5. Lean availability means a usable toolchain, not a stray shim"
check "the helper resolves the home directory at runtime" \
  "grep -q 'Path.home()' tests/_lean_env.py"
check "the helper probes the toolchain rather than trusting the binary" \
  "grep -q -- '--version' tests/_lean_env.py"
check "the probe result is cached" \
  "grep -q 'lru_cache' tests/_lean_env.py"

section "6. Lint gates the expensive jobs"
check "a lint job exists" \
  "grep -q '^  lint:' .github/workflows/tests.yml"
check "pytest waits on it" \
  "grep -q '^    needs: lint$' .github/workflows/tests.yml"
check "ruff runs in CI" \
  "grep -q 'uv run ruff check' .github/workflows/tests.yml"
check "mypy runs in CI" \
  "grep -q 'uv run mypy' .github/workflows/tests.yml"
check "ruff is clean on the tracked tree" \
  "git ls-files '*.py' | xargs uv run ruff check"
check "mypy is clean on the package" \
  "uv run mypy lms/"

section "7. The suite behaves on this machine"
check "full suite passes" \
  "uv run pytest -q"
check "Lean-dependent modules skip rather than fail when Lean is unusable" \
  "test -n \"\$(uv run pytest tests/test_lean_real.py -q --no-header 2>&1 | grep -E 'skipped|passed')\""
check "build() degrades instead of raising when lake is absent" \
  "uv run pytest tests/test_lean_project.py -k lake_is_absent -q"
check "build() guards the subprocess launch" \
  "grep -q 'except FileNotFoundError' lms/lean/project.py"

check "the suite leaves the tracked Lean corpus untouched" \
  "test -z \"\$(git status --porcelain lean/LMS/)\""

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf 'PASS 26Q3-INFRA-03\n'
else
  printf 'FAIL 26Q3-INFRA-03\n'
fi
exit "$fail"
