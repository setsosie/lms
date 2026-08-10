#!/usr/bin/env bash
# Verification script for 26Q3-HARN-09: verified work reaches the next generation.
#
# Discrimination note: at the merge base `Society.persist_foundation` and
# `LeanProject.rebuild_changed_sources` did not exist, `foundation.save()` ran
# only from `Society.save()` at a 10-generation checkpoint, `lean` was invoked
# without strictness flags, and `get_context_for_agent` emitted
# "structure Category / fields: Hom" with no parameter list.
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

section "1. Persistence is part of finishing a generation"
check "Society.persist_foundation exists" \
  "uv run python -c \"from lms.society import Society; assert callable(Society.persist_foundation)\""
check "run_generation persists without waiting for a checkpoint" \
  "uv run pytest tests/test_foundation_persistence.py::test_run_generation_makes_verified_work_importable -q"
check "a generation that verified nothing does not rebuild" \
  "uv run pytest tests/test_foundation_persistence.py::test_run_generation_persists_only_when_something_verified -q"
check "save() is no longer the only writer of the foundation" \
  "uv run python -c \"
import inspect
from lms.society import Society
src = inspect.getsource(Society.run_generation)
assert 'persist_foundation' in src, src
\""

section "1b. Runs start independent of each other"
check "Society.reset_foundation exists" \
  "uv run python -c \"from lms.society import Society; assert callable(Society.reset_foundation)\""
check "run_experiment clears the foundation before generation 1" \
  "uv run python -c \"
import inspect
from lms import run as r
src = inspect.getsource(r.run_experiment)
assert 'reset_foundation' in src, 'a run inherits the previous run(s) definitions'
\""
check "reset writes an empty but still-importable module" \
  "uv run pytest tests/test_foundation_persistence.py::test_reset_writes_an_empty_but_importable_foundation -q"
check "reset discards a previous run's definitions" \
  "uv run pytest tests/test_foundation_persistence.py::test_reset_discards_a_previous_runs_definitions -q"

section "2. Persisting recompiles, so the .olean is not stale"
check "LeanProject.rebuild_changed_sources exists" \
  "uv run python -c \"from lms.lean.project import LeanProject; assert callable(LeanProject.rebuild_changed_sources)\""
check "it bypasses the new-import heuristic (calls build directly)" \
  "uv run python -c \"
import inspect
from lms.lean.project import LeanProject
body = inspect.getsource(LeanProject.rebuild_changed_sources).split('\\\"\\\"\\\"')[-1]
assert 'self.build()' in body, body
assert 'ensure_built' not in body, body
\""
check "persist_foundation drives the rebuild" \
  "uv run pytest tests/test_foundation_persistence.py::test_persist_rebuilds_so_the_olean_is_not_stale -q"
check "safe with a verifier that has no project" \
  "uv run pytest tests/test_foundation_persistence.py::test_persist_survives_a_verifier_with_no_project -q"

section "3. autoImplicit is off, on both paths"
check "strictness flags are passed to lean itself" \
  "uv run python -c \"
from lms.lean.real import RealLeanVerifier
f = RealLeanVerifier.STRICTNESS_FLAGS
assert '-DautoImplicit=false' in f, f
assert '-DrelaxedAutoImplicit=false' in f, f
\""
check "the flags reach the actual command, before the file" \
  "uv run pytest 'tests/test_lean_real_env.py::TestVerifierUsesProjectEnvironment::test_strictness_flags_reach_the_command' -q"
check "the foundation header matches the verifier" \
  "uv run python -c \"
from lms.foundation import FoundationFile
h = FoundationFile.FOUNDATION_HEADER
assert 'set_option autoImplicit false' in h
assert 'set_option relaxedAutoImplicit false' in h
\""

section "4. Agents can see the interface they are told to reuse"
check "declaration signature appears in agent context" \
  "uv run pytest tests/test_foundation_persistence.py::test_agent_context_shows_the_declaration_signature -q"
check "the parameter list specifically is visible" \
  "uv run python -c \"
import tempfile, pathlib
from lms.foundation import FoundationFile
from lms.artifacts import Artifact, ArtifactType
from lms.lean.interface import VerificationStatus
d = pathlib.Path(tempfile.mkdtemp())
f = FoundationFile(d / 'Foundation.lean')
f.add_artifact(Artifact(
    id='definition-Category-x', type=ArtifactType.DEFINITION,
    natural_language='c', lean_code='structure Category (obj : Type u) where\n  Hom : obj → obj → Type v',
    status=VerificationStatus.VERIFIED_LEAN, created_by='a', generation=0))
ctx = f.get_context_for_agent()
assert '(obj : Type u)' in ctx, ctx
\""

section "5. Tests and lint"
check "pytest tests/test_foundation_persistence.py" \
  "uv run pytest tests/test_foundation_persistence.py -q"
check "pytest (full suite)" "uv run pytest -q"
check "pytest leaves lean/ clean" "git diff --quiet -- lean/"
check "ruff on changed source files" \
  "uv run ruff check lms/society.py lms/lean/real.py lms/lean/project.py tests/test_foundation_persistence.py"
check "mypy adds nothing new in changed files" \
  "test \$(uv run mypy lms/society.py lms/foundation.py lms/lean/real.py lms/lean/project.py 2>&1 | grep -cE '^lms/(society|foundation|lean/real|lean/project)\.py') -le 4"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-09: PASS"
else
  echo "26Q3-HARN-09: FAIL"
fi
exit "$fail"
