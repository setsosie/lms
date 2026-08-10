#!/usr/bin/env bash
# Verification script for 26Q3-HARN-01: verified status carries verifier provenance.
#
# Discrimination note: every check below fails at the merge base, where
# `VerificationStatus` did not exist, `Artifact.verified` was a settable bool,
# `MockLeanVerifier` results were indistinguishable from real ones, and the
# archived mock run re-scored to 48/52 verified.
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

section "1. Provenance is required and governs status"
check "VerificationResult rejects construction without a verifier kind/id" \
  "uv run python -c \"
from lms.lean.interface import VerificationResult
try:
    VerificationResult(success=True, code='theorem t : True := trivial')
except TypeError:
    pass
else:
    raise SystemExit('constructed a result with no provenance')
\""

check "status is derived from kind: mock->heuristic, real/mcp->lean" \
  "uv run python -c \"
from lms.lean.interface import VerificationResult, VerificationStatus as S
mk = lambda k: VerificationResult(success=True, code='x', verifier_kind=k, verifier_id=k)
assert mk('mock').status is S.VERIFIED_HEURISTIC
assert mk('real').status is S.VERIFIED_LEAN
assert mk('mcp').status is S.VERIFIED_LEAN
bad = VerificationResult(success=False, code='x', verifier_kind='mock', verifier_id='m')
assert bad.status is S.FAILED
\""

check "an unknown verifier kind raises rather than defaulting" \
  "uv run python -c \"
from lms.lean.interface import VerificationResult
try:
    VerificationResult(success=True, code='x', verifier_kind='reall', verifier_id='x')
except ValueError:
    pass
else:
    raise SystemExit('typo in kind was accepted')
\""

section "2. The mock has no path to VERIFIED_LEAN"
check "every mock accept/reject path stays below Lean grade" \
  "uv run python -c \"
import asyncio
from lms.lean.mock import MockLeanVerifier
from lms.lean.interface import VerificationStatus as S

codes = ['theorem t : True := trivial', 'def f (n : Nat) : Nat := n',
         'axiom bad : False', 'example : True := trivial', '', 'sorry', 'junk']
v = MockLeanVerifier()
for c in codes:
    r = asyncio.run(v.verify(c))
    assert r.status is not S.VERIFIED_LEAN, (c, r.status)
assert v.verifier_kind == 'mock'
\""

section "3. Artifact.verified cannot be assigned"
check "writing the boolean raises AttributeError" \
  "uv run python -c \"
from lms.artifacts import Artifact, ArtifactType
from lms.lean.interface import VerificationStatus as S
a = Artifact(id='a', type=ArtifactType.THEOREM, natural_language='', created_by='', generation=0)
assert a.verified is False
try:
    a.verified = True
except AttributeError:
    pass
else:
    raise SystemExit('verified was settable')
a.status = S.VERIFIED_LEAN
assert a.verified is True
\""

section "4. Legacy records load demoted, never promoted"
check "provenance-free 'verified: true' becomes VERIFIED_HEURISTIC" \
  "uv run python -c \"
from lms.artifacts import Artifact
from lms.lean.interface import VerificationStatus as S
legacy = dict(id='o', type='theorem', natural_language='', verified=True,
              created_by='agent-0-mock', generation=0)
a = Artifact.from_dict(legacy)
assert a.status is S.VERIFIED_HEURISTIC, a.status
assert a.verified is False
\""

section "5. Gate A: the archived mock run re-scores to zero"
check "stacks_ch4_phase1 reports 0 verified (was 48/52 = 92%)" \
  "uv run python -c \"
from pathlib import Path
from lms.artifacts import ArtifactLibrary
p = Path('experiments/stacks_ch4_phase1/artifacts.json')
if not p.exists():
    raise SystemExit('archived run missing; cannot evaluate Gate A')
lib = ArtifactLibrary.load(p)
counts = lib.count_by_status()
assert len(lib) == 52, len(lib)
assert len(lib.get_verified()) == 0, lib.get_verified()
assert counts['verified_heuristic'] == 48, counts
\""

section "6. Experiment metadata records the verifier"
check "verifier_metadata() reports kind/id/lean_version/mathlib_rev" \
  "uv run python -c \"
from lms.config import ProviderConfig
from lms.lean.mock import MockLeanVerifier
from lms.providers.base import BaseLLMProvider, GenerationResponse, TokenUsage
from lms.society import Society

class P(BaseLLMProvider):
    name = 'mock'
    async def generate(self, messages, system_prompt=None, max_tokens=4096):
        return GenerationResponse(content='', usage=TokenUsage(input_tokens=0, output_tokens=0), provider=self.name)

s = Society(n_agents=1, provider=P(ProviderConfig(api_key='k', model='m')), verifier=MockLeanVerifier())
m = s.verifier_metadata()
assert set(m) == {'kind', 'id', 'lean_version', 'mathlib_rev'}, m
assert m['kind'] == 'mock', m
\""

section "7. The suite no longer mutates the tracked Lean corpus"
check "pytest leaves lean/ clean" \
  "git diff --quiet -- lean/ && uv run pytest -q && git diff --quiet -- lean/"

section "8. Tests and lint"
check "pytest tests/test_verifier_provenance.py" "uv run pytest tests/test_verifier_provenance.py -q"
check "pytest (full suite)" "uv run pytest -q"
check "ruff on changed source files" \
  "uv run ruff check lms/lean/interface.py lms/lean/mock.py lms/lean/real.py lms/artifacts.py tests/test_verifier_provenance.py tests/conftest.py"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-01: PASS"
else
  echo "26Q3-HARN-01: FAIL"
fi
exit "$fail"
