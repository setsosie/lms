#!/usr/bin/env bash
# Verification script for 26Q3-HARN-02: `lean_code` reaches the verifier as Lean.
#
# Discrimination note: every check below fails at the merge base, where
# `_clean_lean_code` did not exist, `_parse_artifacts` stored the raw regex
# capture, `Artifact` had no `lean_code_raw`, and all 52 payloads in
# experiments/stacks_ch4_phase1/artifacts.json began with the literal `"|\n  "`.
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

section "1. The cleaner exists and strips both packaging forms"
check "_clean_lean_code is importable" \
  "uv run python -c \"from lms.agent import _clean_lean_code\""
check "YAML block-scalar header is stripped and the body dedented" \
  "uv run python -c \"
from lms.agent import _clean_lean_code
assert _clean_lean_code('|\n  import Mathlib\n  theorem t : True := trivial') == 'import Mathlib\ntheorem t : True := trivial'
\""
check "chomping and indent indicators are stripped (|-, |+, |2, |2-, >)" \
  "uv run python -c \"
from lms.agent import _clean_lean_code
for h in ('|-', '|+', '|2', '|2-', '>'):
    assert _clean_lean_code(h + '\n  theorem t : True := trivial') == 'theorem t : True := trivial', h
\""
check "markdown fences are stripped, tagged or bare" \
  "uv run python -c \"
from lms.agent import _clean_lean_code
for t in ('', 'lean', 'lean4'):
    assert _clean_lean_code('\\\`\\\`\\\`' + t + '\ntheorem t : True := trivial\n\\\`\\\`\\\`') == 'theorem t : True := trivial', t
\""
check "a fence nested inside a block scalar is stripped too" \
  "uv run python -c \"
from lms.agent import _clean_lean_code
assert _clean_lean_code('|\n  \\\`\\\`\\\`lean\n  theorem t : True := trivial\n  \\\`\\\`\\\`') == 'theorem t : True := trivial'
\""
check "clean source passes through unchanged" \
  "uv run python -c \"
from lms.agent import _clean_lean_code
assert _clean_lean_code('theorem t : True := trivial') == 'theorem t : True := trivial'
\""

section "2. The parser is wired to the cleaner"
check "_parse_artifacts emits unpackaged lean_code" \
  "uv run python -c \"
from lms.agent import Agent
block = '<artifact>\ntype: theorem\nname: t\ndescription: d\nlean: |\n  import Mathlib\n  theorem t : True := trivial\nreferences: []\n</artifact>'
proposed, _ = Agent(id='v', provider=None)._parse_artifacts(block)
assert len(proposed) == 1
assert not proposed[0].lean_code.startswith(('|', '>', '\\\`'))
\""
check "the raw capture is retained on the artifact" \
  "uv run python -c \"
from lms.agent import Agent
block = '<artifact>\ntype: theorem\nname: t\ndescription: d\nlean: |\n  theorem t : True := trivial\nreferences: []\n</artifact>'
proposed, _ = Agent(id='v', provider=None)._parse_artifacts(block)
assert proposed[0].lean_code_raw.startswith('|')
assert proposed[0].lean_code == 'theorem t : True := trivial'
\""
check "lean_code_raw survives a serialization round trip" \
  "uv run python -c \"
from lms.artifacts import Artifact
from lms.agent import Agent
block = '<artifact>\ntype: theorem\nname: t\ndescription: d\nlean: |\n  theorem t : True := trivial\nreferences: []\n</artifact>'
proposed, _ = Agent(id='v', provider=None)._parse_artifacts(block)
assert Artifact.from_dict(proposed[0].to_dict()).lean_code_raw.startswith('|')
\""

section "3. Every recorded December payload cleans up"
check "all 52 archived payloads start as packaging" \
  "uv run python -c \"
import json
a = json.load(open('experiments/stacks_ch4_phase1/artifacts.json'))['artifacts']
assert len(a) == 52
assert all(x['lean_code'].startswith('|') for x in a)
\""
check "none of them still start as packaging after cleaning" \
  "uv run python -c \"
import json
from lms.agent import _clean_lean_code
a = json.load(open('experiments/stacks_ch4_phase1/artifacts.json'))['artifacts']
for x in a:
    c = _clean_lean_code(x['lean_code'])
    assert c and not c.startswith(('|', '>', '\\\`')) and not c.startswith(' '), x['id']
\""
check "at least 5 real payloads are checked in as fixtures" \
  "test \$(ls tests/fixtures/malformed_lean/*.txt | wc -l) -ge 5"

section "4. Re-extraction rewrites archives without touching them"
check "reextract script exists" "test -f scripts/reextract_lean_code.py"
check "it writes a sibling and leaves the input byte-identical" \
  "uv run python -c \"
import hashlib, pathlib, subprocess, sys
src = pathlib.Path('experiments/stacks_ch4_phase1/artifacts.json')
# Clear any output from an earlier run first: the check below reads this file,
# and a stale copy would let it pass on a tree where the script is missing.
src.with_name('artifacts.reextracted.json').unlink(missing_ok=True)
before = hashlib.sha256(src.read_bytes()).hexdigest()
subprocess.run([sys.executable, 'scripts/reextract_lean_code.py', str(src)], check=True, capture_output=True)
assert hashlib.sha256(src.read_bytes()).hexdigest() == before, 'input was modified'
out = src.with_name('artifacts.reextracted.json')
assert out.exists()
\""
check "the rewrite resets stale verdicts rather than carrying them forward" \
  "uv run python -c \"
import json
d = json.load(open('experiments/stacks_ch4_phase1/artifacts.reextracted.json'))['artifacts']
assert sum(1 for x in d if x['verified']) == 0
assert sum(1 for x in d if x.get('prior_verified')) == 48
\""

section "5. Tests and lint"
check "pytest tests/test_lean_extraction.py" "uv run pytest tests/test_lean_extraction.py -q"
check "pytest (full suite)" "uv run pytest -q"
check "pytest leaves lean/ clean" "git diff --quiet -- lean/"
check "ruff on changed source files" \
  "uv run ruff check lms/agent.py lms/artifacts.py tests/test_lean_extraction.py scripts/reextract_lean_code.py"
# Scoped to the files this card touches: mypy follows imports, and the repo
# carries pre-existing errors in config.py, providers/ and society.py that this
# card does not claim to fix. An unscoped check could never go green here.
check "mypy reports nothing in the changed source files" \
  "! uv run mypy lms/agent.py lms/artifacts.py 2>&1 | grep -qE '^lms/(agent|artifacts)\.py'"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-02: PASS"
else
  echo "26Q3-HARN-02: FAIL"
fi
exit "$fail"
