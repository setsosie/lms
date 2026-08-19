#!/bin/bash
# Verification for 26Q3-HARN-03: T2 non-vacuity and T4 axiom/sorry gates.
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-03.sh
#
# Behaviour assertions live in tests/test_gates_axioms.py and
# tests/test_gates_vacuity.py, not here. This script checks structure and
# delegates behaviour to pytest.
set -e

# Resolve the repo root from git, not from $0 -- /pre-merge's discrimination
# check replays this script from a throwaway worktree.
cd "$(git rev-parse --show-toplevel)"

# 1. The gate package exists: T4 in axioms.py, T2 in vacuity.py.
test -f lms/gates/axioms.py
test -f lms/gates/vacuity.py
uv run python -c "from lms.gates import AxiomGate, VacuityGate, GateOutcome, GateResult, default_gate_runner"

# 2. INCONCLUSIVE is a first-class outcome and is never a pass.
uv run python -c "
from lms.gates import GateOutcome, GateResult
r = GateResult(gate='T2.duplicate', outcome=GateOutcome.INCONCLUSIVE, reason='x')
assert not r.passed
"

# 3. Artifacts carry gate_results and a strict gates_passed.
grep -q "gate_results" lms/artifacts.py
grep -q "def gates_passed" lms/artifacts.py

# 4. Gates run after verification in all three Society paths
#    (helper + three call sites).
test "$(grep -c "_apply_gates" lms/society.py)" -ge 4

# 5. The gate-failure histogram is a first-class metrics output.
grep -q "def gate_failure_histogram" lms/metrics.py
grep -q "def gate_inconclusive_histogram" lms/metrics.py

# 6. Behaviour, including the card's required regression: the December-run
#    trivial `example` must be rejected.
uv run pytest tests/test_gates_axioms.py tests/test_gates_vacuity.py -q

echo "26Q3-HARN-03: verification passed"
