#!/usr/bin/env bash
# Verification script for 26Q3-HARN-05: per-statement cost accounting.
#
# Behavioural assertions live in tests/test_accounting.py, not here. This
# script checks that the pieces exist, the old inflating counter is gone, and
# the ledger is actually wired, then runs the tests that do the real work.
#
# Discrimination note: at the merge base lms/accounting.py did not exist,
# iterative mode counted len(response.attempts) as artifacts_created, no
# generation call was ledgered, and Society.save wrote no attempts.json.
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

TESTS=tests/test_accounting.py

section "1. The pieces exist and import"
check "AttemptRecord / CostLedger / cvfn_report / statement_key" \
  "uv run python -c 'from lms.accounting import AttemptRecord, CostLedger, cvfn_report, statement_key, OVERHEAD_KEY'"
check "calculate_cvfn re-exported by metrics" \
  "uv run python -c 'from lms.metrics import calculate_cvfn, cvfn_report'"
check "GenerationResult reports attempts and wall-clock" \
  "uv run python -c 'from lms.society import GenerationResult; GenerationResult(0,0,0,0,0).attempts_total; GenerationResult(0,0,0,0,0).wall_clock_s'"
check "test module is present" "test -f $TESTS"

section "2. The old attribution defects are gone"
check "attempts are no longer counted as artifacts_created" \
  "! grep -q 'artifacts_created += len(response.attempts)' lms/society.py"
check "iterative agents write to the society ledger" \
  "grep -q 'ledger=self.ledger' lms/society.py"
check "zero-artifact responses are recorded, not dropped" \
  "grep -q 'record_overhead' lms/agent.py"
check "Society.save persists the ledger" \
  "grep -q 'attempts.json' lms/society.py"

section "3. Behaviour (tests/test_accounting.py)"
check "conservation: sum(attempts) + overhead == society total" \
  "uv run pytest $TESTS -q -k 'conservation or overhead'"
check "retry chain attributes to one statement key" \
  "uv run pytest $TESTS -q -k 'retry'"
check "attempts and library entries are distinct" \
  "uv run pytest $TESTS -q -k 'attempts_are_not_artifacts'"
check "cvfn_report runs on fresh and archived runs" \
  "uv run pytest $TESTS -q -k 'report'"
check "the whole module" "uv run pytest $TESTS -q"

section "4. Suite and hygiene"
check "pytest (full suite)" "uv run pytest -q"
check "pytest leaves lean/ clean" "git diff --quiet -- lean/"
check "ruff on changed source files" \
  "uv run ruff check lms/accounting.py lms/agent.py lms/society.py lms/metrics.py $TESTS"
check "mypy adds nothing new in changed files" \
  "test \$(uv run mypy lms/accounting.py lms/agent.py lms/society.py lms/metrics.py 2>&1 | grep -cE '^lms/(accounting|agent|society|metrics)\.py') -le 3"

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "26Q3-HARN-05: PASS"
else
  echo "26Q3-HARN-05: FAIL"
fi
exit "$fail"
