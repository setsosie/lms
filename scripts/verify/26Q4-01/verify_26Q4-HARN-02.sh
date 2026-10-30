#!/bin/bash
# Verification for 26Q4-HARN-02: Wire TraceStore so conversations.json records transcripts
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-02.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
#
# Discrimination: at main @ 80da9f5 no call site takes a trace store and all
# five tests fail on an empty store.
set -e

# 1. The store is threaded beside the ledger to every call site
grep -q 'trace_store: TraceStore | None' lms/agent.py
grep -q 'trace_store: TraceStore | None' lms/planning.py
grep -q 'trace_store: TraceStore | None' lms/working_group.py
grep -q 'trace_store=self.trace_store' lms/society.py

# 2. Behaviour
uv run pytest tests/test_trace_store_wiring.py tests/test_traces.py tests/test_accounting.py -q

# 3. Hygiene
uv run ruff check lms/agent.py lms/planning.py lms/working_group.py lms/society.py lms/traces.py tests/test_trace_store_wiring.py
uv run mypy lms/agent.py lms/planning.py lms/working_group.py lms/society.py lms/traces.py

echo "26Q4-HARN-02: verification passed"
