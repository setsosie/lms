#!/bin/bash
# Verification for 26Q3-HARN-20 Part 1: a search stage that could not have
# matched does not vote "absent".
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-20.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
#
# Discrimination: every pytest line below fails at the merge base, where
# query-less stages, non-elaborating probes and label names all counted as
# searches that ran and found nothing.
set -e

# 1. The probe reads Lean's output through one function
grep -q "def read_output" lms/novelty/mathlib_search.py

# 2. Stale cached probe outcomes are retired, not replayed
grep -q 'exact_probe/v2:' lms/novelty/mathlib_search.py

# 3. The arcs declare label names, and the control is committed
grep -q '"names_are_labels": true' data/ant_arcs/core_arc.json
grep -q '"names_are_labels": true' data/ant_arcs/ramification_arc.json
test -f data/novelty_control/gate_a_control_arc.json

# 4. Behavior proven in pytest
uv run pytest tests/test_novelty.py -q -k "ExactProbeOutput or NoQuery or ceiling or reach_decisive"
uv run pytest tests/test_measure_n1_density.py -q

echo "26Q3-HARN-20 Part 1: verification passed"
