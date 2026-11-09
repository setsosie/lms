#!/bin/bash
# Verification for 26Q4-CHORE-03: The ANT arc drafts elaborate at the pin
# Run: bash scripts/verify/26Q4-01/verify_26Q4-CHORE-03.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. The check frames drafts as the probe does and blames errors correctly
uv run pytest tests/test_check_arc_elaboration.py -q

# 2. CI runs it
grep -q "check_arc_elaboration.py" .github/workflows/lean.yml
grep -q "data/ant_arcs/\*\*" .github/workflows/lean.yml

# 3. The neighbours that read the arc files
uv run pytest tests/test_measure_n1_density.py tests/test_novelty.py -q

# 4. Quality
uv run ruff check scripts/check_arc_elaboration.py tests/test_check_arc_elaboration.py
uv run mypy scripts/check_arc_elaboration.py

# 5. Real Lean, where lean/.lake holds the pinned library (CI's Lean job runs it everywhere else)
if [ -d lean/.lake/packages/mathlib ]; then
  uv run python scripts/check_arc_elaboration.py
else
  echo "skipped: no lean/.lake here; the Lean CI job runs the elaboration check"
fi

echo "26Q4-CHORE-03: verification passed"
