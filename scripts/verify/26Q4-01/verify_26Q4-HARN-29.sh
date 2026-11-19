#!/bin/bash
# Verification for 26Q4-HARN-29: The foundation keeps the imports its entries verified with
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-29.sh
#
# This script checks that the work landed. It is NOT a test suite.
#   Do:    test -f / grep -q / uv run pytest <file> / ruff / mypy / one-line import smoke
#   Don't: inline assertions, python heredocs, pass/fail counters, threshold math
# Behavior is proven in tests/ (CI runs it every commit) — call pytest on it from here.
set -e

# 1. Behaviour
uv run pytest tests/test_foundation_imports.py tests/test_foundation_universes.py \
  tests/test_foundation_rollback.py tests/test_resume_foundation.py tests/test_seed.py -q

# 2. The pieces are where a run reaches them
grep -q 'entry.imports = own_imports' lms/foundation.py
grep -q '"seed_imports": list(self.seed_imports)' lms/foundation.py
grep -q 'imports=tuple(d.get("imports", ()))' lms/foundation.py
grep -q 'for line in lean_error_lines(output)' lms/lean/project.py

# 3. Quality
uv run ruff check lms/foundation.py lms/lean/project.py tests/test_foundation_imports.py
uv run mypy lms/foundation.py lms/lean/project.py

echo "26Q4-HARN-29: verification passed"
