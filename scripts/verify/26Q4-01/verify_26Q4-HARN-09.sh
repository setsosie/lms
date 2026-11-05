#!/bin/bash
# Verification for 26Q4-HARN-09: The novelty control can fail, and semantic hits stop reading as N1
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh
set -e

# 1. Required content present
grep -q "_SEMANTIC_MATCH_SHARE" lms/novelty/__init__.py
grep -q "max_n1_confidence_reached" lms/novelty/__init__.py
grep -q "exact_probe/v3" lms/novelty/mathlib_search.py
grep -q "names_are_labels=names_are_labels" scripts/measure_n1_density.py
grep -q "max_n1_confidence_reached" docs/runbooks/2026-10-08-box-restore.md
grep -q "statements_by_stages_run" data/novelty_control/README.md

# 2. Tests pass
uv run pytest tests/test_novelty.py tests/test_measure_n1_density.py -v --tb=short

# 3. Code quality
uv run ruff check lms/novelty/ scripts/measure_n1_density.py
uv run mypy lms/novelty/

echo "26Q4-HARN-09: verification passed"
