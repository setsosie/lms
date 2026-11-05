#!/bin/bash
# Verification for 26Q4-HARN-09: The novelty control can fail, and semantic hits stop reading as N1
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-09.sh
set -e

# 1. Required content present
grep -q "_SEMANTIC_MATCH_SHARE" lms/novelty/__init__.py
grep -q "_SEMANTIC_MATCH_MIN_TOKENS" lms/novelty/__init__.py
grep -q "_NEAR_HIT_N1_CONFIDENCE" lms/novelty/__init__.py
grep -q "n1_with_decisive_stages" lms/novelty/__init__.py
grep -q "novelty_candidates" lms/gates/novelty.py
test -f tests/fixtures/novelty/semantic_hits_2026-11-04.json
grep -q "max_n1_confidence_reached" lms/novelty/__init__.py
grep -q "exact_probe/v3" lms/novelty/mathlib_search.py
grep -q "names_are_labels=names_are_labels" scripts/measure_n1_density.py
grep -q "n1_or_inconclusive_density" docs/runbooks/2026-10-08-box-restore.md
grep -q "n1_with_decisive_stages" data/novelty_control/README.md

# 2. Tests pass
uv run pytest tests/test_novelty.py tests/test_measure_n1_density.py tests/test_lean_real.py -v --tb=short

# 3. Code quality
uv run ruff check lms/novelty/ lms/gates/novelty.py lms/artifacts.py scripts/measure_n1_density.py
uv run mypy lms/novelty/ lms/gates/novelty.py lms/artifacts.py

echo "26Q4-HARN-09: verification passed"
