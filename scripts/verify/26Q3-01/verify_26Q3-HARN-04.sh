#!/bin/bash
# Verification for 26Q3-HARN-04: Novelty classifier (N0 / N1)
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-04.sh
#
# Behaviour assertions live in tests/test_novelty.py, not here.
# This script checks structure and delegates behaviour to pytest.
set -e

# Resolve the repo root from git, not from $0 — /pre-merge's discrimination
# check replays this script at the merge base from a different checkout.
cd "$(git rev-parse --show-toplevel)"

# 1. The classifier package exists with the carded API.
grep -q "def classify_novelty" lms/novelty/__init__.py
grep -q "class NoveltyResult" lms/novelty/__init__.py
grep -q "class NoveltyClassifier" lms/novelty/__init__.py

# 2. All four search stages exist, and unavailability is a first-class outcome
#    (a stage that cannot run must say so, not silently pass).
grep -q "class MathlibNameSearch" lms/novelty/mathlib_search.py
grep -q "class LoogleBackend" lms/novelty/mathlib_search.py
grep -q "class ExactProbeBackend" lms/novelty/mathlib_search.py
grep -q "class LeanSearchBackend" lms/novelty/mathlib_search.py
grep -q "def is_available" lms/novelty/mathlib_search.py

# 3. Results are cached on disk and keyed by the Mathlib revision.
grep -q "class DiskCache" lms/novelty/mathlib_search.py
grep -q "mathlib_rev" lms/novelty/mathlib_search.py

# 4. The public services' rate limits are declared, not improvised.
grep -q "LOOGLE_RATE = (3, 30" lms/novelty/mathlib_search.py
grep -q "LEANSEARCH_RATE = (90, 30" lms/novelty/mathlib_search.py

# 5. Gate 4 wrapper exists and artifacts carry the auditable record.
grep -q "def apply_novelty_gate" lms/gates/novelty.py
grep -q "novelty_level" lms/artifacts.py
grep -q "novelty_evidence" lms/artifacts.py

# 6. Slice-selection mode exists and parses (the §4 decision tool).
test -f scripts/measure_n1_density.py
uv run python scripts/measure_n1_density.py --help > /dev/null

# 7. AC behaviour: short-circuit order, cache, confidence routing, gate,
#    density report, recorded-fixture replay of the validation set.
uv run pytest tests/test_novelty.py -q

# 8. No regressions where this task touched shared code.
uv run pytest tests/test_artifacts.py -q

# 9. Touched files are lint-clean (repo-wide debt is pre-existing).
uv run ruff format --check lms/novelty/ lms/gates/ scripts/measure_n1_density.py tests/test_novelty.py
uv run ruff check lms/novelty/ lms/gates/ scripts/measure_n1_density.py tests/test_novelty.py

echo "26Q3-HARN-04: verification passed"
