#!/bin/bash
# Verification for 26Q3-HARN-17: committee_yolo_a post-mortem harness fixes
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-17.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. References are derived from code, not scribe self-report
grep -q "_derive_references" lms/society.py

# 2. Import violations enter the repair loop (no hard-fail branch left)
grep -q "_import_violation" lms/society.py

# 3. Foundation submodule imports are rewritten to the umbrella
grep -q "_rewrite_foundation_imports" lms/agent.py

# 4. Phase 1 declares an import policy
grep -q "ALLOWED_IMPORTS_PHASE_1" lms/goals.py

# 5. Planning tops idle groups back up
grep -q "_top_up" lms/planning.py

# 6. Behavior proven in pytest
uv run pytest tests/test_society.py -k "MechanicalReferenceDerivation or InitialImportViolationRepair" -q
uv run pytest tests/test_goals.py -k "DotAware or Phase1ImportPolicy" -q
uv run pytest tests/test_planning.py -k "TopUp" -q
uv run pytest tests/test_lean_extraction.py -k "FoundationImportRewrite" -q

echo "26Q3-HARN-17: verification passed"
