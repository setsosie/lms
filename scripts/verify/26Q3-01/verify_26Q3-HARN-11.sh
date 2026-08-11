#!/bin/bash
# Verification for 26Q3-HARN-11: Expose the full API of foundation entries to agents
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-11.sh
#
# Behaviour assertions live in tests/test_foundation_api_exposure.py, not here.
# This script checks structure and delegates behaviour to pytest.
set -e

cd "$(dirname "$0")/../../.."

# 1. The renderers read from lean_code, not the lossy signature.
uv run python -c "
from lms.foundation import FoundationEntry
assert hasattr(FoundationEntry, 'declaration_header'), 'AC-1/AC-2: no declaration_header()'
assert hasattr(FoundationEntry, 'field_lines'), 'AC-3/AC-4: no field_lines()'
"

# 2. The duplicated header is gone (signature already carries type + name).
! grep -q '{entry.entry_type} {entry.name}{entry.signature}' lms/foundation.py

# 3. The silent field cap is gone.
! grep -q 'field_match\[:5\]' lms/foundation.py

# 4. The committee summary no longer treats a list as a dict, nor reads .tag.
! grep -q 'self.foundation.entries.values()' lms/society.py
! grep -q 'entries\[:10\]' lms/society.py

# 5. AC-1..AC-6.
uv run pytest tests/test_foundation_api_exposure.py -q

# 6. No regressions in the surrounding suites.
uv run pytest tests/test_foundation.py tests/test_foundation_namespace.py \
    tests/test_foundation_persistence.py tests/test_society.py -q

# 7. Touched files are lint-clean (repo-wide debt is pre-existing).
uv run ruff format --check lms/foundation.py lms/society.py tests/test_foundation_api_exposure.py
uv run ruff check lms/foundation.py lms/society.py tests/test_foundation_api_exposure.py

echo "26Q3-HARN-11: verification passed"
