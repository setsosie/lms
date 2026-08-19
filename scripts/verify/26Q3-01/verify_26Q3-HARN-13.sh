#!/bin/bash
# Verification for 26Q3-HARN-13: verify an artifact in the namespace it will
# be stored in.
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-13.sh
# Structure checks only; behaviour assertions live in
# tests/test_verify_namespace.py.
set -e

# 1. One shared namespace constant exists in foundation.py (AC-4)
grep -q '^FOUNDATION_NAMESPACE = "LMS.Foundation"' lms/foundation.py

# 2. The verifier consumes the constant and carries no second literal (AC-4)
grep -q 'FOUNDATION_NAMESPACE' lms/lean/real.py
! grep -q '"LMS.Foundation"' lms/lean/real.py

# 3. Header and footer derive from the constant, not from literals (AC-4)
grep -q 'namespace {FOUNDATION_NAMESPACE}' lms/foundation.py
grep -q 'end {FOUNDATION_NAMESPACE}' lms/foundation.py

# 4. The import split is shared, not duplicated (AC-2)
grep -q '^def split_imports' lms/foundation.py
grep -q 'split_imports' lms/lean/real.py

# 5. The verifier wraps before writing the temp file (AC-1)
grep -q '_wrap_in_storage_namespace' lms/lean/real.py

# 6. Behaviour tests exist and pass (AC-1..AC-4, AC-6 spot checks)
uv run pytest tests/test_verify_namespace.py -q

echo "26Q3-HARN-13: verification passed"
