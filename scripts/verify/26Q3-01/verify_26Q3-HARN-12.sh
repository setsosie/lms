#!/bin/bash
# Verification for 26Q3-HARN-12: committee architecture reachable + review stage
# Run: bash scripts/verify/26Q3-01/verify_26Q3-HARN-12.sh
# Wiring checks only — behaviour assertions live in tests/test_society.py
# (TestCommitteeMode, TestIterativeReuse).
set -e

# 1. Committee mode has a CLI path
grep -q -- '"--groups"' lms/run.py
grep -q -- '"--n-groups"' lms/run.py

# 2. The flag is read, not just declared: run_generation dispatches on it
grep -A26 'async def run_generation(' lms/society.py | grep -q 'use_working_groups'

# 3. The silent no-goal fallback is gone, replaced by a loud error
! grep -q 'Fallback to regular generation if no goal' lms/society.py
grep -q 'Committee mode requires a goal' lms/society.py

# 4. The review committee stage exists between the groups and the verifier
grep -q 'REVIEW COMMITTEE' lms/society.py

# 5. Iterative mode links references (reuse rate measurable)
awk '/_run_generation_iterative/,/run_generation_with_groups/' lms/society.py | grep -q 'add_reference'

# 6. The behaviour tests for all of the above exist
grep -q 'class TestCommitteeMode' tests/test_society.py
grep -q 'class TestIterativeReuse' tests/test_society.py

echo "26Q3-HARN-12: verification passed"
