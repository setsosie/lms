#!/bin/bash
# Verification for 26Q4-HARN-06: Coordinator–worker mode — one persistent coordinator, fresh workers (the W9 arm)
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-06.sh
set -e

# 1. Files exist
test -f lms/coordinator.py
test -f tests/test_coordinator_mode.py

# 2. Required content present
grep -q -- "--coordinator" lms/run.py
grep -q "coordinator_system" lms/prompts.py
grep -q "coordinator.json" lms/coordinator.py
grep -q "COORDINATOR" scripts/slurm/lms_run.sbatch

# 3. Import and CLI smoke
uv run python -c "import lms.coordinator"
uv run python -m lms.run --help | grep -q -- "--coordinator"

# 4. Tests pass
uv run pytest tests/test_coordinator_mode.py tests/test_planning.py -v --tb=short

# 5. Code quality
uv run ruff check lms/coordinator.py lms/planning.py lms/society.py lms/run.py tests/test_coordinator_mode.py
uv run mypy lms/coordinator.py lms/planning.py lms/society.py lms/run.py

echo "26Q4-HARN-06: verification passed"
