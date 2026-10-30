#!/bin/bash
# Verification for 26Q4-HARN-07: Swarm mode — Prove2Me-style organization on the local DAG (the P9 arm)
# Run: bash scripts/verify/26Q4-01/verify_26Q4-HARN-07.sh
set -e

# 1. Files exist
test -f lms/swarm.py
test -f tests/test_swarm_mode.py

# 2. Required content present
grep -q -- "--swarm" lms/run.py
grep -q -- "--read-k" lms/run.py
grep -q -- "--claim" lms/run.py
grep -q "swarm.json" lms/swarm.py
grep -q "swarm_agent_user" lms/prompts.py
grep -q "SWARM" scripts/slurm/lms_run.sbatch

# 3. Import and CLI smoke
uv run python -c "import lms.swarm"
uv run python -m lms.run --help | grep -q -- "--swarm"

# 4. Tests pass
uv run pytest tests/test_swarm_mode.py -v --tb=short

# 5. Code quality
uv run ruff check lms/swarm.py lms/society.py lms/run.py lms/prompts.py tests/test_swarm_mode.py
uv run mypy lms/swarm.py lms/society.py lms/run.py

echo "26Q4-HARN-07: verification passed"
