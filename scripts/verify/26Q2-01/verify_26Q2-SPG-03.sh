#!/bin/bash
set -e

# 1. Files exist
test -f lms/genome.py
test -f tests/test_genome.py

# 2. Dependency added
grep -q "numpy" pyproject.toml

# 3. Required content present
grep -q "class SoftPrefixGenome" lms/genome.py
grep -q "def random_khot" lms/genome.py
grep -q "def to_embedding" lms/genome.py
grep -q "def mutate" lms/genome.py
grep -q "def to_dict" lms/genome.py
grep -q "def from_dict" lms/genome.py

# 4. Import works
uv run python -c "from lms.genome import SoftPrefixGenome"

# 5. Tests pass
uv run pytest tests/test_genome.py -v --tb=short

# 6. Code quality
uv run ruff check lms/genome.py
uv run mypy lms/genome.py

echo "26Q2-SPG-03: verification passed"
