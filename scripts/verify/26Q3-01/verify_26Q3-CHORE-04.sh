#!/bin/bash
# Verification for 26Q3-CHORE-04: restore code dropped by the #53/#56 squashes
# Run: bash scripts/verify/26Q3-01/verify_26Q3-CHORE-04.sh
#
# This script checks that the work landed. It is NOT a test suite.
# Behavior is proven in tests/ (CI runs it every commit).
set -e

# 1. society.py imports every gate name it uses (dropped by #53's squash)
grep -q "^from lms.gates import GateOutcome, default_gate_runner$" lms/society.py
grep -q "^from lms.gates.lean_source import named_declarations$" lms/society.py
grep -q "^from lms.gates.novelty import apply_novelty_gate, default_novelty_classifier$" lms/society.py

# 2. foundation.py defines the seed installers again (dropped by #56's squash)
grep -q "^def strip_header_universes" lms/foundation.py
grep -q "^    def set_seed" lms/foundation.py

# 3. No undefined names anywhere: the defect class this card exists for.
#    F821 is what a linter would have caught had it run on main.
uv run ruff check lms/society.py lms/foundation.py

# 4. The modules actually import, and a seeded foundation actually seeds
uv run python -c "
import tempfile, pathlib
from lms.foundation import FoundationFile
from lms.seed import load_seed
import lms.society  # noqa: F401 - the import is the assertion
with tempfile.TemporaryDirectory() as d:
    f = FoundationFile(pathlib.Path(d) / 'Foundation.lean')
    assert f.set_seed(load_seed()), 'seed installed no names'
"

# 5. Behavior proven in pytest
uv run pytest tests/test_seed.py -q

echo "26Q3-CHORE-04: verification passed"
