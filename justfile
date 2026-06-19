# Canonical IndigiGenius research-repo verbs.
# /repo-doctor checks that {test, lint, format, check} exist.
# Project-specific verbs (train, eval, container, ...) live below the line.

set shell := ["bash", "-uc"]

# Show available recipes.
default:
    @just --list

# ─── Canonical (required by /repo-doctor) ───────────────────────────────────

# Run the fast test suite.
test *ARGS:
    uv run pytest {{ARGS}}

# Lint with ruff. Use `just lint --fix` to autofix.
lint *ARGS:
    uv run ruff check {{ARGS}} .

# Format with ruff.
format *ARGS:
    uv run ruff format {{ARGS}} .

# Pre-PR gate: lint + format-check + test.
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

# Install all deps (incl. dev group).
install:
    uv sync --all-groups

# Drop caches.
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info

# ─── Project-specific (customize freely) ────────────────────────────────────

# Launch a training run.  CONFIG is a YAML path under configs/.
train CONFIG:
    uv run python -m lms.train --config {{CONFIG}}

# Run evaluation on a checkpoint.
eval CHECKPOINT:
    uv run python -m lms.eval --checkpoint {{CHECKPOINT}}

# Run all tests including slow / integration markers.
test-all:
    uv run pytest -m ""
