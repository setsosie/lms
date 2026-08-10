# Verify scripts

One per task card: `scripts/verify/<SPRINT>/verify_<TASK_ID>.sh`.

A verify script answers one question — *did this card actually land?* — in a
form a reviewer can run in one command and read in one screen.

## What belongs in a verify script

- **Existence and wiring**: `test -f`, `git diff --quiet`, and one-line import
  checks (`uv run python -c 'from lms.society import Society; Society.reset_foundation'`).
  These catch a rename or a deleted module immediately and cheaply.
- **Invocations of the tests that prove the behaviour**, grouped so each line
  names a claim the card makes.
- **Hygiene**: full suite, `ruff`, `mypy`, and any repo-state guard the card
  depends on.

## What does not

**Behavioural assertions written inline as `uv run python -c "..."`.** The
script already runs `pytest`, so an inline assertion is untested code checking
tested code. It has no fixtures, produces no useful failure output, cannot be
run or debugged on its own, and drifts from the implementation silently because
nothing else exercises it.

If a claim is worth verifying, it is worth a test. Write the test, then have the
script run it:

```bash
# no
check "reset discards the previous run's definitions" \
  "uv run python -c \"
from lms.society import Society
...twenty lines of setup...
\""

# yes
check "runs start independent of each other" \
  "uv run pytest tests/test_foundation_persistence.py -q -k 'reset'"
```

A `-k` filter that matches nothing exits 5, so a stale filter fails the script
rather than passing silently.

## Discrimination

Every script carries a note near the top stating what fails at the merge base,
and that claim gets checked before the PR goes up — stash the source changes,
run the script, count the failures. A verify script that passes on `main` is
testing nothing. Hygiene checks (full suite, lint) are expected to pass at the
base; the card-specific ones are not.
