"""A run's verification budget must fit the imports the goal permits (26Q3-HARN-24).

`RealLeanVerifier`'s default timeout was 30s and `run.py` never overrode it.
Measured on a quiet machine with warm oleans, an otherwise empty file costs:

    import LMS.Foundation        2.0s
    import Mathlib.Tactic.Common 3.8s
    import Mathlib.Tactic       31.8s   <-- over the whole budget

`Mathlib.Tactic` is on `ALLOWED_IMPORTS_FOUNDATION`, labelled "All tactics".
An agent that followed the list it was handed timed out before its own code was
read, and the failure was recorded against the agent.
"""

import inspect

from lms.goals import ALLOWED_IMPORTS_FOUNDATION, get_goal
from lms.lean.real import RealLeanVerifier


class TestRunTimeoutBudget:
    def test_run_timeout_exceeds_the_costliest_allowed_import(self):
        """31.8s measured for `Mathlib.Tactic`; the budget must clear it with
        room, since the H100 box runs busier than the machine measured on."""
        assert RealLeanVerifier.RUN_TIMEOUT_S >= 60.0

    def test_run_py_passes_an_explicit_timeout(self):
        """The defect was not the default's value but that `run.py` accepted
        it silently. An explicit argument makes the budget a visible choice."""
        import lms.run

        source = inspect.getsource(lms.run)
        constructions = source.count("RealLeanVerifier(")
        assert constructions > 0
        assert source.count("RealLeanVerifier.RUN_TIMEOUT_S") == constructions

    def test_costly_import_is_still_permitted(self):
        """The fix is to afford the import, not to forbid it: agents reach for
        `Mathlib.Tactic` because it is what they know."""
        assert "Mathlib.Tactic" in ALLOWED_IMPORTS_FOUNDATION

    def test_phase_1_goal_permits_the_costly_import(self):
        """Guards the pairing: if this list ever drops `Mathlib.Tactic`, the
        budget rationale above should be revisited rather than left stale."""
        goal = get_goal("stacks-ch4-phase1")
        assert goal.allowed_imports is not None
        assert "Mathlib.Tactic" in goal.allowed_imports

    def test_constructor_default_is_unchanged(self):
        """Library callers that verify import-free snippets keep the cheap
        default; only a *run* opts into the larger budget."""
        signature = inspect.signature(RealLeanVerifier.__init__)
        assert signature.parameters["timeout"].default == 30.0
