"""Generation-0 axiom layers (26Q3-HARN-23).

The single most load-bearing decision in a run is how `Category` is
represented, and until now it was made by whichever agent happened to submit
first. In `committee_fix_c` that landed on a parameterized `structure`, and
every generation from 5 onward died writing `[C : Category]` against it:
`invalid binder annotation, type is not a class`.

A seed makes that decision once, by a human, in a file that is reviewed and
compiled like any other source. Agents start from it instead of inventing it.

The shipped default is deliberately shaped like Mathlib's `CategoryTheory`
API -- a `class`, with `⟶`, `𝟙` and `≫`. Every Qwen failure in that run was the
model reaching for exactly this idiom; matching it turns the model's strongest
prior from a liability into an asset.
"""

from pathlib import Path

__all__ = ["DEFAULT_SEED", "available_seeds", "load_seed"]

SEED_DIR = Path(__file__).parent

#: Seed used when a run does not name one.
DEFAULT_SEED = "category"


def available_seeds() -> list[str]:
    """Names of the seeds shipped in this package."""
    return sorted(p.stem for p in SEED_DIR.glob("*.lean"))


def load_seed(name: str = DEFAULT_SEED) -> str:
    """Read a seed's Lean source.

    Raises:
        FileNotFoundError: If no seed by that name is shipped. Silently
            returning an empty seed would start the run on an empty foundation
            and look identical to a run that meant to have none.
    """
    path = SEED_DIR / f"{name}.lean"
    if not path.exists():
        raise FileNotFoundError(
            f"No seed named {name!r}. Available: {', '.join(available_seeds()) or 'none'}"
        )
    return path.read_text()
