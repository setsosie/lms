"""A verified artifact closes its tag only when it declares that tag's statement.

The match is on words: the tag name's words (kind word and section title
dropped) against the words of the names the artifact declares.
"""

import pytest

from lms.society import tag_name_matches

FIBRED_GROUPOIDS = (
    "Definition: definition fibred groupoids (Categories fibred in groupoids)"
)

CASES = [
    # Batch q1, r6: both closures were made by the wrong statement.
    (
        "Definition: Functor",
        "structure CategoryDef (obj : Type u) where\n  hom : obj → obj → Type v",
        False,
    ),
    (
        "Definition: Natural Transformation",
        "structure Functor (C D : Type) where\n  obj : C → D",
        False,
    ),
    # The same tags, closed by their own statements.
    (
        "Definition: Functor",
        "structure Functor (C D : Type) where\n  obj : C → D",
        True,
    ),
    (
        "Definition: Natural Transformation",
        "structure NatTrans (F G : Functor C D) where\n  app : ∀ X, F.obj X ⟶ G.obj X",
        True,
    ),
    # Kernel tracks name tags by Stacks label, with a section title in parentheses.
    (
        "Definition: definition categories over C (Categories over categories)",
        "structure CatOverC (C : Type u) where\n  S : Type v",
        True,
    ),
    (
        "Lemma: lemma composition cartesian (Fibred categories)",
        "theorem comp_cartesian {X Y Z : S} : True := trivial",
        True,
    ),
    # Batch q4's 003T artifact names its statement.
    (
        FIBRED_GROUPOIDS,
        "structure CategoryFibredInGroupoids (C : Type u) [Category C] where\n  S : Type v",
        True,
    ),
    # The section title is not part of the statement's name: "Categories" there must not match.
    (
        FIBRED_GROUPOIDS,
        "structure CategoryOver (C : Type u) where\n  S : Type v",
        False,
    ),
]


@pytest.mark.parametrize(("tag_name", "lean_code", "closes"), CASES)
def test_rule_a_name_match(tag_name: str, lean_code: str, closes: bool) -> None:
    assert tag_name_matches(tag_name, lean_code) is closes


def test_short_words_never_match() -> None:
    # `is` is a prefix of `isomorphism`, but a match needs at least 3 letters.
    code = "def IsZero (x : Nat) : Prop := x = 0"
    assert tag_name_matches("Definition: Isomorphism", code) is False


def test_milestone_prefix_is_dropped() -> None:
    code = "theorem yoneda_embedding_faithful : True := trivial"
    assert tag_name_matches("MILESTONE: Yoneda lemma", code) is True
