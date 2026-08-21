"""The generation-0 axiom layer (26Q3-HARN-23).

`Category`'s representation is the most load-bearing decision in a run, and it
was made by whichever agent submitted first. In `committee_fix_c` that landed
on a parameterized `structure`, and generations 5-9 died writing
`[C : Category]` against it: `invalid binder annotation, type is not a class`.

The shipped seed is a `class` with `⟶`/`𝟙`/`≫`, so the idiom the model reaches
for is the correct one.
"""

import pytest

from lms.artifacts import Artifact, ArtifactType
from lms.config import ProviderConfig
from lms.foundation import FOUNDATION_UNIVERSES, FoundationFile, strip_header_universes
from lms.goals import get_goal
from lms.lean.interface import VerificationStatus
from lms.lean.mock import MockLeanVerifier
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.seed import DEFAULT_SEED, available_seeds, load_seed
from lms.society import Society


class SilentProvider(BaseLLMProvider):
    name = "mock"

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> GenerationResponse:
        return GenerationResponse(
            content="No artifacts proposed.",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider=self.name,
        )


def make_society(tmp_path, **kw):
    return Society(
        n_agents=1,
        provider=SilentProvider(ProviderConfig(api_key="k", model="m")),
        verifier=MockLeanVerifier(),
        foundation_path=tmp_path / "Foundation.lean",
        **kw,
    )


class TestSeedLoading:
    def test_default_seed_ships(self):
        assert DEFAULT_SEED in available_seeds()

    def test_unknown_seed_raises_rather_than_returning_empty(self):
        """An empty seed and a missing seed must not look the same: one is a
        deliberate bootstrapping run, the other is a typo."""
        with pytest.raises(FileNotFoundError):
            load_seed("no-such-seed")

    def test_default_seed_is_a_class_with_notation(self):
        source = load_seed()
        assert "class Category" in source
        assert "⟶" in source and "𝟙" in source and "≫" in source

    def test_default_seed_carries_worked_models(self):
        """Lesson 7 of the simulation: a definition shipped with a concrete
        instance is one whose axioms are demonstrably satisfiable."""
        source = load_seed()
        assert "instance typeCategory" in source
        assert "instance punitCategory" in source


class TestStripHeaderUniverses:
    def test_header_names_are_dropped(self):
        assert strip_header_universes(["universe u v"]) == []

    def test_non_header_names_are_kept(self):
        assert strip_header_universes(["universe x"]) == ["universe x"]

    def test_mixed_line_keeps_only_the_exotic_names(self):
        assert strip_header_universes(["universe u x v"]) == ["universe x"]

    def test_other_lines_pass_through(self):
        assert strip_header_universes(["def f := 1"]) == ["def f := 1"]

    def test_every_header_universe_is_stripped(self):
        line = f"universe {' '.join(FOUNDATION_UNIVERSES)}"
        assert strip_header_universes([line]) == []


class TestSetSeed:
    def test_seed_claims_its_declaration_names(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        assert "Category" in f.set_seed(load_seed())

    def test_agent_cannot_redefine_a_seed_name(self, tmp_path):
        """A redefinition must contribute nothing rather than shadow the layer
        everything else is typed against."""
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        art = Artifact(
            id="a-1",
            type=ArtifactType.DEFINITION,
            natural_language="my own Category",
            created_by="agent-0",
            generation=1,
            lean_code="structure Category (C : Type u) where\n  Hom : C → C → Type v",
            status=VerificationStatus.VERIFIED_LEAN,
        )
        assert f.add_artifact(art) is False
        assert f.entries == []

    def test_seed_universes_are_normalised_away(self, tmp_path):
        """The seed sits below the header's `universe u v w`; repeating those
        names is a duplicate-declaration error in the merged file."""
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        assert not any(
            line.strip().startswith("universe ") for line in f.seed_source.split("\n")
        )

    def test_seed_imports_are_stripped(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed("import Foo\n\ndef x : Nat := 1")
        assert "import Foo" not in f.seed_source


class TestSeedInFile:
    def test_seed_written_verbatim_including_notation(self, tmp_path):
        """Notation lines are not declarations, so routing the seed through
        `_extract_entries` would silently drop `⟶`, `𝟙` and `≫`."""
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        f.save()
        text = (tmp_path / "F.lean").read_text()
        assert "class Category" in text
        assert "⟶" in text and "𝟙" in text and "≫" in text

    def test_seed_precedes_agent_entries(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        f.add_artifact(
            Artifact(
                id="a-1",
                type=ArtifactType.DEFINITION,
                natural_language="later",
                created_by="agent-0",
                generation=1,
                lean_code="def later : Nat := 1",
                status=VerificationStatus.VERIFIED_LEAN,
            )
        )
        f.save()
        text = (tmp_path / "F.lean").read_text()
        assert text.index("class Category") < text.index("def later")

    def test_seed_survives_save_load_round_trip(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        f.save()
        reloaded = FoundationFile.load(tmp_path / "F.lean")
        assert "class Category" in reloaded.seed_source
        assert "Category" in reloaded._definition_names


class TestSeedInAgentContext:
    def test_seeded_foundation_is_not_reported_empty(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        assert "FOUNDATION: EMPTY" not in f.get_context_for_agent()

    def test_context_shows_the_seed_verbatim(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        context = f.get_context_for_agent()
        assert "GENERATION-0 SEED" in context
        assert "class Category" in context
        assert "≫" in context

    def test_context_no_longer_calls_the_seed_idiom_unknown(self, tmp_path):
        """The old warning told agents `𝟙` and class-style `Category C` were
        unknown identifiers. With a seed that is false, and a false warning is
        worse than none."""
        f = FoundationFile(tmp_path / "F.lean")
        f.set_seed(load_seed())
        context = f.get_context_for_agent()
        assert (
            "class-style `Category C`, `.Hom`, `𝟙`, `.obj` — is an unknown"
            not in context
        )

    def test_bare_foundation_still_reports_empty(self, tmp_path):
        f = FoundationFile(tmp_path / "F.lean")
        assert "FOUNDATION: EMPTY" in f.get_context_for_agent()


class TestGoalSeeding:
    def test_ch4_phase1_declares_a_seed(self):
        goal = get_goal("stacks-ch4-phase1")
        assert goal.seed == "category"
        assert goal.seeded_tags == ["0013"]

    def test_mark_seeded_marks_only_seeded_tags(self):
        goal = get_goal("stacks-ch4-phase1")
        marked = goal.mark_seeded()
        assert marked == ["0013"]
        by_tag = {d.tag: d for d in goal.definitions}
        assert by_tag["0013"].formalized
        assert not by_tag["0014"].formalized

    def test_mark_seeded_is_idempotent(self):
        goal = get_goal("stacks-ch4-phase1")
        goal.mark_seeded()
        assert goal.mark_seeded() == []

    def test_seeded_tag_records_its_provenance(self):
        goal = get_goal("stacks-ch4-phase1")
        goal.mark_seeded()
        by_tag = {d.tag: d for d in goal.definitions}
        assert "seed" in by_tag["0013"].artifact_ids


class TestSocietySeedSelection:
    def test_explicit_seed_source_wins(self, tmp_path):
        society = make_society(tmp_path, seed_source="def custom : Nat := 1")
        assert society.seed_source == "def custom : Nat := 1"

    def test_empty_string_means_bare_foundation(self, tmp_path):
        """A bootstrapping experiment must still be expressible."""
        society = make_society(tmp_path, seed_source="")
        assert society.seed_source == ""

    def test_goal_seed_is_honoured(self, tmp_path):
        society = make_society(tmp_path, goal=get_goal("stacks-ch4-phase1"))
        assert "class Category" in society.seed_source

    def test_default_when_nothing_specified(self, tmp_path):
        assert "class Category" in make_society(tmp_path).seed_source

    async def test_reset_installs_the_seed(self, tmp_path):
        society = make_society(tmp_path, goal=get_goal("stacks-ch4-phase1"))
        await society.reset_foundation()
        assert "Category" in society.foundation._definition_names

    async def test_reset_marks_seeded_goal_tags(self, tmp_path):
        goal = get_goal("stacks-ch4-phase1")
        society = make_society(tmp_path, goal=goal)
        await society.reset_foundation()
        by_tag = {d.tag: d for d in goal.definitions}
        assert by_tag["0013"].formalized

    async def test_bare_seed_leaves_foundation_empty(self, tmp_path):
        society = make_society(tmp_path, seed_source="")
        await society.reset_foundation()
        assert society.foundation.seed_source == ""
        assert society.foundation._definition_names == set()
