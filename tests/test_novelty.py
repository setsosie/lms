"""Tests for the N0/N1 novelty classifier (26Q3-HARN-04).

Everything here is offline: search behaviour is exercised through
`RecordedBackend`s and fixtures recorded from one live run
(`tests/fixtures/novelty/recorded_searches.json`). No test touches the
network or the Lean toolchain.
"""

import json
from pathlib import Path

import pytest

from lms.artifacts import Artifact, ArtifactType
from lms.gates.novelty import apply_novelty_gate
from lms.novelty import (
    DECISIVE_CONFIDENCE,
    NoveltyClassifier,
    NoveltyLevel,
    measure_density,
)
from lms.novelty.mathlib_search import (
    DiskCache,
    ExactProbeBackend,
    LoogleBackend,
    MathlibNameSearch,
    RateLimiter,
    RecordedBackend,
    SearchHit,
    StageOutcome,
    StatementQuery,
    extract_identifiers,
    name_tokens,
    parse_declaration,
)
from lms.novelty.vocabulary import named_results, project_vocabulary

FIXTURES = Path(__file__).parent / "fixtures" / "novelty"

THEOREM = "theorem Functor.comp_obj (F : Functor C D) (G : Functor D E) (x : C.Obj) :\n    (F.comp G).obj x = G.obj (F.obj x) := rfl"
NOVEL = "theorem sameDenom_eq_iff_exists_postcomp_W {X Y Y' : C} (f g : X ⟶ Y') (s : Y ⟶ Y') (hs : W s) : True := sorry"


def hit(name: str, module: str = "Mathlib.X", sig: str | None = None) -> SearchHit:
    return SearchHit(name=name, module=module, type_signature=sig)


def empty_backend(stage: str) -> RecordedBackend:
    return RecordedBackend(stage, {})


def four_empty_stages() -> list[RecordedBackend]:
    return [empty_backend(s) for s in ("name", "loogle", "exact_probe", "semantic")]


# ---------------------------------------------------------------- parsing


class TestParsing:
    def test_parse_declaration_theorem(self):
        assert parse_declaration(THEOREM) == ("theorem", "Functor.comp_obj")

    def test_parse_declaration_structure(self):
        kind, name = parse_declaration("structure Category where\n  Obj : Type")
        assert (kind, name) == ("structure", "Category")

    def test_parse_declaration_with_attribute(self):
        kind, name = parse_declaration("@[simp] theorem foo_bar : True := trivial")
        assert (kind, name) == ("theorem", "foo_bar")

    def test_parse_declaration_none_for_example(self):
        assert parse_declaration("example : True := trivial") == (None, None)

    def test_extract_identifiers_skips_keywords_and_binders(self):
        idents = extract_identifiers(THEOREM)
        assert "Functor.comp_obj" in idents or "Functor" in idents
        assert "theorem" not in idents
        assert "rfl" not in idents  # lowercase head

    def test_name_tokens_splits_camel_and_snake(self):
        assert name_tokens("sameDenom_eq_iff") == ["same", "denom", "eq", "iff"]


# ------------------------------------------------------------- rate limiter


class TestRateLimiter:
    def test_no_wait_under_limit(self):
        waits: list[float] = []
        clock = iter(range(100))
        rl = RateLimiter(3, 30.0, clock=lambda: float(next(clock)), sleep=waits.append)
        for _ in range(3):
            rl.acquire()
        assert waits == []

    def test_waits_when_window_full(self):
        waits: list[float] = []
        now = [0.0]
        rl = RateLimiter(2, 30.0, clock=lambda: now[0], sleep=waits.append)
        rl.acquire()
        rl.acquire()
        rl.acquire()
        assert len(waits) == 1
        assert waits[0] == pytest.approx(30.0)


# ------------------------------------------------------------------ cache


class TestDiskCache:
    def test_round_trip(self, tmp_path):
        cache = DiskCache(tmp_path)
        outcome = StageOutcome("loogle", available=True, hits=[hit("Nat.add_comm")])
        key = DiskCache.key("loogle", "q", "rev1")
        cache.put(key, outcome)
        loaded = cache.get(key)
        assert loaded is not None
        assert loaded.from_cache is True
        assert loaded.hits[0].name == "Nat.add_comm"

    def test_key_depends_on_mathlib_rev(self):
        assert DiskCache.key("s", "q", "rev1") != DiskCache.key("s", "q", "rev2")

    def test_classifier_uses_cache_instead_of_backend(self, tmp_path):
        backend = RecordedBackend(
            "loogle", {"Functor.comp_obj": StageOutcome("loogle", True)}
        )
        classifier = NoveltyClassifier(
            [backend], cache=DiskCache(tmp_path), mathlib_rev="r"
        )
        classifier.classify(THEOREM)
        classifier.classify(THEOREM)
        assert len(backend.calls) == 1

    def test_unavailable_outcomes_are_not_cached(self, tmp_path):
        backend = RecordedBackend("exact_probe", {}, available=False)
        classifier = NoveltyClassifier(
            [backend], cache=DiskCache(tmp_path), mathlib_rev="r"
        )
        classifier.classify(THEOREM)
        classifier.classify(THEOREM)
        assert len(backend.calls) == 2


# ------------------------------------------------------------- classifier


class TestClassifier:
    def test_exact_name_match_is_decisive_n0(self):
        name = RecordedBackend(
            "name",
            {
                "Functor.comp_obj": StageOutcome(
                    "name", True, hits=[hit("CategoryTheory.Functor.comp_obj")]
                )
            },
        )
        later = empty_backend("loogle")
        result = NoveltyClassifier([name, later]).classify(THEOREM)
        assert result.level is NoveltyLevel.N0
        assert result.confidence >= DECISIVE_CONFIDENCE
        assert result.decisive_stage == "name"
        assert any("comp_obj" in e for e in result.evidence)
        # Short-circuit: the later stage never ran.
        assert later.calls == []

    def test_exact_probe_close_is_decisive_n0(self):
        probe = RecordedBackend(
            "exact_probe",
            {
                "Functor.comp_obj": StageOutcome(
                    "exact_probe", True, closed_by="exact rfl"
                )
            },
        )
        result = NoveltyClassifier([probe]).classify(THEOREM)
        assert result.level is NoveltyLevel.N0
        assert result.decisive_stage == "exact_probe"
        assert "exact rfl" in result.evidence[0]

    def test_all_stages_empty_is_confident_n1(self):
        result = NoveltyClassifier(four_empty_stages()).classify(NOVEL)
        assert result.level is NoveltyLevel.N1
        assert result.confidence == pytest.approx(0.9)
        assert result.needs_review is False
        assert result.stages_available == ["name", "loogle", "exact_probe", "semantic"]

    def test_two_stages_empty_is_low_confidence_n1_needing_review(self):
        stages = [empty_backend("loogle"), empty_backend("semantic")]
        result = NoveltyClassifier(stages).classify(NOVEL)
        assert result.level is NoveltyLevel.N1
        assert result.confidence == pytest.approx(0.6)
        assert result.needs_review is True

    def test_one_stage_empty_is_inconclusive(self):
        result = NoveltyClassifier([empty_backend("semantic")]).classify(NOVEL)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.needs_review is True

    def test_no_stages_available_is_inconclusive(self):
        stages = [RecordedBackend("loogle", {}, available=False)]
        result = NoveltyClassifier(stages).classify(NOVEL)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.confidence == 0.0
        assert result.stages_unavailable == ["loogle"]

    def test_weak_semantic_hit_alone_is_inconclusive_not_n0(self):
        semantic = RecordedBackend(
            "semantic",
            {
                "Functor.comp_obj": StageOutcome(
                    "semantic",
                    True,
                    hits=[
                        hit(
                            "CategoryTheory.Functor.comp_obj",
                            sig="(F.comp G).obj x = G.obj (F.obj x)",
                        )
                    ],
                )
            },
        )
        others = [empty_backend("name"), empty_backend("loogle")]
        result = NoveltyClassifier([*others, semantic]).classify(THEOREM)
        # Same final name component via semantic search: plausible but not
        # decisive — must route to review, never auto-N0.
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.needs_review is True

    def test_mathlib_rev_recorded_on_result(self):
        result = NoveltyClassifier(four_empty_stages(), mathlib_rev="abc123").classify(
            NOVEL
        )
        assert result.mathlib_rev == "abc123"

    def test_to_dict_round_trip_fields(self):
        result = NoveltyClassifier(four_empty_stages()).classify(NOVEL)
        d = result.to_dict()
        assert d["level"] == "N1"
        assert d["needs_review"] is False
        assert set(d) >= {
            "level",
            "confidence",
            "evidence",
            "mathlib_rev",
            "stages_available",
        }


# ------------------------------------------------------------ exact probe


class TestExactProbe:
    def test_probe_source_rewrites_theorem(self):
        src = ExactProbeBackend.probe_source(
            "theorem foo (n : Nat) : n + 0 = n := by simp"
        )
        assert src == "import Mathlib\n\nexample (n : Nat) : n + 0 = n := by exact?\n"

    def test_probe_source_rejects_structure(self):
        assert (
            ExactProbeBackend.probe_source("structure Category where\n  Obj : Type")
            is None
        )

    def test_unavailable_without_mathlib_build(self, tmp_path):
        backend = ExactProbeBackend(project_dir=tmp_path)
        assert backend.is_available() is False


# Lean output recorded 2026-10-08 (toolchain v4.32.1) for three probes: one
# that `exact?` closes, one it cannot, and one whose statement names a type
# that does not exist — which error recovery reduced to a goal `rfl` closes.
CLOSED_TAGGED = "Try this:\n  [apply] exact IsIntegral.add hx hy\n"
MISSED = (
    "probe.lean:5:59: error: `exact?` could not close the goal. "
    "Try `apply?` to see partial suggestions.\n"
)
DID_NOT_ELABORATE = (
    "probe.lean:7:51: error(lean.unknownIdentifier): Unknown identifier `Foo.Bar`\n"
    "Try this:\n  [apply] exact ((fun a => a) ∘ fun a => a) rfl\n"
)


def unelaborated_probe() -> RecordedBackend:
    outcome = ExactProbeBackend.read_output(DID_NOT_ELABORATE)
    return RecordedBackend(
        "exact_probe", {"sameDenom_eq_iff_exists_postcomp_W": outcome}
    )


class TestExactProbeOutput:
    def test_inline_suggestion_closes(self):
        outcome = ExactProbeBackend.read_output("Try this: exact Nat.add_zero n\n")
        assert outcome.closed_by == "exact Nat.add_zero n"

    def test_apply_tagged_suggestion_closes(self):
        outcome = ExactProbeBackend.read_output(CLOSED_TAGGED)
        assert outcome.available is True
        assert outcome.closed_by == "exact IsIntegral.add hx hy"

    def test_a_miss_is_a_search_that_ran(self):
        outcome = ExactProbeBackend.read_output(MISSED)
        assert outcome.available is True
        assert outcome.closed_by is None

    def test_a_statement_that_did_not_elaborate_casts_no_vote(self):
        outcome = ExactProbeBackend.read_output(DID_NOT_ELABORATE)
        assert outcome.available is False
        assert outcome.closed_by is None
        assert (outcome.error or "").startswith("statement did not elaborate")

    def test_it_cannot_lift_n1_to_decisive(self):
        # Three name/semantic stages empty: N1 below the decisive line. Before
        # the probe stopped counting elaboration failures, it made the fourth
        # empty stage and the verdict 0.9, decisive.
        stages = [
            empty_backend("name"),
            empty_backend("loogle"),
            unelaborated_probe(),
            empty_backend("semantic"),
        ]
        result = NoveltyClassifier(stages).classify(NOVEL)
        assert result.level is NoveltyLevel.N1
        assert result.confidence < DECISIVE_CONFIDENCE
        assert result.needs_review is True
        assert result.stages_unavailable == ["exact_probe"]

    def test_the_elaboration_error_reaches_the_result(self):
        # The box run's elaboration check reads these to repair signatures.
        result = NoveltyClassifier([unelaborated_probe()]).classify(NOVEL)
        assert "Unknown identifier `Foo.Bar`" in result.stage_errors["exact_probe"]
        assert result.to_dict()["stage_errors"] == result.stage_errors

    def test_it_cannot_manufacture_an_n0(self):
        result = NoveltyClassifier([unelaborated_probe()]).classify(NOVEL)
        assert result.level is not NoveltyLevel.N0


NAMELESS = "-- a payload with no named declaration\nexample : True := trivial"


class TestExplicitUniverses:
    def test_the_name_stops_before_the_universe_list(self):
        assert parse_declaration("theorem foo.{u} (x : Nat) : x = x := rfl") == (
            "theorem",
            "foo",
        )
        assert parse_declaration("structure Category.{u, v} where") == (
            "structure",
            "Category",
        )


class TestStagesWithNoQuery:
    """A stage that could not form a query did not search, so it cannot vote."""

    def test_name_search_without_a_name(self, tmp_path):
        query = StatementQuery.from_lean(NAMELESS)
        assert MathlibNameSearch(tmp_path).search(query).available is False

    def test_loogle_without_a_name(self):
        query = StatementQuery.from_lean(NAMELESS)
        assert LoogleBackend().search(query).available is False

    def test_exact_probe_on_a_definition(self, tmp_path):
        query = StatementQuery.from_lean("def foo : Nat := 0")
        assert ExactProbeBackend(tmp_path).search(query).available is False

    def test_a_nameless_definition_is_not_confident_n1(self, tmp_path):
        # 11 of the 52 Gate A control payloads look like this. With the three
        # query-less stages voting "absent", it read N1 at 0.9, decisive.
        stages = [
            MathlibNameSearch(tmp_path),
            LoogleBackend(),
            ExactProbeBackend(tmp_path),
            empty_backend("semantic"),
        ]
        result = NoveltyClassifier(stages).classify(NAMELESS)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.stages_available == ["semantic"]


# ------------------------------------------------------------------ gate


def make_artifact(lean_code: str | None) -> Artifact:
    return Artifact(
        id="a1",
        type=ArtifactType.THEOREM,
        natural_language="composition acts on objects",
        created_by="agent-1",
        generation=1,
        lean_code=lean_code,
    )


class TestNoveltyGate:
    def test_decisive_n1_counts_as_novel(self):
        decision = apply_novelty_gate(
            make_artifact(NOVEL), NoveltyClassifier(four_empty_stages())
        )
        assert decision.counts_as_novel is True
        assert decision.needs_human_review is False

    def test_n0_fails_gate(self):
        name = RecordedBackend(
            "name",
            {
                "Functor.comp_obj": StageOutcome(
                    "name", True, hits=[hit("CategoryTheory.Functor.comp_obj")]
                )
            },
        )
        artifact = make_artifact(THEOREM)
        decision = apply_novelty_gate(artifact, NoveltyClassifier([name]))
        assert decision.counts_as_novel is False
        assert artifact.novelty_level == "N0"
        assert artifact.novelty_evidence

    def test_low_confidence_n1_routes_to_review_not_novel(self):
        stages = [empty_backend("loogle"), empty_backend("semantic")]
        decision = apply_novelty_gate(make_artifact(NOVEL), NoveltyClassifier(stages))
        assert decision.counts_as_novel is False
        assert decision.needs_human_review is True

    def test_missing_lean_code_is_inconclusive(self):
        artifact = make_artifact(None)
        decision = apply_novelty_gate(artifact, NoveltyClassifier(four_empty_stages()))
        assert decision.counts_as_novel is False
        assert artifact.novelty_level == "INCONCLUSIVE"

    def test_artifact_novelty_fields_serialize(self):
        artifact = make_artifact(NOVEL)
        apply_novelty_gate(artifact, NoveltyClassifier(four_empty_stages()))
        d = artifact.to_dict()
        assert d["novelty_level"] == "N1"
        assert d["novelty_confidence"] == pytest.approx(0.9)
        loaded = Artifact.from_dict(d)
        assert loaded.novelty_level == "N1"
        assert loaded.novelty_evidence == artifact.novelty_evidence

    def test_legacy_artifact_without_novelty_fields_loads(self):
        d = make_artifact(NOVEL).to_dict()
        for k in ("novelty_level", "novelty_confidence", "novelty_evidence"):
            d.pop(k)
        loaded = Artifact.from_dict(d)
        assert loaded.novelty_level is None
        assert loaded.novelty_evidence == []


# ------------------------------------- statements no Mathlib search can match

YONEDA = json.loads((FIXTURES / "yoneda_bespoke_api.json").read_text())
YONEDA_VOCABULARY = ["Category", "Functor", "TypeCat", "NatTrans", "homFunctor"]


def yoneda_project(tmp_path: Path) -> Path:
    """A Lean project holding the run's foundation, where the gate looks for it."""
    module = tmp_path / "LMS" / "Foundation.lean"
    module.parent.mkdir(parents=True)
    module.write_text(YONEDA["foundation_source"])
    return tmp_path


def yoneda_stages(semantic_hits: bool) -> list[RecordedBackend]:
    """The four stages as the card recorded them: every one ran."""
    stages = [empty_backend(s) for s in ("name", "loogle", "exact_probe")]
    hits = (
        [SearchHit.from_dict(h) for h in YONEDA["semantic_hits"]]
        if semantic_hits
        else []
    )
    stages.append(
        RecordedBackend(
            "semantic", {"yoneda_lemma": StageOutcome("semantic", True, hits=hits)}
        )
    )
    return stages


SEED = (Path(__file__).parent.parent / "lms" / "seed" / "category.lean").read_text()


def seed_project(tmp_path: Path) -> Path:
    """A Lean project whose foundation is the shipped seed: a class plus notation."""
    module = tmp_path / "LMS" / "Foundation.lean"
    module.parent.mkdir(parents=True)
    module.write_text(SEED)
    return tmp_path


class TestProjectVocabulary:
    def test_vocabulary_arriving_as_notation(self, tmp_path):
        code = (
            "import LMS.Foundation\nopen LMS.Foundation\n\n"
            "theorem type_comp_apply {X Y Z : Type u} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) :\n"
            "    (f ≫ g) x = g (f x) := rfl"
        )
        found = project_vocabulary(code, seed_project(tmp_path))
        assert found
        assert "⟶" in found.names

    def test_vocabulary_arriving_through_a_variable(self, tmp_path):
        code = (
            "import LMS.Foundation\nopen LMS.Foundation\n\n"
            "variable {C : Type u} [Category.{v} C]\n\n"
            "theorem id_comp' {X Y : C} (f : X ⟶ Y) : 𝟙 X ≫ f = f := by simp"
        )
        found = project_vocabulary(code, seed_project(tmp_path))
        assert found.names == ["Category", "⟶", "𝟙", "≫"]

    def test_a_declaration_with_explicit_universes(self, tmp_path):
        module = tmp_path / "LMS" / "Foundation.lean"
        module.parent.mkdir(parents=True)
        module.write_text("structure Category.{u, v} where\n  Obj : Type u\n")
        code = "import LMS.Foundation\n\ntheorem t (C : Category.{0, 0}) (X : C.Obj) : True := trivial"
        assert project_vocabulary(code, tmp_path).names == ["Category"]

    def test_an_auto_param_does_not_end_the_header(self, tmp_path):
        code = (
            "import LMS.Foundation\n\n"
            "theorem t (n : Nat) (h : 0 < n := by decide) (C : Category) : True := trivial"
        )
        assert project_vocabulary(code, yoneda_project(tmp_path)).names == ["Category"]

    def test_a_statement_in_foundation_vocabulary(self, tmp_path):
        found = project_vocabulary(YONEDA["lean_code"], yoneda_project(tmp_path))
        assert found.names == YONEDA_VOCABULARY
        assert found.modules == ["LMS.Foundation"]
        assert found.unread == []

    def test_a_statement_over_mathlib_alone(self, tmp_path):
        code = "import Mathlib\n\ntheorem t {A : Type*} [CommRing A] (x : A) : x + 0 = x := by simp"
        assert not project_vocabulary(code, tmp_path)

    def test_a_project_import_the_statement_does_not_use(self, tmp_path):
        code = "import LMS.Foundation\n\ntheorem t (n : Nat) : n + 0 = n := rfl"
        assert not project_vocabulary(code, yoneda_project(tmp_path))

    def test_an_unreadable_project_module_is_not_assumed_safe(self, tmp_path):
        found = project_vocabulary(YONEDA["lean_code"], tmp_path)
        assert found
        assert found.unread == ["LMS.Foundation"]

    def test_no_project_dir_reads_nothing(self):
        assert project_vocabulary(YONEDA["lean_code"], None).unread == [
            "LMS.Foundation"
        ]


class TestYonedaRegression:
    """The card's case: a textbook theorem over a bespoke API is not novel."""

    def test_the_recorded_outcomes_no_longer_read_as_decisive_n1(self, tmp_path):
        assert YONEDA["recorded_verdict"]["level"] == "N1"
        classifier = NoveltyClassifier(
            yoneda_stages(semantic_hits=False), project_dir=yoneda_project(tmp_path)
        )
        result = classifier.classify(YONEDA["lean_code"], informal=YONEDA["informal"])
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.needs_review is True
        assert result.outside_mathlib == YONEDA_VOCABULARY
        assert "LMS.Foundation" in result.evidence[0]

    def test_the_gate_does_not_count_it(self, tmp_path):
        artifact = make_artifact(YONEDA["lean_code"])
        artifact.natural_language = YONEDA["informal"]
        classifier = NoveltyClassifier(
            yoneda_stages(semantic_hits=False), project_dir=yoneda_project(tmp_path)
        )
        decision = apply_novelty_gate(artifact, classifier)
        assert decision.counts_as_novel is False
        assert decision.needs_human_review is True
        assert artifact.novelty_level == "INCONCLUSIVE"

    def test_the_reviewer_sees_mathlibs_yoneda_first(self, tmp_path):
        classifier = NoveltyClassifier(
            yoneda_stages(semantic_hits=True), project_dir=yoneda_project(tmp_path)
        )
        result = classifier.classify(YONEDA["lean_code"], informal=YONEDA["informal"])
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.evidence[0].startswith("semantic: CategoryTheory.yonedaLemma")

    def test_a_novel_statement_over_mathlib_is_unaffected(self, tmp_path):
        classifier = NoveltyClassifier(
            four_empty_stages(), project_dir=yoneda_project(tmp_path)
        )
        result = classifier.classify(NOVEL)
        assert result.level is NoveltyLevel.N1
        assert result.needs_review is False
        assert result.outside_mathlib == []


class TestInformalNamedResults:
    @pytest.mark.parametrize(
        ("informal", "expected"),
        [
            (YONEDA["informal"], [{"yoneda"}]),
            ("Nakayama's lemma for finitely generated modules", [{"nakayama"}]),
            (
                "The Kummer–Dedekind theorem on splitting primes",
                [{"kummer", "dedekind"}],
            ),
            ("Dirichlet's unit theorem", [{"dirichlet"}]),
            ("The integral closure of a Dedekind domain is Dedekind", []),
            ("The main theorem of this section", []),
            (None, []),
            ("Nakayama’s lemma", [{"nakayama"}]),
            ("Gauss' lemma on primitive polynomials", [{"gauss"}]),
            ("The Sylow theorems", [{"sylow"}]),
            ("Lagrange's four-square theorem", [{"lagrange"}]),
            ("The Cauchy—Schwarz inequality", [{"cauchy", "schwarz"}]),
            # A reference or a capitalised common word is not an eponym.
            ("By Lemma 3, the map is injective", []),
            ("Prime number theorem", []),
            ("Central Limit Theorem", []),
            ("First isomorphism theorem", []),
        ],
    )
    def test_named_results(self, informal, expected):
        assert named_results(informal) == [frozenset(e) for e in expected]

    @staticmethod
    def classify(informal: str | None, hit_name: str):
        semantic = RecordedBackend(
            "semantic",
            {"lemma_17": StageOutcome("semantic", True, hits=[hit(hit_name)])},
        )
        others = [empty_backend(s) for s in ("name", "loogle", "exact_probe")]
        return NoveltyClassifier([*others, semantic]).classify(
            "theorem lemma_17 (C : Cat) : True := trivial", informal=informal
        )

    def test_a_hit_carrying_the_eponym_is_plausible_never_decisive(self):
        result = self.classify(
            "Yoneda lemma, stated for Cat", "CategoryTheory.yonedaEquiv"
        )
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert "CategoryTheory.yonedaEquiv" in result.evidence[0]

    def test_without_the_informal_the_same_hit_is_noise(self):
        result = self.classify(None, "CategoryTheory.yonedaEquiv")
        assert result.level is NoveltyLevel.N1

    def test_a_hit_without_the_eponym_changes_nothing(self):
        result = self.classify("Nakayama's lemma", "CategoryTheory.yonedaEquiv")
        assert result.level is NoveltyLevel.N1

    def test_an_eponym_the_statement_uses_as_a_concept_is_not_distinctive(self):
        """Every ANT hit says Dedekind; ram-17's eponym cannot single one out."""
        semantic = RecordedBackend(
            "semantic",
            {
                "ram_17": StageOutcome(
                    "semantic", True, hits=[hit("IsDedekindDomain.HeightOneSpectrum")]
                )
            },
        )
        others = [empty_backend(s) for s in ("name", "loogle", "exact_probe")]
        result = NoveltyClassifier([*others, semantic]).classify(
            "theorem ram_17 {A : Type*} [CommRing A] [IsDedekindDomain A] : True := trivial",
            informal="Dedekind's theorem on the different",
        )
        assert result.level is NoveltyLevel.N1


# ------------------------------------------------------- density measurement


def arc_doc() -> dict:
    return {
        "arc": "test",
        "source": "unit test",
        "statements": [
            {"id": "s1", "name": "Functor.comp_obj", "lean_statement": THEOREM},
            {"id": "s2", "name": "sameDenom", "lean_statement": NOVEL},
        ],
    }


class TestMeasureDensity:
    def test_density_counts_and_review_queue(self):
        name = RecordedBackend(
            "name",
            {
                "Functor.comp_obj": StageOutcome(
                    "name", True, hits=[hit("CategoryTheory.Functor.comp_obj")]
                )
            },
        )
        stages = [
            name,
            empty_backend("loogle"),
            empty_backend("exact_probe"),
            empty_backend("semantic"),
        ]
        report = measure_density(arc_doc(), NoveltyClassifier(stages, mathlib_rev="r1"))
        assert report["total_statements"] == 2
        assert report["counts"] == {"N0": 1, "N1": 1, "INCONCLUSIVE": 0}
        assert report["n1_density"] == pytest.approx(0.5)
        assert report["n1_density_decisive"] == pytest.approx(0.5)
        assert report["needs_review"] == []
        assert report["mathlib_rev"] == "r1"

    def test_confidence_distribution_sums_to_total(self):
        report = measure_density(arc_doc(), NoveltyClassifier(four_empty_stages()))
        assert (
            sum(report["confidence_distribution"].values())
            == report["total_statements"]
        )

    def test_report_states_the_n1_ceiling_of_its_ladder(self):
        stages = [empty_backend("exact_probe"), empty_backend("semantic")]
        report = measure_density(arc_doc(), NoveltyClassifier(stages))
        assert report["stages_run"] == ["exact_probe", "semantic"]
        assert report["max_n1_confidence"] < DECISIVE_CONFIDENCE
        # Two empty stages read N1, but none of it is decisive.
        assert report["counts"]["N1"] == 2
        assert report["n1_density_decisive"] == 0.0
        assert report["needs_review"] == ["s1", "s2"]

    def test_four_stages_can_reach_decisive(self):
        report = measure_density(arc_doc(), NoveltyClassifier(four_empty_stages()))
        assert report["max_n1_confidence"] >= DECISIVE_CONFIDENCE


# ------------------------------------------------- recorded live fixtures


@pytest.mark.skipif(
    not (FIXTURES / "recorded_searches.json").exists(),
    reason="live fixtures not recorded",
)
class TestRecordedFixtures:
    """Replays of one real loogle + leansearch run (2026-08-19).

    Pins the classifier's behaviour on the card's validation set without
    touching the network. Recorded against the Mathlib rev in the fixture.
    """

    @pytest.fixture(scope="class")
    def recorded(self) -> dict:
        return json.loads((FIXTURES / "recorded_searches.json").read_text())

    def make_classifier(self, recorded: dict) -> NoveltyClassifier:
        backends = []
        for stage in ("loogle", "semantic"):
            outcomes = {
                key: StageOutcome.from_dict(o)
                for key, o in recorded["stages"].get(stage, {}).items()
            }
            backends.append(RecordedBackend(stage, outcomes))
        return NoveltyClassifier(backends, mathlib_rev=recorded["mathlib_rev"])

    def test_expected_labels(self, recorded):
        classifier = self.make_classifier(recorded)
        mismatches = []
        for case in recorded["validation"]:
            result = classifier.classify(
                case["lean_statement"], informal=case.get("informal")
            )
            if result.level.value not in case["acceptable_levels"]:
                mismatches.append(
                    (case["name"], result.level.value, case["acceptable_levels"])
                )
        assert mismatches == []

    def test_no_known_n1_is_ever_called_n0(self, recorded):
        """The one unacceptable error: novel work classified as re-derivation."""
        classifier = self.make_classifier(recorded)
        for case in recorded["validation"]:
            if case["expected"] != "N1":
                continue
            result = classifier.classify(
                case["lean_statement"], informal=case.get("informal")
            )
            assert result.level is not NoveltyLevel.N0, case["name"]
