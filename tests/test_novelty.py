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
    RateLimiter,
    RecordedBackend,
    SearchHit,
    StageOutcome,
    extract_identifiers,
    name_tokens,
    parse_declaration,
)

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
