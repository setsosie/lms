"""Tests for the N0/N1 novelty classifier (26Q3-HARN-04).

Everything here is offline: search behaviour is exercised through
`RecordedBackend`s and fixtures recorded from one live run
(`tests/fixtures/novelty/recorded_searches.json`). No test touches the
network or the Lean toolchain.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from lms.accounting import counts_as_novel
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
    LeanSearchBackend,
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

ROOT = Path(__file__).resolve().parent.parent
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
        assert src == (
            "import Mathlib\n\n"
            "theorem lms_novelty_probe (n : Nat) : n + 0 = n := by exact?\n"
        )

    def test_probe_stops_before_a_where_body(self):
        src = ExactProbeBackend.probe_source(
            "theorem foo : A ≃ B where\n  toFun := f\n  invFun := g\n"
        )
        assert src.endswith("theorem lms_novelty_probe : A ≃ B := by exact?\n")

    def test_probe_keeps_a_default_valued_binder(self):
        src = ExactProbeBackend.probe_source(
            "theorem foo (n : Nat) (h : 0 < n := by decide) : n ≠ 0 := sorry"
        )
        assert "(h : 0 < n := by decide) : n ≠ 0 := by exact?" in src

    def test_probe_carries_an_explicit_universe_list(self):
        src = ExactProbeBackend.probe_source(
            "theorem foo.{u} (X : Type u) : X = X := rfl"
        )
        assert "theorem lms_novelty_probe.{u} (X : Type u) : X = X :=" in src

    def test_probe_keeps_every_scope_command_ahead_of_the_declaration(self):
        statement = (
            "section Closed\n"
            "variable (p : Prop)\n"
            "end Closed\n"
            "noncomputable section\n"
            "variable (q : Prop) (h : q) (k : q)\n"
            "include h\n"
            "omit k\n"
            "theorem foo : q := h\n"
        )
        lines = ExactProbeBackend.probe_source(statement).splitlines()
        kept = lines[
            1 : next(i for i, s in enumerate(lines) if s.startswith("theorem"))
        ]
        assert [s for s in kept if s] == statement.splitlines()[:7]

    def test_probe_refuses_a_header_with_no_type(self):
        assert ExactProbeBackend.probe_source("theorem foo := trivial") is None

    def test_probe_does_not_stop_at_a_word_containing_where(self):
        src = ExactProbeBackend.probe_source(
            "theorem foo : elsewhere x ∧ where_ok y := sorry"
        )
        assert ": elsewhere x ∧ where_ok y := by exact?" in src

    def test_a_comment_or_blank_line_does_not_end_a_preamble_command(self):
        src = ExactProbeBackend.probe_source(
            "variable {R : Type} -- ring\n  -- module\n  [Inhabited R]\n\n"
            "  variable (r : R)\n\ntheorem foo : r = r := rfl\n"
        )
        assert "[Inhabited R]" in src
        assert "variable (r : R)" in src

    @pytest.mark.parametrize(
        ("name", "probe"),
        [
            ("Foo.bar", "Foo.lms_novelty_probe"),
            ("_root_.Foo.bar", "_root_.Foo.lms_novelty_probe"),
        ],
    )
    def test_probe_keeps_the_namespace_in_a_dotted_name(self, name, probe):
        src = ExactProbeBackend.probe_source(f"theorem {name} : baz := sorry")
        assert f"theorem {probe} : baz := by exact?" in src

    def test_probe_refuses_a_def(self):
        assert ExactProbeBackend.probe_source("def foo : Nat := 0") is None

    def test_a_command_name_later_in_a_line_is_not_a_preamble(self):
        src = ExactProbeBackend.probe_source(
            'notation "open" => 1\n\ntheorem foo : True := trivial\n'
        )
        assert "notation" not in src

    def test_probe_source_rejects_structure(self):
        assert (
            ExactProbeBackend.probe_source("structure Category where\n  Obj : Type")
            is None
        )

    def test_unavailable_without_mathlib_build(self, tmp_path):
        backend = ExactProbeBackend(project_dir=tmp_path)
        assert backend.is_available() is False

    def test_probe_keeps_the_preamble_the_statement_needs(self):
        lines = ExactProbeBackend.probe_source(ARTIFACT).splitlines()
        preamble = lines[
            : next(i for i, s in enumerate(lines) if s.startswith("theorem"))
        ]
        for kept in (
            "namespace Stacks",
            "open CategoryTheory",
            "universe v₁ u₁",
            "variable {C : Type u₁} [Category.{v₁} C]",
            "  {D : Type u₁} [Category.{v₁} D]",
        ):
            assert kept in preamble

    def test_probe_imports_the_whole_library_and_nothing_narrower(self):
        lines = ExactProbeBackend.probe_source(ARTIFACT).splitlines()
        assert [s for s in lines if s.startswith("import")] == ["import Mathlib"]

    def test_probe_does_not_mistake_comment_text_for_a_preamble(self):
        src = ExactProbeBackend.probe_source(ARTIFACT)
        assert "open questions" not in src
        assert "Functors preserve" not in src

    def test_probe_leaves_out_what_follows_the_declaration(self):
        assert "open Later" not in ExactProbeBackend.probe_source(ARTIFACT)

    def test_probes_built_without_a_preamble_are_not_replayed_from_cache(self):
        query = StatementQuery.from_lean(THEOREM)
        assert ExactProbeBackend().cache_query(query).startswith("exact_probe/v3:")


# A control artifact as agents write it: its names resolve only under its preamble.
ARTIFACT = """import Mathlib.CategoryTheory.Functor.Basic

/-!
open questions about this file go here
-/

namespace Stacks

open CategoryTheory

universe v₁ u₁

variable {C : Type u₁} [Category.{v₁} C]
  {D : Type u₁} [Category.{v₁} D]

/-- Functors preserve isomorphisms. -/
theorem functor_preserves_iso (F : C ⥤ D) {X Y : C} (f : X ⟶ Y) [IsIso f] :
    IsIso (F.map f) :=
  inferInstance

open Later
"""

KERNEL_TIMEOUT = "probe.lean:3:0: error: (kernel) deterministic timeout\n"

# Lean output recorded 2026-11-05 (toolchain v4.32.1) for a control statement written
# as a `lemma` of data type: `exact?` answers, then Lean rejects the declaration.
NOT_A_PROPOSITION = (
    "Try this:\n  [apply] exact hI.uniqueUpToIso hJ\n"
    "probe.lean:8:0: error: type of theorem `Stacks.lms_novelty_probe` "
    "is not a proposition\n"
)

# Lean output recorded 2026-11-04 (toolchain v4.27.0-rc1): `exact?` itself ran out
# of heartbeats, at the column of the tactic, on a statement that elaborated.
HEARTBEAT_LIMIT = (
    "probe.lean:3:105: error: (deterministic) timeout at `whnf`, "
    "maximum number of heartbeats (200000) has been reached\n"
)

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

    def test_a_heartbeat_timeout_is_not_called_an_elaboration_failure(self):
        outcome = ExactProbeBackend.read_output(HEARTBEAT_LIMIT)
        assert outcome.available is False
        assert outcome.closed_by is None
        assert "heartbeat" in (outcome.error or "")
        assert not (outcome.error or "").startswith("statement did not elaborate")

    def test_a_close_on_a_data_goal_does_not_count(self):
        outcome = ExactProbeBackend.read_output(NOT_A_PROPOSITION)
        assert outcome.available is False
        assert outcome.closed_by is None
        assert (outcome.error or "").startswith("statement did not elaborate")

    @pytest.mark.parametrize(
        "output",
        ["", "INTERNAL PANIC: out of memory\n", "uncaught exception: boom\n"],
        ids=["killed", "out-of-memory", "other"],
    )
    def test_no_answer_from_lean_is_not_a_miss(self, output):
        outcome = ExactProbeBackend.read_output(output)
        assert outcome.available is False
        assert outcome.closed_by is None
        assert (outcome.error or "").startswith("probe gave no answer")

    def test_a_kernel_timeout_is_a_heartbeat_limit_too(self):
        outcome = ExactProbeBackend.read_output(KERNEL_TIMEOUT)
        assert outcome.available is False
        assert "heartbeat" in (outcome.error or "")

    def test_a_real_error_outranks_a_heartbeat_limit_after_it(self):
        # Once the statement has failed, a later timeout is on a recovered goal.
        outcome = ExactProbeBackend.read_output(DID_NOT_ELABORATE + HEARTBEAT_LIMIT)
        assert outcome.available is False
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


class TestSeveralDeclarations:
    """A close on the first of several declarations says nothing about the rest."""

    FILE = (
        "open Nat\n\nlemma aux (n : Nat) : succ n = n + 1 := rfl\n\n"
        "theorem novel_main (p : Nat → Prop) : p 0 := sorry\n"
    )

    @staticmethod
    def stages(lean: str, closed_by: str = "exact Nat.foo") -> list[RecordedBackend]:
        key = StatementQuery.from_lean(lean).display_name
        probe = RecordedBackend(
            "exact_probe",
            {key: StageOutcome("exact_probe", True, closed_by=closed_by)},
        )
        return [empty_backend("name"), empty_backend("loogle"), probe]

    def classify(self, lean: str):
        return NoveltyClassifier(
            [*self.stages(lean), empty_backend("semantic")]
        ).classify(lean)

    def test_it_reads_inconclusive_and_names_the_helper(self):
        result = self.classify(self.FILE)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.needs_review is True
        assert any(
            "`aux`" in e and "first of 2 declarations" in e for e in result.evidence
        )

    def test_the_helper_it_names_is_not_one_from_a_comment(self):
        lean = "/-- The\ntheorem of Dedekind. -/\n" + self.FILE
        result = self.classify(lean)
        assert any("`aux`" in e for e in result.evidence)
        assert not any("`of`" in e for e in result.evidence)

    def test_the_ladder_goes_on_to_semantic_search(self):
        result = self.classify(self.FILE)
        assert "semantic" in result.stages_available

    def test_a_single_declaration_still_reads_n0(self):
        lean = "open Nat\n\ntheorem foo (n : Nat) : succ n = n + 1 := rfl\n"
        result = self.classify(lean)
        assert result.level is NoveltyLevel.N0
        assert result.confidence >= DECISIVE_CONFIDENCE

    def test_a_declaration_named_in_a_comment_does_not_count(self):
        lean = (
            "/-- We extend the\ntheorem of Dedekind to this setting. -/\n"
            "theorem foo (n : Nat) : n = n := rfl\n"
            "-- theorem bar : True := trivial\n"
        )
        assert self.classify(lean).level is NoveltyLevel.N0

    @pytest.mark.parametrize(
        "later",
        [
            "theorem π_x : True := trivial",
            "theorem «two words» : True := trivial",
            "nonrec theorem x : True := trivial",
            "private lemma x : True := trivial",
            "@[simp] theorem x : True := trivial",
            "noncomputable def x : Nat := 0",
            "instance : Inhabited Foo := ⟨0⟩",
            "example : True := trivial",
            "  theorem indented : True := trivial",
            "structure Foo where\n  x : Nat",
            "open Nat in theorem x : True := trivial",
            "set_option maxHeartbeats 0 in theorem x : True := trivial",
            "@[aesop safe (rule_sets := [X])] theorem x : True := trivial",
            "local instance : Inhabited Foo := ⟨0⟩",
            "scoped instance : Inhabited Foo := ⟨0⟩",
            "public theorem x : True := trivial",
            "alias x := aux",
            "irreducible_def x : Nat := 0",
            "axiom x : True",
            "opaque x : Nat",
            "theorem x : True := trivial theorem y : True := trivial",
        ],
    )
    def test_any_later_declaration_counts(self, later):
        lean = f"lemma aux (n : Nat) : n = n := rfl\n\n{later}\n"
        assert self.classify(lean).level is NoveltyLevel.INCONCLUSIVE

    def test_a_proof_body_is_not_a_declaration(self):
        lean = (
            "theorem foo (n : Nat) : n = n := by\n  have h : n = n := rfl\n  exact h\n"
        )
        assert self.classify(lean).level is NoveltyLevel.N0

    def test_a_word_containing_a_keyword_is_not_a_declaration(self):
        lean = (
            "theorem foo (my_def class_size : Nat) (h : Foo.class my_def) : "
            "my_def = class_size := sorry\n"
        )
        assert self.classify(lean).level is NoveltyLevel.N0

    def test_it_cannot_fail_a_novel_artifact_at_the_gate(self):
        artifact = make_artifact(self.FILE)
        classifier = NoveltyClassifier(
            [*self.stages(self.FILE), empty_backend("semantic")]
        )
        decision = apply_novelty_gate(artifact, classifier)
        assert artifact.novelty_level == "INCONCLUSIVE"
        assert decision.counts_as_novel is False
        assert decision.needs_human_review is True


class TestLoogleAnswers:
    """An error from Loogle is a search that did not run, not an empty one."""

    @staticmethod
    def search(monkeypatch, answer) -> StageOutcome:
        backend = LoogleBackend()
        monkeypatch.setattr(backend, "_get_json", lambda *_: answer)
        return backend.search(StatementQuery.from_lean(NOVEL))

    # Both answer shapes recorded from the live service on 2026-11-05.
    def test_an_error_answer_is_not_an_empty_search(self, monkeypatch):
        answer = {"error": "<input>:1:20: unexpected end of input", "heartbeats": 0}
        outcome = self.search(monkeypatch, answer)
        assert outcome.available is False
        assert "unexpected end of input" in (outcome.error or "")

    def test_an_empty_answer_is_an_empty_search(self, monkeypatch):
        answer = {"count": 0, "header": "Found 0", "heartbeats": 0, "hits": []}
        outcome = self.search(monkeypatch, answer)
        assert outcome.available is True
        assert outcome.hits == []

    @pytest.mark.parametrize(
        "answer",
        [
            {},
            [],
            "busy",
            {"count": 0},
            {"hits": None},
            {"hits": [1]},
            {"hits": [{"x": 1}]},
        ],
    )
    def test_an_answer_with_no_hits_field_is_not_an_empty_search(
        self, monkeypatch, answer
    ):
        outcome = self.search(monkeypatch, answer)
        assert outcome.available is False
        assert outcome.error == "unexpected response shape"


class TestCachedFailures:
    """Failures used to be cached as searches that ran, so they must not replay."""

    @staticmethod
    def classify(tmp_path, stage: str, error: str):
        cache = DiskCache(tmp_path)
        query = StatementQuery.from_lean(NOVEL)
        backend = RecordedBackend(stage, {})
        key = DiskCache.key(stage, backend.cache_query(query), None)
        cache.put(key, StageOutcome(stage, True, error=error))
        NoveltyClassifier([backend], cache=cache).classify(NOVEL)
        return backend.calls

    def test_a_cached_search_failure_runs_again(self, tmp_path):
        assert self.classify(tmp_path, "loogle", "boom")

    def test_a_cached_probe_miss_still_replays(self, tmp_path):
        # A miss records Lean's message as its error, and it is a real answer.
        assert not self.classify(tmp_path, "exact_probe", "could not close the goal")


class TestNameSearchFailure:
    def test_a_grep_failure_is_not_an_empty_search(self, tmp_path, monkeypatch):
        (tmp_path / ".lake/packages/mathlib/Mathlib").mkdir(parents=True)
        failed = subprocess.CompletedProcess([], 2, stdout="", stderr="grep: boom")
        monkeypatch.setattr(subprocess, "run", lambda *_, **__: failed)
        outcome = MathlibNameSearch(tmp_path).search(StatementQuery.from_lean(NOVEL))
        assert outcome.available is False
        assert outcome.error == "grep: boom"

    def test_no_matches_is_an_empty_search(self, tmp_path, monkeypatch):
        (tmp_path / ".lake/packages/mathlib/Mathlib").mkdir(parents=True)
        none = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        monkeypatch.setattr(subprocess, "run", lambda *_, **__: none)
        outcome = MathlibNameSearch(tmp_path).search(StatementQuery.from_lean(NOVEL))
        assert outcome.available is True
        assert outcome.hits == []


class TestLeanSearchAnswers:
    """An answer nobody can read is a search that did not run, not an empty one."""

    @staticmethod
    def search(monkeypatch, answer) -> StageOutcome:
        backend = LeanSearchBackend()
        monkeypatch.setattr(backend, "_post_json", lambda *_: answer)
        return backend.search(StatementQuery.from_lean(NOVEL))

    @pytest.mark.parametrize(
        "answer",
        [
            {"error": "busy"},
            [],
            [None],
            "busy",
            [[{"result": {}}]],
            [[{"other": 1}]],
            [""],
        ],
        ids=["dict", "empty", "null", "text", "nameless-result", "no-result", "blank"],
    )
    def test_a_malformed_answer_is_not_an_empty_search(self, monkeypatch, answer):
        outcome = self.search(monkeypatch, answer)
        assert outcome.available is False
        assert outcome.error == "unexpected response shape"

    def test_a_genuinely_empty_answer_is_an_empty_search(self, monkeypatch):
        outcome = self.search(monkeypatch, [[]])
        assert outcome.available is True
        assert outcome.hits == []

    def test_one_nameless_item_does_not_spoil_the_answer(self, monkeypatch):
        named = {"result": {"name": ["Lib", "thing"]}}
        outcome = self.search(monkeypatch, [[named, {"result": {}}]])
        assert [h.name for h in outcome.hits] == ["Lib.thing"]

    def test_a_malformed_answer_cannot_lift_n1_to_decisive(self, monkeypatch):
        backend = LeanSearchBackend()
        monkeypatch.setattr(backend, "_post_json", lambda *_: [None])
        stages = [empty_backend(s) for s in ("name", "loogle", "exact_probe")]
        result = NoveltyClassifier([*stages, backend]).classify(NOVEL)
        assert result.level is NoveltyLevel.N1
        assert result.needs_review is True


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

    def test_report_states_the_ceiling_its_statements_actually_reached(self):
        # Four stages configured, but the probe never ran, so decisive N1 was
        # impossible on this arc however empty the searches came back.
        stages = [
            empty_backend("name"),
            empty_backend("loogle"),
            RecordedBackend("exact_probe", {}, available=False),
            empty_backend("semantic"),
        ]
        report = measure_density(arc_doc(), NoveltyClassifier(stages))
        assert report["max_n1_confidence"] == pytest.approx(0.9)
        assert report["max_n1_confidence_reached"] == pytest.approx(0.75)
        assert report["statements_by_stages_run"] == {"3": 2}
        assert report["counts"]["N1"] == 2
        assert report["n1_with_decisive_stages"] == 0

    def test_stage_counts_include_statements_that_stopped_early(self):
        name = RecordedBackend(
            "name",
            {
                "Functor.comp_obj": StageOutcome(
                    "name", True, hits=[hit("CategoryTheory.Functor.comp_obj")]
                )
            },
        )
        stages = [name, *four_empty_stages()[1:]]
        report = measure_density(arc_doc(), NoveltyClassifier(stages))
        assert report["max_n1_confidence_reached"] == pytest.approx(0.9)
        assert report["statements_by_stages_run"] == {"1": 1, "4": 1}

    def test_report_counts_the_statements_the_decisive_check_could_read(self):
        # Three ran all four stages. One found nothing, one has a near hit that
        # holds it back from decisive, and one matched, so it never could be.
        def statement(name: str) -> dict:
            lean = f"theorem {name} : Ideal.ramificationIdx I P = 1 := sorry"
            return {"id": name, "name": name, "lean_statement": lean}

        semantic = RecordedBackend(
            "semantic",
            {
                "near": StageOutcome(
                    "semantic", True, hits=[hit("Ideal.ramificationIdx.other")]
                ),
                "matched": StageOutcome(
                    "semantic", True, hits=[hit("Ideal.ramificationIdx")]
                ),
            },
        )
        stages = [empty_backend(s) for s in ("name", "loogle", "exact_probe")]
        doc = {
            "arc": "t",
            "statements": [statement(n) for n in ("absent", "near", "matched")],
        }
        report = measure_density(doc, NoveltyClassifier([*stages, semantic]))
        assert report["statements_by_stages_run"] == {"4": 3}
        assert report["counts"] == {"N0": 0, "N1": 2, "INCONCLUSIVE": 1}
        assert report["n1_with_decisive_stages"] == 2
        assert report["n1_density_decisive"] == pytest.approx(1 / 3)

    def test_report_states_density_with_inconclusive_included(self):
        semantic = RecordedBackend(
            "semantic",
            {
                "Functor.comp_obj": StageOutcome(
                    "semantic",
                    True,
                    hits=[hit("Functor.comp_obj", sig="(F : Functor)")],
                )
            },
        )
        stages = [empty_backend(s) for s in ("name", "loogle", "exact_probe")]
        report = measure_density(arc_doc(), NoveltyClassifier([*stages, semantic]))
        assert report["counts"] == {"N0": 0, "N1": 1, "INCONCLUSIVE": 1}
        assert report["n1_density"] == pytest.approx(0.5)
        assert report["n1_or_inconclusive_density"] == pytest.approx(1.0)


class TestLabelNames:
    """A label says nothing about the mathematics, so it takes no part in a match."""

    STATEMENT = (
        "theorem {label} (I : Ideal R) (P : Ideal S) : "
        "Ideal.ramificationIdx I P = 1 := sorry"
    )

    def classify(self, label: str, **kwargs):
        match = hit("Ideal.ramificationIdx", sig="(I : Ideal R) (P : Ideal S)")
        semantic = RecordedBackend(
            "semantic", {label: StageOutcome("semantic", True, hits=[match])}
        )
        stages = [empty_backend("exact_probe"), semantic]
        statement = self.STATEMENT.format(label=label)
        return NoveltyClassifier(stages, **kwargs).classify(statement)

    def test_a_long_label_does_not_hide_a_match(self):
        label = "ant_c17_some_long_label_words_here"
        result = self.classify(label, names_are_labels=True)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert any("ramificationIdx" in e for e in result.evidence)

    def test_the_verdict_does_not_depend_on_the_label(self):
        short = self.classify("ant_c1", names_are_labels=True)
        long = self.classify(
            "ant_c17_some_long_label_words_here", names_are_labels=True
        )
        assert (short.level, short.confidence) == (long.level, long.confidence)

    def test_a_label_cannot_create_a_match(self):
        # The hit's tokens appear only in the label, under its full name.
        name = "Ideal.ramificationIdx_label"
        statement = f"theorem {name} : True := sorry"
        match = hit("Ideal.ramificationIdx")

        def classify(**kwargs):
            semantic = RecordedBackend(
                "semantic", {name: StageOutcome("semantic", True, hits=[match])}
            )
            stages = [empty_backend("exact_probe"), semantic]
            return NoveltyClassifier(stages, **kwargs).classify(statement)

        assert classify().level is NoveltyLevel.INCONCLUSIVE
        assert classify(names_are_labels=True).level is NoveltyLevel.N1

    def test_a_label_equal_to_a_hit_name_is_not_a_match(self):
        statement = "theorem mk : True := sorry"
        match = hit("IsZLattice.mk")

        def classify(**kwargs):
            semantic = RecordedBackend(
                "semantic", {"mk": StageOutcome("semantic", True, hits=[match])}
            )
            stages = [empty_backend("exact_probe"), semantic]
            return NoveltyClassifier(stages, **kwargs).classify(statement)

        assert classify().level is NoveltyLevel.INCONCLUSIVE
        assert classify(names_are_labels=True).level is NoveltyLevel.N1

    def test_real_names_still_count_by_default(self):
        name = "ramificationIdx_eq_one"
        match = hit("Ideal.ramificationIdx")
        statement = f"theorem {name} (I : Ideal R) : True := sorry"

        def classify(**kwargs):
            semantic = RecordedBackend(
                "semantic", {name: StageOutcome("semantic", True, hits=[match])}
            )
            stages = [empty_backend("exact_probe"), semantic]
            return NoveltyClassifier(stages, **kwargs).classify(statement)

        assert classify().level is NoveltyLevel.INCONCLUSIVE
        assert classify(names_are_labels=True).level is NoveltyLevel.N1


class TestSemanticMatch:
    """A LeanSearch match is a candidate for review: never N0, and not "not found"."""

    @staticmethod
    def classify(statement: str, name: str, match, informal: str | None = None):
        semantic = RecordedBackend(
            "semantic", {name: StageOutcome("semantic", True, hits=[match])}
        )
        stages = [empty_backend("name"), empty_backend("loogle"), semantic]
        return NoveltyClassifier(stages).classify(statement, informal=informal)

    def test_a_right_hit_in_a_long_statement_is_not_read_as_absent(self):
        # Recorded 2026-11-04: Jaccard against this whole file scored the correct
        # declaration 0.31, so the control read N1 on a result in the library.
        control = json.loads(
            (ROOT / "data/novelty_control/gate_a_control_arc.json").read_text()
        )
        statement = next(
            s["lean_statement"]
            for s in control["statements"]
            if s["id"].startswith("theorem-right_adjoint_preserves_limits")
        )
        match = hit(
            "CategoryTheory.Adjunction.rightAdjoint_preservesLimits",
            sig=": PreservesLimitsOfSize.{v, u} G",
        )
        result = self.classify(statement, parse_declaration(statement)[1], match)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert any("rightAdjoint_preservesLimits" in e for e in result.evidence)

    def test_a_hit_sharing_only_generic_tokens_stays_below_the_floor(self):
        statement = (
            "theorem foo {C : Type} [CategoryTheory.Category C] (X : C) : True := sorry"
        )
        match = hit("CategoryTheory.Category.assoc_of_comp")
        assert self.classify(statement, "foo", match).level is NoveltyLevel.N1

    def test_a_bare_two_token_definition_is_not_a_match(self):
        statement = (
            "theorem foo {C : Type} [CategoryTheory.Category C] (X : C) : True := sorry"
        )
        match = hit("CategoryTheory.Category")
        assert self.classify(statement, "foo", match).level is NoveltyLevel.N1

    def test_no_semantic_hit_can_produce_n0(self):
        statement = "theorem yonedaLemma (C : CategoryTheory.Cat) : True := sorry"
        match = hit("CategoryTheory.yonedaLemma", sig="(C : CategoryTheory.Cat)")
        result = self.classify(statement, "yonedaLemma", match, informal="Yoneda lemma")
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert result.confidence < DECISIVE_CONFIDENCE

    def test_a_hit_below_the_floor_stays_as_a_candidate_for_review(self):
        statement = (
            "theorem foo {C : Type} [CategoryTheory.Category C] (X : C) : True := sorry"
        )
        match = hit("CategoryTheory.Category.assoc_of_comp", module="Mathlib.Cat")
        result = self.classify(statement, "foo", match)
        assert result.level is NoveltyLevel.N1
        expected = "semantic: CategoryTheory.Category.assoc_of_comp (Mathlib.Cat)"
        assert result.candidates == [expected]
        assert result.to_dict()["candidates"] == [expected]

    def test_a_statement_with_all_four_stages_empty_is_decisive_n1(self):
        result = NoveltyClassifier(four_empty_stages()).classify(NOVEL)
        assert result.level is NoveltyLevel.N1
        assert result.needs_review is False

    def test_candidates_keep_every_hit_in_order(self):
        hits = [hit(f"Library.result_number_{i}") for i in range(8)]
        semantic = RecordedBackend(
            "semantic",
            {
                "sameDenom_eq_iff_exists_postcomp_W": StageOutcome(
                    "semantic", True, hits=hits
                )
            },
        )
        result = NoveltyClassifier([semantic]).classify(NOVEL)
        assert result.candidates == [f"semantic: {h.name} (Mathlib.X)" for h in hits]

    def test_candidates_come_from_semantic_search_only(self):
        loogle = RecordedBackend(
            "loogle",
            {
                "sameDenom_eq_iff_exists_postcomp_W": StageOutcome(
                    "loogle", True, hits=[hit("Library.from_loogle")]
                )
            },
        )
        semantic = RecordedBackend("semantic", {}, available=False)
        result = NoveltyClassifier([loogle, semantic]).classify(NOVEL)
        assert result.candidates == []


# A concise restatement of a library result, recorded by a reviewer: the hit is too
# long for a share of 0.8, and shares more than half the tokens of the pair.
RESTATEMENT = (
    "theorem right_adjoint_preserves_limits (adj : Adjunction F G) : "
    "PreservesLimits G := sorry"
)
RESTATED = SearchHit(
    name="CategoryTheory.Adjunction.rightAdjoint_preservesLimits",
    type_signature=": PreservesLimitsOfSize.{v, u} G",
)


def restatement_stages() -> list[RecordedBackend]:
    semantic = RecordedBackend(
        "semantic",
        {
            "right_adjoint_preserves_limits": StageOutcome(
                "semantic", True, hits=[RESTATED]
            )
        },
    )
    return [*(empty_backend(s) for s in ("name", "loogle", "exact_probe")), semantic]


class TestNearHit:
    """A near hit leaves a statement N1, and takes the N1 out of automatic counting."""

    def test_it_stays_n1_but_needs_review(self):
        result = NoveltyClassifier(restatement_stages()).classify(RESTATEMENT)
        assert result.level is NoveltyLevel.N1
        assert result.confidence < DECISIVE_CONFIDENCE
        assert result.needs_review is True

    def test_its_evidence_names_the_hit(self):
        result = NoveltyClassifier(restatement_stages()).classify(RESTATEMENT)
        assert result.evidence[0].startswith("not found by:")
        assert f"near hit: semantic: {RESTATED.name}" in result.evidence

    def test_it_cannot_reach_cvfn(self):
        # The gate and the CVFN report read different fields, and both must hold.
        artifact = make_artifact(RESTATEMENT)
        classifier = NoveltyClassifier(restatement_stages())
        decision = apply_novelty_gate(artifact, classifier)
        assert decision.counts_as_novel is False
        assert decision.needs_human_review is True
        assert counts_as_novel(artifact.to_dict()) is False

    def test_a_decisive_n1_still_reaches_cvfn(self):
        artifact = make_artifact(NOVEL)
        decision = apply_novelty_gate(artifact, NoveltyClassifier(four_empty_stages()))
        assert decision.counts_as_novel is True
        assert counts_as_novel(artifact.to_dict()) is True

    def test_the_gate_stamps_candidates_on_the_artifact(self):
        artifact = make_artifact(RESTATEMENT)
        apply_novelty_gate(artifact, NoveltyClassifier(restatement_stages()))
        expected = [f"semantic: {RESTATED.name}"]
        assert artifact.novelty_candidates == expected
        assert artifact.to_dict()["novelty_candidates"] == expected
        assert Artifact.from_dict(artifact.to_dict()).novelty_candidates == expected


RECORDED_HITS = {
    row["id"]: row
    for row in json.loads((FIXTURES / "semantic_hits_2026-11-04.json").read_text())[
        "statements"
    ]
}
CONTROL_ARC = "gate_a_control_arc"
ARC_FILES = {
    "core_arc": ROOT / "data/ant_arcs/core_arc.json",
    "ramification_arc": ROOT / "data/ant_arcs/ramification_arc.json",
    CONTROL_ARC: ROOT / "data/novelty_control/gate_a_control_arc.json",
}
ARC_STATEMENTS = {
    arc: {s["id"]: s for s in json.loads(path.read_text())["statements"]}
    for arc, path in ARC_FILES.items()
}
ALL_FOUR = ["name", "loogle", "exact_probe"]


def recorded(prefix: str) -> dict:
    rows = [row for key, row in RECORDED_HITS.items() if key.startswith(prefix)]
    assert len(rows) == 1, prefix
    return rows[0]


def replay(
    row: dict,
    hits: list[SearchHit] | None = None,
    stages: list[str] | None = None,
    lean: str | None = None,
):
    """Classify a recorded statement against recorded hits, offline."""
    statement = ARC_STATEMENTS[row["arc"]][row["id"]]
    lean = lean or statement["lean_statement"]
    informal = statement.get("informal")
    if hits is None:
        hits = [SearchHit(name=h["name"], type_signature=h["sig"]) for h in row["hits"]]
    key = StatementQuery.from_lean(lean, informal=informal).display_name
    semantic = RecordedBackend(
        "semantic", {key: StageOutcome("semantic", True, hits=hits)}
    )
    ladder = [empty_backend(s) for s in (row["stages"] if stages is None else stages)]
    return NoveltyClassifier(
        [*ladder, semantic], names_are_labels=row["arc"] != CONTROL_ARC
    ).classify(lean, informal=informal)


def replay_one_hit(prefix: str, name: str):
    """A recorded statement at four stages, against one of its recorded hits."""
    row = recorded(prefix)
    found = next(h for h in row["hits"] if h["name"] == name)
    only = SearchHit(name=found["name"], type_signature=found["sig"])
    return replay(row, hits=[only], stages=ALL_FOUR)


class TestSemanticRule:
    """The chosen rule against the rejected ones, on pairs LeanSearch really returned."""

    @pytest.mark.parametrize(
        ("statement", "name"),
        [
            pytest.param(
                "core-18", "NumberField.Units.torsion", id="share-exactly-0.8"
            ),
            pytest.param(
                "lemma-yonedaEquiv_apply_id",
                "CategoryTheory.yonedaEquiv",
                id="share-0.8-jaccard-0.36",
            ),
            pytest.param("core-07", "NumberField.RingOfIntegers.rank", id="share-0.88"),
            pytest.param(
                "lemma-functor_preserves_iso",
                "CategoryTheory.Functor.mapIso_hom",
                id="share-0.83",
            ),
            pytest.param("ram-06", "Ideal.ramificationIdx", id="exactly-three-tokens"),
        ],
    )
    def test_these_pairs_match(self, statement, name):
        result = replay_one_hit(statement, name)
        assert result.level is NoveltyLevel.INCONCLUSIVE
        assert any(name in e for e in result.evidence)

    @pytest.mark.parametrize(
        ("statement", "name"),
        [
            pytest.param(
                "theorem-fully_faithful_reflects_iso",
                "CategoryTheory.Functor.FullyFaithful.casesOn",
                id="share-0.78",
            ),
            pytest.param("core-18", "subgroup_units_cyclic", id="share-0.75"),
            pytest.param(
                "lemma-product_symmetric",
                "CategoryTheory.Prod.braiding",
                id="share-0.75-long-statement",
            ),
            pytest.param(
                "core-07",
                "NumberField.RingOfIntegers.isIntegral_coe",
                id="share-0.63",
            ),
            pytest.param(
                "ram-14",
                "IsCyclotomicExtension.Rat.ramificationIdx_of_not_dvd",
                id="smaller-set-0.92-share-0.5",
            ),
            pytest.param("core-18", "instIsCyclicUnitsOfFinite", id="jaccard-0.44"),
        ],
    )
    def test_these_pairs_are_neither_a_match_nor_a_near_hit(self, statement, name):
        result = replay_one_hit(statement, name)
        assert result.level is NoveltyLevel.N1
        assert result.needs_review is False
        assert not any(name in e for e in result.evidence)

    @pytest.mark.parametrize(
        ("statement", "name"),
        [
            pytest.param("core-12", "IsZLattice.mk", id="jaccard-exactly-0.5"),
            pytest.param(
                "core-04",
                "minpoly.isIntegrallyClosed_eq_field_fractions'",
                id="jaccard-0.53-share-0.75",
            ),
        ],
    )
    def test_these_pairs_are_near_hits(self, statement, name):
        result = replay_one_hit(statement, name)
        assert result.level is NoveltyLevel.N1
        assert result.needs_review is True
        assert any(name in e and e.startswith("near hit") for e in result.evidence)

    @pytest.mark.parametrize(
        ("statement", "old", "new"),
        [("core-18", "K", "F"), ("ram-20", "K", "F"), ("ram-20", "L", "E")],
    )
    def test_renaming_a_binder_letter_changes_nothing(self, statement, old, new):
        row = recorded(statement)
        lean = ARC_STATEMENTS[row["arc"]][row["id"]]["lean_statement"]
        renamed = re.sub(rf"\b{old}\b", new, lean)
        assert renamed != lean
        written, after = replay(row), replay(row, lean=renamed)
        assert (written.level, written.confidence) == (after.level, after.confidence)

    def test_the_recorded_run_moves_exactly_these_statements_out_of_n1(self):
        moved = {
            row["id"]
            for row in RECORDED_HITS.values()
            if row["r4_level"] == "N1" and replay(row).level is not NoveltyLevel.N1
        }
        assert moved == {
            "core-18",
            "ram-06",
            "definition-initial_terminal_def-b7579f3b",
            "definition-initial_terminal_def_fixed-4b346e5d",
            "definition-mono_epi_def-b8cb5f94",
            "definition-representable_transformation_def-7cd6ca30",
            "lemma-equalizer_def-fd1cc105",
            "lemma-equivalence_of_categories-d0ff5d5a",
            "lemma-functor_preserves_iso-9a0fdc91",
            "lemma-natural_iso_components_iso-b02cff20",
            "lemma-pullback_comparison_iso-45b77643",
            "lemma-pullback_of_mono_is_mono-d8bb9f5e",
            "lemma-representable_functor_preserves_products-47afce45",
            "lemma-representable_morphism_pullback_exists-be32bec1",
            "theorem-fully_faithful_reflects_iso-55b16f1b",
            "theorem-functor_category_def-ecca97d1",
            "theorem-right_adjoint_preserves_limits-0a6a19e1",
        }

    def test_a_near_hit_never_raises_a_confidence(self):
        assert replay(recorded("core-12")).confidence == pytest.approx(0.6)

    def test_the_recorded_run_has_exactly_these_near_hits(self):
        near = {
            row["id"]
            for row in RECORDED_HITS.values()
            if row["r4_level"] == "N1"
            and any(e.startswith("near hit") for e in replay(row).evidence)
        }
        assert near == {"core-04", "core-12", "ram-04", "ram-14"}

    def test_statements_the_old_tier_held_stay_out_of_n1(self):
        rows = [r for r in RECORDED_HITS.values() if r["r4_level"] == "INCONCLUSIVE"]
        assert len(rows) == 12
        assert all(replay(row).level is NoveltyLevel.INCONCLUSIVE for row in rows)

    def test_no_recorded_hit_reaches_n0(self):
        levels = {replay(row).level for row in RECORDED_HITS.values()}
        assert NoveltyLevel.N0 not in levels


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
