"""Arm comparison over saved run directories: pooled arms, distinct statements,
paired tests.

An arm's denominator is distinct statements clearing the gates, keyed by
source tag, not verified artifacts. A pooled arm pays for every run's tokens
but counts a statement once, however many runs proved it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from lms.accounting import AttemptRecord
from lms.arms import (
    arm_report,
    paired_comparison,
    shared_statements,
    statement_outcomes,
)

ROOT = Path(__file__).resolve().parent.parent


def novel(tag, **kw):
    """A verified artifact that clears Gate 4 (decisive N1)."""
    return {
        "id": f"a-{tag}",
        "status": "verified_lean",
        "stacks_tag": tag,
        "novelty_level": "N1",
        "novelty_confidence": 0.9,
        **kw,
    }


def write_run(run_dir, artifacts, tags=(), goal_name="ant", records=None):
    """A saved run directory; `records` becomes the attempt ledger."""
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps({"generations": [], "checkpoint": {"total_tokens_used": 1_000}})
    )
    (run_dir / "artifacts.json").write_text(json.dumps({"artifacts": artifacts}))
    goal = {
        "name": goal_name,
        "forbidden_imports": [],
        "definitions": [{"tag": t} for t in tags],
    }
    (run_dir / "goal.json").write_text(json.dumps(goal))
    if records is not None:
        (run_dir / "attempts.json").write_text(
            json.dumps({"records": [r.to_dict() for r in records]})
        )
    return run_dir


def rec(key, prompt, completion):
    return AttemptRecord(key, "agent-0", 0, prompt, completion, 1.0, "failed")


def test_the_gate_4_rule_is_public_and_the_old_name_still_works():
    from lms.accounting import _counts_as_novel, counts_as_novel

    assert _counts_as_novel is counts_as_novel


def test_a_tag_verified_twice_in_one_run_counts_once(tmp_path):
    run = write_run(tmp_path / "r", [novel("0013"), novel("0013", id="a-dup")])
    assert statement_outcomes(run) == {"tag:0013": True}

    report = arm_report("S1", [run])
    assert report.distinct_statements == 1
    assert report.artifact_denominator == 2  # what cvfn_report says
    assert report.duplicate_statements == ["tag:0013"]
    assert report.cvfn_tokens_per_statement == 1_000


def test_a_tag_proved_by_two_pooled_runs_counts_once_and_is_a_duplicate(tmp_path):
    r1 = write_run(tmp_path / "r1", [novel("0013")])
    r2 = write_run(tmp_path / "r2", [novel("0013"), novel("0014")])
    report = arm_report("I9", [r1, r2])
    assert report.distinct_statements == 2
    assert report.artifact_denominator == 3
    assert report.duplicate_statements == ["tag:0013"]


def test_tokens_sum_across_pooled_runs_including_overhead(tmp_path):
    from lms.accounting import OVERHEAD_KEY

    r1 = write_run(
        tmp_path / "r1",
        [novel("0013")],
        records=[rec("tag:0013", 100, 50), rec(OVERHEAD_KEY, 30, 20)],
    )
    r2 = write_run(tmp_path / "r2", [], records=[rec("tag:0014", 200, 100)])
    # No ledger: falls back to the society total in results.json.
    r3 = write_run(tmp_path / "r3", [])
    assert arm_report("I9", [r1, r2, r3]).total_tokens == 200 + 300 + 1_000


def test_an_untagged_verified_artifact_is_counted_aside_not_keyed(tmp_path):
    untagged = novel(None, lean_code="theorem foo : True := trivial")
    run = write_run(tmp_path / "r", [novel("0013"), untagged])
    assert statement_outcomes(run) == {"tag:0013": True}

    report = arm_report("S1", [run])
    assert report.distinct_statements == 1
    assert report.untagged_verified == 1
    assert "untagged" in report.format()


def test_novelty_labels_filter_and_their_absence_is_flagged(tmp_path):
    labelled = write_run(
        tmp_path / "labelled", [novel("0013"), novel("0014", novelty_level="N0")]
    )
    assert statement_outcomes(labelled) == {"tag:0013": True, "tag:0014": False}
    assert arm_report("C9", [labelled]).denominator_kind == "verified_novel"

    bare = {"id": "a", "status": "verified_lean", "stacks_tag": "0013"}
    unlabelled = write_run(tmp_path / "unlabelled", [bare])
    assert statement_outcomes(unlabelled) == {"tag:0013": True}
    report = arm_report("C9", [unlabelled])
    assert report.denominator_kind == "verified_lean_unfiltered"


def test_a_goal_forbidding_mathlib_makes_the_arm_unmeasurable(tmp_path):
    run = write_run(tmp_path / "r", [novel("0013")])
    goal = json.loads((run / "goal.json").read_text())
    goal["forbidden_imports"] = ["Mathlib"]
    (run / "goal.json").write_text(json.dumps(goal))

    report = arm_report("C9", [run])
    assert report.denominator_kind == "unmeasurable"
    assert report.cvfn_tokens_per_statement is None
    assert "unmeasurable" in report.format()


TAGS = [f"t{i}" for i in range(1, 11)]


@pytest.fixture
def two_arms(tmp_path):
    # both: t1,t2   A only: t3..t7   B only: t8   neither: t9,t10
    a = write_run(tmp_path / "a", [novel(t) for t in TAGS[:7]], tags=TAGS)
    b = write_run(tmp_path / "b", [novel(t) for t in ("t1", "t2", "t8")], tags=TAGS)
    return arm_report("A", [a]), arm_report("B", [b])


def test_mcnemar_p_matches_a_hand_computed_table(two_arms):
    a, b = two_arms
    result = paired_comparison(a, b, [f"tag:{t}" for t in TAGS])
    assert (result.both, result.a_only, result.b_only, result.neither) == (2, 5, 1, 2)
    # Exact two-sided: 2 * P(X <= 1), X ~ Binomial(6, 1/2) = 2 * 7/64.
    assert result.mcnemar_p == pytest.approx(14 / 64)
    assert result.difference == 4
    assert "between-run variance" in result.format()


def test_bootstrap_interval_is_reproducible_under_its_seed(two_arms):
    a, b = two_arms
    statements = [f"tag:{t}" for t in TAGS]
    first = paired_comparison(a, b, statements, seed=7)
    again = paired_comparison(a, b, statements, seed=7)
    assert first.seed == 7
    assert first.bootstrap_ci == again.bootstrap_ci
    lo, hi = first.bootstrap_ci
    assert lo <= first.difference <= hi


def test_pairing_uses_the_shared_slice_and_refuses_different_goals(tmp_path):
    a = arm_report("A", [write_run(tmp_path / "a", [], tags=["t1", "t2", "t3"])])
    b = arm_report("B", [write_run(tmp_path / "b", [], tags=["t2", "t3", "t4"])])
    assert shared_statements(a, b) == ["tag:t2", "tag:t3"]

    other = write_run(tmp_path / "c", [], tags=["t2"], goal_name="stacks-ch4")
    with pytest.raises(ValueError, match="goal"):
        shared_statements(a, arm_report("C", [other]))


def test_cli_prints_arms_and_pairs_and_writes_the_same_as_json(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "compare_arms", ROOT / "scripts" / "compare_arms.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    a = write_run(tmp_path / "a", [novel("t1")], tags=TAGS)
    b1 = write_run(tmp_path / "b1", [novel("t1")], tags=TAGS)
    b2 = write_run(tmp_path / "b2", [novel("t2")], tags=TAGS)
    out = tmp_path / "out.json"
    argv = ["--arm", f"A={a}", "--arm", f"B={b1},{b2}", "--pair", "A:B"]
    assert cli.main([*argv, "--json", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "A" in printed and "B" in printed and "McNemar" in printed
    data = json.loads(out.read_text())
    assert [arm["name"] for arm in data["arms"]] == ["A", "B"]
    assert data["arms"][1]["distinct_statements"] == 2
    assert len(data["pairs"]) == 1 and "mcnemar_p" in data["pairs"][0]
