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


def write_run(run_dir, artifacts, tags=(), goal_name="ant", records=None, forbidden=()):
    """A saved run directory; `records` becomes the attempt ledger."""
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps({"generations": [], "checkpoint": {"total_tokens_used": 1_000}})
    )
    (run_dir / "artifacts.json").write_text(json.dumps({"artifacts": artifacts}))
    goal = {
        "name": goal_name,
        "forbidden_imports": list(forbidden),
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
    run = write_run(tmp_path / "r", [novel("0013")], forbidden=["Mathlib"])

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


def test_runs_with_different_goals_pool_but_refuse_to_pair(tmp_path):
    a = arm_report("A", [write_run(tmp_path / "a", [], tags=["t1"])])
    r1 = write_run(tmp_path / "r1", [novel("t1")], tags=["t1"])
    r2 = write_run(tmp_path / "r2", [novel("t2")], tags=["t2"], goal_name="other")
    mixed = arm_report("B", [r1, r2])
    assert mixed.distinct_statements == 2
    assert "2 different goals" in mixed.format()

    with pytest.raises(ValueError, match="goals differ") as refused:
        shared_statements(a, mixed)
    assert "'ant'" in str(refused.value) and "'other'" in str(refused.value)


def test_a_quoted_tag_pools_and_pairs_as_the_bare_tag(tmp_path):
    # Flat and iterative agents copy the prompt's quotes into the tag.
    quoted = write_run(tmp_path / "q", [novel('"0013"')], tags=["0013"])
    bare = write_run(tmp_path / "b", [novel("0013")], tags=["0013"])
    assert statement_outcomes(quoted) == {"tag:0013": True}

    pooled = arm_report("I9", [quoted, bare])
    assert pooled.distinct_statements == 1
    assert pooled.duplicate_statements == ["tag:0013"]

    a, b = arm_report("I9", [quoted]), arm_report("C9", [bare])
    assert paired_comparison(a, b, shared_statements(a, b)).both == 1


def test_a_verified_statement_off_the_goal_slice_is_counted_aside(tmp_path):
    run = write_run(tmp_path / "r", [novel("t1"), novel("t9")], tags=["t1"])
    report = arm_report("C9", [run])
    assert report.off_slice_verified == 1
    assert "1 off-slice" in report.format()


def test_an_unmeasurable_arm_refuses_to_pair(tmp_path):
    u = write_run(tmp_path / "u", [novel("t1")], tags=TAGS, forbidden=["Mathlib"])
    a = write_run(tmp_path / "a", [novel("t1")], tags=TAGS)
    u_arm, a_arm = arm_report("U", [u]), arm_report("A", [a])
    with pytest.raises(ValueError, match="U.*unmeasurable"):
        paired_comparison(a_arm, u_arm, shared_statements(a_arm, u_arm))


def test_pairing_different_denominator_kinds_is_refused(tmp_path):
    bare = {"id": "x", "status": "verified_lean", "stacks_tag": "t1"}
    a = arm_report("A", [write_run(tmp_path / "a", [novel("t1")], tags=TAGS)])
    b = arm_report("B", [write_run(tmp_path / "b", [bare], tags=TAGS)])
    with pytest.raises(ValueError, match="verified_novel.*verified_lean_unfiltered"):
        paired_comparison(a, b, shared_statements(a, b))


def test_a_run_that_verified_nothing_does_not_set_the_pooled_kind(tmp_path):
    proved = write_run(tmp_path / "p", [novel("t1")], tags=TAGS)
    empty = write_run(tmp_path / "e", [], tags=TAGS)
    i9 = arm_report("I9", [proved, empty])
    assert i9.denominator_kind == "verified_novel"
    c9 = arm_report("C9", [write_run(tmp_path / "c", [novel("t1")], tags=TAGS)])
    assert paired_comparison(i9, c9, shared_statements(i9, c9)).both == 1
    # An arm that verified nothing counts nothing under either rule.
    zero = arm_report("Z", [empty])
    assert paired_comparison(zero, c9, shared_statements(zero, c9)).b_only == 1

    # The goal makes a run unmeasurable whether or not it verified anything.
    barred = write_run(tmp_path / "b", [], tags=TAGS, forbidden=["Mathlib"])
    assert arm_report("X", [proved, barred]).denominator_kind == "unmeasurable"


def test_an_arm_with_no_runs_is_refused():
    with pytest.raises(ValueError, match="no run"):
        arm_report("E", [])


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "compare_arms", ROOT / "scripts" / "compare_arms.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_refuses_a_repeated_arm_name(tmp_path):
    a = write_run(tmp_path / "a", [], tags=TAGS)
    with pytest.raises(SystemExit):
        load_cli().main(["--arm", f"A={a}", "--arm", f"A={a}"])


@pytest.mark.parametrize("dirs", ["{a},", "{a},{missing}"])
def test_cli_refuses_a_missing_run_directory_as_usage(tmp_path, dirs):
    a = write_run(tmp_path / "a", [], tags=TAGS)
    spec = dirs.format(a=a, missing=tmp_path / "missing")
    with pytest.raises(SystemExit) as exited:
        load_cli().main(["--arm", f"A={spec}"])
    assert exited.value.code == 2


def test_cli_prints_arms_and_pairs_and_writes_the_same_as_json(tmp_path, capsys):
    cli = load_cli()

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
    assert data["pairs"][0]["a_kind"] == "verified_novel"
