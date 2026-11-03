"""Explicit dependency edges from a goal replace section-order inference."""

import pytest

from lms.dependency import DependencyGraph, TaskStatus
from lms.goals import Goal, StacksDefinition


def _goal(*defs: tuple[str, str, list[str] | None]) -> Goal:
    return Goal(
        name="g",
        description="d",
        source="s",
        definitions=[
            StacksDefinition(tag=tag, section=sec, name=tag, content="", requires=req)
            for tag, sec, req in defs
        ],
    )


def test_explicit_edges_replace_inference(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(
        DependencyGraph, "_infer_dependencies", lambda self: calls.append(True)
    )
    graph = DependencyGraph.from_goal(
        _goal(
            ("A", "4.1", []),
            ("B", "4.2", []),
            ("C", "4.3", ["A", "B"]),
            ("D", "5.1", ["A"]),
        )
    )
    assert calls == []
    assert graph.get_node("C").requires == ["A", "B"]
    # No chapter-boundary edge: D needs only what it lists.
    assert graph.get_node("D").requires == ["A"]
    assert sorted(graph.get_node("A").unlocks) == ["C", "D"]
    assert graph.get_node("B").unlocks == ["C"]
    assert sorted(n.tag for n in graph.available_tasks()) == ["A", "B"]


def test_unknown_requires_dropped(capsys):
    graph = DependencyGraph.from_goal(
        _goal(("A", "4.1", []), ("B", "4.2", ["ZZZZ", "A", "YYYY"]))
    )
    assert graph.get_node("B").requires == ["A"]
    named = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "ZZZZ" in line or "YYYY" in line
    ]
    assert len(named) == 1 and "ZZZZ" in named[0] and "YYYY" in named[0]
    # A kept unknown tag would block B forever.
    graph.update_status("A", TaskStatus.DONE)
    assert graph.get_node("B").status == TaskStatus.AVAILABLE


def test_empty_requires_is_a_root():
    # One explicit definition puts the whole goal in explicit mode.
    graph = DependencyGraph.from_goal(
        _goal(("A", "4.1", None), ("B", "4.2", []), ("C", "4.3", ["B"]))
    )
    assert graph.get_node("B").requires == []
    assert graph.get_node("B").status == TaskStatus.AVAILABLE
    assert graph.get_node("C").requires == ["B"]


def test_cycle_is_a_goal_defect():
    goal = _goal(
        ("02XA", "4.1", ["02XC"]),
        ("02XB", "4.2", ["02XA"]),
        ("02XC", "4.3", ["02XB"]),
        ("02XD", "4.4", []),
    )
    with pytest.raises(ValueError, match=r"(?i)cycle") as err:
        DependencyGraph.from_goal(goal)
    for tag in ("02XA", "02XB", "02XC"):
        assert tag in str(err.value)
