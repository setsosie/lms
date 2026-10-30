"""The checked-in kernel goal files carry their source's dependency edges."""

import json
from pathlib import Path

from lms.dependency import DependencyGraph
from lms.goals import Goal

GOALS = Path(__file__).resolve().parent.parent / "goals"


def _graph(fname: str) -> DependencyGraph:
    path = GOALS / fname
    # Every definition carries an explicit edge list, roots included.
    assert all("requires" in d for d in json.loads(path.read_text())["definitions"])
    return DependencyGraph.from_goal(Goal.load(path))


def test_kernel_goals_are_dags():
    track_b = _graph("stacks_kernel_track_b.json")
    assert len(track_b.available_tasks()) == 18
    assert track_b.get_node("02XO").requires == ["02XK", "02XL"]
    assert track_b.get_node("02XN").requires == []
    assert track_b.get_node("004B").requires == ["003V", "02XN"]

    track_a = _graph("stacks_kernel_track_a.json")
    assert len(track_a.available_tasks()) == 16
    assert track_a.get_node("012W").requires == ["012V"]
    assert track_a.get_node("0132").requires == ["012W"]
