"""Tests for the dependency graph module."""

import json
import tempfile
from pathlib import Path

import pytest

from lms.dependency import (
    DependencyGraph,
    DependencyNode,
    TaskStatus,
)
from lms.goals import Goal, StacksDefinition


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test status enum has expected values."""
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.AVAILABLE.value == "available"
        assert TaskStatus.BLOCKED.value == "blocked"

    def test_status_from_string(self):
        """Test creating status from string value."""
        assert TaskStatus("done") == TaskStatus.DONE
        assert TaskStatus("blocked") == TaskStatus.BLOCKED


class TestDependencyNode:
    """Tests for DependencyNode dataclass."""

    def test_create_node(self):
        """Test basic node creation."""
        node = DependencyNode(
            tag="CH4-CAT",
            name="Category",
            chapter=4,
            section="4.2",
        )
        assert node.tag == "CH4-CAT"
        assert node.name == "Category"
        assert node.chapter == 4
        assert node.section == "4.2"
        assert node.status == TaskStatus.BLOCKED
        assert node.requires == []
        assert node.unlocks == []
        assert node.artifact_id is None

    def test_node_defaults(self):
        """Test default values."""
        node = DependencyNode(tag="X", name="X", chapter=1, section="1.0")
        assert node.requires == []
        assert node.unlocks == []
        assert node.status == TaskStatus.BLOCKED
        assert node.artifact_id is None

    def test_priority_score_no_unlocks(self):
        """Test priority score with no unlocks."""
        node = DependencyNode(tag="X", name="X", chapter=1, section="1.0")
        assert node.priority_score() == 0

    def test_priority_score_with_unlocks(self):
        """Test priority score increases with unlocks."""
        node = DependencyNode(
            tag="X",
            name="X",
            chapter=1,
            section="1.0",
            unlocks=["A", "B", "C"],
        )
        assert node.priority_score() == 3

    def test_to_dict(self):
        """Test serialization to dict."""
        node = DependencyNode(
            tag="CH4-CAT",
            name="Category",
            chapter=4,
            section="4.2",
            requires=["CH4-PREREQ"],
            unlocks=["CH4-FUNCTOR"],
            status=TaskStatus.DONE,
            artifact_id="artifact-123",
        )
        data = node.to_dict()
        assert data["tag"] == "CH4-CAT"
        assert data["name"] == "Category"
        assert data["chapter"] == 4
        assert data["section"] == "4.2"
        assert data["requires"] == ["CH4-PREREQ"]
        assert data["unlocks"] == ["CH4-FUNCTOR"]
        assert data["status"] == "done"
        assert data["artifact_id"] == "artifact-123"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "tag": "CH4-CAT",
            "name": "Category",
            "chapter": 4,
            "section": "4.2",
            "requires": ["CH4-PREREQ"],
            "unlocks": ["CH4-FUNCTOR"],
            "status": "available",
            "artifact_id": None,
        }
        node = DependencyNode.from_dict(data)
        assert node.tag == "CH4-CAT"
        assert node.name == "Category"
        assert node.status == TaskStatus.AVAILABLE
        assert node.requires == ["CH4-PREREQ"]

    def test_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = DependencyNode(
            tag="TEST",
            name="Test Node",
            chapter=5,
            section="5.3",
            requires=["A", "B"],
            unlocks=["C"],
            status=TaskStatus.IN_PROGRESS,
            artifact_id="art-456",
        )
        data = original.to_dict()
        restored = DependencyNode.from_dict(data)
        assert restored.tag == original.tag
        assert restored.name == original.name
        assert restored.chapter == original.chapter
        assert restored.section == original.section
        assert restored.requires == original.requires
        assert restored.unlocks == original.unlocks
        assert restored.status == original.status
        assert restored.artifact_id == original.artifact_id


class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = DependencyGraph()
        assert len(graph.nodes) == 0
        assert graph.progress() == 0.0

    def test_add_node(self):
        """Test adding a node."""
        graph = DependencyGraph()
        node = DependencyNode(tag="A", name="A", chapter=1, section="1.0")
        graph.add_node(node)
        assert "A" in graph.nodes
        assert graph.nodes["A"] == node

    def test_get_node(self):
        """Test getting a node by tag."""
        graph = DependencyGraph()
        node = DependencyNode(tag="A", name="A", chapter=1, section="1.0")
        graph.add_node(node)
        assert graph.get_node("A") == node
        assert graph.get_node("NONEXISTENT") is None

    def test_update_status(self):
        """Test updating a node's status."""
        graph = DependencyGraph()
        node = DependencyNode(tag="A", name="A", chapter=1, section="1.0")
        graph.add_node(node)

        graph.update_status("A", TaskStatus.DONE, "artifact-1")
        assert graph.nodes["A"].status == TaskStatus.DONE
        assert graph.nodes["A"].artifact_id == "artifact-1"

    def test_update_status_nonexistent_tag(self):
        """Test updating nonexistent tag does nothing."""
        graph = DependencyGraph()
        # Should not raise
        graph.update_status("NONEXISTENT", TaskStatus.DONE)

    def test_available_tasks_empty(self):
        """Test available tasks on empty graph."""
        graph = DependencyGraph()
        assert graph.available_tasks() == []

    def test_available_tasks_all_blocked(self):
        """Test available tasks when all are blocked."""
        graph = DependencyGraph()
        node_a = DependencyNode(
            tag="A", name="A", chapter=1, section="1.0", requires=["B"]
        )
        graph.add_node(node_a)
        assert graph.available_tasks() == []

    def test_available_tasks_no_dependencies(self):
        """Test node with no dependencies is available."""
        graph = DependencyGraph()
        node = DependencyNode(
            tag="A",
            name="A",
            chapter=1,
            section="1.0",
            status=TaskStatus.AVAILABLE,
        )
        graph.add_node(node)
        available = graph.available_tasks()
        assert len(available) == 1
        assert available[0].tag == "A"

    def test_available_tasks_sorted_by_priority(self):
        """Test available tasks are sorted by priority (unlocks count)."""
        graph = DependencyGraph()
        node_low = DependencyNode(
            tag="LOW",
            name="Low Priority",
            chapter=1,
            section="1.0",
            unlocks=["X"],
            status=TaskStatus.AVAILABLE,
        )
        node_high = DependencyNode(
            tag="HIGH",
            name="High Priority",
            chapter=1,
            section="1.1",
            unlocks=["A", "B", "C", "D", "E"],
            status=TaskStatus.AVAILABLE,
        )
        graph.add_node(node_low)
        graph.add_node(node_high)

        available = graph.available_tasks()
        assert len(available) == 2
        assert available[0].tag == "HIGH"  # Higher priority first
        assert available[1].tag == "LOW"

    def test_recalculate_availability(self):
        """Test that availability is recalculated when dependencies are met."""
        graph = DependencyGraph()

        # B depends on A
        node_a = DependencyNode(
            tag="A",
            name="A",
            chapter=1,
            section="1.0",
            status=TaskStatus.BLOCKED,
        )
        node_b = DependencyNode(
            tag="B",
            name="B",
            chapter=1,
            section="1.1",
            requires=["A"],
            status=TaskStatus.BLOCKED,
        )
        graph.add_node(node_a)
        graph.add_node(node_b)

        # Initially, A should be available (no dependencies), B blocked
        graph._recalculate_availability()
        assert graph.nodes["A"].status == TaskStatus.AVAILABLE
        assert graph.nodes["B"].status == TaskStatus.BLOCKED

        # Mark A as done
        graph.update_status("A", TaskStatus.DONE)

        # Now B should be available
        assert graph.nodes["B"].status == TaskStatus.AVAILABLE

    def test_blocked_tasks(self):
        """Test getting blocked tasks."""
        graph = DependencyGraph()
        node_blocked = DependencyNode(
            tag="B",
            name="B",
            chapter=1,
            section="1.0",
            requires=["A"],
            status=TaskStatus.BLOCKED,
        )
        graph.add_node(node_blocked)
        blocked = graph.blocked_tasks()
        assert len(blocked) == 1
        assert blocked[0].tag == "B"

    def test_in_progress_tasks(self):
        """Test getting in-progress tasks."""
        graph = DependencyGraph()
        node = DependencyNode(
            tag="A",
            name="A",
            chapter=1,
            section="1.0",
            status=TaskStatus.IN_PROGRESS,
        )
        graph.add_node(node)
        in_progress = graph.in_progress_tasks()
        assert len(in_progress) == 1
        assert in_progress[0].tag == "A"

    def test_done_tasks(self):
        """Test getting done tasks."""
        graph = DependencyGraph()
        node = DependencyNode(
            tag="A",
            name="A",
            chapter=1,
            section="1.0",
            status=TaskStatus.DONE,
            artifact_id="art-1",
        )
        graph.add_node(node)
        done = graph.done_tasks()
        assert len(done) == 1
        assert done[0].tag == "A"

    def test_progress_calculation(self):
        """Test progress calculation."""
        graph = DependencyGraph()
        for i in range(4):
            node = DependencyNode(
                tag=f"N{i}",
                name=f"Node {i}",
                chapter=1,
                section=f"1.{i}",
                status=TaskStatus.DONE if i < 2 else TaskStatus.BLOCKED,
            )
            graph.add_node(node)

        assert graph.progress() == 0.5  # 2 out of 4

    def test_progress_summary(self):
        """Test progress summary string."""
        graph = DependencyGraph()
        node_done = DependencyNode(
            tag="D", name="Done", chapter=1, section="1.0", status=TaskStatus.DONE
        )
        node_avail = DependencyNode(
            tag="A", name="Avail", chapter=1, section="1.1", status=TaskStatus.AVAILABLE
        )
        node_blocked = DependencyNode(
            tag="B",
            name="Blocked",
            chapter=1,
            section="1.2",
            requires=["X"],
            status=TaskStatus.BLOCKED,
        )
        graph.add_node(node_done)
        graph.add_node(node_avail)
        graph.add_node(node_blocked)

        summary = graph.progress_summary()
        assert "1/3" in summary
        assert "33%" in summary
        assert "Available: 1" in summary
        assert "Blocked: 1" in summary


class TestDependencyGraphFromGoal:
    """Tests for creating DependencyGraph from Goal."""

    def test_from_goal_empty(self):
        """Test creating graph from goal with no definitions."""
        goal = Goal(name="Test", description="...", source="...")
        graph = DependencyGraph.from_goal(goal)
        assert len(graph.nodes) == 0

    def test_from_goal_single_definition(self):
        """Test creating graph from goal with single definition."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="A", section="1.0", name="A", content="..."),
            ],
        )
        graph = DependencyGraph.from_goal(goal)
        assert len(graph.nodes) == 1
        assert "A" in graph.nodes
        # Single node with no deps should be available
        assert graph.nodes["A"].status == TaskStatus.AVAILABLE

    def test_from_goal_preserves_formalized_status(self):
        """Test that formalized definitions become DONE."""
        defn = StacksDefinition(
            tag="A",
            section="1.0",
            name="A",
            content="...",
            formalized=True,
            artifact_ids=["art-1"],
        )
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[defn],
        )
        graph = DependencyGraph.from_goal(goal)
        assert graph.nodes["A"].status == TaskStatus.DONE
        assert graph.nodes["A"].artifact_id == "art-1"

    def test_from_goal_infers_section_dependencies(self):
        """Test that dependencies are inferred from section ordering."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="A", section="1.1", name="A", content="..."),
                StacksDefinition(tag="B", section="1.2", name="B", content="..."),
                StacksDefinition(tag="C", section="1.3", name="C", content="..."),
            ],
        )
        graph = DependencyGraph.from_goal(goal)

        # B should require A
        assert "A" in graph.nodes["B"].requires
        # C should require A and B
        assert "A" in graph.nodes["C"].requires
        assert "B" in graph.nodes["C"].requires
        # A should unlock B and C
        assert "B" in graph.nodes["A"].unlocks
        assert "C" in graph.nodes["A"].unlocks

    def test_from_goal_chapter_dependencies(self):
        """Test that first node of chapter N depends on last node of chapter N-1."""
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="CH4-A", section="4.1", name="Ch4 A", content="..."),
                StacksDefinition(tag="CH4-B", section="4.2", name="Ch4 B", content="..."),
                StacksDefinition(tag="CH5-A", section="5.1", name="Ch5 A", content="..."),
            ],
        )
        graph = DependencyGraph.from_goal(goal)

        # CH5-A should depend on CH4-B (last of ch4)
        assert "CH4-B" in graph.nodes["CH5-A"].requires
        # CH4-B should unlock CH5-A
        assert "CH5-A" in graph.nodes["CH4-B"].unlocks

    def test_from_goal_availability_cascade(self):
        """Test that availability cascades when dependencies are done."""
        defn_a = StacksDefinition(
            tag="A",
            section="1.1",
            name="A",
            content="...",
            formalized=True,  # Done
            artifact_ids=["art-a"],
        )
        defn_b = StacksDefinition(
            tag="B", section="1.2", name="B", content="..."  # Not done
        )
        goal = Goal(
            name="Test",
            description="...",
            source="...",
            definitions=[defn_a, defn_b],
        )
        graph = DependencyGraph.from_goal(goal)

        # A is done, so B should be available
        assert graph.nodes["A"].status == TaskStatus.DONE
        assert graph.nodes["B"].status == TaskStatus.AVAILABLE


class TestDependencyGraphSaveLoad:
    """Tests for saving and loading dependency graphs."""

    def test_save_empty_graph(self):
        """Test saving an empty graph."""
        graph = DependencyGraph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            graph.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["nodes"] == []

    def test_save_with_nodes(self):
        """Test saving a graph with nodes."""
        graph = DependencyGraph()
        graph.add_node(
            DependencyNode(
                tag="A",
                name="A",
                chapter=1,
                section="1.0",
                requires=["X"],
                unlocks=["Y", "Z"],
                status=TaskStatus.DONE,
                artifact_id="art-1",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            graph.save(path)
            data = json.loads(path.read_text())
            assert len(data["nodes"]) == 1
            assert data["nodes"][0]["tag"] == "A"
            assert data["nodes"][0]["status"] == "done"

    def test_load_graph(self):
        """Test loading a graph."""
        data = {
            "nodes": [
                {
                    "tag": "A",
                    "name": "A",
                    "chapter": 1,
                    "section": "1.0",
                    "requires": ["X"],
                    "unlocks": ["Y"],
                    "status": "available",
                    "artifact_id": None,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            path.write_text(json.dumps(data))
            graph = DependencyGraph.load(path)
            assert len(graph.nodes) == 1
            assert graph.nodes["A"].tag == "A"
            assert graph.nodes["A"].status == TaskStatus.AVAILABLE

    def test_save_load_roundtrip(self):
        """Test save/load roundtrip preserves data."""
        original = DependencyGraph()
        original.add_node(
            DependencyNode(
                tag="A",
                name="Node A",
                chapter=4,
                section="4.2",
                requires=["X", "Y"],
                unlocks=["B", "C", "D"],
                status=TaskStatus.IN_PROGRESS,
                artifact_id=None,
            )
        )
        original.add_node(
            DependencyNode(
                tag="B",
                name="Node B",
                chapter=4,
                section="4.3",
                requires=["A"],
                unlocks=[],
                status=TaskStatus.BLOCKED,
                artifact_id=None,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            original.save(path)
            loaded = DependencyGraph.load(path)

            assert len(loaded.nodes) == 2
            assert loaded.nodes["A"].tag == "A"
            assert loaded.nodes["A"].name == "Node A"
            assert loaded.nodes["A"].requires == ["X", "Y"]
            assert loaded.nodes["A"].unlocks == ["B", "C", "D"]
            assert loaded.nodes["A"].status == TaskStatus.IN_PROGRESS
            assert loaded.nodes["B"].requires == ["A"]


class TestDependencyGraphPromptContext:
    """Tests for generating prompt context from dependency graph."""

    def test_to_prompt_context_empty(self):
        """Test prompt context for empty graph."""
        graph = DependencyGraph()
        context = graph.to_prompt_context()
        assert "Dependency Graph Status" in context
        assert "Completed (0)" in context

    def test_to_prompt_context_with_tasks(self):
        """Test prompt context includes all task states."""
        graph = DependencyGraph()
        graph.add_node(
            DependencyNode(
                tag="DONE-1",
                name="Done Task",
                chapter=1,
                section="1.0",
                status=TaskStatus.DONE,
            )
        )
        graph.add_node(
            DependencyNode(
                tag="AVAIL-1",
                name="Available Task",
                chapter=1,
                section="1.1",
                unlocks=["X", "Y", "Z"],
                status=TaskStatus.AVAILABLE,
            )
        )
        graph.add_node(
            DependencyNode(
                tag="BLOCKED-1",
                name="Blocked Task",
                chapter=1,
                section="1.2",
                requires=["MISSING"],
                status=TaskStatus.BLOCKED,
            )
        )

        context = graph.to_prompt_context()

        assert "[DONE] DONE-1" in context
        assert "[AVAILABLE] AVAIL-1" in context
        assert "unlocks 3 tasks" in context
        assert "[BLOCKED] BLOCKED-1" in context
        assert "needs: MISSING" in context

    def test_to_prompt_context_limits_output(self):
        """Test that prompt context limits long lists."""
        graph = DependencyGraph()
        # Add many done tasks
        for i in range(15):
            graph.add_node(
                DependencyNode(
                    tag=f"DONE-{i}",
                    name=f"Done Task {i}",
                    chapter=1,
                    section=f"1.{i}",
                    status=TaskStatus.DONE,
                )
            )

        context = graph.to_prompt_context()
        # Should show first 10 and indicate more
        assert "DONE-0" in context
        assert "DONE-9" in context
        assert "... and 5 more" in context


class TestDependencyGraphIntegration:
    """Integration tests with real-like scenarios."""

    def test_stacks_chapter_workflow(self):
        """Test a workflow similar to Stacks Project chapters."""
        # Create a mini goal with ch4 and ch5 definitions
        goal = Goal(
            name="Mini Stacks",
            description="...",
            source="...",
            definitions=[
                StacksDefinition(tag="CH4-CAT", section="4.2", name="Category", content="..."),
                StacksDefinition(tag="CH4-FUNC", section="4.2", name="Functor", content="..."),
                StacksDefinition(tag="CH4-NAT", section="4.2", name="NatTrans", content="..."),
                StacksDefinition(tag="CH5-TOP", section="5.1", name="TopSpace", content="..."),
                StacksDefinition(tag="CH5-CONT", section="5.2", name="Continuous", content="..."),
            ],
        )

        graph = DependencyGraph.from_goal(goal)

        # Initially, CH4-CAT should be available (it's first)
        assert graph.nodes["CH4-CAT"].status == TaskStatus.AVAILABLE
        # Others in ch4 should be blocked
        assert graph.nodes["CH4-FUNC"].status == TaskStatus.BLOCKED
        assert graph.nodes["CH4-NAT"].status == TaskStatus.BLOCKED
        # CH5 items should be blocked
        assert graph.nodes["CH5-TOP"].status == TaskStatus.BLOCKED

        # Mark CH4-CAT as done
        graph.update_status("CH4-CAT", TaskStatus.DONE, "art-cat")
        assert graph.nodes["CH4-FUNC"].status == TaskStatus.AVAILABLE

        # Mark CH4-FUNC as done
        graph.update_status("CH4-FUNC", TaskStatus.DONE, "art-func")
        assert graph.nodes["CH4-NAT"].status == TaskStatus.AVAILABLE

        # Mark CH4-NAT as done - should unblock CH5-TOP
        graph.update_status("CH4-NAT", TaskStatus.DONE, "art-nat")
        assert graph.nodes["CH5-TOP"].status == TaskStatus.AVAILABLE

        # Progress should be 3/5
        assert graph.progress() == 0.6
