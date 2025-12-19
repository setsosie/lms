"""Dependency graph for goal-directed formalization.

The dependency graph tracks which definitions depend on which, enabling:
- Topological ordering (work on leaves first)
- Blocking detection (can't work on X until Y is done)
- Priority assignment (X unblocks 5 things, Y unblocks 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lms.goals import Goal


class TaskStatus(Enum):
    """Status of a task in the dependency graph."""

    DONE = "done"  # Verified and in Foundation.lean
    IN_PROGRESS = "in_progress"  # Assigned to a WorkingGroup this gen
    AVAILABLE = "available"  # Dependencies met, ready to work on
    BLOCKED = "blocked"  # Dependencies not yet met


@dataclass
class DependencyNode:
    """A single task in the dependency graph."""

    tag: str  # "CH4-LIMITS"
    name: str  # "Limits and Colimits"
    chapter: int  # 4
    section: str  # "4.4"
    requires: list[str] = field(default_factory=list)  # ["CH4-CAT", "CH4-FUNCTOR"]
    unlocks: list[str] = field(default_factory=list)  # ["CH6-PRESHEAF", "CH7-SITE"]
    status: TaskStatus = TaskStatus.BLOCKED
    artifact_id: str | None = None  # If DONE, which artifact solved it

    def priority_score(self) -> int:
        """Higher score = more things unlocked = higher priority."""
        return len(self.unlocks)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tag": self.tag,
            "name": self.name,
            "chapter": self.chapter,
            "section": self.section,
            "requires": self.requires,
            "unlocks": self.unlocks,
            "status": self.status.value,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DependencyNode:
        """Create from dictionary."""
        return cls(
            tag=data["tag"],
            name=data["name"],
            chapter=data["chapter"],
            section=data["section"],
            requires=data.get("requires", []),
            unlocks=data.get("unlocks", []),
            status=TaskStatus(data["status"]),
            artifact_id=data.get("artifact_id"),
        )


@dataclass
class DependencyGraph:
    """The full graph of tasks and their dependencies."""

    nodes: dict[str, DependencyNode] = field(default_factory=dict)

    def add_node(self, node: DependencyNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.tag] = node

    def get_node(self, tag: str) -> DependencyNode | None:
        """Get a node by tag."""
        return self.nodes.get(tag)

    def update_status(
        self, tag: str, status: TaskStatus, artifact_id: str | None = None
    ) -> None:
        """Update a node's status and recalculate available tasks."""
        if tag in self.nodes:
            self.nodes[tag].status = status
            if artifact_id is not None:
                self.nodes[tag].artifact_id = artifact_id
            self._recalculate_availability()

    def _recalculate_availability(self) -> None:
        """Recalculate which nodes are available based on dependencies."""
        for node in self.nodes.values():
            if node.status in (TaskStatus.DONE, TaskStatus.IN_PROGRESS):
                continue

            # Check if all dependencies are DONE
            all_deps_done = all(
                self.nodes.get(dep) is not None
                and self.nodes[dep].status == TaskStatus.DONE
                for dep in node.requires
            )

            if all_deps_done:
                node.status = TaskStatus.AVAILABLE
            else:
                node.status = TaskStatus.BLOCKED

    def available_tasks(self) -> list[DependencyNode]:
        """Return tasks that are ready to be worked on, sorted by priority."""
        available = [n for n in self.nodes.values() if n.status == TaskStatus.AVAILABLE]
        return sorted(available, key=lambda n: n.priority_score(), reverse=True)

    def blocked_tasks(self) -> list[DependencyNode]:
        """Return tasks that are waiting on dependencies."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.BLOCKED]

    def in_progress_tasks(self) -> list[DependencyNode]:
        """Return tasks currently being worked on."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.IN_PROGRESS]

    def done_tasks(self) -> list[DependencyNode]:
        """Return completed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.DONE]

    def progress(self) -> float:
        """Return fraction of tasks completed."""
        if not self.nodes:
            return 0.0
        done = len(self.done_tasks())
        return done / len(self.nodes)

    def progress_summary(self) -> str:
        """Return a human-readable progress summary."""
        total = len(self.nodes)
        done = len(self.done_tasks())
        available = len(self.available_tasks())
        blocked = len(self.blocked_tasks())
        in_progress = len(self.in_progress_tasks())
        return (
            f"Progress: {done}/{total} ({self.progress():.0%}) | "
            f"Available: {available} | In Progress: {in_progress} | Blocked: {blocked}"
        )

    @classmethod
    def from_goal(cls, goal: Goal) -> DependencyGraph:
        """Build a DependencyGraph from a Goal's definitions.

        This infers dependencies from section ordering within chapters,
        and makes first nodes of later chapters depend on last nodes of
        earlier chapters.
        """
        graph = cls()

        # First pass: create all nodes
        for defn in goal.definitions:
            # Parse chapter from section (e.g., "4.4" -> 4)
            try:
                chapter = int(defn.section.split(".")[0]) if "." in defn.section else 0
            except (ValueError, IndexError):
                chapter = 0

            node = DependencyNode(
                tag=defn.tag,
                name=defn.name,
                chapter=chapter,
                section=defn.section,
                status=TaskStatus.DONE if defn.formalized else TaskStatus.BLOCKED,
                artifact_id=defn.artifact_ids[0] if defn.artifact_ids else None,
            )
            graph.add_node(node)

        # Second pass: infer dependencies from chapter/section ordering
        graph._infer_dependencies()
        graph._recalculate_availability()

        return graph

    def _infer_dependencies(self) -> None:
        """Infer dependencies based on chapter/section ordering.

        Within each chapter, earlier sections are dependencies of later ones.
        The first node of chapter N depends on the last node of chapter N-1.
        """
        # Group by chapter
        by_chapter: dict[int, list[DependencyNode]] = {}
        for node in self.nodes.values():
            by_chapter.setdefault(node.chapter, []).append(node)

        # Sort chapters
        sorted_chapters = sorted(by_chapter.keys())

        for chapter in sorted_chapters:
            nodes = by_chapter[chapter]
            # Sort by section string (works for "4.1", "4.2", etc.)
            sorted_nodes = sorted(nodes, key=lambda n: n.section)

            for i, node in enumerate(sorted_nodes):
                # Reset requires and unlocks for fresh inference
                node.requires = []
                node.unlocks = []

            for i, node in enumerate(sorted_nodes):
                # Depend on all earlier nodes in same chapter
                for earlier in sorted_nodes[:i]:
                    if earlier.tag not in node.requires:
                        node.requires.append(earlier.tag)
                    if node.tag not in earlier.unlocks:
                        earlier.unlocks.append(node.tag)

            # First node of chapter N depends on last node of chapter N-1
            if chapter > min(sorted_chapters) and sorted_nodes:
                prev_chapter = max(c for c in sorted_chapters if c < chapter)
                prev_chapter_nodes = by_chapter.get(prev_chapter, [])
                if prev_chapter_nodes:
                    last_prev = sorted(prev_chapter_nodes, key=lambda n: n.section)[-1]
                    first_curr = sorted_nodes[0]
                    if last_prev.tag not in first_curr.requires:
                        first_curr.requires.append(last_prev.tag)
                    if first_curr.tag not in last_prev.unlocks:
                        last_prev.unlocks.append(first_curr.tag)

    def save(self, path: Path) -> None:
        """Save graph to JSON."""
        data = {"nodes": [n.to_dict() for n in self.nodes.values()]}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> DependencyGraph:
        """Load graph from JSON."""
        data = json.loads(path.read_text())
        graph = cls()
        for item in data["nodes"]:
            node = DependencyNode.from_dict(item)
            graph.add_node(node)
        return graph

    def to_prompt_context(self) -> str:
        """Generate context string for agent prompts."""
        lines = ["## Dependency Graph Status", ""]

        # Group by status
        done = self.done_tasks()
        available = self.available_tasks()
        in_progress = self.in_progress_tasks()
        blocked = self.blocked_tasks()

        lines.append(f"### Completed ({len(done)})")
        for node in done[:10]:  # Limit output
            lines.append(f"- [DONE] {node.tag}: {node.name}")
        if len(done) > 10:
            lines.append(f"  ... and {len(done) - 10} more")

        lines.append(f"\n### Available ({len(available)})")
        for node in available:
            lines.append(
                f"- [AVAILABLE] {node.tag}: {node.name} (unlocks {len(node.unlocks)} tasks)"
            )

        lines.append(f"\n### In Progress ({len(in_progress)})")
        for node in in_progress:
            lines.append(f"- [IN_PROGRESS] {node.tag}: {node.name}")

        lines.append(f"\n### Blocked ({len(blocked)})")
        for node in blocked[:5]:  # Limit output
            requires_str = ", ".join(node.requires[:3])
            if len(node.requires) > 3:
                requires_str += f", ... +{len(node.requires) - 3} more"
            lines.append(f"- [BLOCKED] {node.tag}: {node.name} (needs: {requires_str})")
        if len(blocked) > 5:
            lines.append(f"  ... and {len(blocked) - 5} more blocked tasks")

        return "\n".join(lines)
