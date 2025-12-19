# Specification: Interactive Society & Working Groups

**Status**: Draft v2
**Date**: 2025-12-18
**Authors**: Gemini 3 (initial), Claude Opus 4.5 (expansion)
**Goal**: Enable synchronous, conversational collaboration between agents to solve complex formalization tasks that require negotiation and rapid feedback.

---

## 1. The Problem: "Letter Writing" is Too Slow

The current architecture (v2) models agents as 18th-century mathematicians exchanging letters (artifacts) once per generation.

*   **Pros**: Highly parallel, scales linearly, robust to hallucinations (verification filters bad letters).
*   **Cons**: No negotiation. If Agent A defines `Category` slightly differently than Agent B, they don't find out until the "Conference" (Foundation merge) fails. They cannot say "Wait, let's align on structure X first."
*   **Result**: "Definition Drift" and wasted generations fixing import errors.

### 1.1 Evidence from Experiments

In the 2025-12-18 experiment (3 agents, 5 generations):
- Gen 0: 2 verified (TopologicalSpace)
- Gen 1-4: 0 verified (all attempting CH4-LIMITS)
- Root cause: Agents discovered via textbook that `Category` is a `structure` not `class`, but couldn't coordinate on the correct usage pattern.

The textbook showed clear collective debugging across generations, but the asynchronous nature meant 4 generations were spent on what could have been a 1-turn conversation.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GENERATION N                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         PLANNING PANEL                                  │ │
│  │                    (1 Chair + 3 Voting Members)                         │ │
│  │                                                                         │ │
│  │  Inputs:                                                                │ │
│  │  ├── DependencyGraph (what depends on what)                             │ │
│  │  ├── Foundation.lean (what's already verified)                          │ │
│  │  ├── Goal progress (what's [DONE] vs [TODO])                            │ │
│  │  └── Last generation's textbook (failures, insights)                    │ │
│  │                                                                         │ │
│  │  Process:                                                               │ │
│  │  1. Chair proposes task assignments                                     │ │
│  │  2. Members discuss and vote                                            │ │
│  │  3. Majority decides (Chair breaks ties)                                │ │
│  │                                                                         │ │
│  │  Outputs:                                                               │ │
│  │  └── List[WorkingGroupAssignment]                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ WorkingGroup │ │ WorkingGroup │ │ WorkingGroup │ │ WorkingGroup │       │
│  │      1       │ │      2       │ │      3       │ │     ...      │       │
│  │              │ │              │ │              │ │              │       │
│  │ Task: Limits │ │ Task: Adjoint│ │ Task: TopBas │ │              │       │
│  │ Members: 3   │ │ Members: 3   │ │ Members: 3   │ │              │       │
│  │ Turns: 5     │ │ Turns: 5     │ │ Turns: 5     │ │              │       │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘       │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        VERIFICATION PHASE                               │ │
│  │                                                                         │ │
│  │  For each WorkingGroup artifact:                                        │ │
│  │  1. Verify against Foundation.lean (MCP verifier)                       │ │
│  │  2. If success: merge into Foundation.lean                              │ │
│  │  3. If failure: log to textbook with error details                      │ │
│  │  4. Update DependencyGraph status                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Data Structures

### 3.1 DependencyGraph

The dependency graph tracks which definitions depend on which, enabling:
- Topological ordering (work on leaves first)
- Blocking detection (can't work on X until Y is done)
- Priority assignment (X unblocks 5 things, Y unblocks 1)

```python
# lms/dependency.py

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    DONE = "done"           # Verified and in Foundation.lean
    IN_PROGRESS = "in_progress"  # Assigned to a WorkingGroup this gen
    AVAILABLE = "available"  # Dependencies met, ready to work on
    BLOCKED = "blocked"      # Dependencies not yet met


@dataclass
class DependencyNode:
    """A single task in the dependency graph."""

    tag: str                          # "CH4-LIMITS"
    name: str                          # "Limits and Colimits"
    chapter: int                       # 4
    section: str                       # "4.4"
    requires: list[str] = field(default_factory=list)  # ["CH4-CAT", "CH4-FUNCTOR"]
    unlocks: list[str] = field(default_factory=list)   # ["CH6-PRESHEAF", "CH7-SITE"]
    status: TaskStatus = TaskStatus.BLOCKED
    artifact_id: str | None = None     # If DONE, which artifact solved it

    def priority_score(self) -> int:
        """Higher score = more things unlocked = higher priority."""
        return len(self.unlocks)


@dataclass
class DependencyGraph:
    """The full graph of tasks and their dependencies."""

    nodes: dict[str, DependencyNode] = field(default_factory=dict)

    def add_node(self, node: DependencyNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.tag] = node

    def update_status(self, tag: str, status: TaskStatus, artifact_id: str | None = None) -> None:
        """Update a node's status and recalculate available tasks."""
        if tag in self.nodes:
            self.nodes[tag].status = status
            self.nodes[tag].artifact_id = artifact_id
            self._recalculate_availability()

    def _recalculate_availability(self) -> None:
        """Recalculate which nodes are available based on dependencies."""
        for node in self.nodes.values():
            if node.status in (TaskStatus.DONE, TaskStatus.IN_PROGRESS):
                continue

            # Check if all dependencies are DONE
            all_deps_done = all(
                self.nodes.get(dep, DependencyNode(tag=dep, name="")).status == TaskStatus.DONE
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

    def done_tasks(self) -> list[DependencyNode]:
        """Return completed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.DONE]

    def progress(self) -> float:
        """Return fraction of tasks completed."""
        if not self.nodes:
            return 0.0
        done = len(self.done_tasks())
        return done / len(self.nodes)

    @classmethod
    def from_goal(cls, goal: "Goal") -> "DependencyGraph":
        """Build a DependencyGraph from a Goal's definitions."""
        graph = cls()

        # First pass: create all nodes
        for defn in goal.definitions:
            node = DependencyNode(
                tag=defn.tag,
                name=defn.name,
                chapter=int(defn.section.split(".")[0]) if "." in defn.section else 0,
                section=defn.section,
                status=TaskStatus.DONE if defn.formalized else TaskStatus.BLOCKED,
                artifact_id=defn.artifact_ids[0] if defn.artifact_ids else None,
            )
            graph.add_node(node)

        # Second pass: infer dependencies from chapter/section ordering
        # (Can be overridden with explicit requires in goal definitions)
        graph._infer_dependencies()
        graph._recalculate_availability()

        return graph

    def _infer_dependencies(self) -> None:
        """Infer dependencies based on chapter/section ordering."""
        # Group by chapter
        by_chapter: dict[int, list[DependencyNode]] = {}
        for node in self.nodes.values():
            by_chapter.setdefault(node.chapter, []).append(node)

        # Within each chapter, earlier sections are dependencies of later ones
        for chapter, nodes in by_chapter.items():
            sorted_nodes = sorted(nodes, key=lambda n: n.section)
            for i, node in enumerate(sorted_nodes):
                # Depend on all earlier nodes in same chapter
                node.requires = [n.tag for n in sorted_nodes[:i]]
                # Earlier nodes unlock this one
                for earlier in sorted_nodes[:i]:
                    if node.tag not in earlier.unlocks:
                        earlier.unlocks.append(node.tag)

            # First node of chapter N depends on last node of chapter N-1
            if chapter > 4 and sorted_nodes:  # Ch4 is the root
                prev_chapter_nodes = by_chapter.get(chapter - 1, [])
                if prev_chapter_nodes:
                    last_prev = sorted(prev_chapter_nodes, key=lambda n: n.section)[-1]
                    sorted_nodes[0].requires.append(last_prev.tag)
                    last_prev.unlocks.append(sorted_nodes[0].tag)

    def save(self, path: "Path") -> None:
        """Save graph to JSON."""
        import json
        data = {
            "nodes": [
                {
                    "tag": n.tag,
                    "name": n.name,
                    "chapter": n.chapter,
                    "section": n.section,
                    "requires": n.requires,
                    "unlocks": n.unlocks,
                    "status": n.status.value,
                    "artifact_id": n.artifact_id,
                }
                for n in self.nodes.values()
            ]
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: "Path") -> "DependencyGraph":
        """Load graph from JSON."""
        import json
        data = json.loads(path.read_text())
        graph = cls()
        for item in data["nodes"]:
            node = DependencyNode(
                tag=item["tag"],
                name=item["name"],
                chapter=item["chapter"],
                section=item["section"],
                requires=item["requires"],
                unlocks=item["unlocks"],
                status=TaskStatus(item["status"]),
                artifact_id=item.get("artifact_id"),
            )
            graph.add_node(node)
        return graph
```

### 3.2 PlanningPanel

The planning panel is a small group that decides how to allocate work each generation.

```python
# lms/planning.py

from dataclasses import dataclass, field
from enum import Enum


class Vote(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class WorkingGroupAssignment:
    """An assignment of a task to a working group."""

    group_id: int
    task_tag: str
    task_name: str
    priority: int  # 1 = highest
    guidance: str  # Specific advice from panel, e.g., "Use structure not class"
    backup_task: str | None = None  # If primary is solved quickly


@dataclass
class PlanningProposal:
    """A proposal from the Chair for how to allocate work."""

    assignments: list[WorkingGroupAssignment]
    rationale: str


@dataclass
class PanelVote:
    """A vote from a panel member on a proposal."""

    member_id: str
    vote: Vote
    comment: str  # Why they voted this way


@dataclass
class PlanningSession:
    """State for a planning panel session."""

    generation: int
    chair_id: str
    member_ids: list[str]  # 3 voting members

    # Inputs
    available_tasks: list[DependencyNode]
    last_gen_failures: list[str]  # Summary of what failed
    foundation_summary: str  # What's in Foundation.lean

    # Process
    proposals: list[PlanningProposal] = field(default_factory=list)
    votes: list[PanelVote] = field(default_factory=list)
    discussion: list[dict] = field(default_factory=list)  # Chat history

    # Output
    final_assignments: list[WorkingGroupAssignment] = field(default_factory=list)
    approved: bool = False


class PlanningPanel:
    """Orchestrates the planning panel discussion."""

    def __init__(
        self,
        provider: "Provider",
        graph: DependencyGraph,
        textbook: "Textbook",
        foundation_summary: str,
        n_groups: int = 3,
    ):
        self.provider = provider
        self.graph = graph
        self.textbook = textbook
        self.foundation_summary = foundation_summary
        self.n_groups = n_groups

    async def run_session(self, generation: int) -> list[WorkingGroupAssignment]:
        """Run a planning panel session and return assignments."""

        # 1. Gather context
        available = self.graph.available_tasks()
        if not available:
            return []  # Nothing to do

        # Get recent failures from textbook
        recent_failures = self._get_recent_failures()

        # 2. Create session
        session = PlanningSession(
            generation=generation,
            chair_id="panel-chair",
            member_ids=["panel-member-1", "panel-member-2", "panel-member-3"],
            available_tasks=available,
            last_gen_failures=recent_failures,
            foundation_summary=self.foundation_summary,
        )

        # 3. Chair proposes
        proposal = await self._get_chair_proposal(session)
        session.proposals.append(proposal)

        # 4. Members discuss and vote
        for member_id in session.member_ids:
            vote = await self._get_member_vote(session, member_id, proposal)
            session.votes.append(vote)

        # 5. Tally votes
        approvals = sum(1 for v in session.votes if v.vote == Vote.APPROVE)

        if approvals >= 2:  # Majority
            session.final_assignments = proposal.assignments
            session.approved = True
        else:
            # Chair revises based on feedback
            revised = await self._get_revised_proposal(session)
            session.proposals.append(revised)
            session.final_assignments = revised.assignments
            session.approved = True  # Chair decides on tie

        # 6. Mark tasks as in_progress
        for assignment in session.final_assignments:
            self.graph.update_status(assignment.task_tag, TaskStatus.IN_PROGRESS)

        return session.final_assignments

    def _get_recent_failures(self) -> list[str]:
        """Get summaries of recent failures from textbook."""
        failures = []
        for entry in self.textbook.entries[-20:]:  # Last 20 entries
            if "[FAILED]" in entry.title:
                failures.append(f"{entry.title}: {entry.content[:200]}...")
        return failures

    async def _get_chair_proposal(self, session: PlanningSession) -> PlanningProposal:
        """Get the Chair's initial proposal."""

        prompt = self._build_chair_prompt(session)
        response = await self.provider.generate([
            {"role": "system", "content": PLANNING_CHAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return self._parse_proposal(response)

    async def _get_member_vote(
        self, session: PlanningSession, member_id: str, proposal: PlanningProposal
    ) -> PanelVote:
        """Get a member's vote on the proposal."""

        prompt = self._build_vote_prompt(session, proposal)
        response = await self.provider.generate([
            {"role": "system", "content": PLANNING_MEMBER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return self._parse_vote(response, member_id)

    async def _get_revised_proposal(self, session: PlanningSession) -> PlanningProposal:
        """Get Chair's revised proposal after feedback."""

        prompt = self._build_revision_prompt(session)
        response = await self.provider.generate([
            {"role": "system", "content": PLANNING_CHAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return self._parse_proposal(response)

    def _build_chair_prompt(self, session: PlanningSession) -> str:
        """Build the prompt for the Chair."""

        available_str = "\n".join([
            f"- {t.tag}: {t.name} (unlocks {len(t.unlocks)} tasks, requires: {t.requires})"
            for t in session.available_tasks[:10]  # Top 10 by priority
        ])

        failures_str = "\n".join(session.last_gen_failures[:5])

        return f"""# Planning Panel - Generation {session.generation}

## Available Tasks (sorted by priority)
{available_str}

## Recent Failures to Learn From
{failures_str}

## Foundation Summary
{session.foundation_summary}

## Your Task
Propose assignments for {self.n_groups} working groups. Each group should:
1. Work on ONE primary task
2. Have a backup task if primary is completed quickly
3. Receive specific guidance based on past failures

Format your proposal as:
<proposal>
<rationale>Why this allocation makes sense</rationale>
<assignments>
<group id="1" task="TAG" backup="BACKUP_TAG" priority="1">
Specific guidance for this group...
</group>
...
</assignments>
</proposal>
"""

    def _build_vote_prompt(self, session: PlanningSession, proposal: PlanningProposal) -> str:
        """Build the prompt for a voting member."""

        return f"""# Planning Panel Vote - Generation {session.generation}

## Chair's Proposal
{proposal.rationale}

## Proposed Assignments
{self._format_assignments(proposal.assignments)}

## Your Task
Vote on this proposal. Consider:
1. Are the priorities correct?
2. Is the guidance helpful based on past failures?
3. Are there conflicts or duplications?

Format your vote as:
<vote>
<decision>APPROVE|REJECT|ABSTAIN</decision>
<comment>Your reasoning...</comment>
</vote>
"""

    def _format_assignments(self, assignments: list[WorkingGroupAssignment]) -> str:
        """Format assignments for display."""
        lines = []
        for a in assignments:
            lines.append(f"Group {a.group_id}: {a.task_tag} ({a.task_name})")
            lines.append(f"  Priority: {a.priority}")
            lines.append(f"  Backup: {a.backup_task or 'None'}")
            lines.append(f"  Guidance: {a.guidance}")
            lines.append("")
        return "\n".join(lines)

    def _parse_proposal(self, response: str) -> PlanningProposal:
        """Parse a proposal from LLM response."""
        # TODO: Implement XML parsing
        # For now, return a placeholder
        return PlanningProposal(assignments=[], rationale=response)

    def _parse_vote(self, response: str, member_id: str) -> PanelVote:
        """Parse a vote from LLM response."""
        # TODO: Implement XML parsing
        vote = Vote.APPROVE if "APPROVE" in response.upper() else Vote.REJECT
        return PanelVote(member_id=member_id, vote=vote, comment=response)


# Prompts for planning panel
PLANNING_CHAIR_SYSTEM_PROMPT = """You are the Chair of the LMS Planning Panel.

Your role is to allocate work to Working Groups for this generation. You do NOT write code.

You must:
1. Prioritize tasks that unblock the most downstream work
2. Learn from past failures - include specific guidance
3. Avoid assigning the same task to multiple groups
4. Consider dependencies - don't assign blocked tasks

You are decisive but open to feedback from panel members."""


PLANNING_MEMBER_SYSTEM_PROMPT = """You are a voting member of the LMS Planning Panel.

Your role is to review the Chair's proposal and vote. You do NOT write code.

You should:
1. Check for conflicts or duplications
2. Verify priorities make sense
3. Ensure guidance is actionable
4. Vote APPROVE if the proposal is reasonable, REJECT if it has serious flaws

Be constructive - if you REJECT, explain what should change."""
```

### 3.3 WorkingGroup

The working group manages a small team conversation.

```python
# lms/working_group.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable


class Role(Enum):
    CHAIR = "chair"       # Facilitates, doesn't write code
    SCRIBE = "scribe"     # Compiles final artifact
    RESEARCHER = "researcher"  # Proposes code, debates


@dataclass
class Message:
    """A message in the working group conversation."""

    sender_id: str
    role: Role
    content: str
    turn: int
    timestamp: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class WorkingGroupConfig:
    """Configuration for a working group."""

    group_id: int
    task_tag: str
    task_name: str
    task_content: str  # The full definition from Goal
    guidance: str      # From planning panel
    max_turns: int = 5
    members_per_role: dict[Role, int] = field(
        default_factory=lambda: {Role.CHAIR: 1, Role.SCRIBE: 1, Role.RESEARCHER: 1}
    )


@dataclass
class WorkingGroupState:
    """The evolving state of a working group session."""

    config: WorkingGroupConfig
    messages: list[Message] = field(default_factory=list)
    blackboard: str = ""  # Current shared draft
    current_turn: int = 0
    status: str = "discussing"  # "discussing", "drafting", "finalizing", "done"
    final_artifact: dict | None = None


class WorkingGroup:
    """Orchestrates a working group conversation."""

    def __init__(
        self,
        config: WorkingGroupConfig,
        provider: "Provider",
        foundation_summary: str,
    ):
        self.config = config
        self.provider = provider
        self.foundation_summary = foundation_summary
        self.state = WorkingGroupState(config=config)

        # Create member IDs
        self.members: list[tuple[str, Role]] = []
        for role, count in config.members_per_role.items():
            for i in range(count):
                member_id = f"group-{config.group_id}-{role.value}-{i}"
                self.members.append((member_id, role))

    async def run_session(self) -> dict | None:
        """Run the full working group session. Returns final artifact or None."""

        # Turn 0: Chair opens
        await self._chair_opening()

        # Turns 1 to N-1: Discussion
        for turn in range(1, self.config.max_turns - 1):
            self.state.current_turn = turn
            await self._discussion_round()

            # Check if we have consensus
            if self._has_consensus():
                break

        # Final turn: Scribe compiles
        self.state.status = "finalizing"
        await self._scribe_finalize()

        self.state.status = "done"
        return self.state.final_artifact

    async def _chair_opening(self) -> None:
        """Chair sets the agenda."""

        chair_id = next(m for m, r in self.members if r == Role.CHAIR)[0]

        prompt = f"""# Working Group {self.config.group_id} - Opening

## Your Task
{self.config.task_tag}: {self.config.task_name}

{self.config.task_content}

## Guidance from Planning Panel
{self.config.guidance}

## Foundation Summary
{self.foundation_summary}

## Your Role
You are the CHAIR. Set the agenda for this discussion:
1. Summarize what we need to accomplish
2. Propose a strategy (e.g., "Let's first agree on the structure signature")
3. Ask specific questions to guide the Researchers

Do NOT write code. Facilitate the discussion."""

        response = await self.provider.generate([
            {"role": "system", "content": CHAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        self.state.messages.append(Message(
            sender_id=chair_id,
            role=Role.CHAIR,
            content=response,
            turn=0,
        ))

    async def _discussion_round(self) -> None:
        """One round of discussion among all members."""

        # Each researcher responds
        researchers = [(m, r) for m, r in self.members if r == Role.RESEARCHER]

        for member_id, role in researchers:
            context = self._build_context()

            prompt = f"""{context}

## Your Turn
You are a RESEARCHER. Based on the discussion so far:
1. Propose code or critique existing proposals
2. Reference Foundation.lean definitions correctly
3. Be specific about types, universes, and structure signatures

If you agree with the current blackboard draft, say "I agree with the current proposal."
If you have changes, provide the updated code."""

            response = await self.provider.generate([
                {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])

            self.state.messages.append(Message(
                sender_id=member_id,
                role=Role.RESEARCHER,
                content=response,
                turn=self.state.current_turn,
            ))

            # Update blackboard if code was proposed
            if "```lean" in response:
                self._update_blackboard(response)

        # Chair summarizes
        await self._chair_summary()

    async def _chair_summary(self) -> None:
        """Chair summarizes the round and checks for consensus."""

        chair_id = next(m for m, r in self.members if r == Role.CHAIR)[0]
        context = self._build_context()

        prompt = f"""{context}

## Your Turn (Chair Summary)
Summarize this round:
1. What was agreed upon?
2. What disagreements remain?
3. Is the group ready to finalize?

If ready, say "CONSENSUS REACHED" and summarize the agreed approach.
If not, pose the next question to resolve."""

        response = await self.provider.generate([
            {"role": "system", "content": CHAIR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        self.state.messages.append(Message(
            sender_id=chair_id,
            role=Role.CHAIR,
            content=response,
            turn=self.state.current_turn,
        ))

    async def _scribe_finalize(self) -> None:
        """Scribe compiles the final artifact."""

        scribe_id = next(m for m, r in self.members if r == Role.SCRIBE)[0]
        context = self._build_context()

        prompt = f"""{context}

## Your Turn (Final Compilation)
You are the SCRIBE. Compile the final artifact:
1. Use the agreed-upon code from the blackboard
2. Ensure all imports are correct
3. Format as a proper <artifact> block
4. Include notes summarizing the group's discussion

The artifact MUST be complete and ready for LEAN verification."""

        response = await self.provider.generate([
            {"role": "system", "content": SCRIBE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        self.state.messages.append(Message(
            sender_id=scribe_id,
            role=Role.SCRIBE,
            content=response,
            turn=self.state.current_turn + 1,
        ))

        # Parse the artifact
        self.state.final_artifact = self._parse_artifact(response)

    def _build_context(self) -> str:
        """Build context from conversation history."""

        lines = [
            f"# Working Group {self.config.group_id}",
            f"## Task: {self.config.task_tag} - {self.config.task_name}",
            "",
            "## Conversation History",
        ]

        for msg in self.state.messages:
            lines.append(f"### [{msg.role.value.upper()}] Turn {msg.turn}")
            lines.append(msg.content)
            lines.append("")

        if self.state.blackboard:
            lines.append("## Current Blackboard (Draft Code)")
            lines.append("```lean")
            lines.append(self.state.blackboard)
            lines.append("```")

        return "\n".join(lines)

    def _update_blackboard(self, response: str) -> None:
        """Extract code from response and update blackboard."""
        import re
        match = re.search(r"```lean\n(.*?)```", response, re.DOTALL)
        if match:
            self.state.blackboard = match.group(1)

    def _has_consensus(self) -> bool:
        """Check if the group has reached consensus."""
        last_msg = self.state.messages[-1] if self.state.messages else None
        if last_msg and "CONSENSUS REACHED" in last_msg.content.upper():
            return True
        return False

    def _parse_artifact(self, response: str) -> dict | None:
        """Parse artifact from Scribe's response."""
        # Reuse existing artifact parsing logic
        # TODO: Import from agent.py
        return {"raw": response, "blackboard": self.state.blackboard}


# Role-specific system prompts
CHAIR_SYSTEM_PROMPT = """You are the CHAIR of an LMS Working Group.

Your role is to FACILITATE, not to write code:
1. Keep the group focused on the assigned task
2. Summarize agreements and disagreements
3. Identify when consensus is reached
4. Ask clarifying questions

You are neutral and ensure all voices are heard."""


RESEARCHER_SYSTEM_PROMPT = """You are a RESEARCHER in an LMS Working Group.

Your role is to propose and critique code:
1. Write LEAN 4 code that addresses the task
2. Use existing Foundation.lean definitions correctly
3. Debate with colleagues - disagree if you see issues
4. Be specific about types, universes, and structure signatures

Do NOT use `sorry`. Only propose complete, verifiable code."""


SCRIBE_SYSTEM_PROMPT = """You are the SCRIBE of an LMS Working Group.

Your role is to compile the final artifact:
1. Take the agreed-upon code from the discussion
2. Format it as a proper <artifact> block
3. Ensure imports and namespace are correct
4. Add notes summarizing the group's key decisions

The artifact must be ready for LEAN verification."""
```

---

## 4. Integration with Existing System

### 4.1 Modified Society Class

```python
# In lms/society.py - new method

async def run_generation_with_groups(self, generation: int) -> GenerationResult:
    """Run a generation using the Working Group architecture."""

    # 1. Planning Phase
    panel = PlanningPanel(
        provider=self.provider,
        graph=self.dependency_graph,
        textbook=self.textbook,
        foundation_summary=self._get_foundation_summary(),
        n_groups=self.config.n_working_groups,
    )

    assignments = await panel.run_session(generation)

    if not assignments:
        return GenerationResult(generation=generation, artifacts=[], message="No tasks available")

    # 2. Working Group Phase (parallel)
    groups = []
    for assignment in assignments:
        config = WorkingGroupConfig(
            group_id=assignment.group_id,
            task_tag=assignment.task_tag,
            task_name=assignment.task_name,
            task_content=self._get_task_content(assignment.task_tag),
            guidance=assignment.guidance,
        )
        group = WorkingGroup(
            config=config,
            provider=self.provider,
            foundation_summary=self._get_foundation_summary(),
        )
        groups.append(group)

    # Run all groups in parallel
    results = await asyncio.gather(*[g.run_session() for g in groups])

    # 3. Verification Phase
    artifacts = []
    for result, group in zip(results, groups):
        if result and result.get("blackboard"):
            artifact = self._create_artifact_from_group(group, result)
            verified = await self.verifier.verify(artifact)

            if verified:
                self._add_to_foundation(artifact)
                self.dependency_graph.update_status(
                    group.config.task_tag,
                    TaskStatus.DONE,
                    artifact["id"]
                )
            else:
                self.dependency_graph.update_status(
                    group.config.task_tag,
                    TaskStatus.AVAILABLE  # Can try again next gen
                )

            artifacts.append(artifact)

    # 4. Update textbook with group discussions
    for group in groups:
        self._add_group_to_textbook(group)

    return GenerationResult(
        generation=generation,
        artifacts=artifacts,
        verified=sum(1 for a in artifacts if a.get("verified")),
    )
```

### 4.2 Configuration

```python
# In lms/config.py - new fields

@dataclass
class SocietyConfig:
    # Existing fields...

    # Working Group settings
    use_working_groups: bool = False
    n_working_groups: int = 3
    group_size: int = 3  # Members per group (1 chair + 1 scribe + N-2 researchers)
    max_turns_per_group: int = 5
    use_planning_panel: bool = True
```

---

## 5. Experiments

### 5.1 A/B Test: Async vs Working Groups

| Metric | Async (Control) | Working Groups (Exp) |
|--------|-----------------|----------------------|
| Agents | 15 | 9 (3 groups × 3) |
| Generations | 5 | 5 |
| Tokens/gen | ~800K | ~2.4M (3× more) |
| Expected verifications | 2-3 | 3-5 |
| Definition drift | High | Low |

### 5.2 Validation Criteria

The Working Groups architecture is successful if:
1. **Higher verification rate**: >20% of artifacts verify (vs current ~13%)
2. **Less definition drift**: Groups don't redefine existing structures
3. **Faster convergence**: Hard problems (CH4-LIMITS) solved in fewer generations
4. **Better knowledge transfer**: Textbook entries are more actionable

---

## 6. Future Extensions

### 6.1 Dynamic Group Formation
Instead of fixed groups, form groups based on task similarity or past collaboration success.

### 6.2 Cross-Group Communication
Allow groups to "consult" each other mid-session for dependency questions.

### 6.3 Specialist Roles
Add domain-specific roles: "Universe Expert", "Tactic Specialist", etc.

---

## Appendix A: File Structure

```
lms/
├── dependency.py      # DependencyGraph, DependencyNode, TaskStatus
├── planning.py        # PlanningPanel, PlanningSession, WorkingGroupAssignment
├── working_group.py   # WorkingGroup, WorkingGroupState, Message, Role
├── society.py         # Modified to support run_generation_with_groups()
├── config.py          # Extended with SocietyConfig.use_working_groups
└── prompts.py         # New prompts for Chair, Scribe, Researcher roles
```

## Appendix B: CLI Changes

```bash
# New flags for run.py
python -m lms.run \
  --provider google \
  --agents 9 \
  --generations 5 \
  --goal stacks-ch4-done \
  --working-groups \           # Enable working group mode
  --n-groups 3 \               # Number of parallel groups
  --group-turns 5              # Max turns per group
```
