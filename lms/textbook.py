"""Collective textbook - accumulated wisdom searchable by keyword.

Agents write entries with descriptive titles that help future agents
find relevant knowledge. Titles should answer: "What would have helped
me at the start if I had known this?"
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextbookEntry:
    """A single piece of wisdom in the textbook.

    Agents should write entries with:
    - title: A descriptive, searchable title (what would have helped you?)
    - content: Full explanation (not truncated)
    - topics: Tags for categorical search (goal tags, "error", "strategy", etc.)
    """

    content: str
    author: str
    generation: int
    topics: list[str] = field(default_factory=list)  # Keywords/tags
    title: str = ""  # Descriptive title for searchability
    entry_type: str = "insight"  # "insight", "error", "strategy", "success", "writeup"


class Textbook:
    """Searchable collection of accumulated wisdom.

    Agents can add insights and search for relevant wisdom.
    Keeps context small by only returning relevant entries.
    """

    def __init__(self) -> None:
        self.entries: list[TextbookEntry] = []

    def add(
        self,
        content: str,
        author: str,
        generation: int,
        topics: list[str] | None = None,
        title: str = "",
        entry_type: str = "insight",
    ) -> None:
        """Add an insight to the textbook.

        Args:
            content: Full content (not truncated)
            author: Agent ID who wrote this
            generation: Generation number
            topics: Tags for search (goal tags, "error", etc.)
            title: Descriptive title - what would help future agents find this?
            entry_type: Type of entry ("insight", "error", "strategy", "success", "writeup")
        """
        self.entries.append(TextbookEntry(
            content=content,
            author=author,
            generation=generation,
            topics=topics or [],
            title=title,
            entry_type=entry_type,
        ))

    def search(self, query: str, max_results: int = 5) -> list[TextbookEntry]:
        """Search for relevant entries by keyword.

        Searches title, content, and topics. Title matches are prioritized.
        """
        query_lower = query.lower()
        title_matches = []
        content_matches = []

        for entry in self.entries:
            # Title matches are highest priority
            if entry.title and query_lower in entry.title.lower():
                title_matches.append(entry)
            # Check content and topics
            elif query_lower in entry.content.lower():
                content_matches.append(entry)
            elif any(query_lower in topic.lower() for topic in entry.topics):
                content_matches.append(entry)

        # Combine: title matches first, then content matches
        matches = title_matches + content_matches

        # Return most recent matches first within each priority
        matches.sort(key=lambda e: (-1 if e in title_matches else 0, -e.generation))
        return matches[:max_results]

    def list_titles(self, entry_type: str | None = None) -> list[tuple[str, str, int]]:
        """List all entry titles for browsing.

        Args:
            entry_type: Optional filter by entry type

        Returns:
            List of (title, author, generation) tuples
        """
        entries = self.entries
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]

        return [
            (e.title or f"[Untitled: {e.content[:50]}...]", e.author, e.generation)
            for e in sorted(entries, key=lambda e: -e.generation)
        ]

    def get_for_context(self, tags: list[str] | None = None, max_tokens: int = 800) -> str:
        """Get relevant textbook entries for prompt context.

        Shows titles first so agents can identify relevant entries,
        then includes full content for the most relevant ones.

        Args:
            tags: Optional stacks tags to search for
            max_tokens: Rough limit on output size (chars / 4)

        Returns:
            Formatted string with browsable titles and relevant content
        """
        if not self.entries:
            return ""

        relevant = []

        # If tags provided, search for them
        if tags:
            for tag in tags:
                relevant.extend(self.search(tag, max_results=3))

        # Also include recent successes
        successes = [e for e in self.entries if e.entry_type == "success"]
        recent_successes = sorted(successes, key=lambda e: e.generation, reverse=True)[:2]
        for entry in recent_successes:
            if entry not in relevant:
                relevant.append(entry)

        # Also include recent general insights
        recent = sorted(self.entries, key=lambda e: e.generation, reverse=True)[:3]
        for entry in recent:
            if entry not in relevant:
                relevant.append(entry)

        if not relevant:
            return ""

        # Format with titles for browsability
        lines = ["## Collective Wisdom (from previous generations)"]
        lines.append("")
        lines.append("### Available Entries (by title):")

        # Show all titles so agents can see what's available
        all_titles = self.list_titles()[:15]  # Limit to recent 15
        for title, author, gen in all_titles:
            lines.append(f"  - [{gen}] {title}")

        lines.append("")
        lines.append("### Relevant Details:")

        char_count = len("\n".join(lines))
        max_chars = max_tokens * 4

        for entry in relevant:
            title_part = f"**{entry.title}**\n" if entry.title else ""
            entry_text = f"\n- [{entry.author}, gen {entry.generation}] {title_part}{entry.content}"
            if char_count + len(entry_text) > max_chars:
                break
            lines.append(entry_text)
            char_count += len(entry_text)

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save textbook to JSON."""
        data = {
            "entries": [
                {
                    "content": e.content,
                    "author": e.author,
                    "generation": e.generation,
                    "topics": e.topics,
                    "title": e.title,
                    "entry_type": e.entry_type,
                }
                for e in self.entries
            ]
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Textbook":
        """Load textbook from JSON."""
        data = json.loads(path.read_text())
        textbook = cls()
        for entry_data in data["entries"]:
            # Handle old format without title/entry_type
            if "title" not in entry_data:
                entry_data["title"] = ""
            if "entry_type" not in entry_data:
                entry_data["entry_type"] = "insight"
            textbook.entries.append(TextbookEntry(**entry_data))
        return textbook

    def __len__(self) -> int:
        return len(self.entries)
