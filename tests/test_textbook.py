"""Tests for collective textbook - accumulated wisdom."""

import json
from pathlib import Path

import pytest

from lms.textbook import Textbook, TextbookEntry


class TestTextbookEntry:
    """Tests for TextbookEntry dataclass."""

    def test_create_entry(self):
        """TextbookEntry holds all fields."""
        entry = TextbookEntry(
            content="Use funext for function extensionality",
            author="agent-0-anthropic",
            generation=5,
            topics=["DEF-HOMFUNCTOR", "tactic"],
            title="How to prove HomFunctor.map_id",
            entry_type="success",
        )
        assert entry.content == "Use funext for function extensionality"
        assert entry.author == "agent-0-anthropic"
        assert entry.title == "How to prove HomFunctor.map_id"
        assert entry.entry_type == "success"

    def test_entry_defaults(self):
        """TextbookEntry has sensible defaults."""
        entry = TextbookEntry(
            content="Some insight",
            author="agent-1",
            generation=0,
        )
        assert entry.topics == []
        assert entry.title == ""
        assert entry.entry_type == "insight"


class TestTextbook:
    """Tests for Textbook class."""

    def test_add_entry(self):
        """Can add entries to textbook."""
        textbook = Textbook()
        textbook.add(
            content="Important lesson learned",
            author="agent-0",
            generation=3,
            topics=["error"],
            title="Namespace collisions cause ambiguous term errors",
            entry_type="error",
        )

        assert len(textbook) == 1
        assert textbook.entries[0].title == "Namespace collisions cause ambiguous term errors"
        assert textbook.entries[0].entry_type == "error"

    def test_search_by_content(self):
        """Search finds entries matching content."""
        textbook = Textbook()
        textbook.add("Use funext for function proofs", "agent-0", 1)
        textbook.add("Namespace issues are common", "agent-1", 2)
        textbook.add("Category theory basics", "agent-2", 3)

        results = textbook.search("funext")
        assert len(results) == 1
        assert "funext" in results[0].content

    def test_search_by_topic(self):
        """Search finds entries matching topics."""
        textbook = Textbook()
        textbook.add("Some content", "agent-0", 1, topics=["DEF-CAT"])
        textbook.add("Other content", "agent-1", 2, topics=["LEM-YONEDA"])

        results = textbook.search("DEF-CAT")
        assert len(results) == 1

    def test_search_prioritizes_title(self):
        """Search prioritizes title matches over content matches."""
        textbook = Textbook()
        textbook.add(
            content="Something about funext",
            author="agent-0",
            generation=1,
            title="Category theory basics",
        )
        textbook.add(
            content="Other content here",
            author="agent-1",
            generation=2,
            title="How to use funext correctly",
        )

        results = textbook.search("funext")
        # Title match should come first
        assert results[0].title == "How to use funext correctly"

    def test_list_titles(self):
        """Can list all titles for browsing."""
        textbook = Textbook()
        textbook.add("Content 1", "agent-0", 1, title="First entry")
        textbook.add("Content 2", "agent-1", 2, title="Second entry")
        textbook.add("Content 3", "agent-0", 3, title="")  # No title

        titles = textbook.list_titles()
        assert len(titles) == 3
        # Most recent first
        assert titles[0][0].startswith("[Untitled:")  # gen 3, no title
        assert titles[1][0] == "Second entry"  # gen 2
        assert titles[2][0] == "First entry"  # gen 1

    def test_list_titles_by_type(self):
        """Can filter titles by entry type."""
        textbook = Textbook()
        textbook.add("s1", "a", 1, title="Success 1", entry_type="success")
        textbook.add("e1", "a", 2, title="Error 1", entry_type="error")
        textbook.add("s2", "a", 3, title="Success 2", entry_type="success")

        successes = textbook.list_titles(entry_type="success")
        assert len(successes) == 2

        errors = textbook.list_titles(entry_type="error")
        assert len(errors) == 1

    def test_get_for_context_includes_titles(self):
        """Context for prompts includes browsable titles."""
        textbook = Textbook()
        textbook.add("Content 1", "agent-0", 1, title="How to fix namespace errors")
        textbook.add("Content 2", "agent-1", 2, title="Using funext correctly")

        context = textbook.get_for_context(max_tokens=500)

        assert "Available Entries" in context
        assert "How to fix namespace errors" in context
        assert "Using funext correctly" in context

    def test_get_for_context_prioritizes_successes(self):
        """Context includes recent successes."""
        textbook = Textbook()
        textbook.add("Failed attempt", "a", 1, entry_type="writeup")
        textbook.add("Success story", "a", 2, entry_type="success", title="Breakthrough")
        textbook.add("Another writeup", "a", 3, entry_type="writeup")

        context = textbook.get_for_context(max_tokens=800)

        # Success should be included
        assert "Success story" in context or "Breakthrough" in context

    def test_save_and_load(self, tmp_path: Path):
        """Textbook can be saved and loaded."""
        textbook = Textbook()
        textbook.add(
            content="Important lesson",
            author="agent-0",
            generation=5,
            topics=["DEF-CAT", "error"],
            title="How to fix Category definition",
            entry_type="success",
        )
        textbook.add(
            content="Another entry",
            author="agent-1",
            generation=6,
        )

        path = tmp_path / "textbook.json"
        textbook.save(path)

        # Verify file contents
        data = json.loads(path.read_text())
        assert len(data["entries"]) == 2
        assert data["entries"][0]["title"] == "How to fix Category definition"
        assert data["entries"][0]["entry_type"] == "success"

        # Load and verify
        loaded = Textbook.load(path)
        assert len(loaded) == 2
        assert loaded.entries[0].title == "How to fix Category definition"
        assert loaded.entries[0].entry_type == "success"

    def test_load_old_format(self, tmp_path: Path):
        """Can load old format without title/entry_type."""
        # Simulate old format
        old_data = {
            "entries": [
                {
                    "content": "Old entry",
                    "author": "agent-0",
                    "generation": 1,
                    "topics": ["test"],
                }
            ]
        }
        path = tmp_path / "textbook.json"
        path.write_text(json.dumps(old_data))

        # Should load without error
        loaded = Textbook.load(path)
        assert len(loaded) == 1
        assert loaded.entries[0].title == ""  # Default
        assert loaded.entries[0].entry_type == "insight"  # Default
