"""Foundation file management - accumulated verified Lean code.

The foundation file is the collective memory of verified definitions that
agents can import and build upon. Each generation adds verified artifacts
to this shared foundation, allowing subsequent generations to reuse them
via `import LMS.Foundation` instead of re-implementing from scratch.

This implements the "collective brain" vision: knowledge accumulates across
generations, with each agent building on the verified work of predecessors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lms.artifacts import Artifact

#: A `by` ending a line opens a tactic block, whether written `:= by`,
#: `(by`, or bare `by`.
_BY_TAIL = re.compile(r"\bby$")


@dataclass
class FoundationEntry:
    """A single definition/theorem in the foundation file.

    Attributes:
        artifact_id: ID of the artifact that contributed this entry
        name: Name of the definition (e.g., "Category", "CFunctor")
        entry_type: Type of entry (structure, def, theorem, lemma, etc.)
        signature: The type signature line
        lean_code: Full Lean code for this entry
        generation: Generation when this was added
        author: Agent that created this
    """

    artifact_id: str
    name: str
    entry_type: str  # structure, def, theorem, lemma, etc.
    signature: str
    lean_code: str
    generation: int
    author: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "entry_type": self.entry_type,
            "signature": self.signature,
            "lean_code": self.lean_code,
            "generation": self.generation,
            "author": self.author,
        }

    #: Declarations whose indented body *is* the API an agent needs. An
    #: `instance ... where` qualifies too; that is decided by the `where`
    #: suffix in `body_is_api()`, not by the keyword.
    BODY_IS_API = frozenset({"structure", "class", "inductive"})

    #: Declarations whose body is a proof or a value -- signature only.
    SIGNATURE_ONLY = frozenset({"theorem", "lemma", "def", "abbrev", "instance"})

    #: Keywords that begin a *new* declaration or command. `_extract_entries`
    #: slices to the next `DEFINITION_PATTERN` match, and that pattern misses
    #: these -- so without this stop list they render as fields of whatever
    #: entry precedes them.
    FOREIGN_TOKENS = frozenset(
        {
            "section",
            "end",
            "variable",
            "variables",
            "open",
            "namespace",
            "example",
            "noncomputable",
            "private",
            "protected",
            "partial",
            "mutual",
            "attribute",
            "deriving",
            "macro",
            "notation",
            "syntax",
            # Declaration keywords too: an `@[ext] theorem` is indented and
            # carries an attribute, so `DEFINITION_PATTERN`'s `^\s*` never
            # matches it and it is never extracted as its own entry.
            "theorem",
            "lemma",
            "def",
            "structure",
            "class",
            "instance",
            "abbrev",
            "inductive",
        }
    )

    #: Bracket pairs `:=` can hide inside. Lean 4 named arguments (`(f := f)`)
    #: and autoparams (`(h : P := by simp)` ) sit at depth > 0.
    _OPEN_BRACKETS = "([{⟨"
    _CLOSE_BRACKETS = ")]}⟩"

    def _code(self) -> str:
        """`lean_code` with any leading comments removed.

        `lean_code` does not begin at the declaration. `_strip_block_comments`
        blanks `/-- ... -/` to spaces before matching, and
        `DEFINITION_PATTERN`'s leading `^\\s*` then reaches back across the
        blanked region -- but `_extract_entries` slices the *original* source
        from that offset. So every doc-commented declaration carries its
        comment as line 1, and roughly half the corpus is doc-commented.

        Fixing the slice boundary instead would change what `save()` writes
        into `Foundation.lean`, which the accumulated corpus cannot absorb.

        Like `_strip_block_comments`, this does not handle *nested* block
        comments. The two agree by construction: a nested comment leaves
        residual non-whitespace that stops `DEFINITION_PATTERN` reaching back,
        so `lean_code` starts at the declaration and this method is not
        needed. Make one of them nest and the other must follow.
        """
        text = self.lean_code
        while True:
            stripped = text.lstrip()
            if stripped.startswith("/-"):
                close = stripped.find("-/")
                if close == -1:
                    return ""
                text = stripped[close + 2 :]
                continue
            if stripped.startswith("--"):
                newline = stripped.find("\n")
                if newline == -1:
                    return ""
                text = stripped[newline + 1 :]
                continue
            return stripped

    def declaration_header(self) -> str:
        """The declaration line as LEAN accepted it.

        Read from `lean_code`, never from `signature`. `signature` is a lossy
        reconstruction: `_extract_entries` strips the whitespace between the
        name and its parameters, so `Category (obj : Type u)` comes back as
        `Category(obj : Type u)`. `lean_code` is what was verified and what
        `save()` writes, so it cannot drift from the file agents are told to
        import.
        """
        for line in self._code().splitlines():
            if line.strip():
                return line.strip()
        return ""

    def body_is_api(self) -> bool:
        """Whether the indented body is the declaration's API surface.

        Keyed on the `where` suffix as well as the keyword: an
        `instance TypeCat : Category (Type u) where` has fields exactly like a
        structure, and classifying it as signature-only rendered one bare
        field name and silently dropped the other five.
        """
        return (
            self.entry_type in self.BODY_IS_API
            or self.declaration_header().endswith("where")
        )

    @classmethod
    def _trim_unbalanced(cls, text: str) -> str:
        """Drop a trailing unclosed bracket group.

        Cutting a line at its proof can leave `... → map (` behind. An
        unbalanced fragment presented as the API is worse than a shorter one.
        """
        while True:
            depth = 0
            last_open = -1
            for i, char in enumerate(text):
                if char in cls._OPEN_BRACKETS:
                    if depth == 0:
                        last_open = i
                    depth += 1
                elif char in cls._CLOSE_BRACKETS:
                    depth = max(0, depth - 1)
            if depth == 0 or last_open == -1:
                return text.rstrip(" \t:=,→⟶")
            text = text[:last_open]

    @classmethod
    def _scan_for_assign(cls, text: str, depth: int) -> tuple[int, int | None]:
        """Track bracket depth across `text`, returning the top-level `:=`.

        A blind `find(":=")` cuts inside Lean 4 named arguments -- it turned
        `pullback.fst (f := f) (g := g) ≫ f = pullback.snd ≫ g` into
        `pullback.fst (f`, an unbalanced fragment presented as the API.
        """
        for i, char in enumerate(text):
            if char in cls._OPEN_BRACKETS:
                depth += 1
            elif char in cls._CLOSE_BRACKETS:
                depth = max(0, depth - 1)
            elif depth == 0 and text.startswith(":=", i):
                return depth, i
        return depth, None

    def _continuation_lines(self, *, stop_at_body: bool) -> list[str]:
        """Indented lines belonging to the declaration, comments dropped.

        Relative indentation is preserved: a wrapped field type flattened to
        the same column reads as a separate field.
        """
        lines = self._code().splitlines()
        if not lines:
            return []

        depth, header_assign = self._scan_for_assign(lines[0], 0)
        if stop_at_body and header_assign is not None:
            # `:= by` on the declaration line: the body starts there, so no
            # continuation line carries a `:=` to stop at and the whole proof
            # would otherwise render.
            return []

        kept: list[str] = []
        in_comment = False
        proof_indent: int | None = None
        for line in lines[1:]:
            text = line.strip()
            if in_comment:
                if "-/" in text:
                    in_comment = False
                continue
            if not text or text.startswith("--"):
                continue
            if text.startswith("/-"):
                in_comment = "-/" not in text[2:]
                continue
            if not line[0].isspace():
                break
            indent = len(line) - len(line.lstrip())
            if proof_indent is not None:
                # Inside a field's tactic proof. Anything indented deeper than
                # the `:= by` line belongs to the proof, not to the API. A line
                # that simply *wrapped* is not deeper than its own field.
                if indent > proof_indent:
                    continue
                proof_indent = None
            if (
                text.startswith("@[")
                or text.split(maxsplit=1)[0] in self.FOREIGN_TOKENS
            ):
                break
            if stop_at_body:
                if text == "by" or text.startswith("by "):
                    break
                depth, assign = self._scan_for_assign(text, depth)
                if assign is not None:
                    head = line[: len(line) - len(line.lstrip()) + assign].rstrip()
                    if head.strip():
                        kept.append(head)
                    break
            # A field whose value is a tactic proof: keep the field's own
            # signature, drop the proof under it.
            proof_start = _BY_TAIL.search(text)
            if proof_start:
                _, assign = self._scan_for_assign(text, 0)
                cut = assign if assign is not None else proof_start.start()
                head = self._trim_unbalanced(text[:cut])
                kept.append(" " * indent + head if head else line.rstrip())
                proof_indent = indent
                continue
            kept.append(line.rstrip())

        if not kept:
            return []
        base = min(len(k) - len(k.lstrip()) for k in kept)
        return [k[base:] for k in kept]

    def field_lines(self) -> list[str]:
        """Fields of a structure/class, or constructors of an inductive.

        A bare `Hom` does not tell an agent the arity or the argument order;
        `Hom : obj → obj → Type v` does. That gap is what let a generation-2
        agent treat `Category` as a bare type and fail to elaborate. In source
        order, none elided -- a silent cap here is the bug, not the fix.
        """
        return self._continuation_lines(stop_at_body=False)

    def statement_lines(self) -> list[str]:
        """The rest of a theorem or def signature, without its proof or value.

        For a theorem the *statement* is the API, and it routinely wraps:
        rendering line 1 alone leaves an agent half the binders and no
        conclusion. The proof below it is noise that grows without bound.
        """
        return self._continuation_lines(stop_at_body=True)

    @classmethod
    def from_dict(cls, d: dict) -> FoundationEntry:
        """Create from dictionary."""
        return cls(
            artifact_id=d["artifact_id"],
            name=d["name"],
            entry_type=d["entry_type"],
            signature=d["signature"],
            lean_code=d["lean_code"],
            generation=d["generation"],
            author=d["author"],
        )


class FoundationFile:
    """Manages the accumulated foundation of verified Lean definitions.

    The foundation file grows across generations:
    - Gen 0: Agent defines Category → saved to Foundation.lean
    - Gen 1: Agent imports Foundation, defines Functor → appended
    - Gen 2: Agent imports Foundation (has Cat, Functor), defines NatTrans → appended
    - ...

    Agents receive context about what's available for import, encouraging
    reuse over re-implementation.

    CONFLICT RESOLUTION:
    When multiple agents define the same core concept (Category, Functor, etc.)
    with incompatible structures, only the FIRST verified definition is kept.
    This "first-mover wins" rule prevents compilation errors from incompatible
    definitions. Future: implement voting on conflicts.
    """

    # Core concepts that should only have one definition
    # Different naming conventions map to the same concept
    CORE_CONCEPTS = {
        "Category": "category",
        "Cat": "category",
        "Functor": "functor",
        "CFunctor": "functor",
        "Fun": "functor",
        "Funct": "functor",
        "NatTrans": "nattrans",
        "NaturalTransformation": "nattrans",
    }

    # Patterns to extract definitions from Lean code
    # Matches: structure Name, def Name, theorem Name, lemma Name, etc.
    # Allows leading whitespace for indented code
    DEFINITION_PATTERN = re.compile(
        r"^\s*(structure|def|theorem|lemma|abbrev|instance|class|inductive)\s+"
        r"([A-Za-z_][A-Za-z0-9_\.]*)"
        r"([^\n]*)",
        re.MULTILINE,
    )

    # Standard imports for the foundation file
    FOUNDATION_HEADER = """/-
LMS Foundation - Accumulated Verified Definitions
This file is auto-generated by the LLM Mathematical Society.
Each section was verified by LEAN before being added.

Usage: import LMS.Foundation
-/

import Mathlib.Tactic.Common
import Mathlib.Logic.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.Ring.Basic

universe u v w

-- Matches the flags RealLeanVerifier passes to `lean` (STRICTNESS_FLAGS).
-- An entry is verified standalone and then recompiled here as part of the
-- library; if the two disagree on autoImplicit, code can pass verification
-- and then fail the foundation build, or vice versa.
set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace LMS.Foundation

"""

    FOUNDATION_FOOTER = """
end LMS.Foundation
"""

    def __init__(self, path: Path) -> None:
        """Initialize foundation file manager.

        Args:
            path: Path to the Foundation.lean file
        """
        self.path = Path(path)
        self.entries: list[FoundationEntry] = []
        self._artifact_ids: set[str] = set()  # Track added artifacts
        self._definition_names: set[str] = (
            set()
        )  # Track defined names to avoid duplicates
        self._claimed_concepts: set[str] = (
            set()
        )  # Track which core concepts have definitions

    def __len__(self) -> int:
        """Return number of entries in foundation."""
        return len(self.entries)

    def add_artifact(self, artifact: Artifact) -> None:
        """Add a verified artifact to the foundation.

        Args:
            artifact: The artifact to add (must be verified with Lean code)

        Raises:
            ValueError: If artifact is not verified or has no Lean code
        """
        if not artifact.verified:
            raise ValueError("Foundation accepts only verified artifacts")
        if not artifact.lean_code:
            raise ValueError("Artifact has no Lean code")

        # Skip duplicates
        if artifact.id in self._artifact_ids:
            return

        # Clean the lean code (remove YAML multiline markers and embedded imports)
        clean_code = artifact.lean_code
        if clean_code.startswith("|"):
            clean_code = clean_code[1:].lstrip("\n")

        # Remove problematic statements that conflict with Foundation wrapper:
        # - import statements (must be at file top, Foundation has common imports)
        # - universe declarations (Foundation header has them)
        # - namespace/end statements (conflict with LMS.Foundation namespace)
        lines = clean_code.split("\n")
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import "):
                continue
            if stripped.startswith("universe "):
                continue
            # Remove namespace and end statements (they break Foundation wrapper)
            if stripped.startswith("namespace "):
                continue
            if stripped.startswith("end "):
                continue
            clean_lines.append(line)
        clean_code = "\n".join(clean_lines).strip()

        # Extract definitions from the code
        new_entries = self._extract_entries(
            clean_code,
            artifact.id,
            artifact.generation,
            artifact.created_by,
        )

        # Check for INCOMPATIBLE concept conflicts (different names, same concept)
        # e.g., Foundation has "Category", artifact defines "Cat" → reject entire artifact
        # This is different from DUPLICATE definitions (same name) which are just skipped
        for entry in new_entries:
            if entry.entry_type == "structure" and entry.name in self.CORE_CONCEPTS:
                concept = self.CORE_CONCEPTS[entry.name]
                if concept in self._claimed_concepts:
                    # Check if it's a true conflict (different name) or just a duplicate (same name)
                    # Find existing structure name for this concept
                    existing_name = None
                    for existing in self.entries:
                        if (
                            existing.entry_type == "structure"
                            and existing.name in self.CORE_CONCEPTS
                            and self.CORE_CONCEPTS[existing.name] == concept
                        ):
                            existing_name = existing.name
                            break

                    if existing_name and existing_name != entry.name:
                        # TRUE CONFLICT: Different names for same concept (e.g., Category vs Cat)
                        # Reject entire artifact to avoid incompatible definitions
                        self._artifact_ids.add(
                            artifact.id
                        )  # Mark as seen to prevent retry
                        return
                    # If same name, it's just a duplicate - will be skipped below

        # Filter and add entries
        unique_entries = []
        for entry in new_entries:
            # Skip if name already exists (duplicate)
            if entry.name in self._definition_names:
                continue

            # Claim any core concepts defined by this entry
            if entry.entry_type == "structure" and entry.name in self.CORE_CONCEPTS:
                self._claimed_concepts.add(self.CORE_CONCEPTS[entry.name])

            unique_entries.append(entry)
            self._definition_names.add(entry.name)

        self.entries.extend(unique_entries)
        self._artifact_ids.add(artifact.id)

    # Pattern to match block comments /- ... -/
    BLOCK_COMMENT_PATTERN = re.compile(r"/-.*?-/", re.DOTALL)

    def _strip_block_comments(self, code: str) -> str:
        """Remove block comments from code for definition detection.

        Block comments are replaced with whitespace of the same length
        to preserve character positions for slicing original code.

        Args:
            code: Lean source code

        Returns:
            Code with block comments replaced by spaces
        """
        result = code
        for match in self.BLOCK_COMMENT_PATTERN.finditer(code):
            # Replace with spaces to preserve positions
            replacement = " " * len(match.group())
            result = result[: match.start()] + replacement + result[match.end() :]
        return result

    def _extract_entries(
        self,
        code: str,
        artifact_id: str,
        generation: int,
        author: str,
    ) -> list[FoundationEntry]:
        """Extract definition entries from Lean code.

        Args:
            code: Lean source code
            artifact_id: ID of the source artifact
            generation: Generation number
            author: Agent ID

        Returns:
            List of extracted FoundationEntry objects
        """
        entries = []

        # Strip block comments to avoid matching definitions inside comments
        code_without_comments = self._strip_block_comments(code)

        # Find all definition positions (using comment-stripped code)
        matches = list(self.DEFINITION_PATTERN.finditer(code_without_comments))

        for i, match in enumerate(matches):
            entry_type = match.group(1)  # structure, def, theorem, etc.
            name = match.group(2)  # Name of the definition
            rest = match.group(3).strip()  # Rest of signature line

            signature = f"{entry_type} {name}{rest}"

            # Extract just this definition's code
            # From this match's start to the next match's start (or end of code)
            start = match.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(code)

            definition_code = code[start:end].strip()

            entries.append(
                FoundationEntry(
                    artifact_id=artifact_id,
                    name=name,
                    entry_type=entry_type,
                    signature=signature,
                    lean_code=definition_code,  # Store only this definition's code
                    generation=generation,
                    author=author,
                )
            )

        # If no definitions found, create a single entry for the whole artifact
        if not entries:
            # Use last 12 chars of artifact_id to avoid collisions
            # (all artifacts start with "definition-" so first chars collide)
            unique_suffix = artifact_id[-12:] if len(artifact_id) >= 12 else artifact_id
            entries.append(
                FoundationEntry(
                    artifact_id=artifact_id,
                    name=f"artifact_{unique_suffix}",
                    entry_type="code",
                    signature="",
                    lean_code=code,
                    generation=generation,
                    author=author,
                )
            )

        return entries

    #: Namespace every foundation entry is written inside (FOUNDATION_HEADER).
    NAMESPACE = "LMS.Foundation"

    def get_import_statement(self) -> str:
        """Return the import statement agents should use.

        Returns:
            Import statement string
        """
        return f"import {self.NAMESPACE}"

    def get_preamble(self) -> str:
        """Return the lines agents need to actually *use* the foundation.

        Importing alone is not enough, and that was the whole failure. Entries
        are written inside `namespace LMS.Foundation`, so a verified `Category`
        is `LMS.Foundation.Category`. An agent that imports and then writes
        `Category` gets `unknown identifier` -- the module resolved, the name
        never did.

        This was invisible until `autoImplicit` was turned off in
        26Q3-HARN-09: before that the unresolved name was silently auto-bound
        as an implicit variable, and the error surfaced somewhere else entirely
        as a type mismatch.

        Returns:
            The import and open lines, newline separated.
        """
        return f"{self.get_import_statement()}\nopen {self.NAMESPACE}"

    def get_available_definitions(self) -> str:
        """Return a summary of available definitions.

        Returns:
            Human-readable summary of what's in the foundation
        """
        if not self.entries:
            return "No definitions available yet."

        lines = []
        # Group by type
        by_type: dict[str, list[FoundationEntry]] = {}
        for entry in self.entries:
            if entry.entry_type not in by_type:
                by_type[entry.entry_type] = []
            by_type[entry.entry_type].append(entry)

        for entry_type, type_entries in sorted(by_type.items()):
            lines.append(f"\n{entry_type.upper()}S:")
            for entry in type_entries:
                lines.append(f"  - {entry.name} (gen {entry.generation})")
                # The declaration, not a `signature[:80]` amputated mid-token.
                # Same renderer as the agent-facing context, so the two cannot
                # disagree about what the foundation contains.
                header = entry.declaration_header()
                if header:
                    lines.append(f"    {header}")

        return "\n".join(lines)

    def get_context_for_agent(self) -> str:
        """Get full context string for agent prompts.

        This tells agents what's available for import and how to use it.

        Returns:
            Context string to include in agent prompts
        """
        if not self.entries:
            return """═══════════════════════════════════════════════════════════════════════════════
                            FOUNDATION: EMPTY
═══════════════════════════════════════════════════════════════════════════════
No verified definitions yet. You are starting from scratch.
Create foundational definitions that future generations can build upon.
═══════════════════════════════════════════════════════════════════════════════"""

        lines = [
            "═══════════════════════════════════════════════════════════════════════════════",
            "                    VERIFIED FOUNDATION (import LMS.Foundation)",
            "═══════════════════════════════════════════════════════════════════════════════",
            "",
            "The following definitions are VERIFIED and available for import.",
            "USE THEM! Do not redefine what already exists.",
            "",
            "To use, put BOTH of these at the top of your code:",
            "",
            f"    {self.get_preamble()}",
            "",
            f"The `open` is required. Every definition below lives inside the "
            f"`{self.NAMESPACE}` namespace, so with only the import you must "
            f"write `{self.NAMESPACE}.Category` in full; a bare `Category` is "
            f"an unknown identifier.",
            "",
        ]

        # Group entries by generation for clarity
        by_gen: dict[int, list[FoundationEntry]] = {}
        for entry in self.entries:
            if entry.generation not in by_gen:
                by_gen[entry.generation] = []
            by_gen[entry.generation].append(entry)

        omitted = 0
        for gen in sorted(by_gen.keys()):
            gen_entries = by_gen[gen]
            lines.append(f"── Generation {gen} ──")
            for entry in gen_entries:
                # The declaration exactly as verified. The previous rendering
                # concatenated `entry_type`, `name` and `signature`, but
                # `signature` already begins with the first two -- so agents
                # were handed `structure Categorystructure Category(obj :
                # Type u) where`, which is not valid Lean and appears nowhere
                # in Foundation.lean. An agent cannot use what it cannot see
                # the shape of.
                header = entry.declaration_header()
                if not header:
                    omitted += 1
                    continue
                lines.append(f"  {header}")

                # The rest of the API surface, with types. Names alone hide
                # arity and argument order, and the old `[:5]` dropped the
                # sixth field onward behind a bare `...`.
                if entry.body_is_api():
                    lines.extend(f"    {field}" for field in entry.field_lines())
                elif entry.entry_type in FoundationEntry.SIGNATURE_ONLY:
                    lines.extend(f"    {rest}" for rest in entry.statement_lines())
            lines.append("")

        # An elision the agent cannot see is the bug this card exists to
        # remove, so say how many rather than quietly shortening the list.
        if omitted:
            plural = "entry" if omitted == 1 else "entries"
            lines.append(
                f"({omitted} {plural} omitted: no declaration found in the code)"
            )
            lines.append("")

        lines.append(
            "═══════════════════════════════════════════════════════════════════════════════"
        )

        return "\n".join(lines)

    def save(self) -> None:
        """Save foundation to Lean file and metadata JSON."""
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Build Lean file content
        lean_content = self.FOUNDATION_HEADER

        # Write each unique entry's code
        # Group by artifact but only write code for unique definitions
        seen_artifacts: set[str] = set()
        written_names: set[str] = set()

        for entry in self.entries:
            # Skip if this definition name was already written
            if entry.name in written_names:
                continue

            if entry.artifact_id not in seen_artifacts:
                seen_artifacts.add(entry.artifact_id)
                lean_content += f"\n-- From {entry.artifact_id} (gen {entry.generation}, {entry.author})\n"

            # Write only this entry's code
            lean_content += entry.lean_code
            lean_content += "\n"
            written_names.add(entry.name)

        lean_content += self.FOUNDATION_FOOTER

        # Write Lean file
        self.path.write_text(lean_content)

        # Write metadata JSON
        metadata = {
            "entries": [e.to_dict() for e in self.entries],
            "artifact_ids": list(self._artifact_ids),
        }
        metadata_path = self.path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Path) -> FoundationFile:
        """Load foundation from saved metadata.

        Args:
            path: Path to Foundation.lean file

        Returns:
            Loaded FoundationFile
        """
        foundation = cls(path)

        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            foundation.entries = [
                FoundationEntry.from_dict(e) for e in metadata.get("entries", [])
            ]
            foundation._artifact_ids = set(metadata.get("artifact_ids", []))
            # Rebuild definition names from loaded entries
            foundation._definition_names = {e.name for e in foundation.entries}
            # Rebuild claimed concepts from loaded entries
            for entry in foundation.entries:
                if entry.entry_type == "structure" and entry.name in cls.CORE_CONCEPTS:
                    foundation._claimed_concepts.add(cls.CORE_CONCEPTS[entry.name])

        return foundation
