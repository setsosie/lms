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
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lms.artifacts import Artifact

#: A `by` ending a line opens a tactic block, whether written `:= by`,
#: `(by`, or bare `by`.
_BY_TAIL = re.compile(r"\bby$")

#: The namespace every foundation entry is stored inside. The verifier
#: (`lms/lean/real.py`) elaborates candidates inside this same namespace so
#: the oracle and the store agree on what a name collides with -- two string
#: literals that had to agree is how the verify/store asymmetry arose
#: (26Q3-HARN-13).
FOUNDATION_NAMESPACE = "LMS.Foundation"

#: Universe names the foundation header binds, above its `namespace` line.
#: `FoundationFile.add_artifact` drops these from an entry's own `universe`
#: lines on the grounds that they are already in scope at the destination --
#: so the verifier has to put them in scope too, or it rejects code that would
#: compile once stored (26Q3-HARN-21).
FOUNDATION_UNIVERSES: tuple[str, ...] = ("u", "v", "w")


_UNIVERSE_NAME = re.compile(r"[^\W\d][\w'!?]*")


def _universe_line_names(stripped: str) -> list[str]:
    """The names a stripped `universe ...` line binds, in order.

    Reading stops at the first token that isn't a name: a trailing comment, or
    the `in` of `universe u in`, which scopes the line to one command.
    """
    names: list[str] = []
    rest = re.sub(r"/-.*?-/", " ", stripped[len("universe ") :])
    for token in rest.split():
        if token == "in" or not _UNIVERSE_NAME.fullmatch(token):
            break
        names.append(token)
    return names


def declared_universe_order(lines: Iterable[str]) -> list[str]:
    """Universe names that `universe ...` lines bind, first-declared first.

    Order is meaning: Lean orders a declaration's level parameters by it, so a
    different order changes what `f.{0, 1}` refers to and can still compile.
    """
    names: dict[str, None] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("universe "):
            names.update(dict.fromkeys(_universe_line_names(stripped)))
    return list(names)


def declared_universe_names(lines: Iterable[str]) -> set[str]:
    """Universe names that a `universe ...` line in `lines` already binds.

    The verifier adds only the header universes a candidate has *not* declared
    for itself: adding one twice is `error: a universe level named 'u' has
    already been declared`, which is the same class of failure in the opposite
    direction.
    """
    return set(declared_universe_order(lines))


def strip_header_universes(lines: Iterable[str]) -> list[str]:
    """Drop universe names the foundation header already binds.

    Inlined text sits below `universe u v w`, so repeating any of those names
    is `error: a universe level named 'u' has already been declared`. A line
    binding names the header does *not* cover keeps exactly those, so a seed
    or artifact using an exotic level still works.
    """
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("universe "):
            out.append(line)
            continue
        kept = [
            name
            for name in _universe_line_names(stripped)
            if name not in FOUNDATION_UNIVERSES
        ]
        if kept:
            out.append(f"universe {' '.join(kept)}")
    return out


def split_imports(code: str) -> tuple[list[str], list[str]]:
    """Partition Lean source lines into `import` lines and everything else.

    Lean rejects an `import` inside a `namespace`, so any wrapper must keep
    imports above the `namespace` line. Both `FoundationFile.add_artifact`
    and `RealLeanVerifier` need this exact split; a second copy of it is the
    kind of divergence 26Q3-HARN-13 exists to remove.
    """
    import_lines: list[str] = []
    body_lines: list[str] = []
    for line in code.split("\n"):
        if line.strip().startswith("import "):
            import_lines.append(line)
        else:
            body_lines.append(line)
    return import_lines, body_lines


@dataclass(frozen=True)
class FoundationSnapshot:
    """The foundation's full mutable state at a point in time.

    Taken after every successful build so a generation whose additions break
    the merged module can be undone (26Q3-HARN-22). Every set is copied on
    capture and on restore: sharing them would let a later `add_artifact`
    mutate the snapshot that is supposed to be the way back.
    """

    entries: tuple[FoundationEntry, ...]
    artifact_ids: frozenset[str]
    definition_names: frozenset[str]
    claimed_concepts: frozenset[str]


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
        universes: Universe names its artifact bound beyond the header. On the
            entry so a rollback or a checkpoint keeps them with what uses them.
    """

    artifact_id: str
    name: str
    entry_type: str  # structure, def, theorem, lemma, etc.
    signature: str
    lean_code: str
    generation: int
    author: str
    universes: tuple[str, ...] = ()

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
            "universes": list(self.universes),
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
            # The notation family. These carry a precedence suffix
            # (`infixr:80`, `notation:max`) or a scope modifier
            # (`scoped notation`, `local infixl`), so a whole-token match
            # walked straight past them -- see `_is_foreign`.
            "infix",
            "infixl",
            "infixr",
            "prefix",
            "postfix",
            "scoped",
            "local",
            "set_option",
            "universe",
            "universes",
            "elab",
            "initialize",
            "opaque",
            "axiom",
            "export",
            "alias",
            "omit",
            "include",
            "run_cmd",
            # Declaration keywords too: `DEFINITION_PATTERN` can't read every
            # form, such as a nested attribute (`@[to_additive [x]]`), and an
            # unread declaration is not extracted as its own entry.
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

    #: Binders that consume a `:=` inside a *type*. ART/CNF-style Lean opens a
    #: return type with `letI := toMathlib C; …`, and cutting the statement at
    #: that `:=` left the return type as the bare token `letI`.
    _BINDER_TOKENS = frozenset({"let", "letI", "have", "haveI", "suffices"})

    #: Bracket pairs `:=` can hide inside. Lean 4 named arguments (`(f := f)`)
    #: and autoparams (`(h : P := by simp)` ) sit at depth > 0.
    _OPEN_BRACKETS = "([{⟨"
    _CLOSE_BRACKETS = ")]}⟩"

    #: Brackets, `:=`, and identifiers -- the only tokens `_scan_for_assign`
    #: needs to distinguish. Scanning raw characters could not tell a binder
    #: keyword from a fragment of one.
    _TOKENS = re.compile(r"[([{⟨)\]}⟩]|:=|[A-Za-z_][A-Za-z_0-9'!?]*")

    @classmethod
    def _strip_line_comment(cls, text: str) -> str:
        """Drop a trailing `--` comment, respecting string literals.

        Only whole-line comments were dropped before, so a trailing comment
        stayed in the text that `_scan_for_assign`, `body_is_api` and
        `_BY_TAIL` all read. One `(` inside such a comment held bracket depth
        above zero for the rest of the declaration, and the top-level `:=` was
        then never found -- the entire proof body rendered as the API.
        """
        in_string = False
        i = 0
        while i < len(text):
            char = text[i]
            if char == '"' and (i == 0 or text[i - 1] != "\\"):
                in_string = not in_string
            elif not in_string and text.startswith("--", i):
                return text[:i].rstrip()
            i += 1
        return text

    @classmethod
    def _is_foreign(cls, text: str) -> bool:
        """Whether `text` begins a new declaration or command.

        Matching the whole first token missed every precedence-suffixed and
        `#`-prefixed command: `infixr:80`, `notation:max`, `#check`. Those
        rendered as fields of the preceding structure, and an agent that reads
        them as fields writes `{ Hom := …, infixr := … }`, which Lean rejects.
        """
        if text.startswith(("@[", "#")):
            return True
        first = text.split(maxsplit=1)[0]
        return (
            first in cls.FOREIGN_TOKENS or first.split(":", 1)[0] in cls.FOREIGN_TOKENS
        )

    def _code(self) -> str:
        """`lean_code` with leading comments and own-line attributes removed.

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
            first = stripped.split("\n", 1)[0].strip()
            if "\n" in stripped and re.fullmatch(r"@\[[^\]]*\]", first):
                text = stripped.split("\n", 1)[1]
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

        The header stops at the first top-level `:=`. 18 corpus entries write
        the proof on the declaration line, and returning that line raw handed
        agents `lemma mem_span_singleton … :=` -- a declaration with no body,
        which does not compile if copied.
        """
        for line in self._code().splitlines():
            text = self._strip_line_comment(line.strip())
            if not text:
                continue
            _, assign = self._scan_for_assign(text, 0)
            if assign is not None:
                return self._trim_unbalanced(text[:assign])
            return text
        return ""

    def body_is_api(self) -> bool:
        """Whether the indented body is the declaration's API surface.

        Keyed on the `where` suffix as well as the keyword: an
        `instance TypeCat : Category (Type u) where` has fields exactly like a
        structure, and classifying it as signature-only rendered one bare
        field name and silently dropped the other five.

        The exception is keyed on `where`, not on the keyword: a
        `theorem … where` or `def … where` builds a structure term, and its
        fields are as much the API as a structure's. What the review caught was
        not the routing but the *output* -- `_continuation_lines` fell back to
        the raw source line whenever its trim came back empty, so a wrapped
        field value like `{ app := by` reached agents with an unclosed brace.
        That fallback is gone; an unrenderable line is dropped instead.

        The suffix is looked for across the whole header *region*, not just
        line 1. When the binders wrap, `where` lands on line 2 -- checking
        only the first line reintroduced the same one-bare-field bug on 10.6%
        of corpus entries. The region ends at the first top-level `:=`, which
        is where a proof or value body begins.
        """
        if self.entry_type in self.BODY_IS_API:
            return True
        depth = 0
        in_comment = False
        for line in self._code().splitlines():
            text = line.strip()
            if in_comment:
                if "-/" in text:
                    in_comment = False
                continue
            if text.startswith("/-"):
                # Only the *first* line of a block comment was skipped before,
                # so interior lines were scanned for brackets as if code.
                in_comment = "-/" not in text[2:]
                continue
            text = self._strip_line_comment(text)
            if not text:
                continue
            depth, assign = self._scan_for_assign(text, depth)
            if assign is not None:
                return False
            if depth == 0 and text.endswith("where"):
                return True
        return False

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
    def _bracket_delta(cls, text: str) -> int:
        """Net bracket depth change across `text`, trailing comment ignored."""
        text = cls._strip_line_comment(text)
        delta = 0
        for char in text:
            if char in cls._OPEN_BRACKETS:
                delta += 1
            elif char in cls._CLOSE_BRACKETS:
                delta -= 1
        return delta

    @classmethod
    def _scan_for_assign(cls, text: str, depth: int) -> tuple[int, int | None]:
        """Track bracket depth across `text`, returning the top-level `:=`.

        A blind `find(":=")` cuts inside Lean 4 named arguments -- it turned
        `pullback.fst (f := f) (g := g) ≫ f = pullback.snd ≫ g` into
        `pullback.fst (f`, an unbalanced fragment presented as the API.

        A `:=` claimed by a `letI`/`have` binder in the *type* is skipped for
        the same reason: `… : letI := toMathlib C; CategoryTheory.Functor …`
        is one return type, not a type and a body, and cutting at the binder
        rendered six shipped `Compat.lean` entries as `… : letI`.

        The trailing `--` comment is dropped first. `assign` indexes into
        `text`, and stripping only truncates the tail, so callers slicing the
        original line with the returned index stay correct.
        """
        text = cls._strip_line_comment(text)
        pending_binders = 0
        for match in cls._TOKENS.finditer(text):
            token = match.group()
            if token in cls._OPEN_BRACKETS:
                depth += 1
            elif token in cls._CLOSE_BRACKETS:
                depth = max(0, depth - 1)
            elif token == ":=":
                if depth != 0:
                    continue
                if pending_binders:
                    pending_binders -= 1
                    continue
                return depth, match.start()
            elif depth == 0 and token in cls._BINDER_TOKENS:
                pending_binders += 1
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
        skip_depth = 0
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
            if skip_depth > 0:
                # Inside a bracket group whose opening line was cut. Its
                # closers sit at the field's own indent, so `proof_indent`
                # cannot see them; count brackets until the group closes.
                skip_depth = max(0, skip_depth + self._bracket_delta(line.strip()))
                if skip_depth == 0:
                    proof_indent = None
                continue
            if proof_indent is not None:
                # Inside a field's tactic proof. Anything indented deeper than
                # the `:= by` line belongs to the proof, not to the API. A line
                # that simply *wrapped* is not deeper than its own field.
                if indent > proof_indent:
                    continue
                proof_indent = None
            text = self._strip_line_comment(text)
            if not text:
                continue
            if self._is_foreign(text):
                break
            if stop_at_body:
                if text == "by" or text.startswith("by "):
                    break
                # `def f : Nat → Nat` / `| 0 => 1` -- an equation-style def has
                # no top-level `:=` to stop at, so its whole value body used to
                # render as the statement.
                if text.startswith("|"):
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
                # Falling back to the raw line here is what put `{ app := by`
                # -- an unclosed brace and a bare `by` -- in front of agents.
                # If nothing balanced survives the trim, there is no API line
                # to show, and a fragment is worse than nothing.
                if head:
                    kept.append(" " * indent + head)
                # The unclosed group can sit on either side of the cut:
                # `map (by` leaves it in the head, `:= funext (λ g => by`
                # leaves it in the dropped tail. The head is balanced after
                # the trim, so the whole line's delta is what stays open.
                skip_depth = max(0, self._bracket_delta(text))
                proof_indent = indent
                continue

            # Never emit a line that leaves a bracket group open. `map_id :
            # (x : C.Obj) → map (by` cut to `... → map`, and the group's
            # closing `)` sat on a later line at this same indent -- an indent
            # test cannot see it, so it rendered as a bare `)` "field". 22 of
            # the 23 remaining unbalanced corpus blocks were this.
            delta = self._bracket_delta(text)
            if delta > 0:
                head = self._trim_unbalanced(text)
                if head:
                    kept.append(" " * indent + head)
                skip_depth = delta
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
            universes=tuple(d.get("universes", ())),
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

    The same goes for any name declared twice with different bodies, and the
    later artifact contributes nothing at all: its other entries were verified
    against its own declaration, not the one they would bind to here. An
    identical re-declaration is harmless, so only that entry is skipped.
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
        r"^\s*(?:@\[(?:[^\[\]]|\[[^\]]*\])*\]\s*"
        r"|(?:private|protected|noncomputable|partial|unsafe|nonrec|scoped|local)\s+)*"
        r"(class\s+inductive|class\s+abbrev|structure|def|theorem|lemma|abbrev|instance|class|inductive)\s+"
        r"(?:\(\s*priority\s*:=[^)]*\)\s*)?"
        r"([^\W\d][\w.'!?]*)"
        r"([^\n]*)",
        re.MULTILINE,
    )

    # Standard imports for the foundation file. The namespace line comes from
    # FOUNDATION_NAMESPACE -- the same constant the verifier wraps with.
    FOUNDATION_HEADER = f"""/-
LMS Foundation - Accumulated Verified Definitions
This file is auto-generated by the LLM Mathematical Society.
Each section was verified by LEAN before being added.

Usage: import {FOUNDATION_NAMESPACE}
-/

import Mathlib.Tactic.Common
import Mathlib.Logic.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.Ring.Basic

universe {" ".join(FOUNDATION_UNIVERSES)}

-- Matches the flags RealLeanVerifier passes to `lean` (STRICTNESS_FLAGS).
-- An entry is verified standalone and then recompiled here as part of the
-- library; if the two disagree on autoImplicit, code can pass verification
-- and then fail the foundation build, or vice versa.
set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace {FOUNDATION_NAMESPACE}

"""

    FOUNDATION_FOOTER = f"""
end {FOUNDATION_NAMESPACE}
"""

    def __init__(self, path: Path) -> None:
        """Initialize foundation file manager.

        Args:
            path: Path to the Foundation.lean file
        """
        self.path = Path(path)
        #: Hand-written generation-0 Lean, written verbatim below the header
        #: and above every agent entry (26Q3-HARN-23). Verbatim rather than
        #: parsed into entries because `save()` only writes entry bodies, and
        #: the seed's `scoped notation` lines are not declarations -- routing
        #: them through `_extract_entries` would silently drop `⟶`, `𝟙`, `≫`.
        self.seed_source: str = ""
        #: The seed's declarations, for agent context and name-claiming only.
        self.seed_entries: list[FoundationEntry] = []
        self.entries: list[FoundationEntry] = []
        self._artifact_ids: set[str] = set()  # Track added artifacts
        self._definition_names: set[str] = (
            set()
        )  # Track defined names to avoid duplicates
        self._claimed_concepts: set[str] = (
            set()
        )  # Track which core concepts have definitions
        #: Why an artifact was refused, by id, so a caller can report it.
        self._rejections: dict[str, str] = {}
        #: Which generation refused it. A refused artifact has no entries to
        #: date it by, and a rollback to a generation boundary has to forget it.
        self._refused_in: dict[str, int] = {}
        #: Everything each accepted artifact declares, including what it copied
        #: and so didn't add. A retry after a failed build needs it: a copy has
        #: to land if the artifact holding its original is dropped.
        self._accepted: dict[str, list[FoundationEntry]] = {}

    def __len__(self) -> int:
        """Return number of entries in foundation."""
        return len(self.entries)

    def set_seed(self, source: str) -> list[str]:
        """Install the generation-0 axiom layer; return the names it claims.

        The seed is the one part of the foundation no agent wrote. Its names
        are registered in `_definition_names` so an agent redefining `Category`
        contributes nothing rather than shadowing the layer everything else is
        built on -- which is how `committee_fix_c` ended up with four mutually
        incompatible `Functor`s.
        """
        # The seed is written below the header's `universe u v w`, so any
        # header name it rebinds is a duplicate-declaration error. It keeps
        # its own `universe` line so the file stands alone for review and
        # standalone compilation; that line is normalised away on install.
        _, body = split_imports(source.strip())
        self.seed_source = "\n".join(strip_header_universes(body)).strip()
        self.seed_entries = self._extract_entries(
            self.seed_source, artifact_id="seed", generation=0, author="seed"
        )
        for entry in self.seed_entries:
            self._definition_names.add(entry.name)
            if (
                entry.entry_type in ("structure", "class")
                and entry.name in self.CORE_CONCEPTS
            ):
                self._claimed_concepts.add(self.CORE_CONCEPTS[entry.name])
        return [entry.name for entry in self.seed_entries]

    def snapshot(self) -> FoundationSnapshot:
        """Capture the current state so a bad generation can be undone."""
        return FoundationSnapshot(
            entries=tuple(self.entries),
            artifact_ids=frozenset(self._artifact_ids),
            definition_names=frozenset(self._definition_names),
            claimed_concepts=frozenset(self._claimed_concepts),
        )

    def snapshot_before(self, generation: int) -> FoundationSnapshot:
        """The state before `generation` added anything to it.

        A checkpoint taken mid-generation (a walltime kill) holds that
        generation's entries, which never went through a build together. A
        resume treats this boundary state as the one known to compile, so a
        build those entries break rolls back to it like any failed generation.
        """
        kept = tuple(e for e in self.entries if e.generation < generation)
        later_ids = {e.artifact_id for e in self.entries if e.generation >= generation}
        # A copy has no entries of its own to show which generation it's from.
        later_ids |= {
            artifact_id
            for artifact_id, entries in self._accepted.items()
            if entries and entries[0].generation >= generation
        }
        later_ids |= {
            artifact_id
            for artifact_id, refused_in in self._refused_in.items()
            if refused_in >= generation
        }
        # Same claiming rules as `set_seed` (seed) and `load` (entries).
        concepts = {
            self.CORE_CONCEPTS[e.name]
            for e in self.seed_entries
            if e.entry_type in ("structure", "class") and e.name in self.CORE_CONCEPTS
        } | {
            self.CORE_CONCEPTS[e.name]
            for e in kept
            if e.entry_type == "structure" and e.name in self.CORE_CONCEPTS
        }
        return FoundationSnapshot(
            entries=kept,
            artifact_ids=frozenset(self._artifact_ids - later_ids),
            definition_names=frozenset(
                {e.name for e in self.seed_entries} | {e.name for e in kept}
            ),
            claimed_concepts=frozenset(concepts),
        )

    def restore(self, snapshot: FoundationSnapshot) -> list[str]:
        """Roll back to `snapshot`; return the names of the dropped entries.

        The names are returned rather than logged so the caller can report
        exactly what a failed build cost -- a rollback that silently discards a
        generation's verified work is the kind of thing that makes a run's
        numbers untraceable afterwards.

        `_artifact_ids` is restored too, so a dropped artifact is genuinely
        forgotten: leaving its id behind would make `add_artifact` skip it as a
        duplicate if a later generation resubmitted the same work.
        """
        kept = {id(entry) for entry in snapshot.entries}
        dropped = [entry.name for entry in self.entries if id(entry) not in kept]
        self.entries = list(snapshot.entries)
        self._artifact_ids = set(snapshot.artifact_ids)
        self._definition_names = set(snapshot.definition_names)
        self._claimed_concepts = set(snapshot.claimed_concepts)
        self._accepted = {
            artifact_id: entries
            for artifact_id, entries in self._accepted.items()
            if artifact_id in snapshot.artifact_ids
        }
        self._rejections = {
            artifact_id: reason
            for artifact_id, reason in self._rejections.items()
            if artifact_id in snapshot.artifact_ids
        }
        self._refused_in = {
            artifact_id: refused_in
            for artifact_id, refused_in in self._refused_in.items()
            if artifact_id in snapshot.artifact_ids
        }
        return dropped

    def add_artifact(self, artifact: Artifact) -> bool:
        """Add a verified artifact to the foundation.

        Args:
            artifact: The artifact to add (must be verified with Lean code)

        Returns:
            True if this call contributed at least one new entry. False means
            it contributed nothing -- an id already seen, a conflicting core
            concept, a name declared differently (`rejection_reason` says
            which), or every declaration in it being an identical copy.

            Returning nothing made those three outcomes indistinguishable from
            success at the call site, so a caller reported a promotion while
            the foundation gained nothing (26Q3-HARN-22).

        Raises:
            ValueError: If artifact is not verified or has no Lean code
        """
        if not artifact.verified:
            raise ValueError("Foundation accepts only verified artifacts")
        if not artifact.lean_code:
            raise ValueError("Artifact has no Lean code")

        # Skip duplicates
        if artifact.id in self._artifact_ids:
            return False

        # Clean the lean code (remove YAML multiline markers and embedded imports)
        clean_code = artifact.lean_code
        if clean_code.startswith("|"):
            clean_code = clean_code[1:].lstrip("\n")

        # Remove problematic statements that conflict with Foundation wrapper:
        # - import statements (must be at file top, Foundation has common imports)
        # - universe declarations (the header's, and `save` scopes the rest)
        # - namespace/end statements (conflict with LMS.Foundation namespace)
        _, body_lines = split_imports(clean_code)
        own_universes = tuple(
            name
            for name in declared_universe_order(
                self._strip_block_comments("\n".join(body_lines)).splitlines()
            )
            if name not in FOUNDATION_UNIVERSES
        )
        clean_lines = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.startswith("universe "):
                # `universe u in def f ...` on one line still holds a declaration.
                scoped = re.match(r"universe\s+(?:\S+\s+)*?in\s+(\S.*)", stripped)
                if scoped:
                    clean_lines.append(scoped.group(1))
                continue
            # Remove namespace and end statements (they break Foundation wrapper)
            if stripped.startswith("namespace "):
                continue
            if stripped.startswith("end "):
                continue
            clean_lines.append(line)
        # Dedented so a command ends a declaration whether or not the artifact
        # indented its namespace.
        clean_code = textwrap.dedent("\n".join(clean_lines)).strip()

        # Extract definitions from the code
        new_entries = self._extract_entries(
            clean_code,
            artifact.id,
            artifact.generation,
            artifact.created_by,
        )
        for entry in new_entries:
            entry.universes = own_universes

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
                        return self._refuse(
                            artifact,
                            f"it declares `{entry.name}`, but the foundation "
                            f"already has `{existing_name}` for that concept",
                        )
                    # If same name, it's just a duplicate - will be skipped below

        # Same name, different body: refuse the whole artifact, because its
        # other entries were verified against its own declaration. The same goes
        # for commands after a copy, which are skipped with it: an instance or
        # notation the foundation lacks could leave its later entries compiling
        # with another meaning.
        declared = {
            entry.name: self._comparable(entry)
            for entry in (*self.seed_entries, *self.entries)
        }
        own: dict[str, tuple[str, str]] = {}
        for entry in new_entries:
            declaration, commands = self._comparable(entry)
            if own.setdefault(entry.name, (declaration, commands)) != (
                declaration,
                commands,
            ):
                return self._refuse(
                    artifact, f"it declares `{entry.name}` twice, differently"
                )
            known = declared.get(entry.name)
            if known is None:
                continue
            if known[0] != declaration:
                return self._refuse(
                    artifact,
                    f"it declares `{entry.name}`, which the foundation already "
                    "declares differently",
                )
            if commands and commands != known[1]:
                return self._refuse(
                    artifact,
                    f"after its copy of `{entry.name}` it runs commands the "
                    "foundation doesn't have there",
                )

        return bool(self._land(artifact.id, new_entries))

    def _refuse(self, artifact: Artifact, reason: str) -> bool:
        """Record why an artifact contributes nothing. Always False, for `add_artifact`."""
        self._artifact_ids.add(artifact.id)
        self._rejections[artifact.id] = reason
        self._refused_in[artifact.id] = artifact.generation
        return False

    def _land(
        self, artifact_id: str, new_entries: list[FoundationEntry]
    ) -> list[FoundationEntry]:
        """Add what `new_entries` doesn't copy; return what it added.

        A skipped copy's commands go with it. They aren't added back, because
        repeating a `notation` makes every later use ambiguous with no build
        failing, and `add_artifact` has already refused any that differ.
        """
        landed: list[FoundationEntry] = []
        for entry in new_entries:
            if entry.name not in self._definition_names:
                self._admit(entry)
                landed.append(entry)
        self.entries.extend(landed)
        self._artifact_ids.add(artifact_id)
        self._rejections.pop(artifact_id, None)
        self._refused_in.pop(artifact_id, None)
        self._accepted[artifact_id] = list(new_entries)
        return landed

    def _split_commands(self, code: str) -> tuple[str, list[str]]:
        """Split a declaration's slice into the declaration and the commands after it.

        Only a command at column 0 ends it: `open X in` inside a proof is indented,
        and a `|` arm at column 0 isn't a command. A `deriving` clause belongs to
        its declaration, `deriving instance` stands alone, and `#[1, 2]` is a value.
        """
        blanked = self._strip_block_comments(code)
        match = self.DEFINITION_PATTERN.search(blanked)
        if match is None:
            return code, []
        offset = match.end()
        starts: list[int] = []
        for line in blanked[offset:].splitlines(keepends=True):
            text = FoundationEntry._strip_line_comment(line.strip())
            is_clause = text.startswith("deriving") and not text.startswith(
                "deriving instance"
            )
            is_value = text.startswith("#") and not re.match(r"#[A-Za-z_]", text)
            if (
                text
                and not is_clause
                and not is_value
                and not line[0].isspace()
                and FoundationEntry._is_foreign(text)
            ):
                starts.append(offset)
            offset += len(line)
        if not starts:
            return code, []
        ends = [*starts[1:], len(code)]
        return code[: starts[0]].rstrip(), [
            code[start:end].strip() for start, end in zip(starts, ends)
        ]

    def _admit(self, entry: FoundationEntry) -> None:
        """Record the name and any core concept that `entry` claims."""
        if entry.entry_type == "structure" and entry.name in self.CORE_CONCEPTS:
            self._claimed_concepts.add(self.CORE_CONCEPTS[entry.name])
        self._definition_names.add(entry.name)

    #: What `_comparable` drops: comments and layout, but nothing in a string.
    _NOT_MEANING = re.compile(r'("(?:[^"\\]|\\.)*")|/-.*?-/|--[^\n]*|\s+', re.DOTALL)

    def _comparable(self, entry: FoundationEntry) -> tuple[str, str]:
        """A declaration, and the commands after it that change anything.

        Enough to tell a copy from a difference. Comments and layout don't count.
        The artifact's own universes that the declaration uses do, because their
        order is part of what it means. `#check` and `example` change nothing.
        """

        def plain(code: str) -> str:
            return self._NOT_MEANING.sub(lambda m: m.group(1) or " ", code).strip()

        declaration, commands = self._split_commands(entry.lean_code)
        text = plain(declaration)
        words = set(re.findall(r"[^\W\d][\w'!?]*", text))
        used = " ".join(name for name in entry.universes if name in words)
        lasting = [
            plain(command)
            for command in commands
            if not re.match(r"#[A-Za-z_]|example\b", command)
        ]
        return f"{used}|{text}", "\n".join(lasting)

    def holds(self, artifact_id: str) -> bool:
        """Whether this artifact's work is here: added, or an identical copy."""
        return artifact_id in self._artifact_ids and artifact_id not in self._rejections

    def rejection_reason(self, artifact_id: str) -> str | None:
        """Why `add_artifact` refused this artifact, or None if it didn't."""
        return self._rejections.get(artifact_id)

    def artifacts_since(
        self, snapshot: FoundationSnapshot
    ) -> list[tuple[str, list[FoundationEntry]]]:
        """Artifacts accepted after `snapshot`, in order, with all they declare."""
        return [
            (artifact_id, entries)
            for artifact_id, entries in self._accepted.items()
            if artifact_id not in snapshot.artifact_ids
        ]

    def readmit(self, artifact_id: str, entries: list[FoundationEntry]) -> list[str]:
        """Put one artifact back after `restore`; return the names it added."""
        return [entry.name for entry in self._land(artifact_id, entries)]

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
            # `class inductive P` is an inductive named `P`, not a class named
            # `inductive`.
            entry_type = match.group(1).split()[-1]  # structure, def, theorem, etc.
            # `f.{u}` reads as `f.`: the universe list isn't part of the name.
            name = match.group(2).rstrip(".")  # Name of the definition
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
    NAMESPACE = FOUNDATION_NAMESPACE

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
                # Same renderer *and the same continuation lines* as the
                # agent-facing context: the header alone is one physical line,
                # so on the 10.6% of entries whose binders wrap this printed a
                # dangling binder list -- no better than the cut it replaced.
                header = entry.declaration_header()
                if header:
                    lines.append(f"    {header}")
                    lines.extend(f"      {rest}" for rest in self._api_lines(entry))

        return "\n".join(lines)

    @staticmethod
    def _api_lines(entry: FoundationEntry) -> list[str]:
        """The entry's API surface below its declaration line.

        One helper for both renderers. Two copies of this decision is how they
        came to disagree about what the foundation contains.
        """
        if entry.body_is_api():
            return entry.field_lines()
        if entry.entry_type in FoundationEntry.SIGNATURE_ONLY:
            return entry.statement_lines()
        return []

    def _seed_context_lines(self) -> list[str]:
        """Render the generation-0 axiom layer, verbatim, above agent entries.

        Verbatim and unabridged on purpose. This is the layer every later
        artifact is typed against, and a truncated or paraphrased rendering of
        it is exactly what produced five generations of `invalid binder
        annotation` -- agents were shown a signature line and guessed the rest
        from their Mathlib prior.
        """
        return [
            "── GENERATION-0 SEED (hand-written; NEVER redefine these) ──",
            "",
            *self.seed_source.split("\n"),
            "",
            "The seed is already compiled and imported. Use it directly:",
            "  `[Category C]` as an instance binder, `X ⟶ Y` for morphisms,",
            "  `𝟙 X` for identities, `f ≫ g` for diagrammatic composition.",
            "Redefining any seed name contributes nothing and will be dropped.",
            "",
        ]

    def get_context_for_agent(self, max_entries: int | None = None) -> str:
        """Get full context string for agent prompts.

        This tells agents what's available for import and how to use it.

        Args:
            max_entries: Render at most this many entries, then say how many
                were left out. Agents get the whole foundation (`None`); the
                committee prompts that embed this string alongside much else
                pass a bound. Unbounded, this section alone outgrows the
                served `max_model_len` on the corpora this program targets.

        Returns:
            Context string to include in agent prompts
        """
        if not self.entries and not self.seed_entries:
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
            "⚠ Mathlib's own category API does NOT exist here — no "
            "`Mathlib.CategoryTheory` import, and none of its lemmas. The "
            "ONLY fields and constants available are the ones printed below. "
            "Writing Mathlib lemma names against these definitions is the "
            "single most common verification failure. Read the declarations "
            "below and use exactly those.",
            "",
        ]

        if self.seed_entries:
            lines.extend(self._seed_context_lines())

        # Group entries by generation for clarity
        by_gen: dict[int, list[FoundationEntry]] = {}
        for entry in self.entries:
            if entry.generation not in by_gen:
                by_gen[entry.generation] = []
            by_gen[entry.generation].append(entry)

        omitted = 0
        rendered = 0
        truncated = 0
        for gen in sorted(by_gen.keys()):
            gen_entries = by_gen[gen]
            lines.append(f"── Generation {gen} ──")
            for entry in gen_entries:
                if max_entries is not None and rendered >= max_entries:
                    truncated += 1
                    continue
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
                rendered += 1

                # A fallback `code` entry has no parsed declaration, so its
                # header is just whatever line the artifact happens to open
                # with. Without its name the agent cannot cite the artifact at
                # all -- and if it recreates that name, `add_artifact` drops
                # the new verified work as a duplicate.
                if entry.entry_type == "code":
                    lines.append(f"  {entry.name} (no declaration parsed)")
                    lines.append(f"    begins: {header}")
                    continue

                lines.append(f"  {header}")

                # The rest of the API surface, with types. Names alone hide
                # arity and argument order, and the old `[:5]` dropped the
                # sixth field onward behind a bare `...`.
                lines.extend(f"    {rest}" for rest in self._api_lines(entry))
            lines.append("")

        if truncated:
            plural = "definition" if truncated == 1 else "definitions"
            lines.append(f"(... and {truncated} more {plural} not shown here)")
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

        # The seed goes verbatim, above every agent entry: later entries are
        # written in insertion order and may depend on it.
        if self.seed_source:
            lean_content += (
                "-- ===== Generation-0 seed (hand-written, not agent output) =====\n"
                f"{self.seed_source}\n"
                "-- ===== End seed =====\n"
            )

        # Write each unique entry's code
        # Group by artifact but only write code for unique definitions
        seen_artifacts: set[str] = set()
        written_names: set[str] = set()
        seed_universes = declared_universe_names(
            self._strip_block_comments(self.seed_source).splitlines()
        )
        section_open = False

        for entry in self.entries:
            # Skip if this definition name was already written
            if entry.name in written_names:
                continue

            if entry.artifact_id not in seen_artifacts:
                seen_artifacts.add(entry.artifact_id)
                if section_open:
                    lean_content += "end\n"
                    section_open = False
                lean_content += f"\n-- From {entry.artifact_id} (gen {entry.generation}, {entry.author})\n"
                # Scoped to the artifact, not hoisted to the header: two
                # artifacts may pick the same name, and a later `def f.{u₁}`
                # fails if `u₁` is already bound around it.
                universes = [n for n in entry.universes if n not in seed_universes]
                if universes:
                    lean_content += f"section\nuniverse {' '.join(universes)}\n"
                    section_open = True

            # Write only this entry's code
            lean_content += entry.lean_code
            lean_content += "\n"
            written_names.add(entry.name)

        if section_open:
            lean_content += "end\n"

        lean_content += self.FOUNDATION_FOOTER

        # Write Lean file
        self.path.write_text(lean_content)
        self.save_metadata(self.path.with_suffix(".json"))

    def save_metadata(self, metadata_path: Path) -> None:
        """Write the state `load` rebuilds from: entries, ids and the seed.

        Callable on its own so a checkpoint can keep a copy in the run
        directory. The Lean file has to live in the project, where each job's
        cleanup and the next run's reset overwrite it.
        """
        metadata = {
            "entries": [e.to_dict() for e in self.entries],
            "artifact_ids": list(self._artifact_ids),
            # Persisted so a resumed run rebuilds the same axiom layer rather
            # than silently dropping it and letting agents redefine `Category`.
            "seed_source": self.seed_source,
            "accepted": {
                artifact_id: [e.to_dict() for e in entries]
                for artifact_id, entries in self._accepted.items()
            },
            # A refused id stays in `artifact_ids`, so a resume needs this to
            # tell it from one the foundation holds.
            "rejections": self._rejections,
            "refused_in": self._refused_in,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Path, metadata_path: Path | None = None) -> FoundationFile:
        """Load foundation from saved metadata.

        Args:
            path: Path to Foundation.lean file
            metadata_path: Metadata to rebuild from, when it is not the
                sidecar next to `path` (a checkpoint's copy)

        Returns:
            Loaded FoundationFile
        """
        foundation = cls(path)

        if metadata_path is None:
            metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            foundation.entries = [
                FoundationEntry.from_dict(e) for e in metadata.get("entries", [])
            ]
            foundation._artifact_ids = set(metadata.get("artifact_ids", []))
            foundation._rejections = dict(metadata.get("rejections", {}))
            foundation._refused_in = dict(metadata.get("refused_in", {}))
            # A checkpoint from before this field records only what landed.
            if "accepted" in metadata:
                foundation._accepted = {
                    artifact_id: [FoundationEntry.from_dict(e) for e in entries]
                    for artifact_id, entries in metadata["accepted"].items()
                }
            else:
                for entry in foundation.entries:
                    foundation._accepted.setdefault(entry.artifact_id, []).append(entry)
            # Rebuild definition names from loaded entries
            foundation._definition_names = {e.name for e in foundation.entries}
            # Rebuild claimed concepts from loaded entries
            for entry in foundation.entries:
                if entry.entry_type == "structure" and entry.name in cls.CORE_CONCEPTS:
                    foundation._claimed_concepts.add(cls.CORE_CONCEPTS[entry.name])
            # Reinstall the seed last: it claims its own names on top of the
            # rebuilt sets, so a resumed run protects the axiom layer exactly
            # as the original run did.
            if metadata.get("seed_source"):
                foundation.set_seed(metadata["seed_source"])

        return foundation
