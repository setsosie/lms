"""Best-effort structural parsing of Lean 4 source for the machine gates.

This is a scanner, not a parser: it recovers declaration names, namespaces and
theorem binders from well-formed agent output. It does not understand macros,
`open ... in`, or string literals containing comment markers. Every consumer
treats a parse it cannot complete as INCONCLUSIVE, never as a pass — so the
failure mode of this module is a statement routed to human review, not a
statement miscounted.
"""

import re
from dataclasses import dataclass

__all__ = [
    "BinderGroup",
    "LeanDeclaration",
    "TheoremSignature",
    "extract_declarations",
    "parse_theorem_signature",
    "strip_comments",
]

# Bracket pairs that can open a binder group.
_OPEN_TO_CLOSE = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}

_DECL_KEYWORDS = (
    "theorem",
    "lemma",
    "example",
    "def",
    "abbrev",
    "structure",
    "class",
    "inductive",
    "instance",
    "axiom",
    "opaque",
)

# Attributes and modifiers that may precede a declaration keyword.
_DECL_RE = re.compile(
    r"^\s*"
    r"(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|scoped|local)\s+)*"
    r"(?:(?P<classind>class\s+inductive)|(?P<kw>" + "|".join(_DECL_KEYWORDS) + r"))"
    r"(?:\s+(?P<name>[A-Za-z_][A-Za-z0-9_'.!?]*))?"
)

_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)")
_SECTION_RE = re.compile(r"^\s*section(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\s*$")
_END_RE = re.compile(r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\s*$")


def strip_comments(code: str) -> str:
    """Remove `--` line comments and (nested) `/- -/` block comments.

    Comment markers inside string literals are not recognized — acceptable for
    agent-produced mathematics, and an error here only widens what a gate
    scans, it cannot hide anything from it.
    """
    out: list[str] = []
    i = 0
    depth = 0
    n = len(code)
    while i < n:
        two = code[i : i + 2]
        if depth == 0 and two == "/-":
            depth = 1
            i += 2
        elif depth > 0 and two == "/-":
            depth += 1
            i += 2
        elif depth > 0 and two == "-/":
            depth -= 1
            i += 2
        elif depth > 0:
            # Preserve newlines so line numbers survive stripping.
            if code[i] == "\n":
                out.append("\n")
            i += 1
        elif two == "--":
            while i < n and code[i] != "\n":
                i += 1
        else:
            out.append(code[i])
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class LeanDeclaration:
    """One top-level declaration found in a source snippet."""

    keyword: str
    name: str | None  # None for anonymous (example, bare instance)
    full_name: str | None  # namespace-qualified, when a name exists
    line: int  # 1-indexed line in the original source

    @property
    def anonymous(self) -> bool:
        return self.name is None


def extract_declarations(code: str) -> list[LeanDeclaration]:
    """Scan for declarations, tracking the namespace stack for full names."""
    decls: list[LeanDeclaration] = []
    # Stack of (kind, name) for kind in {"namespace", "section"}.
    stack: list[tuple[str, str | None]] = []

    for lineno, line in enumerate(strip_comments(code).splitlines(), start=1):
        ns_match = _NAMESPACE_RE.match(line)
        if ns_match:
            stack.append(("namespace", ns_match.group(1)))
            continue
        if _SECTION_RE.match(line):
            stack.append(("section", _SECTION_RE.match(line).group(1)))  # type: ignore[union-attr]
            continue
        if _END_RE.match(line):
            if stack:
                stack.pop()
            continue

        m = _DECL_RE.match(line)
        if not m:
            continue
        keyword = "inductive" if m.group("classind") else m.group("kw")
        name = m.group("name")
        # `example` never has a name; the regex can capture the start of its
        # statement (`example : ...` captures nothing, but `example foo : ...`
        # would). Treat every example as anonymous.
        if keyword == "example":
            name = None
        # `instance : Foo` is anonymous; the regex's name group only matches
        # identifiers, and `:` stops it, so this is already None.
        full_name = None
        if name is not None:
            prefix = ".".join(n for k, n in stack if k == "namespace" and n)
            full_name = f"{prefix}.{name}" if prefix else name
        decls.append(
            LeanDeclaration(
                keyword=keyword, name=name, full_name=full_name, line=lineno
            )
        )
    return decls


@dataclass(frozen=True)
class BinderGroup:
    """One binder group of a theorem signature, e.g. `(x : T)` or `[Group G]`."""

    bracket: str  # "(", "{", "[" or "⦃"
    names: str | None  # left of the top-level colon; None if no colon
    type: str  # right of the top-level colon, or the whole group


@dataclass(frozen=True)
class TheoremSignature:
    """Binders and goal of a `theorem`/`lemma`, when the scan succeeds."""

    name: str
    binders: tuple[BinderGroup, ...]
    goal: str


def _split_binder(inner: str) -> tuple[str | None, str]:
    """Split a binder group's contents at its first top-level colon."""
    depth = 0
    for i, ch in enumerate(inner):
        if ch in _OPEN_TO_CLOSE:
            depth += 1
        elif ch in _OPEN_TO_CLOSE.values():
            depth -= 1
        elif ch == ":" and depth == 0:
            # Don't split at `:=` — a default value means the "type" would be
            # a term, and the callers treat that group as unparseable anyway.
            if inner[i : i + 2] == ":=":
                return (None, inner.strip())
            return (inner[:i].strip() or None, inner[i + 1 :].strip())
    return (None, inner.strip())


def parse_theorem_signature(
    code: str, decl: LeanDeclaration
) -> TheoremSignature | None:
    """Recover binders and goal for a named theorem/lemma declaration.

    Returns None when the scan cannot complete (unbalanced brackets, no
    top-level `:` before `:=`). Callers must map None to INCONCLUSIVE.
    """
    if decl.keyword not in ("theorem", "lemma") or decl.name is None:
        return None

    stripped = strip_comments(code)
    lines = stripped.splitlines()
    if decl.line - 1 >= len(lines):
        return None
    # The signature may span lines; scan from the declaration line onward.
    # Re-match the declaration header so the scan starts right after the
    # name — a plain string split would hit the name's letters inside the
    # keyword ("t" is in "theorem").
    text = "\n".join(lines[decl.line - 1 :])
    m = _DECL_RE.match(text)
    if not m or m.group("name") != decl.name:
        return None
    rest = text[m.end() :]

    binders: list[BinderGroup] = []
    i = 0
    n = len(rest)
    while i < n:
        ch = rest[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _OPEN_TO_CLOSE:
            close = _OPEN_TO_CLOSE[ch]
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if rest[j] == ch:
                    depth += 1
                elif rest[j] == close:
                    depth -= 1
                j += 1
            if depth != 0:
                return None
            names, type_ = _split_binder(rest[i + 1 : j - 1])
            binders.append(BinderGroup(bracket=ch, names=names, type=type_))
            i = j
            continue
        if ch == ":":
            if rest[i : i + 2] == ":=":
                return None  # no statement colon before the body
            goal_region = rest[i + 1 :]
            # Goal runs to the `:=` or `by` that starts the proof.
            for stop in (":=",):
                idx = goal_region.find(stop)
                if idx != -1:
                    goal_region = goal_region[:idx]
                    break
            return TheoremSignature(
                name=decl.name, binders=tuple(binders), goal=goal_region.strip()
            )
        return None  # something before the statement colon we don't understand
    return None
