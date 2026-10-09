"""What a statement is phrased in, as far as a Mathlib search is concerned.

Two facts the search ladder cannot see from its own results:

- **Project vocabulary.** A statement over `LMS.Foundation`'s hand-rolled
  `Category` shares no names, types or structure with Mathlib's, so every
  stage comes up empty whether or not Mathlib proves the same theorem. Silence
  from such a search is not evidence of absence. The vocabulary can arrive by
  name, through a `variable` binder, or as notation (`f ≫ g`).
- **Named results.** "Yoneda lemma" in the informal statement names a result
  regardless of the API the Lean is written against. A Mathlib hit carrying
  the same eponym is a candidate match the Lean-side scoring cannot see.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from lms.novelty.mathlib_search import (
    declared_names,
    extract_identifiers,
    name_tokens,
    statement_header,
)

__all__ = ["ProjectVocabulary", "named_results", "project_vocabulary"]

# Import roots that are libraries the search ladder can reach (or Lean's own).
# Anything else is a project module.
LIBRARY_ROOTS = frozenset(
    {
        "Aesop",
        "Batteries",
        "ImportGraph",
        "Init",
        "Lake",
        "Lean",
        "LeanSearchClient",
        "Mathlib",
        "Plausible",
        "ProofWidgets",
        "Qq",
        "Std",
    }
)

_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)", re.MULTILINE)
# Dotted names, stopping before an explicit universe list (`Category.{v}`).
_TOKEN_RE = re.compile(r"[A-Za-z_][\w']*(?:\.[\w']+)*")
_VARIABLE_RE = re.compile(r"^\s*variable\b(.*(?:\n[ \t]+\S.*)*)", re.MULTILINE)
_NOTATION_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:scoped\s+|local\s+)?"
    r"(?:notation|infixl|infixr|infix|prefix|postfix)\b([^\n]*?)=>",
    re.MULTILINE,
)

_RESULT_KINDS = (
    "lemma|theorem|embedding|criterion|formula|inequality|identity|bound"
    "|conjecture|correspondence|duality|reciprocity"
)
# An eponym directly before the kind word, allowing up to two lowercase words
# between them: "Yoneda lemma", "Nakayama's lemma", "Lagrange's four-square
# theorem", "the Sylow theorems". Only the kind word ignores case; the eponym
# must be capitalised. A kind word followed by a number ("Lemma 3") is a
# reference, not a name.
_NAMED_RESULT_RE = re.compile(
    rf"\b([A-Z]\w*(?:[-–—][A-Z]\w*)*)(?:['’]s|['’])?(?:\s+[a-z][a-z-]*){{0,2}}\s+"
    rf"(?i:(?:{_RESULT_KINDS})s?)\b(?!\s*\d)"
)
# Capitalised words that precede "theorem" without naming anyone: sentence
# starts, ordinals, and the common nouns of unnamed results.
_NOT_EPONYMS = frozenset(
    {
        "a", "an", "and", "any", "as", "by", "each", "every", "for", "from",
        "hence", "in", "of", "or", "see", "then", "this", "thus", "the", "our",
        "using", "with",
        "first", "second", "third", "last", "final", "previous", "next",
        "main", "key", "basic", "standard", "general", "generalized",
        "fundamental", "following", "strong", "weak",
        "binomial", "central", "classification", "division", "dominated",
        "fixed", "implicit", "intermediate", "inverse", "isomorphism", "limit",
        "mean", "monotone", "prime", "rank", "remainder", "spectral",
        "structure", "uniform", "value",
    }
)  # fmt: skip


@dataclass
class ProjectVocabulary:
    """Names a statement takes from project modules rather than from Mathlib.

    Truthy when the statement uses any, or when an imported project module
    could not be read: vocabulary that cannot be checked is not assumed safe.
    """

    names: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    unread: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.names or self.unread)

    @property
    def reason(self) -> str:
        if self.names:
            return (
                f"phrased over project vocabulary ({', '.join(self.names)} from "
                f"{', '.join(self.modules)}): no Mathlib search can match it"
            )
        return (
            f"imports project module(s) {', '.join(self.unread)} that could not "
            "be read: whether the statement uses them is unknown"
        )


def _module_path(project_dir: Path, module: str) -> Path:
    return project_dir.joinpath(*module.split(".")).with_suffix(".lean")


def _uses(token: str, decl: str) -> bool:
    return token == decl or token.split(".")[0] == decl or token.endswith("." + decl)


def _notation_symbols(source: str) -> list[str]:
    """The literal tokens of every notation a module declares (`⟶`, `𝟙`)."""
    symbols: list[str] = []
    for m in _NOTATION_RE.finditer(source):
        for literal in re.findall(r'"([^"]*)"', m.group(1)):
            symbol = literal.strip()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _statement_context(lean_code: str, header: str) -> str:
    """The header plus the `variable` binders a statement may draw on.

    Every `variable` line counts, even one after the declaration: erring
    toward "uses project vocabulary" only ever withholds an N1.
    """
    return "\n".join([header, *_VARIABLE_RE.findall(lean_code)])


def project_vocabulary(
    lean_code: str, project_dir: Path | str | None
) -> ProjectVocabulary:
    """Which names and notations in a statement come from project modules.

    Only the header and `variable` binders count: a proof may use anything
    without changing what the statement says.
    """
    found = ProjectVocabulary()
    header = statement_header(lean_code)
    imports = [
        mod
        for mod in _IMPORT_RE.findall(lean_code)
        if mod.split(".")[0] not in LIBRARY_ROOTS
    ]
    if header is None or not imports:
        return found

    context = _statement_context(lean_code, header)
    tokens = list(dict.fromkeys(_TOKEN_RE.findall(context)))
    for module in imports:
        path = _module_path(Path(project_dir), module) if project_dir else None
        if path is None or not path.is_file():
            found.unread.append(module)
            continue
        source = path.read_text()
        decls = declared_names(source)
        used = [t for t in tokens if any(_uses(t, d) for d in decls)]
        for symbol in _notation_symbols(source):
            # A word-like notation must match a whole token, not a substring.
            in_use = (
                symbol in tokens
                if re.fullmatch(r"[A-Za-z_][\w']*", symbol)
                else symbol in context
            )
            if in_use:
                used.append(symbol)
        if used:
            found.modules.append(module)
            found.names.extend(n for n in used if n not in found.names)
    return found


def named_results(
    informal: str | None, lean_statement: str | None = None
) -> list[frozenset[str]]:
    """Eponyms of the named results an informal statement mentions.

    Each result is the set of name tokens a Mathlib declaration must carry to
    match it ("Kummer–Dedekind" needs both). An eponym the Lean statement
    already uses as vocabulary is dropped: "Dedekind's theorem" stated over
    `IsDedekindDomain` shares "Dedekind" with half of Mathlib's number theory,
    so it singles nothing out.
    """
    if not informal:
        return []
    header = statement_header(lean_statement or "") or ""
    used = {t for ident in extract_identifiers(header) for t in name_tokens(ident)}
    results: list[frozenset[str]] = []
    for m in _NAMED_RESULT_RE.finditer(informal):
        eponym = m.group(1)
        if eponym.lower() in _NOT_EPONYMS:
            continue
        tokens = frozenset(name_tokens(eponym))
        if tokens and not tokens <= used and tokens not in results:
            results.append(tokens)
    return results
