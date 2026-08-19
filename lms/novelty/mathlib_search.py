"""Mathlib search backends for the novelty classifier (26Q3-HARN-04).

Four search stages, tried in order and short-circuited by the classifier:

1. **name** — declaration-name match over a local Mathlib source checkout
   (`.lake/packages/mathlib`). Cheap and exact, but only available where the
   Lean project has fetched its packages (the box; not every dev machine).
2. **loogle** — type/name pattern search against the public Loogle API.
3. **exact_probe** — ask Lean itself whether `exact?` closes the statement's
   goal from Mathlib alone. The strongest N0 signal there is; requires a
   working `lake` + Mathlib build.
4. **semantic** — LeanSearch (and optionally LeanFinder) natural-language
   fallback. Weak signal on its own; never decisive by itself.

Every backend reports availability honestly instead of pretending: a stage
that cannot run on this machine returns ``available=False`` and the classifier
lowers its confidence accordingly. Results are cached on disk keyed by
(stage, query, mathlib_rev) so re-scoring archived runs costs nothing, and the
HTTP backends respect the public services' rate limits.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

LOOGLE_URL = "https://loogle.lean-lang.org"
LEANSEARCH_URL = "https://leansearch.net"

# Public-service rate limits, mirroring the lean-lsp MCP server's own caps.
LOOGLE_RATE = (3, 30.0)
LEANSEARCH_RATE = (90, 30.0)
LEANFINDER_RATE = (10, 30.0)

_DECL_KEYWORDS = (
    "theorem",
    "lemma",
    "def",
    "abbrev",
    "structure",
    "class",
    "instance",
    "inductive",
)

_DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+|@\[[^\]]*\]\s*)*"
    rf"({'|'.join(_DECL_KEYWORDS)})\s+([A-Za-z_][\w.']*)",
    re.MULTILINE,
)

_LEAN_KEYWORDS = frozenset(
    {
        "theorem",
        "lemma",
        "def",
        "abbrev",
        "structure",
        "class",
        "instance",
        "inductive",
        "where",
        "fun",
        "let",
        "by",
        "sorry",
        "Type",
        "Prop",
        "Sort",
        "exists",
        "forall",
        "if",
        "then",
        "else",
        "match",
        "with",
        "do",
        "return",
    }
)


def parse_declaration(lean_code: str) -> tuple[str | None, str | None]:
    """Extract ``(kind, name)`` of the first declaration in a Lean snippet.

    Returns ``(None, None)`` when no declaration header is recognizable
    (e.g. a bare ``example``).
    """
    m = _DECL_RE.search(lean_code)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def extract_identifiers(lean_code: str) -> list[str]:
    """Capitalized / dotted identifiers appearing in a statement.

    Used to build loogle queries and to score semantic hits. Binder names and
    Lean keywords are excluded; order of first appearance is preserved.
    """
    seen: dict[str, None] = {}
    for tok in re.findall(r"[A-Za-z_][\w.']*", lean_code):
        if tok in _LEAN_KEYWORDS or tok in seen:
            continue
        head = tok.split(".")[0]
        if not head[:1].isupper():
            continue
        seen[tok] = None
    return list(seen)


def name_tokens(name: str) -> list[str]:
    """Lowercase word tokens of a declaration name, for fuzzy comparison."""
    parts = re.split(r"[._']", name)
    tokens: list[str] = []
    for part in parts:
        tokens.extend(t.lower() for t in re.findall(r"[A-Za-z][a-z]*|\d+", part))
    return [t for t in tokens if t]


@dataclass(frozen=True)
class SearchHit:
    """A single declaration returned by a search backend."""

    name: str
    module: str | None = None
    type_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "type_signature": self.type_signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchHit":
        return cls(
            name=d["name"],
            module=d.get("module"),
            type_signature=d.get("type_signature"),
        )


@dataclass
class StageOutcome:
    """What one search stage found (or why it could not run)."""

    stage: str
    available: bool
    hits: list[SearchHit] = field(default_factory=list)
    # Set by the exact-probe stage: the `exact?` invocation that closed the goal.
    closed_by: str | None = None
    error: str | None = None
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "available": self.available,
            "hits": [h.to_dict() for h in self.hits],
            "closed_by": self.closed_by,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageOutcome":
        return cls(
            stage=d["stage"],
            available=d["available"],
            hits=[SearchHit.from_dict(h) for h in d.get("hits", [])],
            closed_by=d.get("closed_by"),
            error=d.get("error"),
        )


@dataclass(frozen=True)
class StatementQuery:
    """Everything the backends need to know about one statement."""

    lean_statement: str
    name: str | None = None
    informal: str | None = None

    @classmethod
    def from_lean(cls, lean_code: str, informal: str | None = None) -> "StatementQuery":
        _, decl_name = parse_declaration(lean_code)
        return cls(lean_statement=lean_code, name=decl_name, informal=informal)

    @property
    def display_name(self) -> str:
        return self.name or self.lean_statement[:60]


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per ``window`` seconds.

    ``clock``/``sleep`` are injectable so tests never actually wait.
    """

    def __init__(
        self,
        max_calls: int,
        window: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls = max_calls
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            wait = self.window - (now - self._calls[0])
            if wait > 0:
                self._sleep(wait)
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.window:
                self._calls.popleft()
        self._calls.append(self._clock())


class DiskCache:
    """One JSON file per (stage, query, mathlib_rev) triple.

    Mathlib moves; a stale N1 becomes an N0 when upstream lands it. Keying the
    cache on the Mathlib revision means a bumped pin re-queries instead of
    replaying answers about a Mathlib that no longer exists.
    """

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def key(stage: str, query: str, mathlib_rev: str | None) -> str:
        payload = json.dumps(
            {"stage": stage, "query": query, "mathlib_rev": mathlib_rev},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> StageOutcome | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            outcome = StageOutcome.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None
        outcome.from_cache = True
        return outcome

    def put(self, key: str, outcome: StageOutcome) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(outcome.to_dict(), indent=2))


class SearchBackend(Protocol):
    """One stage of the search ladder."""

    stage: str

    def is_available(self) -> bool: ...

    def cache_query(self, query: StatementQuery) -> str:
        """The string identifying this query for cache purposes."""
        ...

    def search(self, query: StatementQuery) -> StageOutcome: ...


def detect_mathlib_rev(project_dir: Path | str) -> str | None:
    """Read the pinned Mathlib revision from the project's lake manifest."""
    manifest = Path(project_dir) / "lake-manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None
    for pkg in data.get("packages", []):
        if pkg.get("name") == "mathlib":
            rev = pkg.get("rev")
            return str(rev) if rev else None
    return None


class MathlibNameSearch:
    """Stage 1: declaration-name grep over a local Mathlib source tree."""

    stage = "name"

    def __init__(self, project_dir: Path | str = "lean") -> None:
        self.mathlib_dir = Path(project_dir) / ".lake" / "packages" / "mathlib"
        self._grep = shutil.which("grep")

    def is_available(self) -> bool:
        return self._grep is not None and (self.mathlib_dir / "Mathlib").is_dir()

    def cache_query(self, query: StatementQuery) -> str:
        return f"name:{query.name or ''}"

    def search(self, query: StatementQuery) -> StageOutcome:
        if not query.name:
            return StageOutcome(self.stage, available=True, error="no declaration name")
        if not self.is_available():
            return StageOutcome(
                self.stage, available=False, error="no local Mathlib source"
            )
        # Final name component: Mathlib namespacing means `Functor.comp` lives
        # in `namespace CategoryTheory.Functor` as `comp`.
        last = query.name.split(".")[-1]
        pattern = (
            rf"^\s*(theorem|lemma|def|abbrev|structure|class|instance|inductive)"
            rf"\s+([A-Za-z_][A-Za-z0-9_.']*\.)?{re.escape(last)}\b"
        )
        assert self._grep is not None
        proc = subprocess.run(
            [
                self._grep,
                "-rInE",
                "--include=*.lean",
                pattern,
                str(self.mathlib_dir / "Mathlib"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # grep exits 1 on "no matches", which is a result, not a failure.
        if proc.returncode not in (0, 1):
            return StageOutcome(
                self.stage, available=True, error=proc.stderr.strip()[:500]
            )
        hits = []
        for line in proc.stdout.splitlines()[:50]:
            try:
                file_part, _line_no, text = line.split(":", 2)
            except ValueError:
                continue
            _kind, decl_name = parse_declaration(text)
            if decl_name is None:
                continue
            module = (
                Path(file_part)
                .relative_to(self.mathlib_dir)
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
            )
            hits.append(SearchHit(name=decl_name, module=module, type_signature=None))
        return StageOutcome(self.stage, available=True, hits=hits)


class _HttpBackend:
    """Shared plumbing for the JSON-over-HTTP backends."""

    def __init__(self, limiter: RateLimiter, timeout: float = 20.0) -> None:
        self._limiter = limiter
        self._timeout = timeout

    def _get_json(self, url: str) -> Any:
        self._limiter.acquire()
        req = urllib.request.Request(url, headers={"User-Agent": "lms-novelty/0.1"})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())

    def _post_json(self, url: str, body: dict[str, Any]) -> Any:
        self._limiter.acquire()
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "lms-novelty/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())


class LoogleBackend(_HttpBackend):
    """Stage 2: Loogle name/pattern search.

    The auto-derived query is a *name-substring* search (``"foo"``). Deriving a
    reliable type-pattern query from arbitrary agent Lean is future work; a
    name miss here simply means the ladder continues.
    """

    stage = "loogle"

    def __init__(
        self,
        base_url: str = LOOGLE_URL,
        limiter: RateLimiter | None = None,
        timeout: float = 20.0,
    ) -> None:
        super().__init__(limiter or RateLimiter(*LOOGLE_RATE), timeout)
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _query_string(query: StatementQuery) -> str | None:
        if not query.name:
            return None
        return f'"{query.name.split(".")[-1]}"'

    def cache_query(self, query: StatementQuery) -> str:
        return f"loogle:{self._query_string(query) or ''}"

    def search(self, query: StatementQuery) -> StageOutcome:
        q = self._query_string(query)
        if q is None:
            return StageOutcome(self.stage, available=True, error="no declaration name")
        url = f"{self.base_url}/json?q={urllib.parse.quote(q)}"
        try:
            data = self._get_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return StageOutcome(self.stage, available=False, error=str(exc)[:500])
        if isinstance(data, dict) and data.get("error"):
            return StageOutcome(
                self.stage, available=True, error=str(data["error"])[:500]
            )
        hits = [
            SearchHit(
                name=h.get("name", ""),
                module=h.get("module"),
                type_signature=h.get("type"),
            )
            for h in (data.get("hits") or [])[:20]
            if h.get("name")
        ]
        return StageOutcome(self.stage, available=True, hits=hits)


class ExactProbeBackend:
    """Stage 3: does ``exact?`` close the goal from Mathlib alone?

    A closed goal is the strongest N0 signal available. An *elaboration* error
    is recorded but proves nothing either way: statements phrased against
    project-local definitions will not even elaborate in a pure-Mathlib
    context, which is a naming mismatch, not novelty.
    """

    stage = "exact_probe"

    def __init__(
        self, project_dir: Path | str = "lean", timeout: float = 120.0
    ) -> None:
        self.project_dir = Path(project_dir)
        self.timeout = timeout
        self._lake = shutil.which("lake")

    def is_available(self) -> bool:
        return (
            self._lake is not None
            and (self.project_dir / "lakefile.toml").exists()
            and (self.project_dir / ".lake" / "packages" / "mathlib").is_dir()
        )

    def cache_query(self, query: StatementQuery) -> str:
        return f"exact_probe:{query.lean_statement}"

    @staticmethod
    def probe_source(lean_statement: str) -> str | None:
        """Rewrite ``theorem foo ... := proof`` as an ``exact?`` probe."""
        m = _DECL_RE.search(lean_statement)
        if not m or m.group(1) not in ("theorem", "lemma"):
            return None
        rest = lean_statement[m.end() :]
        body_start = rest.find(":=")
        header = rest[:body_start] if body_start != -1 else rest
        header = header.strip()
        if ":" not in header:
            return None
        return f"import Mathlib\n\nexample {header} := by exact?\n"

    def search(self, query: StatementQuery) -> StageOutcome:
        source = self.probe_source(query.lean_statement)
        if source is None:
            return StageOutcome(
                self.stage, available=True, error="statement not probeable"
            )
        if not self.is_available():
            return StageOutcome(
                self.stage, available=False, error="no lake or Mathlib build"
            )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            dir=self.project_dir,
            delete=False,
        ) as f:
            f.write(source)
            probe_path = Path(f.name)
        try:
            assert self._lake is not None
            proc = subprocess.run(
                [
                    self._lake,
                    "env",
                    "lean",
                    "-DautoImplicit=false",
                    "-DrelaxedAutoImplicit=false",
                    str(probe_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.project_dir,
            )
        except subprocess.TimeoutExpired:
            return StageOutcome(self.stage, available=True, error="probe timed out")
        finally:
            probe_path.unlink(missing_ok=True)
        output = proc.stdout + proc.stderr
        m = re.search(r"Try this:\s*(exact .+)", output)
        if m:
            return StageOutcome(
                self.stage, available=True, closed_by=m.group(1).strip()
            )
        return StageOutcome(
            self.stage,
            available=True,
            error=output.strip()[:500] or None,
        )


class LeanSearchBackend(_HttpBackend):
    """Stage 4: LeanSearch natural-language search."""

    stage = "semantic"

    def __init__(
        self,
        base_url: str = LEANSEARCH_URL,
        limiter: RateLimiter | None = None,
        timeout: float = 30.0,
        num_results: int = 8,
    ) -> None:
        super().__init__(limiter or RateLimiter(*LEANSEARCH_RATE), timeout)
        self.base_url = base_url.rstrip("/")
        self.num_results = num_results

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _query_string(query: StatementQuery) -> str:
        return query.informal or query.lean_statement[:300]

    def cache_query(self, query: StatementQuery) -> str:
        return f"semantic:{self._query_string(query)}"

    def search(self, query: StatementQuery) -> StageOutcome:
        body = {"query": [self._query_string(query)], "num_results": self.num_results}
        try:
            data = self._post_json(f"{self.base_url}/search", body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return StageOutcome(self.stage, available=False, error=str(exc)[:500])
        hits: list[SearchHit] = []
        try:
            for item in data[0][: self.num_results]:
                result = item.get("result", {})
                name_parts = result.get("name") or []
                module_parts = result.get("module_name") or []
                if not name_parts:
                    continue
                hits.append(
                    SearchHit(
                        name=".".join(name_parts),
                        module=".".join(module_parts) or None,
                        type_signature=result.get("signature") or result.get("type"),
                    )
                )
        except (IndexError, TypeError, AttributeError):
            return StageOutcome(
                self.stage, available=True, error="unexpected response shape"
            )
        return StageOutcome(self.stage, available=True, hits=hits)


class RecordedBackend:
    """Replays recorded outcomes; used in tests and offline fixture replay."""

    def __init__(
        self, stage: str, outcomes: dict[str, StageOutcome], available: bool = True
    ):
        self.stage = stage
        self._outcomes = outcomes
        self._available = available
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def cache_query(self, query: StatementQuery) -> str:
        return f"{self.stage}:{query.display_name}"

    def search(self, query: StatementQuery) -> StageOutcome:
        key = query.display_name
        self.calls.append(key)
        if not self._available:
            return StageOutcome(
                self.stage, available=False, error="recorded as unavailable"
            )
        outcome = self._outcomes.get(key)
        if outcome is None:
            return StageOutcome(self.stage, available=True, hits=[])
        return outcome


def default_backends(
    project_dir: Path | str = "lean",
    stages: Iterable[str] | None = None,
) -> list[SearchBackend]:
    """The production search ladder, in classification order."""
    ladder: list[SearchBackend] = [
        MathlibNameSearch(project_dir),
        LoogleBackend(),
        ExactProbeBackend(project_dir),
        LeanSearchBackend(),
    ]
    if stages is None:
        return ladder
    wanted = set(stages)
    return [b for b in ladder if b.stage in wanted]
