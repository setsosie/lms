"""Novelty classification (N0 / N1) for Lean statements — 26Q3-HARN-04.

The novelty ladder (`specs/faithfulness_protocol.md` §6.1):

| Level | Meaning |
|-------|---------|
| N0    | Re-proof of an existing Mathlib result |
| N1    | Known textbook statement absent from Mathlib |
| N2/N3 | Require a *literature* search — out of scope here, by design |

`classify_novelty` runs the four-stage Mathlib search ladder from
`lms.novelty.mathlib_search`, short-circuiting on a confident N0 hit. A
statement is only labelled N1 when the available stages all came up empty, and
the confidence of that N1 scales with how many stages could actually run: an
"absent from Mathlib" claim backed by one reachable backend is worth much less
than one backed by all four. INCONCLUSIVE (and low-confidence N1) routes to D4
human review — it is never counted as novel without sign-off.

Every result records the Mathlib revision it was computed against. Mathlib
moves; yesterday's N1 is today's N0.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from lms.novelty.mathlib_search import (
    DiskCache,
    SearchBackend,
    SearchHit,
    StageOutcome,
    StatementQuery,
    default_backends,
    detect_mathlib_rev,
    extract_identifiers,
    name_tokens,
)

__all__ = [
    "DECISIVE_CONFIDENCE",
    "NoveltyClassifier",
    "NoveltyLevel",
    "NoveltyResult",
    "classify_novelty",
    "measure_density",
]

# An N0 claim at or above this confidence short-circuits the ladder; an N1
# below it routes to D4 review instead of being counted as novel.
DECISIVE_CONFIDENCE = 0.8

# Confidence granted to an "absent from Mathlib" (N1) verdict, by how many
# search stages were actually able to run. All four stages empty is a strong
# absence claim; one stage empty is barely evidence at all.
_N1_CONFIDENCE_BY_STAGES = {4: 0.9, 3: 0.75, 2: 0.6, 1: 0.45, 0: 0.0}

# Weakest N0 evidence still worth flagging for human eyes: below decisive but
# above this floor the result is INCONCLUSIVE rather than N1.
_N0_CANDIDATE_FLOOR = 0.5


class NoveltyLevel(str, Enum):
    """Where a statement sits on the novelty ladder (machine-checkable part)."""

    N0 = "N0"
    N1 = "N1"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class NoveltyResult:
    """Outcome of classifying one Lean statement.

    ``confidence`` is confidence *in the assigned level*. ``evidence`` names
    the matching Mathlib declarations for an N0 verdict so the claim is
    auditable; for N1 it names the stages that came up empty.
    """

    level: NoveltyLevel
    confidence: float
    evidence: list[str] = field(default_factory=list)
    mathlib_rev: str | None = None
    decisive_stage: str | None = None
    stages_available: list[str] = field(default_factory=list)
    stages_unavailable: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """True when this result must not be counted without D4 sign-off."""
        if self.level is NoveltyLevel.INCONCLUSIVE:
            return True
        return self.level is NoveltyLevel.N1 and self.confidence < DECISIVE_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "mathlib_rev": self.mathlib_rev,
            "decisive_stage": self.decisive_stage,
            "stages_available": self.stages_available,
            "stages_unavailable": self.stages_unavailable,
            "needs_review": self.needs_review,
        }


def _hit_evidence(stage: str, hit: SearchHit) -> str:
    module = f" ({hit.module})" if hit.module else ""
    return f"{stage}: {hit.name}{module}"


def _token_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NoveltyClassifier:
    """Runs the search ladder and turns stage outcomes into a NoveltyResult."""

    def __init__(
        self,
        backends: Sequence[SearchBackend],
        cache: DiskCache | None = None,
        mathlib_rev: str | None = None,
    ) -> None:
        self.backends = list(backends)
        self.cache = cache
        self.mathlib_rev = mathlib_rev

    def classify(
        self,
        lean_statement: str,
        informal: str | None = None,
    ) -> NoveltyResult:
        query = StatementQuery.from_lean(lean_statement, informal=informal)
        best_confidence = 0.0
        evidence: list[str] = []
        decisive_stage: str | None = None
        stages_available: list[str] = []
        stages_unavailable: list[str] = []

        for backend in self.backends:
            outcome = self._run_stage(backend, query)
            if not outcome.available:
                stages_unavailable.append(backend.stage)
                continue
            stages_available.append(backend.stage)
            confidence, stage_evidence = self._score(outcome, query)
            evidence.extend(stage_evidence)
            if confidence > best_confidence:
                best_confidence = confidence
                decisive_stage = backend.stage
            if best_confidence >= DECISIVE_CONFIDENCE:
                break

        return self._verdict(
            best_confidence,
            evidence,
            decisive_stage,
            stages_available,
            stages_unavailable,
        )

    def _run_stage(self, backend: SearchBackend, query: StatementQuery) -> StageOutcome:
        key = None
        if self.cache is not None:
            key = DiskCache.key(
                backend.stage, backend.cache_query(query), self.mathlib_rev
            )
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        outcome = backend.search(query)
        # Only genuine answers are cached: an unavailable backend today may be
        # available tomorrow (e.g. after `lake build` on the box).
        if self.cache is not None and key is not None and outcome.available:
            self.cache.put(key, outcome)
        return outcome

    def _score(
        self, outcome: StageOutcome, query: StatementQuery
    ) -> tuple[float, list[str]]:
        """Map one stage's outcome to (N0 confidence, evidence strings)."""
        if outcome.stage == "exact_probe":
            if outcome.closed_by:
                return 0.95, [f"exact_probe: closed by `{outcome.closed_by}`"]
            return 0.0, []

        if not outcome.hits:
            return 0.0, []

        last = (query.name or "").split(".")[-1]
        query_tokens = set(name_tokens(query.name or ""))
        query_tokens.update(
            t.lower()
            for ident in extract_identifiers(query.lean_statement)
            for t in name_tokens(ident)
        )

        best = 0.0
        scored: list[tuple[float, str]] = []
        for hit in outcome.hits[:10]:
            hit_last = hit.name.split(".")[-1]
            if outcome.stage == "name":
                score = 0.95 if hit_last == last else 0.85
            elif outcome.stage == "loogle":
                if hit_last == last:
                    score = 0.9
                elif last and last.lower() in hit.name.lower():
                    score = 0.75
                else:
                    score = 0.55
            else:  # semantic
                hit_tokens = set(name_tokens(hit.name))
                if hit.type_signature:
                    hit_tokens.update(
                        t.lower()
                        for ident in extract_identifiers(hit.type_signature)
                        for t in name_tokens(ident)
                    )
                overlap = _token_overlap(query_tokens, hit_tokens)
                if hit_last == last:
                    score = 0.75
                elif overlap >= 0.8:
                    score = 0.7
                elif overlap >= 0.5:
                    score = 0.55
                else:
                    score = 0.3
            if score > best:
                best = score
            if score >= _N0_CANDIDATE_FLOOR:
                scored.append((score, _hit_evidence(outcome.stage, hit)))
        # Best match first: `evidence[0]` is what a D4 reviewer reads.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return best, [e for _, e in scored[:5]]

    def _verdict(
        self,
        best_confidence: float,
        evidence: list[str],
        decisive_stage: str | None,
        stages_available: list[str],
        stages_unavailable: list[str],
    ) -> NoveltyResult:
        if best_confidence >= DECISIVE_CONFIDENCE:
            level, confidence = NoveltyLevel.N0, best_confidence
        elif best_confidence >= _N0_CANDIDATE_FLOOR:
            # Plausible Mathlib match that no stage could confirm decisively.
            level, confidence = NoveltyLevel.INCONCLUSIVE, 0.5
            decisive_stage = None
        else:
            n1_confidence = _N1_CONFIDENCE_BY_STAGES.get(len(stages_available), 0.0)
            if n1_confidence < _N0_CANDIDATE_FLOOR:
                # Too few stages could run for "not found" to mean anything.
                level, confidence = NoveltyLevel.INCONCLUSIVE, 0.0
                decisive_stage = None
                if not evidence:
                    evidence = [
                        f"insufficient search coverage: only {stages_available or 'none'} ran"
                    ]
            else:
                level, confidence = NoveltyLevel.N1, n1_confidence
                decisive_stage = None
                evidence = [f"not found by: {', '.join(stages_available)}"]
        return NoveltyResult(
            level=level,
            confidence=confidence,
            evidence=evidence,
            mathlib_rev=self.mathlib_rev,
            decisive_stage=decisive_stage,
            stages_available=stages_available,
            stages_unavailable=stages_unavailable,
        )


def classify_novelty(
    lean_decl: str,
    *,
    project_dir: Path | str = "lean",
    cache_dir: Path | str | None = None,
    informal: str | None = None,
    classifier: NoveltyClassifier | None = None,
) -> NoveltyResult:
    """Classify one Lean declaration against Mathlib.

    Convenience wrapper that builds the default production ladder. Pass a
    ``classifier`` to reuse backends/cache across many statements (the density
    script does; so should any batch caller).
    """
    if classifier is None:
        cache = DiskCache(cache_dir) if cache_dir is not None else None
        classifier = NoveltyClassifier(
            default_backends(project_dir),
            cache=cache,
            mathlib_rev=detect_mathlib_rev(project_dir),
        )
    return classifier.classify(lean_decl, informal=informal)


def measure_density(
    doc: dict[str, Any],
    classifier: NoveltyClassifier,
) -> dict[str, Any]:
    """Classify every statement of an arc file and report N1 density.

    ``doc`` follows the arc-statements schema (see
    ``scripts/measure_n1_density.py``). The report carries the full confidence
    distribution, not just the mean: on a *density* measurement, systematic
    bias matters more than per-item error.
    """
    statements = doc.get("statements", [])
    results: list[dict[str, Any]] = []
    counts = {level: 0 for level in NoveltyLevel}
    decisive_n1 = 0
    review_queue: list[str] = []

    for stmt in statements:
        result = classifier.classify(
            stmt["lean_statement"],
            informal=stmt.get("informal"),
        )
        counts[result.level] += 1
        if result.level is NoveltyLevel.N1 and not result.needs_review:
            decisive_n1 += 1
        if result.needs_review:
            review_queue.append(stmt["id"])
        results.append({"id": stmt["id"], "name": stmt.get("name"), **result.to_dict()})

    total = len(statements)
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for r in results:
        c = min(r["confidence"], 0.999)
        low = int(c * 5) * 2
        buckets[f"{low / 10:.1f}-{(low + 2) / 10:.1f}"] += 1

    return {
        "arc": doc.get("arc"),
        "source": doc.get("source"),
        "mathlib_rev": classifier.mathlib_rev,
        "total_statements": total,
        "counts": {level.value: n for level, n in counts.items()},
        # Upper bound: every N1 label, decisive or not.
        "n1_density": (counts[NoveltyLevel.N1] / total) if total else 0.0,
        # Lower bound: only N1 labels confident enough to stand without D4.
        "n1_density_decisive": (decisive_n1 / total) if total else 0.0,
        "confidence_distribution": buckets,
        "needs_review": review_queue,
        "statements": results,
    }
