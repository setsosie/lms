"""Gate 4 — novelty (N0 / N1) — 26Q3-HARN-04.

Wraps `lms.novelty.classify_novelty` as a pass/fail gate over an `Artifact`
and stamps the classification onto the artifact so `artifacts.json` carries
the auditable record.

Gate semantics: only a *decisive* N1 counts as novel. N0 fails the gate.
INCONCLUSIVE and low-confidence N1 do not pass — they route to D4 human
review, because a wrong novelty label is worse than no label.
"""

from dataclasses import dataclass
from pathlib import Path

from lms.artifacts import Artifact
from lms.lean.interface import LEAN_GRADE_KINDS, LeanVerifier
from lms.novelty import NoveltyClassifier, NoveltyLevel, NoveltyResult
from lms.novelty.mathlib_search import (
    DiskCache,
    default_backends,
    detect_mathlib_rev,
)

__all__ = [
    "NoveltyGateDecision",
    "apply_novelty_gate",
    "default_novelty_classifier",
]


def default_novelty_classifier(
    verifier: LeanVerifier | None,
) -> NoveltyClassifier | None:
    """Build the production search ladder from a lean-grade verifier, else None.

    Mirrors `LeanProbeRunner.from_verifier`: without a real Lean project there
    is no local Mathlib to search and no revision to pin a verdict to, so the
    honest answer is "this gate cannot run" rather than a classifier whose
    stages all report unavailable.

    The disk cache lives beside the Lean project so repeated runs on the same
    box stop re-querying Loogle for statements they have already classified.
    """
    if verifier is None or verifier.verifier_kind not in LEAN_GRADE_KINDS:
        return None
    project = getattr(verifier, "project", None)
    project_dir = getattr(project, "project_dir", None)
    if project_dir is None:
        return None
    return NoveltyClassifier(
        default_backends(project_dir),
        cache=DiskCache(Path(project_dir) / ".lake" / "novelty-cache"),
        mathlib_rev=detect_mathlib_rev(project_dir),
    )


@dataclass
class NoveltyGateDecision:
    """What Gate 4 concluded about one artifact."""

    result: NoveltyResult
    # True only for decisive N1: this artifact may count toward CVFN.
    counts_as_novel: bool
    # True when the label cannot stand without D4 sign-off.
    needs_human_review: bool
    reason: str


def apply_novelty_gate(
    artifact: Artifact,
    classifier: NoveltyClassifier,
) -> NoveltyGateDecision:
    """Classify an artifact's Lean code and record the result on the artifact.

    Artifacts with no `lean_code` are INCONCLUSIVE by definition — there is
    nothing to search for.
    """
    if not artifact.lean_code:
        result = NoveltyResult(
            level=NoveltyLevel.INCONCLUSIVE,
            confidence=0.0,
            evidence=["no lean_code to classify"],
            mathlib_rev=classifier.mathlib_rev,
        )
    else:
        result = classifier.classify(
            artifact.lean_code,
            informal=artifact.natural_language,
        )

    artifact.novelty_level = result.level.value
    artifact.novelty_confidence = result.confidence
    artifact.novelty_evidence = list(result.evidence)

    counts = result.level is NoveltyLevel.N1 and not result.needs_review
    if counts:
        reason = f"decisive N1 (confidence {result.confidence:.2f})"
    elif result.level is NoveltyLevel.N0:
        reason = (
            f"N0 — found in Mathlib: {'; '.join(result.evidence[:2]) or 'see evidence'}"
        )
    elif result.needs_review:
        reason = f"{result.level.value} at confidence {result.confidence:.2f} — routed to D4 review"
    else:
        reason = f"{result.level.value} (confidence {result.confidence:.2f})"

    return NoveltyGateDecision(
        result=result,
        counts_as_novel=counts,
        needs_human_review=result.needs_review,
        reason=reason,
    )
