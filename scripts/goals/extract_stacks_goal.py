"""Generate shared-kernel goal JSONs from the Stacks Project TeX source.

Reads a local clone of the Stacks Project (untracked, expected at
``references/stacks-project/``) and emits goal files under ``goals/`` in the
exact schema ``lms.goals.Goal.save``/``Goal.load`` use. Every statement is the
verbatim TeX of the tagged definition/lemma, so the goal content is
source-anchored rather than paraphrased.

Dependency edges are source-anchored too: each definition's ``requires`` lists
the in-track tags its statement or proof cites with ``\\ref``. Only edges to
entries earlier in the curation order are kept; a forward reference is printed
and dropped, so a mis-ordered curation table shows up at regeneration time.

Usage:
    uv run python scripts/goals/extract_stacks_goal.py

Curation lives in the TRACKS table below: each entry is a Stacks tag plus the
role it plays in the track. Regenerating after editing the table is the
supported way to change a track's scope.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# The Stacks Project clone is untracked; override with STACKS_PROJECT_DIR when
# it lives outside this checkout (e.g. when running from a worktree).
STACKS = Path(
    os.environ.get("STACKS_PROJECT_DIR", REPO / "references" / "stacks-project")
)
OUT = REPO / "goals"

MAX_BODY_LINES = 40  # statement bodies are trimmed; proofs are read for refs only

REF_RE = re.compile(r"\\ref\{([^}]+)\}")

# ---------------------------------------------------------------------------
# Track curation: (tag, milestone) in dependency order. Dependency order is the
# Stacks Project's own presentation order, which is already topologically
# sorted; do not reorder entries within a file.
# ---------------------------------------------------------------------------

TRACK_B_CATEGORIES = [
    # 4.32 Categories over categories
    ("003Y", False),
    ("02XH", False),
    ("0040", False),
    ("02XI", False),
    # 4.33 Fibred categories
    ("02XK", False),
    ("02XL", False),
    ("06N4", False),
    ("02XM", False),
    ("02XN", False),
    ("02XO", False),
    ("042G", False),
    ("02XP", False),
    ("02XQ", False),
    ("02XR", False),
    # 4.34 Inertia
    ("034I", False),
    ("04Z6", False),
    # 4.35 Categories fibred in groupoids
    ("003T", False),
    ("003V", False),
    ("03WQ", False),
    ("02XS", False),
    ("0041", False),
    ("003Z", False),
    ("04Z7", False),
    ("02XT", False),
    # 4.36-4.37 Presheaves of categories / groupoids (split fibred categories)
    ("02XW", False),
    ("02XX", True),
    ("04TL", False),
    ("02XY", False),
    # 4.38-4.39 Categories fibred in sets / setoids
    ("0043", False),
    ("02Y2", False),
    ("04SA", False),
    ("04SC", False),
    # 4.40 Representable categories fibred in groupoids
    ("0046", False),
    ("02Y3", False),
    # 4.41 The 2-Yoneda lemma
    ("004B", True),
]

TRACK_B_STACKS = [
    # 8.3 Descent data in fibred categories
    ("026B", False),
    ("02ZD", False),
    ("02ZE", False),
    ("026D", False),
    ("026E", True),
]

TRACK_A_HOMOLOGY = [
    # 12.20 Spectral sequences: basic notions
    ("011N", False),
    ("011O", False),
    # 12.21 exact couples
    ("011Q", False),
    ("011R", False),
    ("011S", False),
    ("011T", False),
    # 12.22 differential objects
    ("011V", False),
    ("011X", False),
    # 12.23 filtered differential objects
    ("012B", False),
    ("012C", False),
    # 12.24 filtered complexes
    ("012L", False),
    ("012M", False),
    ("012P", False),
    ("012U", False),
    ("012V", False),
    ("012W", True),
    # 12.25 double complexes
    ("0130", False),
    ("0131", False),
    ("0132", True),
    ("0133", False),
]


def load_tags() -> dict[str, str]:
    """label -> tag, from the Stacks tags file."""
    out: dict[str, str] = {}
    for line in (STACKS / "tags" / "tags").read_text().splitlines():
        if line.startswith("#") or "," not in line:
            continue
        tag, label = line.split(",", 1)
        out[label] = tag
    return out


def resolve_refs(text: str, stem: str, label_to_tag: dict[str, str]) -> set[str]:
    """Tags of the statements ``text`` cites with ``\\ref``.

    Within a file the Stacks source writes labels without the file stem
    (``definition-fibred-category``); across files it writes the full label
    (``categories-definition-fibred-category``), which is the tags file's key.
    Labels the tags file does not know are dropped.
    """
    tags: set[str] = set()
    for label in REF_RE.findall(text):
        tag = label_to_tag.get(f"{stem}-{label}") or label_to_tag.get(label)
        if tag:
            tags.add(tag)
    return tags


def extract_file(fname: str, chapter: int) -> dict[str, dict]:
    """tag -> {section, name, content, refs} for every tagged statement in fname.

    ``refs`` resolves the references in the statement and in the proof that
    immediately follows it; nothing from the proof enters ``content``.
    """
    label_to_tag = load_tags()
    stem = fname.removesuffix(".tex")
    lines = (STACKS / fname).read_text().splitlines()

    results: dict[str, dict] = {}
    sec_no = 0
    sec_title = ""
    i = 0
    env_re = re.compile(r"\\begin\{(definition|lemma|proposition|theorem)\}")
    while i < len(lines):
        line = lines[i]
        if line.startswith("\\section{"):
            sec_no += 1
            sec_title = line[len("\\section{") : -1]
        m = env_re.match(line)
        if m:
            env = m.group(1)
            j = i + 1
            body: list[str] = []
            label = None
            while j < len(lines) and not lines[j].startswith(f"\\end{{{env}}}"):
                lm = re.match(r"\\label\{(.+)\}", lines[j])
                if lm:
                    label = lm.group(1)
                elif not lines[j].startswith("\\reference"):
                    body.append(lines[j])
                j += 1
            k = j + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            proof: list[str] = []
            if k < len(lines) and lines[k].startswith("\\begin{proof}"):
                while k < len(lines) and not lines[k].startswith("\\end{proof}"):
                    proof.append(lines[k])
                    k += 1
            refs = resolve_refs("\n".join(body + proof), stem, label_to_tag)
            if label:
                tag = label_to_tag.get(f"{stem}-{label}", "")
                if tag:
                    if len(body) > MAX_BODY_LINES:
                        body = body[:MAX_BODY_LINES] + [
                            "[... statement trimmed; see the tag online ...]"
                        ]
                    results[tag] = {
                        "refs": refs,
                        "section": f"{chapter}.{sec_no}",
                        "name": f"{env.capitalize()}: {label.replace('-', ' ')} ({sec_title})",
                        "content": "\n".join(body).strip()
                        + f"\n\n[Stacks Tag {tag}, {stem}.tex, https://stacks.math.columbia.edu/tag/{tag}]",
                    }
            i = j
        i += 1
    return results


def build_goal(
    name: str,
    description: str,
    source: str,
    picks: list[tuple[list[tuple[str, bool]], dict[str, dict]]],
    preamble: str,
) -> dict:
    definitions = []
    position = {tag: i for i, (tag, _) in enumerate(t for tr, _ in picks for t in tr)}
    for track, extracted in picks:
        for tag, milestone in track:
            if tag not in extracted:
                raise SystemExit(
                    f"tag {tag} not found in source — check the curation table"
                )
            e = extracted[tag]
            cited = sorted(r for r in e["refs"] if r in position)
            for ref in cited:
                if position[ref] > position[tag]:
                    print(f"{tag}: dropped forward reference to {ref}")
            definitions.append(
                {
                    "tag": tag,
                    "section": e["section"],
                    "name": ("MILESTONE: " if milestone else "") + e["name"],
                    "content": e["content"],
                    "formalized": False,
                    "artifact_ids": [],
                    "requires": [r for r in cited if position[r] < position[tag]],
                }
            )
    return {
        "name": name,
        "description": description,
        "source": source,
        "definitions": definitions,
        # NOTE: Goal.load() currently reads only the four fields above; the
        # fields below are carried for the registration shim to apply (see
        # goals/README.md) and are ignored by a plain Goal.load().
        "allowed_imports": None,
        "forbidden_imports": [],
        "preamble": preamble,
    }


PREAMBLE_KERNEL = """/-
  SHARED KERNEL: build ON Mathlib, do not rebuild it.

  - Import Mathlib freely; prefer Mathlib.CategoryTheory.* as the substrate.
  - Extend what exists (e.g. CategoryTheory.FiberedCategory.*) instead of
    redefining it. A statement Mathlib already has is calibration only (N0)
    and must not be re-proved from scratch.
  - Cite the Stacks tag of the statement you formalize in a comment:
    `-- Stacks Tag 02XM`.
  - The compiled LMS corpus (LMS.Foundation) is importable and openable.
-/"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cats = extract_file("categories.tex", 4)
    stx = extract_file("stacks.tex", 8)
    hom = extract_file("homology.tex", 12)

    track_b = build_goal(
        name="Shared Kernel Track B: Fibred Categories and Descent",
        description=(
            "Formalize the fibred-category layer of the shared kernel: categories over a "
            "category, (strongly) cartesian morphisms, fibred categories and their 2-category, "
            "inertia, categories fibred in groupoids/sets/setoids, split fibred categories vs "
            "presheaves of categories, representable CFGs, the 2-Yoneda lemma, and descent data "
            "in fibred categories. This bridges the compiled WC-3 2-category corpus to the "
            "sites/stacks layer. Build ON Mathlib (extend CategoryTheory.FiberedCategory.*); "
            "novelty is measured by the N0/N1 classifier, not assumed."
        ),
        source="The Stacks Project, Chapter 4 §32-41 and Chapter 8 §3 (https://stacks.math.columbia.edu)",
        picks=[(TRACK_B_CATEGORIES, cats), (TRACK_B_STACKS, stx)],
        preamble=PREAMBLE_KERNEL,
    )
    (OUT / "stacks_kernel_track_b.json").write_text(
        json.dumps(track_b, indent=2) + "\n"
    )

    track_a = build_goal(
        name="Shared Kernel Track A: Spectral Sequences",
        description=(
            "Formalize spectral sequences from the Stacks Project homological algebra chapter: "
            "the basic notions, exact couples and their derived couples, the spectral sequence "
            "of a differential object, of a filtered complex, and of a double complex, with "
            "convergence statements. Mathlib coverage of spectral sequences is essentially zero; "
            "this track is the committee-unanimous #1 kernel priority. Build ON Mathlib "
            "(homology API); novelty is measured by the N0/N1 classifier, not assumed."
        ),
        source="The Stacks Project, Chapter 12 §20-25 (https://stacks.math.columbia.edu)",
        picks=[(TRACK_A_HOMOLOGY, hom)],
        preamble=PREAMBLE_KERNEL,
    )
    (OUT / "stacks_kernel_track_a.json").write_text(
        json.dumps(track_a, indent=2) + "\n"
    )

    for f, g in (
        ("stacks_kernel_track_b.json", track_b),
        ("stacks_kernel_track_a.json", track_a),
    ):
        print(f"{f}: {len(g['definitions'])} statements")


if __name__ == "__main__":
    main()
