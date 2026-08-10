### 26Q3-HARN-02: Fix `lean_code` extraction

**User Story**: As the verifier, I want to receive exactly the Lean source the
agent wrote, so that compilation failures reflect the mathematics rather than a
parsing artifact.

| Field | Value |
|-------|-------|
| **Story Points** | 2 |
| **Priority** | CRITICAL |
| **Status** | ✅ DONE |
| **Branch** | `26Q3-HARN-02-lean-code-extraction` |
| **Dependencies** | 26Q3-CHORE-01 |
| **PR Size Target** | <200 lines |

---

#### Context

Every `lean_code` field in `experiments/stacks_ch4_phase1/artifacts.json` begins
with the literal characters `"|\n  "` — a YAML block-scalar marker captured as
source code.

Cause, `lms/agent.py:104-115`:

```python
ARTIFACT_PATTERN = re.compile(
    ...
    r"(?:lean:\s*(?P<lean>.+?)\s*)?"
    ...
    re.DOTALL,
)
```

When an agent writes `lean: |` (YAML block style), `\s*` stops at the `|`, which
then becomes the first character of the `lean` group. `lean_code.strip()`
(`lms/agent.py:481`) removes whitespace but not the pipe. The body also keeps the
block's indentation.

Consequences:
- Under the real verifier every such artifact fails to compile for a reason that
  has nothing to do with the math — a plausible contributor to the 0–6%
  verification rates in the December runs.
- Under the mock verifier it passes anyway, because the regex anchors on `^\s*`
  and matches `import`-free indented `example` lines further down.

**This bug makes both the failure numbers and the success numbers untrustworthy.**

---

#### Acceptance Criteria

- [x] A leading YAML block-scalar marker (`|`, `|-`, `|+`, `>`, `>-`, `>+`,
      optionally followed by an indent indicator) is stripped from the `lean` group
- [x] Common markdown fencing (` ```lean ` … ` ``` `) is also stripped
- [x] Block indentation is dedented (`textwrap.dedent` after marker removal)
- [x] Extraction is a named, separately tested function
      (`lms/agent.py::_clean_lean_code`) rather than inline regex behavior
- [x] Table-driven tests over the real malformed payloads: pull ≥5 samples from
      `experiments/stacks_ch4_phase1/artifacts.json` as fixtures
- [x] A canary test asserts no extracted `lean_code` starts with `|`, `>`, or ```` ``` ````
- [x] **Re-extraction script**: `scripts/reextract_lean_code.py` rewrites archived
      `artifacts.json` files into a `*.reextracted.json` sibling (never in place),
      so Gate A can re-score historical runs on clean source

---

#### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lms/agent.py` | MODIFY | `_clean_lean_code`; call from `_parse_artifacts` |
| `tests/test_lean_extraction.py` | CREATE | Table-driven + canary tests |
| `tests/fixtures/malformed_lean/` | CREATE | Real payloads from the Dec runs |
| `scripts/reextract_lean_code.py` | CREATE | Re-extraction for archived runs |

---

#### Implementation Notes

- Do not try to make the prompt stop emitting YAML block scalars instead —
  the parser must be robust to what models actually emit. Prompt changes can come
  later and independently.
- Keep the original raw capture on the artifact (`lean_code_raw`) so extraction
  bugs stay diagnosable.

---

#### Definition of Done

- [x] All acceptance criteria checked off
- [x] `uv run pytest` clean — 476 passed
- [x] `uv run ruff check` and `uv run mypy` clean **on the files this card
      touches**. Repo-wide they are not: 62 ruff errors (mostly unused imports
      in `tests/`) and 12 mypy errors (`config.py`, `providers/`, `society.py`,
      `run.py`) pre-date this branch and are unchanged by it. Verified by
      diffing both against the merge base rather than by reading the totals.
