# Cluster runbook — calibration Phase B

**For**: the 4×H100 NVL box (376 GB). **You run these; Claude does not.**
**Phase**: `docs/planning/calibration-program.md` §3 Phase B (Aug 10 – Aug 21)
**Goal**: Gate B — one agent, one statement, real Lean verification, end to end.
**Hard checkpoint**: **Aug 21.** If Gate B isn't green, invoke the pre-committed
API fallback ($300 cap) so the Sep 30 verdict date holds.

Paste output back into the session after each checkpoint. Every step has an
explicit expected result; if you get something else, stop at that step rather than
continuing.

---

## Step 0 — Prerequisites

```bash
nvidia-smi                       # expect 4x H100 NVL, ~94GB each
df -h /scratch                   # need >= 100 GB free (model ~61 GB + Mathlib ~20 GB + toolchain)
uv --version                     # everything Python in this repo goes through uv
uv run python --version          # >= 3.12
```

**Checkpoint 0**: 4 GPUs visible, ≥100 GB free on the scratch mount, `uv` present.
Never invoke bare `python`/`pip` anywhere in this runbook — always `uv run`.

---

## Step 0.5 — Route heavy artifacts to scratch

Everything bulky (model weights, Mathlib cache, elan toolchain, the repo itself
with its `.lake/` and `.venv/`) goes on scratch, not home. Substitute the box's
actual scratch mount if it isn't `/scratch/$USER`.

```bash
cat >> ~/.bashrc <<'EOF'
export SCRATCH=/scratch/$USER        # adjust to the real scratch mount
export HF_HOME=$SCRATCH/hf           # model weights (vLLM downloads land here)
export XDG_CACHE_HOME=$SCRATCH/cache # Mathlib download cache, uv, vLLM/torch caches
export ELAN_HOME=$SCRATCH/elan       # Lean toolchains
EOF
source ~/.bashrc
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$ELAN_HOME"
```

**Checkpoint 0.5**: `echo $HF_HOME` prints a scratch path in a fresh shell.

> If scratch is purged on a schedule, note the purge policy here when you report
> back — a purged Mathlib cache mid-phase would look like a mysteriously broken
> toolchain.

---

## Step 1 — Lean toolchain

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y

# elan does NOT create an `env` file to source (that is rustup, not elan). It
# installs binaries into <install-dir>/bin. Find which dir it actually used:
ls "$ELAN_HOME/bin/elan" 2>/dev/null || ls ~/.elan/bin/elan

# Then put that bin dir on PATH — substitute ~/.elan if that is where it landed:
export PATH="$ELAN_HOME/bin:$PATH"
echo 'export PATH="$ELAN_HOME/bin:$PATH"' >> ~/.bashrc

elan --version
lake --version
```

**Checkpoint 1**: `lake --version` prints a version, and `which lake` points
inside the scratch mount.

> `ELAN_HOME` is honored by the `elan-init` **binary**, not by the shell
> bootstrapper, and it is undocumented in elan's README — so verify rather than
> assume. If elan landed in `~/.elan`, the toolchains and their multi-GB
> `.olean` trees are going to your **home quota**, which is exactly what Step 0.5
> exists to prevent. In that case remove `~/.elan`, confirm `echo $ELAN_HOME`
> is set in the current shell, and re-run the installer.

---

## Step 2 — Build the LMS Lean project

The repo pins Lean `v4.27.0-rc1` and Mathlib rev `fe3134f0c3508d` (Dec 2025).

```bash
git clone git@github.com:setsosie/lms.git "$SCRATCH/lms"   # on scratch: .lake/ and .venv/ are the bulky parts
cd "$SCRATCH/lms/lean"
cat lean-toolchain                  # leanprover/lean4:v4.27.0-rc1

time lake exe cache get             # downloads prebuilt Mathlib olean files
time lake build                     # expect: fast if cache hit, hours if not
```

**Checkpoint 2**: `lake build` exits 0.

> **If `lake exe cache get` misses**, the pinned rev's cache has aged out. Do
> **not** build Mathlib from source (many GPU-hours of CPU time for no benefit).
> Instead report back — the right move is to bump to a current Mathlib rev and
> re-pin, which is a decision with consequences for `lean/LMS/*.lean` compiling
> and should be made deliberately, not inside a runbook.

```bash
# Compile the corpus and record what comes out
lake build 2>&1 | tee "$SCRATCH/lake-build.log"
grep -c "error:" "$SCRATCH/lake-build.log" || true
grep -n "declaration uses 'sorry'" "$SCRATCH/lake-build.log" || true
```

**Checkpoint 2b — this is the first time any of this has been compiled.**
Do not expect a clean build, and do not treat errors here as a failed checkpoint.

Until `lean/lakefile.toml` gained `globs = ["LMS.+"]`, Lake's default built only
the module `LMS` (which imports `LMS.Basic`) — every other file under `LMS/` was
excluded, so `lake build` exited 0 without elaborating them. No CI ever covered
them either: `lean/.github/workflows/` is not a path GitHub Actions reads. The
previously recorded "all compile, 0 errors" baseline has no build behind it.

What we know from source review, to compare against:

- **1 real `sorry`** — `LMS/Categories/Compat.lean:161`, the triangle identity in
  `equivalenceToMathlib`. It is **not fillable as written**: LMS's `Equivalence`
  records no coherence law, so Mathlib's `functor_unitIso_comp` obligation is
  false for equivalences the structure admits. Expect one `declaration uses
  'sorry'` warning here. The fix (`CategoryTheory.Equivalence.mk`, which applies
  `adjointifyη`) is its own task — **do not attempt it on the box.**
- Two further `sorry` tokens in that file are prose inside docstrings (lines 30,
  103). A `grep -c sorry` counts 3; the code contains 1. Count
  `declaration uses 'sorry'` warnings, not tokens.
- `LMS/Temp/verify_f5413690.lean` is harness scratch that got tracked by
  accident and is now inside the glob. If it errors, that's a cleanup task, not
  corpus rot.

**Report the full error list verbatim.** That list is the deliverable of this
step — it tells us what to fix and what to purge. Do not hand-edit any `.lean`
file to make the build pass; a corpus repaired by hand is no longer a measurement
of what the pipeline produced.

---

## Step 3 — Verify Lean is reachable as an oracle from Python

```bash
cd "$SCRATCH/lms"
uv sync
uv run python -c "
from lms.lean.real import RealLeanVerifier
import asyncio, pathlib
v = RealLeanVerifier(project_dir=pathlib.Path('lean'))
code = 'theorem cal_smoke (n : Nat) : n + 0 = n := by simp'
print(asyncio.run(v.verify(code)))
"
```

**Checkpoint 3**: prints a success result. This is the first time in this
project's history that the verification oracle is confirmed live — say so in the
report.

Then confirm it **rejects** what it should:

```bash
uv run python -c "
from lms.lean.real import RealLeanVerifier
import asyncio, pathlib
v = RealLeanVerifier(project_dir=pathlib.Path('lean'))
print(asyncio.run(v.verify('theorem bad (n : Nat) : n + 0 = n := by sorry')))
print(asyncio.run(v.verify('theorem nonsense : 1 = 2 := by rfl')))
"
```

**Checkpoint 3b**: both rejected. An oracle that only ever says yes is the failure
mode this whole program is guarding against.

---

## Step 4 — Serve the model

Per `docs/infrastructure/2026Q2-lean-codegen-base-model-selection.md`, start with
the generalist only — one model is enough for Gate B. Add BFS-Prover-V1-7B for
Phase C.

Run from the repo root, after Step 3's `uv sync` — `uv pip install` targets the
project's `.venv`, so `vllm` lands there and must be invoked with `uv run`
(a bare `vllm serve` will not be on `PATH`).

```bash
cd "$SCRATCH/lms"
uv pip install vllm
CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --served-model-name lms-generalist
```

Run it inside `tmux` (or `nohup`) so it survives your SSH session — the model
download (~61 GB to `$HF_HOME`) plus startup can take a while on first run.
Leave it running; use a second shell for the rest.

```bash
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lms-generalist","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":10}'
```

**Checkpoint 4**: the model is listed and the completion returns `ok`.

---

## Step 5 — Point the harness at it

Requires `26Q3-INFRA-01` (Sprint 2) to have landed.

`.env` is gitignored, so the clone starts without one — this creates it. No real
API keys are needed for local serving (`.env.example` is in the repo for
reference; do not copy the laptop's `.env` with live keys onto the box).

```bash
cd "$SCRATCH/lms"
cat >> .env <<'EOF'
LMS_OPENAI_BASE_URL=http://localhost:8000/v1
LMS_OPENAI_MODEL=lms-generalist
OPENAI_API_KEY=EMPTY
EOF

uv run python -c "
from lms.config import Config
c = Config.from_env()
print(c.openai.base_url, c.openai.model)
"
```

**Checkpoint 5**: prints the local URL and model name.

---

## Step 6 — Gate B: end-to-end smoke

```bash
cd "$SCRATCH/lms"
uv run lms \
  --provider openai \
  --verifier real \
  --agents 1 \
  --generations 1 \
  --output experiments/gateB_smoke \
  2>&1 | tee /tmp/gateB.log
```

Then score it through the Sprint 2 gates. Requires `26Q3-HARN-01`, `-03`, `-04`
and `-05` to have landed — as of 2026-07-28 `lms/metrics.py` has no
`cvfn_report` and no `__main__`, so this command does not exist yet:

```bash
uv run python -m lms.metrics cvfn_report experiments/gateB_smoke
```

**Checkpoint 6 — Gate B passes when**:

- [ ] ≥1 artifact reaches `VERIFIED_LEAN` (not `VERIFIED_HEURISTIC`)
- [ ] `metadata.json` records `verifier.kind == "real"` plus Lean/Mathlib versions
- [ ] gate results are populated for every artifact (pass or fail, with reasons)
- [ ] the novelty classifier emits a level for each verified artifact
- [ ] `cvfn_report` produces a number, even if that number is terrible

**A terrible number here is a pass.** Gate B tests that the instrument works, not
that the pipeline is good. Phase C measures the pipeline.

---

## What to report back

Paste: checkpoint results 0–6, `wall-clock` for `lake exe cache get` and
`lake build`, the `cvfn_report` output, and the gate-failure histogram. That plus
GPU-hours consumed is everything needed to configure Phase C.

## What not to do

- Don't build Mathlib from source on a cache miss — report instead (Step 2).
- Don't raise the Phase C token budget to make a run "work". Budget exhaustion at
  zero verified statements is a result we need recorded, not a problem to spend
  past.
- Don't fix agent output by hand mid-run. Anything hand-repaired is no longer a
  measurement of the pipeline.
