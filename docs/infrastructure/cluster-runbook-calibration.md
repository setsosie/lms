# Cluster runbook — calibration Phase B

**For**: the 4×H100 NVL box (376 GB). **You run these; Claude does not.**
**Phase**: `docs/planning/calibration-program.md` §3 Phase B (Aug 10 – Aug 21)
**Goal**: Gate B — one agent, one statement, real Lean verification, end to end.
**Hard checkpoint**: **Aug 21.** If Gate B isn't green, invoke the pre-committed
API fallback ($300 cap) so the Sep 30 verdict date holds.

Paste output back into the session after each checkpoint. Every step has an
explicit expected result; if you get something else, stop at that step rather than
continuing.

## State of the box as of 2026-08-10

Steps 0 through 2c ran early, on 2026-07-28. **Start at Step 2 (branch switch),
then go to Step 3.**

| Step | State |
|---|---|
| 0, 0.5, 1 — prereqs, scratch routing, elan | ✅ done |
| 2 — Mathlib cache + corpus build | ✅ done (0 errors, 1 `sorry`). Needs only the switch to `main` |
| 2c — axiom audit | ✅ done. Four Localization theorems clean on `[propext, Classical.choice, Quot.sound]` |
| 3 — Lean reachable from Python | ⬜ **today.** Expectation changed: 3c now **passes** (`26Q3-HARN-07` landed) |
| 4 — serve the model | ⬜ **today.** `--max-model-len` corrected below |
| 5 — point the harness at it | ⬜ **today.** Unblocked: `26Q3-INFRA-01` merged as #18 |
| 6 — Gate B | ⬜ **partial today.** Only checkbox 1 of 5 is reachable; see Step 6 |

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

Everything bulky — model weights, Mathlib cache, elan toolchains, Lake build
artifacts — goes on scratch. **The repo itself stays in home** (see Step 2): code
and git history belong on a backed-up mount, not a purgeable one. Substitute the
box's actual scratch mount if it isn't `/scratch/$USER`.

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

**`ELAN_HOME` must be exported in the shell that runs the installer.** elan reads
it at *runtime* for `$ELAN_HOME/toolchains`, but the piped bootstrapper does not
see it unless it is already in the environment. Get this wrong and you get a
split brain — binary and shims in `~/.elan`, toolchain root pointed at an empty
scratch path — whose symptom is `could not canonicalize path:
'<scratch>/elan/toolchains'` followed by `no default toolchain configured`.

```bash
echo "ELAN_HOME=$ELAN_HOME"   # must be non-empty BEFORE the next line
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y

# elan does NOT create an `env` file to source (that is rustup, not elan). It
# installs binaries into <install-dir>/bin. Find which dir it actually used:
ls "$ELAN_HOME/bin/elan" 2>/dev/null || ls ~/.elan/bin/elan

# Then put that bin dir on PATH — substitute ~/.elan if that is where it landed:
export PATH="$ELAN_HOME/bin:$PATH"
echo 'export PATH="$ELAN_HOME/bin:$PATH"' >> ~/.bashrc

# Install the PINNED toolchain. Do NOT run `elan default stable`, which is what
# elan's own error message suggests — this repo pins v4.27.0-rc1 and the Step 2
# Mathlib cache is built against that pin. `stable` guarantees a cache miss.
elan toolchain install leanprover/lean4:v4.27.0-rc1
elan default leanprover/lean4:v4.27.0-rc1

elan --version
lake --version
which lake
```

**Checkpoint 1**: `lake --version` prints a version, and `which lake` points
inside the scratch mount.

> **Recovery if elan already installed to `~/.elan`.** Check the size first —
> `du -sh ~/.elan`. Do **not** assume it is just the binary and shims: a first
> install sets a default toolchain, so it is routinely 2-3 GB. `no default
> toolchain configured` does not mean nothing is installed, only that nothing is
> installed *at `$ELAN_HOME`*, which is where elan is looking. Deleting it costs
> a re-download.
>
> Then `mkdir -p "$ELAN_HOME" && rm -rf ~/.elan`, re-run the installer with
> `ELAN_HOME` exported, and **`hash -r`** — bash caches absolute paths of
> commands it has already run, so without it the shell keeps trying the deleted
> `~/.elan/bin/elan` and reports `No such file or directory` even though the new
> install is fine. Check your shell rc for a stale `~/.elan/bin` PATH line the
> first installer appended; note that `grep -n elan ~/.bashrc` will not show a
> line written as `$ELAN_HOME/bin` — grep for both.

> `ELAN_HOME` is honored by the `elan-init` **binary**, not by the shell
> bootstrapper, and it is undocumented in elan's README — so verify rather than
> assume. If elan landed in `~/.elan`, the toolchains and their multi-GB
> `.olean` trees are going to your **home quota**, which is exactly what Step 0.5
> exists to prevent. In that case remove `~/.elan`, confirm `echo $ELAN_HOME`
> is set in the current shell, and re-run the installer.

---

## Step 2 — Build the LMS Lean project

The repo pins Lean `v4.27.0-rc1` and Mathlib rev `fe3134f0c3508d` (Dec 2025).
`main` carries the **same** pin as the July branches, so if the box already ran
Step 2 on 2026-07-28 its `$SCRATCH/lake-artifacts` cache is still valid — the
switch to `main` costs a re-elaboration of the LMS modules, not of Mathlib.

**The repo itself lives in home, not scratch.** Only two directories inside it
are bulky — `lean/.lake/` (unpacked Mathlib oleans, ~20-30 GB) and `.venv/`.
Everything else is source and git history, which is the last thing you want on a
purgeable mount. Symlink the bulk out instead. Keeping artifacts off `/` also
protects the root filesystem: a full `/` destabilizes the box, a full scratch
mount only costs a re-download.

**Use `main`.** As of 2026-08-10 both chore branches have landed, and `main` is a
strict superset of the merge the July run used: all 19 modules including
`TwoCat.lean` and `TwoFibreProduct.lean` (which `chore/wire-lean-build` was
missing), the `globs = ["LMS.+"]` line, and `LMS/Temp/` dropped from the corpus
and gitignored. Do **not** re-create the July two-branch merge.

```bash
git clone git@github.com:setsosie/lms.git ~/code/lms   # or `git fetch origin` if it exists
cd ~/code/lms
git checkout main && git pull origin main

# Route build artifacts to scratch BEFORE the first cache get
mkdir -p "$SCRATCH/lake-artifacts"
ln -s "$SCRATCH/lake-artifacts" lean/.lake

cd ~/code/lms/lean
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

**Result, 2026-07-28 (first run):** 19 modules, **0 errors**, **exactly one**
`declaration uses 'sorry'` at `Compat.lean:108` — the `equivalenceToMathlib`
declaration, whose `sorry` token sits at line 161. The corpus compiles against
pinned Mathlib `fe3134f0`.

---

## Step 2c — Axiom audit

**A 0-error build is not verification.** A proof admitted through an `aesop_cat`
autoParam produces no error and no `sorry` warning, and a `grep` for `sorry`
cannot see a `sorryAx` reached through an import. Only `#print axioms` settles
what a proof term actually depends on. This is the manual form of the T4 gate in
`26Q3-HARN-03`.

Note the invocation: **`lake env lean`**, not bare `lean`. Bare `lean` has no
`LEAN_PATH` and cannot resolve a single import.

```bash
cd ~/code/lms/lean
cat > "$SCRATCH/axcheck.lean" <<'EOF'
import LMS.Categories.Localization
import LMS.Categories.Compat

#print axioms LMS.Categories.Localization.sameDenom_eq_iff_exists_postcomp_W
#print axioms LMS.Categories.Localization.sameDenom_eq_iff_exists_postcomp
#print axioms LMS.Categories.Localization.exists_lift_commSq
#print axioms LMS.Categories.Localization.isIso_map_iff_inSaturation
#print axioms LMS.Categories.Compat.equivalenceToMathlib
EOF
lake env lean "$SCRATCH/axcheck.lean"
```

**Checkpoint 2c**: every declaration should report exactly
`[propext, Classical.choice, Quot.sound]`. Anything else — above all `sorryAx` —
means an obligation was silently admitted.

**Result, 2026-07-28:** all four `Localization` theorems (Stacks Tags 04VB, 04VD,
05Q2) clean on the standard three. `sorryAx` appears only in
`Compat.equivalenceToMathlib` and does not leak into the Localization results.
This is the project's first genuinely verified output.

> `TwoFibreProduct.lean` declares into `namespace LMS.Categories.IsoComma`, not a
> namespace of its own — so `mapFunctor` and the `assoc*` family are
> `LMS.Categories.IsoComma.*`. The `IsoComma`/`TwoFibreProduct`/`TwoCat`
> declarations are the ones carrying `aesop_cat` obligations and are worth their
> own batch.

---

## Step 3 — Verify Lean is reachable as an oracle from Python

```bash
cd ~/code/lms
uv sync
uv run python -c "
from lms.lean.real import RealLeanVerifier
import asyncio, pathlib
v = RealLeanVerifier(project_dir=pathlib.Path('lean').resolve())
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
v = RealLeanVerifier(project_dir=pathlib.Path('lean').resolve())
print(asyncio.run(v.verify('theorem bad (n : Nat) : n + 0 = n := by sorry')))
print(asyncio.run(v.verify('theorem nonsense : 1 = 2 := by rfl')))
"
```

**Checkpoint 3b**: both rejected. An oracle that only ever says yes is the failure
mode this whole program is guarding against.

> Note which of those two rejections is load-bearing. `sorry` is caught by
> `if "sorry" in code` in `lms/lean/real.py:86` — a Python substring test that
> short-circuits before Lean is ever invoked, and returns the same answer on a
> machine with no Lean installed. Only `1 = 2 := by rfl` exercises the compiler.

Now the test that actually matters — code with an import:

```bash
uv run python -c "
from lms.lean.real import RealLeanVerifier
import asyncio, pathlib
v = RealLeanVerifier(project_dir=pathlib.Path('lean').resolve())
code = '''import Mathlib.Logic.Basic
theorem cal_import_smoke (p : Prop) [Decidable p] : ¬¬p ↔ p := not_not'''
print(asyncio.run(v.verify(code)))
"
```

**Checkpoint 3c**: **this must now PASS.** It is the single most important
checkpoint in the runbook.

It was expected to fail when this runbook was written on 2026-07-28:
`RealLeanVerifier.verify` invoked bare `lean <temp_path>` with no `LEAN_PATH` and
no `cwd`, so nothing put the project's `.lake/packages/*/build/lib/lean`
directories on the module search path, and the verifier could only check
**import-free** Lean. Every real agent proof imports Mathlib, so the CVFN
numerator was structurally 0 for reasons having nothing to do with the models.

`26Q3-HARN-07` fixed it (PR #12, merged 2026-07-28): `real.py` now runs
`lake env lean` from the Lean project directory, the same invocation the Step 2c
axiom audit uses.

**If 3c still fails with `unknown module prefix 'Mathlib'`, stop.** Either the
checkout is behind `main` or the `lake env` path is broken on this box, and
everything downstream would measure the environment instead of the pipeline.
Note that 3 and 3b passing tells you nothing here — their snippets import
nothing, and `sorry` is caught by a Python substring test in `lms/lean/real.py`
that returns the same answer on a machine with no Lean at all.

### 3d — Confirm it rejects an import-bearing statement too

3b only ever exercised import-free code, so it could not tell a working oracle
from one that says no for the wrong reason. Run the negative case through the
*same* path as 3c:

```bash
uv run python -c "
from lms.lean.real import RealLeanVerifier
import asyncio, pathlib
v = RealLeanVerifier(project_dir=pathlib.Path('lean').resolve())
code = '''import Mathlib.Logic.Basic
theorem cal_import_reject (p : Prop) : p ↔ ¬p := not_not'''
print(asyncio.run(v.verify(code)))
"
```

**Checkpoint 3d**: `success=False` carrying a real elaborator message
(`Type mismatch ... has type ¬¬?m.1 ↔ ?m.1 but is expected to have type p ↔ ¬p`).
A bare `no such file or directory` is **not** a pass — see the note below.

> **`.resolve()` is load-bearing in these snippets.** `RealLeanVerifier` runs
> Lean with `cwd=project_dir` and separately hands it a temp path derived from
> the same value. With a relative `project_dir`, Lean runs inside `lean/` and is
> given `lean/.lake/verify-temp/x.lean`, so it looks for `lean/lean/.lake/...`
> and returns `no such file or directory (error code: 2)` — a false negative
> shaped exactly like a failed proof. Fixed in `LeanProject.__init__` (PR #21);
> the `.resolve()` here is belt-and-braces and works either way.
> `lms/run.py:146` always built an absolute path, so Step 6 was never affected.

**Result, 2026-08-10 (first live run of the oracle):** 3c `success=True`, 3d
`success=False` with the type mismatch above. Both adjudicated by the compiler
against pinned Mathlib. This is the first time the project has verified
import-bearing Lean end to end — every previously recorded "verified" figure came
from the mock or from import-free snippets.

---

## Step 4 — Serve the model

Per `docs/infrastructure/2026Q2-lean-codegen-base-model-selection.md`, start with
the generalist only — one model is enough for Gate B. Add BFS-Prover-V1-7B for
Phase C.

**vLLM gets its own virtualenv, and it is the one thing in this repo you do not
launch with `uv run`.** Confirmed on the box 2026-08-10.

An earlier version of this step said to `uv pip install vllm` into the project's
`.venv` and launch it with `uv run vllm serve`. That cannot work. vLLM requires a
newer `openai` than `uv.lock` pins (2.12.0), so the install upgrades it — and then
`uv run` re-syncs the environment to the lockfile *before* running the command,
silently downgrading `openai` back. vLLM then dies during import:

```
Uninstalled 2 packages in 2ms
Installed 2 packages in 28ms
ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'
```

That "uninstalled/installed" pair immediately above the traceback is the tell.
Re-installing does not help; `uv run` undoes it every launch.

The two sides have no reason to share an interpreter — since `26Q3-INFRA-01` they
communicate over HTTP. Keeping them apart also puts vLLM's ~10 GB of wheels on
scratch instead of the home quota, and makes a later `uv sync` unable to break a
running server.

### 4a — CUDA: pin vLLM below 0.20 while the driver is 570

**`mahpiya` runs driver 570.207, which caps CUDA at 12.8.** vLLM moved its default
PyPI wheel to CUDA 13.0 at **v0.20.0** (to match torch 2.11), and CUDA 13 requires
driver **≥ 580.65.06**. So the current default wheel cannot run on this box.

This fails in two stages, and the first fix does not reveal the second:

1. Plain `uv pip install vllm` gets torch built for **cu130**. Engine startup dies
   in `init_device` with `RuntimeError: The NVIDIA driver on your system is too
   old`.
2. Adding `--torch-backend=auto` fixes *torch* (it picks cu129) but not vLLM's own
   compiled extension, which is still built against CUDA 13. Now it dies earlier,
   at import: `ImportError: libcudart.so.13: cannot open shared object file`.

`libcudart.so.13` cannot be supplied by a package — CUDA 13 needs a 13-capable
driver. Pinning below 0.20 is the fix that works today:

```bash
cd ~
echo "$SCRATCH"                     # must be non-empty
export UV_LINK_MODE=copy            # cache and venv are on different filesystems
uv self update && hash -r           # see the uv note below
uv venv "$SCRATCH/vllm-env" --python 3.12
uv pip install --python "$SCRATCH/vllm-env/bin/python" "vllm<0.20" --torch-backend=auto
```

> **`uv` must be current.** An older `uv` does not know CUDA backends past
> `cu126`, and `--torch-backend=cu128` fails with `invalid value`, listing only
> ancient options. Do not pick `cu126` from that list to get past it — vLLM pins
> an exact torch version and torch drops old CUDA targets, so you get an
> unresolvable or ABI-mismatched environment. Update `uv` and use `auto`, which
> reads the driver and chooses. `hash -r` matters: bash caches the old binary's
> path.

**Verified working on the box, 2026-08-10:** vLLM `0.19.1`, torch `2.11.0+cu129`,
driver `570.207`. Qwen3 MoE support long predates 0.20, so the pin costs nothing
we need.

**Retire this pin when the driver reaches 580.65.06+**, at which point plain
`uv pip install vllm --torch-backend=auto` is correct again. That bump is
scheduled maintenance on a shared box — root, a reboot, and displacing other
jobs — not a debugging step. Do not do it mid-phase.

### 4b — Check the GPUs before loading 61 GB of weights

Both checks below are the exact failures above. Each takes seconds; a failed
model load takes minutes.

```bash
"$SCRATCH/vllm-env/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
"$SCRATCH/vllm-env/bin/python" -c "import vllm; print(vllm.__version__)"
```

Want `2.11.0+cu129 12.9 True 4` from the first — CUDA 12.9 on a 12.8 driver is
fine, since minor versions are compatible within CUDA 12; it was the **major**
jump to 13 that broke. The second must print a version rather than raise
`ImportError`.

### 4c — Serve

```bash
CUDA_VISIBLE_DEVICES=0,1 "$SCRATCH/vllm-env/bin/vllm" serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --served-model-name lms-generalist \
  2>&1 | tee "$SCRATCH/vllm.log"
```

Healthy progress, in order: `world_size=2` with an `nccl` backend →
`Starting to load model` → `Loading safetensors checkpoint shards: N%` →
`GPU KV cache size: ... tokens` → `Starting vLLM API server on http://0.0.0.0:8000`.
Until that last line, `curl` returning `Connection refused` is expected. Use
`curl -sS`, not `-s`, or connection failures are silent.

**Result, 2026-08-10 (first successful serve):** `Available KV cache memory:
52.19 GiB` → **`GPU KV cache size: 1,139,984 tokens`**; engine init (profile,
create KV cache, warmup) 40.7 s after weights loaded.

At `--max-model-len 131072` that is ~8.7 full-length sequences resident
simultaneously, so a 1-agent Gate B run uses roughly 11% of KV capacity.
**Phase C sizing follows from this**: there is headroom for substantially more
concurrent agents on GPUs 0–1, or for BFS-Prover-V1-7B on GPUs 2–3, without
revisiting `--gpu-memory-utilization` (left at its 0.9 default) or
`--max-model-len`. *(Superseded 2026-08-19: Step 8 and Phase C serve on all
four GPUs instead — see 4d.)*

If you already ran `uv pip install vllm` against the project, `cd ~/code/lms &&
uv sync` prunes vLLM and torch back out of `.venv` and restores the pinned
`openai`. The harness side keeps using `uv run` exactly as before — Steps 3, 5
and 6 are unchanged.

> **tmux, if you don't use it often.** `Ctrl-b` is a prefix: press and *release*
> `Ctrl-b`, then tap the next key on its own — `Ctrl-b Ctrl-c` is a different
> (unbound) sequence. `c` new window, `0`/`1` jump to window N, `n`/`p` next and
> previous, `d` detach, `[` scroll mode (`q` exits). The status bar names the
> session and marks the current window with `*`. To read a pane you can't scroll,
> `tmux capture-pane -t <session> -p -S -5000 > file`. `echo "$TMUX"` is the
> reliable test for whether you are inside a session — it is empty outside.

> **Sizing `--max-model-len`.** Keep
> `--max-model-len ≥ longest prompt + LMS_OPENAI_MAX_TOKENS` (Step 5 sets the
> cap to 16384): vLLM rejects any request where
> `prompt_tokens + max_tokens > max_model_len`, and the symptom is a run that
> fails on the first generation with a 400 that looks like a malformed request.
> 131072 is comfortable — inside Qwen3-Coder-30B-A3B's native 262144 window,
> with room for the largest agent prompts — and costs nothing at rest: KV is
> ~96 KB/token (48 layers × 4 KV heads × 128 dim × 2 × 2 bytes), so a
> full-length sequence is ~12.6 GB, ~6.3 GB per GPU at TP=2, against 94 GB
> H100 NVLs already holding ~30.5 GB of weights each.
>
> *(History: before `26Q3-INFRA-02` the harness had no cap lever — every
> request asked for 64,000 completion tokens, sized to Claude Opus's output
> limit, which is why this note used to mandate 131072 as a workaround.)*

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

Check `max_model_len` in the `/v1/models` response, not just that the call
succeeded — it is the field that would otherwise 400 every agent request. Want
`"max_model_len":131072`.

**Result, 2026-08-10:** ✅ both. `/v1/models` reported `lms-generalist`
(root `Qwen/Qwen3-Coder-30B-A3B-Instruct`) with `max_model_len: 131072`, and the
completion returned `ok` in 2 tokens.

### 4d — Serve on all four GPUs (go-forward, decided 2026-08-19)

4c's TP=2 serve stays above as the recorded 2026-08-10 result; from Step 8
onward, serve on all four. Two reasons, one caution:

- **KV headroom is the point.** At TP=2 the cache is 1,139,984 tokens and the
  Phase C 9-agent config can ask for 9 × 131,072 = 1,179,648 — slightly over.
  Near the ceiling vLLM preempts and recomputes sequences, degrading exactly
  the config the population hypothesis leans on. TP=4 roughly doubles the
  cache (weights per GPU halve to ~15 GB), so expect ~2.3M tokens.
- **Speed is not the point.** Decode was never the bottleneck — Lean
  elaboration is — so do not read a faster run into this change.
- **This claims the whole box.** Check the GPUs are idle before loading
  (`nvidia-smi`), the same check the sbatch preflight enforces.

Record the topology once — it is the fact that would justify two TP=2
instances over one TP=4 later, if serving evidence ever demands that:

```bash
nvidia-smi topo -m
```

H100 NVL cards are NVLink-bridged in pairs: expect `NV#` between GPUs 0–1 and
between 2–3, and PCIe (`PIX`/`PHB`/`SYS`) across the pairs. TP=4 then crosses
PCIe on the all-reduce, which is acceptable here precisely because decode is
not the bottleneck. (The faster-comms alternative — two TP=2 instances, one
per pair — needs a request router the harness does not have; card it only on
evidence.)

**Result, 2026-08-19:** GPU0↔GPU1 and GPU2↔GPU3 are `NV12`; every cross-pair
path is `SYS` — not merely PCIe but the UPI socket interconnect, since each
pair sits on its own NUMA node (node 0: CPUs 0–31,64–95; node 1: 32–63,
96–127). TP=4 comms take the slowest path on the legend; accepted per the
above. Also recorded: 128 logical CPUs total, so the sbatch's
`--cpus-per-task=16` has ample room to grow if concurrent Lean elaboration
becomes the constraint.

Then serve exactly as 4c with two changed values:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$SCRATCH/vllm-env/bin/vllm" serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000 --tensor-parallel-size 4 --max-model-len 131072 --served-model-name lms-generalist 2>&1 | tee "$SCRATCH/vllm.log"
```

Healthy progress is as in 4c but with `world_size=4`. **Record the
`GPU KV cache size: ... tokens` line** — expected ~2.3M. If it is not roughly
double 4c's 1,139,984, stop and report rather than continuing.

**Result, 2026-08-19 (first TP=4 serve):** `GPU KV cache size: 2,903,296
tokens` — 2.55× the TP=2 figure, above prediction because halving the weights
per GPU (~30.5 → ~15.3 GB) frees budget on all four cards. That is ~22
full-length sequences at `--max-model-len 131072`; the 9-agent Phase C config
worst-cases at ~40% of capacity. Checkpoint 4 re-run green: `/v1/models`
listed `lms-generalist` with `max_model_len: 131072`.

Checkpoint 4 applies verbatim: same curl checks, same
`"max_model_len":131072`.

Bookkeeping consequences:

- **GPU-hours for CVFN are now 4 × wall-clock**, not 2 ×. The cost model
  charges the allocation, not the utilization.
- BFS-Prover-V1-7B co-residency on GPUs 2–3 (the option 4c noted) is off the
  table while TP=4 holds the box. It was a post-CVFN idea anyway.

`scripts/slurm/lms_run.sbatch` defaults now match (`--gres=gpu:h100:4`,
`TP_SIZE=4`, 8 h walltime). To hand half the box back on a day it is shared:
`sbatch --gres=gpu:h100:2 --time=04:00:00 --export=ALL,TP_SIZE=2 scripts/slurm/lms_run.sbatch`

---

## Step 5 — Point the harness at it

`26Q3-INFRA-01` landed as PR #18 on 2026-08-10, so this step is unblocked. It
threads `base_url` from the environment onto `ProviderConfig` and into
`AsyncOpenAI`; an unset or blank value leaves hosted-OpenAI behavior untouched.

`.env` is gitignored, so the clone starts without one — this creates it. No real
API keys are needed for local serving (`.env.example` is in the repo for
reference; do not copy the laptop's `.env` with live keys onto the box).

```bash
cd ~/code/lms
cat >> .env <<'EOF'
LMS_OPENAI_BASE_URL=http://localhost:8000/v1
LMS_OPENAI_MODEL=lms-generalist
LMS_OPENAI_MAX_TOKENS=16384
OPENAI_API_KEY=EMPTY
EOF

uv run python -c "
from lms.config import Config
c = Config.from_env()
print(c.openai.base_url, c.openai.model, c.openai.max_tokens)
"
```

**Checkpoint 5**: prints the local URL, model name, and the 16384 cap.

`LMS_OPENAI_MAX_TOKENS` (landed with `26Q3-INFRA-02`) is the per-request
completion cap; unset it falls back to the 64k hosted-Claude default, which is
more than any single Lean artifact needs and forces `--max-model-len` past 64k.
16384 is generous for proof generation while keeping
`prompt + cap` far inside the served window.

---

## Step 6 — Gate B: end-to-end smoke

**Only checkbox 1 of the 5 below is reachable on 2026-08-10.** `26Q3-HARN-01`
(provenance) is open as PR #14; `-03` (T2/T4 gates), `-04` (novelty) and `-05`
(cost accounting) are not started. Today's target is therefore **Gate B-minus**:
the loop closes and at least one artifact clears *real* Lean. That is a genuine
first — the pipeline has never once run against a live oracle — but it is not
Gate B, and the Aug 21 checkpoint still judges the full five.

> ### Guard the corpus before you run
>
> `Society.__init__` defaults its accumulator to `FoundationFile(Path("lean/LMS/Foundation.lean"))`
> (`lms/society.py:122`) and `lms/run.py` never passes `foundation_path`, so
> **the run overwrites the tracked 127-line WC-3 corpus file.** This is issue #19
> in a second guise — it is by design that verified artifacts accumulate there,
> but that file is also a build input, so a later `lake build` would be
> elaborating harness-mutated source and Step 2's result would stop meaning what
> it says.
>
> ```bash
> cd ~/code/lms
> git status --short lean/          # expect clean before the run
> cp lean/LMS/Foundation.lean "$SCRATCH/Foundation.lean.orig"
> ```
>
> After the run, keep the mutated copy as evidence, then restore:
>
> ```bash
> cp lean/LMS/Foundation.lean "$SCRATCH/Foundation.lean.after-gateB"
> git checkout -- lean/LMS/Foundation.lean
> git status --short lean/          # must be clean again
> ```

Invoke the module, **not** `uv run lms`. The project declares
`[project.scripts] lms` but has no `[build-system]` and no
`tool.uv.package = true`, so `uv sync` prints *"Skipping installation of entry
points (`project.scripts`) because this project is not packaged"* and `uv run
lms` dies with `Failed to spawn: lms`. `python -m lms.run` takes the identical
arguments and is verified working.

Run this from a shell that has the elan PATH. `RealLeanVerifier.__init__` calls
`_find_lean()`, which raises `FileNotFoundError` when `lean` is not on `PATH` — so
a fresh `tmux` pane that never sourced `~/.bashrc` crashes before a single token
is generated. Check with `which lean lake` first.

```bash
cd ~/code/lms
which lean lake                     # both must resolve, inside $ELAN_HOME
uv run python -m lms.run \
  --provider openai \
  --verifier real \
  --agents 1 \
  --generations 1 \
  --output experiments/gateB_smoke \
  2>&1 | tee "$SCRATCH/gateB.log"
```

Then score it through the Sprint 2 gates. Requires `26Q3-HARN-01`, `-03`, `-04`
and `-05` to have landed — as of 2026-07-28 `lms/metrics.py` has no
`cvfn_report` and no `__main__`, so this command does not exist yet:

```bash
uv run python -m lms.metrics cvfn_report experiments/gateB_smoke
```

**Checkpoint 6 — Gate B passes when**:

- [ ] ≥1 artifact reaches `VERIFIED_LEAN` (not `VERIFIED_HEURISTIC`) — *reachable today*
- [ ] `metadata.json` records `verifier.kind == "real"` plus Lean/Mathlib versions — *blocked on `26Q3-HARN-01`, PR #14*
- [ ] gate results are populated for every artifact (pass or fail, with reasons) — *blocked on `26Q3-HARN-03`*
- [ ] the novelty classifier emits a level for each verified artifact — *blocked on `26Q3-HARN-04`*
- [ ] `cvfn_report` produces a number, even if that number is terrible — *blocked on `26Q3-HARN-05`*

**Checkpoint 6-minus — what to check today instead.** Since `metadata.json` does
not yet carry provenance, the only way to tell a real verification from a
heuristic one is that you launched with `--verifier real` and Checkpoint 3c
passed. Record both facts alongside the artifacts, because this run's outputs
will otherwise be indistinguishable from the mock-verified runs that produced the
retracted roadmap numbers.

```bash
uv run python -c "
import json, collections
d = json.load(open('experiments/gateB_smoke/artifacts.json'))
a = d['artifacts']
print('artifacts     :', len(a))
print('verified      :', collections.Counter(str(x.get('verified')) for x in a))
print('by type       :', collections.Counter(x.get('type') for x in a))
print('with an error :', sum(1 for x in a if x.get('verification_error')))
print('total tokens  :', d.get('total_tokens_used'))
for x in a:
    if x.get('verification_error'):
        print('---', x['id'], x['verification_error'][:300])
"
```

Report that histogram, the wall-clock, and the GPU-hours. **A histogram of
all-failures is a pass for today** — it means the instrument ran end to end
against a live oracle. What would *not* be a pass is a high verified count: the
archived `experiments/mixed_3llm` reads 21/25 `verified: true` and every one of
those came from the mock. If this run reports a similar rate, suspect the
verifier wiring before believing the number.

**A terrible number here is a pass.** Gate B tests that the instrument works, not
that the pipeline is good. Phase C measures the pipeline.

---

## Step 7 — N1-density measurement (Sprint 3 DoD item 1)

**Prereqs**: PRs #36 (arc statement lists) and #38 (novelty classifier) merged
to `main`; the box repo on `main` and pulled; Step 2's Mathlib build present —
the name-grep and `exact?`-probe stages need `.lake/packages/mathlib` and
report `available=False` without it; outbound HTTPS (loogle and LeanSearch are
hosted services).

No GPU needed: this is CPU-bound Lean elaboration plus rate-limited HTTP. The
login node is fine; vLLM does not need to be running.

```bash
cd ~/lms && git switch main && git pull
```

```bash
mkdir -p experiments/n1_density
```

```bash
uv run python scripts/measure_n1_density.py data/ant_arcs/core_arc.json --json-out experiments/n1_density/core.json
```

```bash
uv run python scripts/measure_n1_density.py data/ant_arcs/ramification_arc.json --json-out experiments/n1_density/ramification.json
```

Rate limits (loogle 3/30s) put a full ~20-statement arc at roughly 10–25
minutes. Results are disk-cached by statement hash, so re-runs after signature
repairs only pay for what changed.

**Expected**: a per-statement N0/N1/INCONCLUSIVE table with confidence and
evidence; an arc-level N1 density (upper and decisive) with the confidence
distribution; a D4 review queue. All four search stages should report
available — if name-grep or the exact-probe say unavailable, stop at this
step: the classifier isn't finding the Mathlib checkout.

**Also expected — not a failure**: per-statement elaboration errors. All 41
Lean drafts shipped unvalidated (0/41 type-checked locally; stale olean
cache), so this run doubles as their elaboration check. Paste the failure
list; the signatures get repaired in one pass and re-run against the cache.

**Checkpoint 7**: two density numbers recorded. This resolves the
`calibration-program.md` §4 slice decision (pick the higher-N1 arc) and closes
Sprint 3 DoD item 1.

## Step 7b — Gate A novelty control (Sprint 3 DoD item 3)

**Run this before trusting Step 7's densities.** The archived
`stacks_ch4_phase1` artifacts are known Mathlib reimplementations, so the
classifier must read them ~all N0. The failure mode this catches: search
queries that never match anything produce all-stages-empty, high-confidence
N1 verdicts *everywhere* — indistinguishable from a genuinely novel arc until
a known-N0 corpus goes through the same pipeline.

Clean the historical records first (pre-HARN-02 payloads carry the YAML
block-scalar leak; searching on `|`-prefixed source tests the leak, not
Mathlib):

```bash
uv run python scripts/reextract_lean_code.py experiments/stacks_ch4_phase1
```

```bash
uv run python scripts/make_novelty_control.py experiments/stacks_ch4_phase1/artifacts.reextracted.json --out experiments/n1_density/gate_a_control_arc.json
```

```bash
uv run python scripts/measure_n1_density.py experiments/n1_density/gate_a_control_arc.json --json-out experiments/n1_density/gate_a_control.json
```

52 statements against loogle's 3/30s limit is a while, but confident N0 hits
short-circuit the ladder, so a healthy run is much faster than 52 full
searches.

**Checkpoint 7b**: N0 accounts for essentially all 52. Some INCONCLUSIVE is
tolerable (that is the D4 route working); a substantial N1 count is not — it
means the instrument is broken, **both Step 7 densities are void**, and the
fix is in the classifier's query construction, not in the arcs. An N1-heavy
result here specifically suggests name-bias: the artifacts reimplement
Mathlib under custom names, so only the content-level stages (`exact?` probe,
semantic search) can catch them.

---

## Step 8 — the first full committee run

**Prereqs**: PR #35 (committee wiring) merged; #33 strongly recommended first
(without it the oracle still false-negatives on core-name collisions); #34 and
#40 if targeting the Stacks kernel goal; vLLM serving and the harness pointed
at it exactly as Steps 4–5 configured — **serve per 4d (all four GPUs)** from
here on. **Step 6's corpus guard applies
verbatim here — run it before and after.**

Smoke on the historical goal first, to validate the wiring against a known
target:

```bash
uv run python -m lms.run --groups --n-groups 3 --agents 3 --goal stacks-ch4-phase1 --generations 2 --verifier real --provider openai --output experiments/committee_smoke
```

Then the real target — the shared-kernel fibred-categories track:

```bash
uv run python -m lms.run --groups --n-groups 3 --agents 3 --goal stacks-kernel-track-b --generations 2 --verifier real --provider openai --output experiments/committee_trackB
```

**Expected**: `Mode: COMMITTEE (3 groups)` in the banner; each generation the
three groups attempt three *different* tags (check `artifacts.json` for
distinct `stacks_tag` values across `group-*` creators — in every earlier run
N agents were N copies of one agent); possibly `Rejected by review committee`
entries, which are the review stage working, not a failure. Once, deliberately
run `--groups` without `--goal` and confirm it errors loudly instead of
silently degrading to flat mode.

**Checkpoint 8**: `artifacts.json` shows ≥2 distinct tags across groups and
the run summary shows `reviews_total > 0`.

> **Criterion amended 2026-08-20**: for `stacks-ch4-phase1` the dependency
> graph gates every task behind `0013`, so `available_tasks()` at gen 0 is
> exactly `['0013']` and distinct tags are unsatisfiable *by construction* —
> all groups on the sole available tag is the graph working, not the panel
> failing. The criterion is: every artifact's tag is a **real goal tag** that
> was available when assigned, and `reviews_total > 0`.

**Result, 2026-08-19/20 (five smoke runs, four harness fixes):** Checkpoint 8
**green** on `committee_smoke_e` under the amended criterion. The road there —
each run's failure became a merged fix (#44, #45, #46):

1. `committee_smoke` (run 1): every LLM call died — committee classes passed
   dicts where the provider reads `m.role` by attribute; summary printed
   "0 tokens used". → #44.
2. `committee_smoke_b`: 111,389 tokens, pipeline executed end to end, review
   REJECT worked — but the chair invented tags (`LOGIC-001`), the
   `Define <tag>` fallback masked it, and payloads carried the YAML
   block-scalar leak. → #45 (tag enforcement + creation-site cleaning).
3. `committee_smoke_c`: accidental clean replication of run 2's defects (ran
   on pre-#45 code — **always verify `git log` after pulling**).
4. `committee_smoke_d`: leak survived #45 — the review prompt requests
   `modified_code: |` and a MODIFY overwrote cleaned code with the uncleaned
   capture (5/6 artifacts). Scribes decorated tags (`CAT-0013`). → #46.
5. `committee_smoke_e`: 6/6 artifacts on real tag `0013`, clean code, real
   imports (5/6), and **every failure is mathematics**: 3× `unknown universe
   level u`, one genuine type error, one syntax slip, one guessed Mathlib
   module. 0 verified — the known structural gap: groups get one shot with no
   Lean feedback (the iterative agent loop gets 5 attempts with error
   feedback; committee groups get none). That is the next card, not a
   wiring defect.

**What to paste back for Steps 7–8**: both density outputs (or their
`--json-out` files), the elaboration-failure list, the committee run banner,
the `artifacts.json` tag/creator/verified summary, every `verification_error`
and review-rejection string, wall-clock, and `git status --short lean/` after
the guard's restore.

---

## What to report back

Paste: checkpoint results 0–6, `wall-clock` for `lake exe cache get` and
`lake build`, the `cvfn_report` output, and the gate-failure histogram. That plus
GPU-hours consumed is everything needed to configure Phase C.

**For the 2026-08-10 Gate B-minus run specifically**, `cvfn_report` does not
exist yet, so paste instead:

1. Checkpoint 3c verbatim — pass/fail decides whether anything after it means
   something.
2. The Step 4 vLLM startup banner (it prints the resolved `max_model_len`, the
   KV-cache blocks allocated, and the dtype).
3. The Step 6 histogram, plus every `verification_error` string. **The error
   strings are the most valuable output of the day**: they say whether the models
   are producing malformed Lean, unprovable goals, or hitting the `lean_code`
   YAML block-scalar leak that `26Q3-HARN-02` is meant to fix.
4. Wall-clock for the run and `nvidia-smi` peak memory.
5. `git status --short lean/` after the restore in Step 6's guard.

## What not to do

- Don't build Mathlib from source on a cache miss — report instead (Step 2).
- Don't raise the Phase C token budget to make a run "work". Budget exhaustion at
  zero verified statements is a result we need recorded, not a problem to spend
  past.
- Don't fix agent output by hand mid-run. Anything hand-repaired is no longer a
  measurement of the pipeline.
