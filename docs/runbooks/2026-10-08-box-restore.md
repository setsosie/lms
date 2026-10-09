# Box restore runbook — 2026-10-08

**You run these; Claude does not.** Other people's jobs run on this box, so
Claude prepares the commands and reads what you paste back. Every code block
is **one command on one line**: multi-line pastes get mangled in the terminal
and produce plausible-looking wrong output.

**Prerequisite:** PR #63 merged to `main`. It carries the classifier fix this
measurement depends on, the committed Gate A control arc, and the repaired arc
drafts. The R4 batch job refuses to run on a checkout without it.

**Also merge before R4: #66** (`26Q3-HARN-20` Part 2), so DoD 1's numbers come
from the classifier as it will stand. It does not move the arcs much: no arc
draft uses project vocabulary, and the eponym rule can reach only core-13,
core-14 and core-20. If R4 already ran, re-running after the merge is cheap.
Search results are cached in `.novelty_cache/`, so only scoring repeats.

**R6 needs this runbook's own PR merged:** it adds the committee-mode knob to
`lms_run.sbatch`.

## What changed

The box came back **restored to an earlier state**, and per-user scratch moved:

| Was | Now |
|---|---|
| `/scratch/$USER/…` | `/scratch/users/$USER/…` |
| — | `/scratch/shared/{hf,datasets,containers}` (shared, read-mostly) |
| — | `/scratch/slurm-tmp` |

Every path `docs/infrastructure/cluster-runbook-calibration.md` and both batch
scripts used under `/scratch/$USER` is therefore gone: elan toolchains, the
library build, the vLLM environment, model weights, run logs. What survived in
home is unknown. **R0 finds out before anything is installed or deleted.**

## What this closes

Sprint 3's Definition of Done has three open items, all of them server work:

| DoD | Item | Step | Needs a GPU |
|---|---|---|---|
| 1 | N1 density for both ANT arcs | R4 | no |
| 3 | Gate A novelty control | R4 (runs first) | no |
| 4 | CVFN denominator on one real run | R5 | only if no ledger survived the restore |
| — | Committee re-run: harness vs model (not DoD) | R6 | yes |

---

## R0 — Inventory (read-only)

Nothing here changes anything. Paste all of it back.

```bash
hostname; nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv 2>&1 | head -6
```

```bash
sinfo -o "%P %a %l %D %G %N"
```

```bash
sacctmgr -n show assoc user=$USER format=account%20,partition%20 2>&1 | head
```

```bash
ls -ld /scratch/users/$USER /scratch/$USER 2>&1; ls /scratch/users/$USER 2>&1
```

```bash
ls /scratch/shared/hf/hub 2>/dev/null | grep -i qwen
```

```bash
ls -d ~/code/lms ~/lms 2>&1
```

```bash
cd ~/code/lms && git log --oneline -1 && git status --short | head -20
```

```bash
ls -la ~/code/lms/lean/.lake
```

```bash
grep -n 'SCRATCH\|HF_HOME\|ELAN_HOME\|XDG_CACHE_HOME\|elan' ~/.bashrc
```

```bash
which uv lean lake elan; elan show 2>&1 | head -3
```

```bash
ls ~/code/lms/experiments/; ls ~/code/lms/experiments/*/attempts.json 2>/dev/null
```

```bash
for u in https://leansearch.net https://loogle.lean-lang.org; do curl -sS --max-time 15 -o /dev/null -w "$u %{http_code}\n" "$u"; done
```

**Checkpoint R0.** What each answer decides:

- **`sinfo` / `sacctmgr`** → the `-p`, `--gres` and `--account` flags for every
  `sbatch` below. The August site config (one `gpu` partition,
  `gpu:h100:4`, no accounting) is not assumed to hold.
- **`/scratch/$USER`** should be gone. If it still exists, say so before R1.
- **`lean/.lake`** was a symlink into `/scratch/$USER/lake-artifacts`; expect it
  to dangle now. If it is a **real directory**, stop: that is tens of GB in
  home, and moving it is a decision, not a runbook step.
- **`~/.bashrc`** lines are what R1 rewrites. If any of them serve another
  project on this account, edit by hand instead of running R1's `sed`.
- **`attempts.json`** under any surviving run means R5a can close DoD 4 with no
  GPU time.
- **`git status`** must be clean before R2 pulls. If it is not, paste it; do
  not discard anything.

---

## R1 — Scratch routing under the new layout

Back up, drop the old four exports, append the new ones:

```bash
cp ~/.bashrc ~/.bashrc.pre-restore
```

```bash
sed -i '/^export \(SCRATCH\|HF_HOME\|XDG_CACHE_HOME\|ELAN_HOME\)=/d' ~/.bashrc
```

```bash
printf '%s\n' 'export SCRATCH=/scratch/users/$USER' 'export HF_HOME=$SCRATCH/hf' 'export XDG_CACHE_HOME=$SCRATCH/cache' 'export ELAN_HOME=$SCRATCH/elan' 'export PATH="$ELAN_HOME/bin:$PATH"' >> ~/.bashrc
```

```bash
source ~/.bashrc && hash -r && mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$ELAN_HOME" && echo "$SCRATCH | $HF_HOME | $ELAN_HOME"
```

**Checkpoint R1:** three paths, all under `/scratch/users/<you>`.

The `sed` matches the variable names, so it removes the old lines whether they
were written as `/scratch/$USER` or with the username expanded.

---

## R2 — Lean toolchain, repo, library build

Pull first: the toolchain pin lives in the repo.

```bash
cd ~/code/lms && git fetch origin && git switch main && git pull --ff-only origin main && git log --oneline -1
```

```bash
echo "ELAN_HOME=$ELAN_HOME"
```

The line above must print a non-empty scratch path before the installer runs;
elan reads it to decide where toolchains go. This is CI's invocation:
`--no-modify-path` because R1 already put `$ELAN_HOME/bin` on `PATH`, and the
pinned default rather than `stable`, which would miss the library cache.

```bash
curl -sSfL https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --no-modify-path --default-toolchain "$(cat ~/code/lms/lean/lean-toolchain)"
```

If R0 showed elan already installed inside the new `$ELAN_HOME`, skip the
installer and run this instead:

```bash
cd ~/code/lms/lean && elan toolchain install "$(cat lean-toolchain)" && elan default "$(cat lean-toolchain)"
```

```bash
hash -r && which lean lake && lean --version
```

**Checkpoint R2a:** `lean --version` prints `4.27.0-rc1`, and `which` points
inside `$ELAN_HOME`. Check the version, not just that `lean` resolves: elan's
shim exists before any toolchain does, and a shim with no default fails every
call. That gap broke ten tests on the first CI run (2026-09-08).

Re-point the library build at scratch. `rm` on a path with **no trailing
slash** removes the symlink only. Run this only if R0 showed a symlink:

```bash
mkdir -p "$SCRATCH/lake-artifacts" && cd ~/code/lms && rm lean/.lake && ln -s "$SCRATCH/lake-artifacts" lean/.lake && ls -la lean/.lake
```

```bash
cd ~/code/lms/lean && time lake exe cache get
```

```bash
cd ~/code/lms/lean && time lake build 2>&1 | tail -5
```

**Checkpoint R2b:** `lake build` exits 0 with exactly one `declaration uses
'sorry'` (`Compat.lean`; known and unfillable as written). **On a cache miss,
stop.** Do not build the library from source; the pinned rev aging out of the
cache is a re-pin decision.

---

## R3 — Python environment and suite

```bash
cd ~/code/lms && uv sync --frozen && uv run pytest -q 2>&1 | tail -3
```

```bash
cd ~/code/lms && git status --short lean/
```

**Checkpoint R3:** no failures, and `lean/` clean afterwards. With a working
toolchain the 16 Lean-dependent tests run instead of skipping; the skip count
should drop accordingly.

---

## R4 — N1 density and the Gate A control (DoD 1 and 3)

CPU only: one batch job, no GPU requested. Substitute the partition (and
`--account=…` if R0 showed accounting) from R0:

```bash
cd ~/code/lms && mkdir -p logs && sbatch -p PARTITION scripts/slurm/n1_density.sbatch
```

```bash
squeue -u $USER
```

```bash
tail -n 40 ~/code/lms/logs/lms-n1-density-JOBID.out
```

The job checks the toolchain, the library checkout, that the checkout includes
the classifier fix, and outbound HTTPS from the compute node. Then it runs the
control first, then both arcs. Expect roughly an hour: about 70 probes, each
loading the whole library once, plus loogle's 3-per-30-seconds limit on the
control's 52 statements. The walltime is 4 h.

Summary, one line per report:

```bash
cd ~/code/lms && uv run python -c "import json,glob; [print(p.split('/')[-1], r['counts'], 'upper=%.2f decisive=%.2f cap=%.2f review=%d' % (r['n1_density'], r['n1_density_decisive'], r['max_n1_confidence'], len(r['needs_review']))) for p in sorted(glob.glob('experiments/n1_density/*.report.json')) for r in [json.load(open(p))]]"
```

Drafts that did not elaborate at the pin:

```bash
cd ~/code/lms && uv run python -c "import json,glob; [print(s['id'], s['stage_errors']['exact_probe'][:220]) for p in sorted(glob.glob('experiments/n1_density/*.report.json')) for s in json.load(open(p))['statements'] if s.get('stage_errors', {}).get('exact_probe', '').startswith('statement did not elaborate')]"
```

**Checkpoint R4 — read it in this order.**

1. **Control.** `decisive` should be **0.00**. Inspect every decisive N1 by
   hand: each one is either a classifier defect or a genuinely new result in a
   run designed to have none. If more than a couple appear, **the arc
   densities are void**; stop and paste. Expect a large INCONCLUSIVE/review
   share. 11 of the 52 have no parseable name and 31 are not theorems, so
   for many of them only semantic search can run. That is the instrument
   declining to guess, not failing. The N0 count is the control's *recall*;
   record it.
2. **Arcs.** `cap=0.60` and `decisive=0.00` are expected **by construction**.
   The drafts' names are labels, so only `exact?` and semantic search run, and
   two stages cannot reach the 0.8 decisive line. The numbers that decide the
   slice are `upper` and the review queue. Cross-check against each
   statement's `notes`: one the drafter marked "Expected N0" that reads N1 is
   a classifier miss worth a look. Since #66, core-13 and core-14 (Minkowski)
   and core-20 (Dirichlet) can read INCONCLUSIVE with a semantic hit carrying
   the eponym as `evidence[0]`. That is the instrument working; their notes
   expect N0.
3. **Elaboration at the pin.** The 2026-10-08 pre-screen against a newer
   library build expects only SCHEMATIC drafts here (13 of them). Paste any
   non-SCHEMATIC id: it gets repaired in one pass through a PR, and a re-run
   costs only the changed statements, since results are cached by statement
   text.

Paste back: the summary lines, the elaboration list, the job's
`.out`/`.err`, and the three `*.report.json` files (or `scp` them off). That
closes DoD 1 and 3. The slice decision in `calibration-program.md` §4 is
then made on measured numbers.

---

## R5 — A CVFN denominator from a real run (DoD 4)

"Tokens and wall-clock attributed per statement, including failed attempts" —
that is the `attempts.json` ledger, which every run writes since 2026-08-19.

### R5a — from a run that survived the restore

If R0 listed any `attempts.json`, substitute that run's directory:

```bash
cd ~/code/lms && uv run python -m lms.accounting experiments/RUN
```

```bash
cd ~/code/lms && uv run python -c "import json,collections as C; r=json.load(open('experiments/RUN/attempts.json'))['records']; n=C.Counter(); t=C.Counter(); w=C.Counter(); [(n.update([x['statement_key']]), t.update({x['statement_key']: x['prompt_tokens'] + x['completion_tokens']}), w.update({x['statement_key']: x['wall_clock_s']})) for x in r]; print(len(r), 'attempts over', len(n), 'statements'); [print(k, n[k], t[k], round(w[k], 1)) for k in sorted(t, key=t.get, reverse=True)[:15]]"
```

**Checkpoint R5a:** the report shows an `overhead tokens:` count rather than
`n/a (run predates the ledger)`, `total wall-clock` is non-zero, and the
second command attributes tokens and seconds to named statements. That is
DoD 4, whatever the CVFN value is. **`CVFN: undefined` is still a pass**: the
DoD asks for the denominator's inputs, not a non-zero numerator. So is
**`CVFN: unmeasurable — goal forbids Mathlib.CategoryTheory`**, which any run on
a `stacks-ch4-*` goal prints since #66: those goals make agents rebuild the
library from scratch, which no novelty search can match.

### R5b — a small real run, if nothing survived

This one needs a GPU and the vLLM environment, both of which the restore
removed. **GPU work goes through `sbatch` only.** A server started outside
Slurm is invisible to the scheduler and collides with other people's jobs.
The tmux serve in the old runbook (Steps 4c/4d) does not apply here.

Driver version on a GPU node decides the vLLM pin (old runbook, Step 4a):
below 580.65.06, keep `vllm<0.20`.

```bash
srun -p GPU_PARTITION --gres=GPU_GRES:1 --time=00:05:00 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

```bash
cd ~ && export UV_LINK_MODE=copy && uv self update && hash -r && uv venv "$SCRATCH/vllm-env" --python 3.12
```

Driver below 580.65.06:

```bash
uv pip install --python "$SCRATCH/vllm-env/bin/python" "vllm<0.20" --torch-backend=auto
```

Driver 580.65.06 or newer:

```bash
uv pip install --python "$SCRATCH/vllm-env/bin/python" vllm --torch-backend=auto
```

**Weights.** ADR 0001's local arm is Qwen3.6-27B-FP8, but
`scripts/slurm/lms_run.sbatch` still defaults `MODEL_REPO` to
Qwen3-Coder-30B-A3B-Instruct. Pass `MODEL_REPO` explicitly: either the
repository id the August committee runs served, or, if R0 found it under
`/scratch/shared/hf/hub`, the local `snapshots/<hash>` path. A local path
skips a multi-GB download into your scratch.

Then one agent, one generation, on the smallest allocation that has served
before (TP=2):

```bash
cd ~/code/lms && mkdir -p logs && sbatch -p GPU_PARTITION --gres=GPU_GRES:2 --time=02:00:00 --export=ALL,TP_SIZE=2,AGENTS=1,GENERATIONS=1,RUN_TAG=dod4_cost_smoke,MODEL_REPO=MODEL scripts/slurm/lms_run.sbatch
```

Then run R5a's two commands with `RUN=dod4_cost_smoke`. The batch script
guards and restores the tracked foundation files on every exit path. Its last
lines must read `lean/ clean after restore`.

---

## R6 — Committee re-run: harness or model? (after R4 and R5)

Not a DoD item. It is the first experiment in the post-box queue: *re-run
committee mode with the August fixes merged, then judge the model.* The last
committee run on this box, `committee_fix_c` (2026-08-20), ended in harness
defects, not in a verdict on the model. Its generations 5–9 verified 0 of 33,
on a foundation API mismatch. Three "verified" artifacts were the scribe's
prompt scaffold, and Gate 4 never ran. All of that is fixed on `main`.
Re-running the same configuration separates the two explanations.

Same configuration as `committee_fix_c` (3 agents, 3 groups,
`stacks-ch4-phase1`), 10 generations. It ran at TP=4; on a shared node, TP=2
serves 3 agents (about 1.1M tokens of KV cache). That changes wall-clock, not
results. Substitute `GPU_PARTITION`, `GPU_GRES` and `MODEL` as in R5b:

```bash
cd ~/code/lms && git pull --ff-only origin main && mkdir -p logs && sbatch -p GPU_PARTITION --gres=GPU_GRES:2 --time=08:00:00 --export=ALL,TP_SIZE=2,AGENTS=3,N_GROUPS=3,ITERATIVE=0,GENERATIONS=10,GOAL=stacks-ch4-phase1,RUN_TAG=committee_rerun_a,MODEL_REPO=MODEL scripts/slurm/lms_run.sbatch
```

```bash
grep -m1 '^agents=' ~/code/lms/logs/lms-run-JOBID.out
```

It must end `n_groups=3`. If the field is missing, the checkout predates the
knob and the run is iterative, not committee: `scancel` it. **Do not use
`--resume`** if the job hits its walltime. A resumed run points its foundation
at the output directory, where Lean cannot import it (known, not yet carded).
A partial run is still a result.

Per generation: created, verified, tokens, seconds:

```bash
cd ~/code/lms && uv run python -c "import json; [print(g['generation'], g['artifacts_created'], g['artifacts_verified'], g['tokens_used'], round(g.get('wall_clock_s', 0))) for g in json.load(open('experiments/committee_rerun_a/results.json'))['generations']]"
```

Status, novelty and reuse over the whole run:

```bash
cd ~/code/lms && uv run python -c "import json,collections as C; a=json.load(open('experiments/committee_rerun_a/artifacts.json'))['artifacts']; v=[x for x in a if x.get('status')=='verified_lean']; print(len(a), 'created', len(v), 'verified'); print('status', dict(C.Counter(x.get('status') for x in a))); print('novelty on verified', dict(C.Counter(x.get('novelty_level') for x in v))); print('verified citing earlier work', sum(1 for x in v if x.get('references')))"
```

```bash
cd ~/code/lms && uv run python -m lms.accounting experiments/committee_rerun_a
```

**Checkpoint R6**, against `committee_fix_c`: 71 created, 10 verified (3 of
them the scaffold), generations 5–9 at 0 of 33, `novelty_level` unset on all
71, about 5.6M tokens.

- **Generations 5–9 verify something.** The foundation API fixes held.
- **Every verified artifact has a `novelty_level`.** Gate 4 is wired. Expect
  N0 and INCONCLUSIVE only: on this goal a statement over the foundation's own
  `Category` cannot read N1 since #66.
- **`verified citing earlier work` above 0.** Reuse is being recorded.
- **The CVFN line reads `unmeasurable`.** That is by construction on this
  goal. This run measures the harness, not novelty.
- **Then judge the model.** The report's `gate failures (ledger)` histogram
  separates harness rejections from Lean failures. If verification stays rare,
  and the failures are Lean errors in the agents' own mathematics, then the
  local model is the constraint. ADR 0001's control arm measures how much.

---

## What to paste back

R0 in full; the R1 echo; R2a/R2b versions, cache wall-clock and the build
tail; R3's tail; R4's summary lines, elaboration list and job logs; R5a's two
outputs; R6's `agents=` line and three read-backs. Paste after each checkpoint
rather than at the end, so a failure stops the sequence where it happened.

## What not to do

- Don't build the library from source on a cache miss (R2b).
- Don't run Lean-heavy work on the login node. R4 is a batch job for that
  reason: each probe holds ~5 GB.
- Don't start vLLM outside Slurm (R5b).
- Don't touch `/scratch/shared`. Read from it only.
- Don't hand-edit an arc draft on the box to make a probe pass. Repairs go
  through a PR, so the measured statement is the one in the repo.
