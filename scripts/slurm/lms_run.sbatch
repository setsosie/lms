#!/usr/bin/env bash
# Self-contained LMS run: bring up vLLM, wait for it, run the experiment, tear down.
#
# Everything in here is a gotcha already paid for on the box -- see
# docs/infrastructure/cluster-runbook-calibration.md. Read that before editing.
#
#   sbatch scripts/slurm/lms_run.sbatch
#   sbatch --export=ALL,AGENTS=6,GENERATIONS=5 scripts/slurm/lms_run.sbatch
#   sbatch --export=ALL,N_GROUPS=3,ITERATIVE=0 scripts/slurm/lms_run.sbatch   # committee mode
#   sbatch --export=ALL,RESUME=1,RUN_TAG=<tag>,... scripts/slurm/lms_run.sbatch  # after a walltime
#
# ---------------------------------------------------------------------------
# SITE CONFIG -- from the restore runbook's Step R0, run 2026-11-04:
#   one node, default partition gpu* (2-day limit), four GPUs whose gres name
#   is gpu:nvidia_h100_nvl. The name before the restore, gpu:h100, is refused.
#   No --account is needed.
# Command-line flags override the directives below.
# ---------------------------------------------------------------------------
#SBATCH --job-name=lms-run
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# All four GPUs by default (decided 2026-08-19; runbook 4d) — the 9-agent
# config slightly exceeds TP=2's 1,139,984-token KV cache. Walltime sized for
# a 2M-token budget run (est. up to ~6 h for the 1-agent config). On a shared
# day: sbatch --gres=gpu:nvidia_h100_nvl:2 --time=04:00:00 --export=ALL,TP_SIZE=2 ...
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# --- Experiment knobs (override via --export) -------------------------------
AGENTS="${AGENTS:-3}"
GENERATIONS="${GENERATIONS:-3}"
GOAL="${GOAL:-stacks-ch4-phase1}"
ITERATIVE="${ITERATIVE:-1}"          # 1 = pass --iterative
# Committee mode (--groups) when > 0. Not GROUPS: bash owns that name and
# silently ignores assignments to it.
N_GROUPS="${N_GROUPS:-0}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-2}"  # committee mode only
# Token budget for the whole run (--max-tokens); empty = no limit. The harness
# checks it when a generation starts, so a run stops at the first generation
# boundary past the budget, not at the exact token.
MAX_TOKENS="${MAX_TOKENS:-}"
RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID:-manual}}"
# 1 = continue experiments/$RUN_TAG from its checkpoint, to GENERATIONS total.
# Pass the same mode knobs as the original job; the checkpoint does not keep them.
RESUME="${RESUME:-0}"
# Every generation, so a walltime kill loses at most the one in flight. The
# CLI default (10) let a 10-generation job die with nothing to resume from.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"

REPO="${REPO:-$HOME/code/lms}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-lms-generalist}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TP_SIZE="${TP_SIZE:-4}"
# Seconds to wait for the server to list its model. A cold load took about
# 3 minutes on the box; the old loop said 600 s and really waited up to 20 min.
READY_TIMEOUT="${READY_TIMEOUT:-900}"
# vLLM 0.33.0 compiles its DeepGEMM and FlashInfer kernels at load, which needs a
# CUDA toolkit this node doesn't have on its default paths: it died in
# find_cuda_toolkit. Its GDN prefill also picks FlashInfer without checking that
# it's there, so a run gets past startup and fails on the first real request.
# Its sampler does the same: it checks the GPU but not the toolkit, and died in
# the first sampling pass (round R5d).
# These use the kernels that need no toolkit. Override them if one is installed.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
# Anything else for `vllm serve`, split on spaces (e.g. "--enforce-eager").
LMS_EXTRA_VLLM_ARGS="${LMS_EXTRA_VLLM_ARGS:-}"

# --- Scratch routing --------------------------------------------------------
# A batch job does NOT source ~/.bashrc, so nothing exported interactively in
# Step 0.5 survives into here. Every one of these must be set explicitly or the
# job will redownload weights to $HOME and blow the quota.
# Per-user scratch moved under /scratch/users/ when the box was restored.
export SCRATCH="${SCRATCH:-/scratch/users/$USER}"
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/cache"
export ELAN_HOME="$SCRATCH/elan"

# Same reason: elan's bin dir is on PATH only because .bashrc put it there.
# RealLeanVerifier does fall back to $ELAN_HOME/bin (26Q3-HARN-07), but `lake`
# is also invoked directly, so put it on PATH rather than relying on that.
export PATH="$ELAN_HOME/bin:$HOME/.elan/bin:$PATH"

# --- Harness -> local server wiring -----------------------------------------
# Exported rather than read from .env: a batch job should not depend on a
# gitignored file that may or may not exist on this checkout.
export LMS_OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export LMS_OPENAI_MODEL="$SERVED_NAME"
export OPENAI_API_KEY="EMPTY"
# Through `tee`, Python block-buffers stdout: progress lines (and the
# runbook's resume check) would reach the log only in 8 KB chunks.
export PYTHONUNBUFFERED=1

LOG_DIR="$SCRATCH/lms-runs/$RUN_TAG"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
OUT_DIR="$REPO/experiments/$RUN_TAG"

cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 1; }

echo "=== LMS run $RUN_TAG ==="
echo "node=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset-by-slurm}"
echo "agents=$AGENTS generations=$GENERATIONS goal=$GOAL iterative=$ITERATIVE n_groups=$N_GROUPS"
echo "resume=$RESUME checkpoint_interval=$CHECKPOINT_INTERVAL out=$OUT_DIR"
echo "max_tokens=${MAX_TOKENS:-unlimited}"

# Before vLLM loads, like the check below: a fresh run over a checkpoint
# overwrites it, and a resume with no checkpoint has nothing to continue.
if [ "$RESUME" = "1" ]; then
  # foundation.json is absent from checkpoints older than resume support,
  # which lms.run refuses -- but only after vLLM has loaded.
  for f in results.json metadata.json foundation.json; do
    [ -f "$OUT_DIR/$f" ] || { echo "FATAL: RESUME=1 but $OUT_DIR has no $f"; exit 2; }
  done
fi
if [ "$RESUME" != "1" ] && [ -f "$OUT_DIR/results.json" ]; then
  echo "FATAL: $OUT_DIR already holds a run; pass RESUME=1 or a new RUN_TAG"
  exit 2
fi

# Same reason: argparse would reject it, but only after vLLM has loaded.
if [ -n "$MAX_TOKENS" ] && ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: MAX_TOKENS=$MAX_TOKENS is not a positive whole number"
  exit 2
fi

# lms.run refuses --groups with --iterative, but only after vLLM has loaded.
if [ "$N_GROUPS" -gt 0 ] && [ "$ITERATIVE" = "1" ]; then
  echo "FATAL: committee mode (N_GROUPS=$N_GROUPS) needs ITERATIVE=0"
  exit 2
fi
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "logs=$LOG_DIR"

# --- Guard the tracked Lean corpus ------------------------------------------
# A run rewrites the foundation, which is a tracked build input (issue #19).
# Back it up so a crashed job cannot leave the corpus mutated.
#
# BOTH files, not just the .lean. `FoundationFile.save()` writes the Lean
# source and its `Foundation.json` metadata sidecar, and both are tracked.
# Backing up only the .lean left every run ending with a dirty tree --
# observed on shakedown_3x3_e, where the restore reported success and
# `git status` still showed ` M lean/LMS/Foundation.json`.
FOUNDATION_FILES="lean/LMS/Foundation.lean lean/LMS/Foundation.json"
for f in $FOUNDATION_FILES; do
  if [ -f "$f" ]; then
    cp "$f" "$LOG_DIR/$(basename "$f").bak"
    echo "backed up $f"
  fi
done

# --- Preflight --------------------------------------------------------------
command -v lake >/dev/null 2>&1 || { echo "FATAL: lake not on PATH; check ELAN_HOME=$ELAN_HOME"; exit 1; }
[ -d "lean/.lake" ] || { echo "FATAL: lean/.lake missing -- Mathlib not built on this node"; exit 1; }
[ -x "$SCRATCH/vllm-env/bin/vllm" ] || { echo "FATAL: no vLLM venv at $SCRATCH/vllm-env"; exit 1; }

# Refuse to load onto a GPU that is already busy.
#
# A server started outside Slurm is invisible to the scheduler, so Slurm can
# allocate a GPU it is already holding. Without this check the failure is a
# confusing mid-load CUDA OOM rather than "something else is on this GPU".
#
# Only the GPUs Slurm allocated are checked. Other people's jobs share this
# node; a GPU busy with theirs is not ours to refuse over.
echo "--- GPU state (allocated: ${CUDA_VISIBLE_DEVICES:-all}) ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' -v vis="${CUDA_VISIBLE_DEVICES:-}" '
      BEGIN { n = split(vis, ids, ","); for (i = 1; i <= n; i++) mine[ids[i]] = 1 }
      (n == 0 || ($1 in mine)) && $2 > 1024 { printf "%s ", $1 }')
if [ -n "$BUSY" ]; then
  echo "FATAL: GPU(s) $BUSY already hold >1 GiB. A vLLM started outside Slurm"
  echo "       (e.g. in tmux) is invisible to the scheduler and will collide."
  echo "       Stop it first, or free those GPUs, then resubmit."
  exit 1
fi

# The port has to be free too, for the same reason.
if curl -sS --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "FATAL: something already answers on port $PORT. Stop it or set PORT=..."
  exit 1
fi

# --- Start vLLM -------------------------------------------------------------
# Absolute path into vLLM's OWN venv. Never `uv run vllm`: uv re-syncs uv.lock,
# which downgrades the openai package the harness needs.
#
# Do NOT hardcode CUDA_VISIBLE_DEVICES here. Under --gres Slurm has already
# set it to the GPUs it allocated -- overriding it would reach outside the
# allocation and collide with another job. Only pick devices when running this
# script outside Slurm.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi
echo "--- starting vLLM on GPUs ${CUDA_VISIBLE_DEVICES:-all} (log: $VLLM_LOG) ---"
"$SCRATCH/vllm-env/bin/vllm" serve "$MODEL_REPO" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --served-model-name "$SERVED_NAME" \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  $LMS_EXTRA_VLLM_ARGS \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Kill the server on any exit path, or a failed job strands the H100s until the
# walltime expires.
cleanup() {
  echo "--- tearing down vLLM (pid $VLLM_PID) ---"
  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# --- Wait for readiness -----------------------------------------------------
# Poll rather than sleep: load time varies with page cache and is ~2-3 min cold.
# vLLM binds the port long before it can answer, so a `curl` during load waits
# out its own limit: time comes from the clock, not from counting loops.
echo "--- waiting for vLLM on port $PORT (up to ${READY_TIMEOUT}s) ---"
READY=0
WAIT_START=$SECONDS
while [ $((SECONDS - WAIT_START)) -lt "$READY_TIMEOUT" ]; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "FATAL: vLLM died during startup. Tail of $VLLM_LOG:"
    tail -40 "$VLLM_LOG"
    exit 1
  fi
  if curl -sS --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "vLLM ready after ~$((SECONDS - WAIT_START))s"
    break
  fi
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "FATAL: vLLM not ready after ${READY_TIMEOUT}s"; tail -40 "$VLLM_LOG"; exit 1; }

grep -m1 "GPU KV cache size" "$VLLM_LOG" || true

# --- Run --------------------------------------------------------------------
ITER_FLAG=""
[ "$ITERATIVE" = "1" ] && ITER_FLAG="--iterative"
MODE_FLAGS=""
[ "$N_GROUPS" -gt 0 ] && MODE_FLAGS="--groups --n-groups $N_GROUPS --repair-attempts $REPAIR_ATTEMPTS"
BUDGET_ARGS=()
[ -n "$MAX_TOKENS" ] && BUDGET_ARGS=(--max-tokens "$MAX_TOKENS")
RESUME_ARGS=()
TEE_ARGS=()
if [ "$RESUME" = "1" ]; then
  RESUME_ARGS=(--resume "$OUT_DIR")
  TEE_ARGS=(-a)  # keep the earlier jobs' log
fi

echo "--- running experiment -> $OUT_DIR ---"
START=$SECONDS
uv run python -m lms.run \
  --provider openai \
  --verifier real \
  --agents "$AGENTS" \
  --generations "$GENERATIONS" \
  --goal "$GOAL" \
  $ITER_FLAG \
  $MODE_FLAGS \
  ${BUDGET_ARGS[@]+"${BUDGET_ARGS[@]}"} \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --output "$OUT_DIR" \
  2>&1 | tee ${TEE_ARGS[@]+"${TEE_ARGS[@]}"} "$LOG_DIR/run.log"
RUN_RC=${PIPESTATUS[0]}
ELAPSED=$((SECONDS - START))

echo "--- experiment exited rc=$RUN_RC after ${ELAPSED}s ---"

# --- Report -----------------------------------------------------------------
# Wall-clock per artifact is the number Phase C sizing needs and the one the
# harness does not yet record itself (26Q3-HARN-05).
if [ -f "$OUT_DIR/artifacts.json" ]; then
  uv run python -c "
import json, sys
d = json.load(open('$OUT_DIR/artifacts.json'))['artifacts']
by = {}
for a in d:
    by[a.get('status', 'unknown')] = by.get(a.get('status', 'unknown'), 0) + 1
print('artifacts:', len(d))
print('by status:', by)
if '$RESUME' == '1':
    # This job ran only part of the run; dividing by every artifact understates it.
    print('elapsed_s (this job):', $ELAPSED, '| per artifact: n/a on a resumed job')
else:
    print('elapsed_s:', $ELAPSED, '| per artifact:', round($ELAPSED / max(len(d), 1), 1))
" | tee -a "$LOG_DIR/run.log"
fi

# Restore the corpus and say plainly whether the run moved it.
for f in $FOUNDATION_FILES; do
  bak="$LOG_DIR/$(basename "$f").bak"
  [ -f "$bak" ] || continue
  if ! diff -q "$bak" "$f" >/dev/null 2>&1; then
    echo "NOTE: the run modified $f; restoring from backup"
    cp "$bak" "$f"
  fi
done

# The restore above is not trusted -- it is verified. A "restored" message on
# a tree that is still dirty is exactly the class of harness dishonesty this
# sprint exists to remove, and it shipped once already.
if [ -n "$(git status --porcelain lean/ 2>/dev/null)" ]; then
  echo "WARN: lean/ still dirty after restore:"
  git status --short lean/
else
  echo "lean/ clean after restore"
fi

echo "=== done: $LOG_DIR ==="
exit "$RUN_RC"
