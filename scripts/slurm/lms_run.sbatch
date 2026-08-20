#!/usr/bin/env bash
# Self-contained LMS run: bring up vLLM, wait for it, run the experiment, tear down.
#
# Everything in here is a gotcha already paid for on the box -- see
# docs/infrastructure/cluster-runbook-calibration.md. Read that before editing.
#
#   sbatch scripts/slurm/lms_run.sbatch
#   sbatch --export=ALL,AGENTS=6,GENERATIONS=5 scripts/slurm/lms_run.sbatch
#
# ---------------------------------------------------------------------------
# SITE CONFIG -- resolved 2026-08-10 on mahpiya:
#   sinfo  -> "gpu* infinite gpu:h100:4 localhost"  (single node, default part.)
#   sacctmgr -> no accounting_storage plugin, so --account must NOT be set
# ---------------------------------------------------------------------------
#SBATCH --job-name=lms-run
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# All four GPUs by default (decided 2026-08-19; runbook 4d) — the 9-agent
# config slightly exceeds TP=2's 1,139,984-token KV cache. Walltime sized for
# a 2M-token budget run (est. up to ~6 h for the 1-agent config). On a shared
# day: sbatch --gres=gpu:h100:2 --time=04:00:00 --export=ALL,TP_SIZE=2 ...
#SBATCH --gres=gpu:h100:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# --- Experiment knobs (override via --export) -------------------------------
AGENTS="${AGENTS:-3}"
GENERATIONS="${GENERATIONS:-3}"
GOAL="${GOAL:-stacks-ch4-phase1}"
ITERATIVE="${ITERATIVE:-1}"          # 1 = pass --iterative
RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID:-manual}}"

REPO="${REPO:-$HOME/code/lms}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-lms-generalist}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TP_SIZE="${TP_SIZE:-4}"

# --- Scratch routing --------------------------------------------------------
# A batch job does NOT source ~/.bashrc, so nothing exported interactively in
# Step 0.5 survives into here. Every one of these must be set explicitly or the
# job will redownload weights to $HOME and blow the quota.
export SCRATCH="${SCRATCH:-/scratch/$USER}"
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

LOG_DIR="$SCRATCH/lms-runs/$RUN_TAG"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
OUT_DIR="$REPO/experiments/$RUN_TAG"

cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 1; }

echo "=== LMS run $RUN_TAG ==="
echo "node=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset-by-slurm}"
echo "agents=$AGENTS generations=$GENERATIONS goal=$GOAL iterative=$ITERATIVE"
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
# This box runs a single-node Slurm over the same four H100s that a tmux vLLM
# may be holding. A server started outside Slurm is invisible to the scheduler,
# so Slurm believes all four are free and can allocate the two already in use.
# Without this check the failure is a confusing mid-load CUDA OOM rather than
# "something else is on this GPU".
echo "--- GPU state ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 > 1024 {printf "%s ", $1}')
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
echo "--- waiting for vLLM on port $PORT ---"
READY=0
for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "FATAL: vLLM died during startup. Tail of $VLLM_LOG:"
    tail -40 "$VLLM_LOG"
    exit 1
  fi
  if curl -sS --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "vLLM ready after ~$((i * 5))s"
    break
  fi
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "FATAL: vLLM not ready after 600s"; tail -40 "$VLLM_LOG"; exit 1; }

grep -m1 "GPU KV cache size" "$VLLM_LOG" || true

# --- Run --------------------------------------------------------------------
ITER_FLAG=""
[ "$ITERATIVE" = "1" ] && ITER_FLAG="--iterative"

echo "--- running experiment -> $OUT_DIR ---"
START=$SECONDS
uv run python -m lms.run \
  --provider openai \
  --verifier real \
  --agents "$AGENTS" \
  --generations "$GENERATIONS" \
  --goal "$GOAL" \
  $ITER_FLAG \
  --output "$OUT_DIR" \
  2>&1 | tee "$LOG_DIR/run.log"
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
