#!/usr/bin/env bash
# Self-contained LMS run: bring up vLLM, wait for it, run the experiment, tear down.
#
# Everything in here is a gotcha already paid for on the box -- see
# docs/infrastructure/cluster-runbook-calibration.md. Read that before editing.
#
#   sbatch scripts/slurm/lms_run.sbatch
#   sbatch --export=ALL,AGENTS=6,GENERATIONS=5 scripts/slurm/lms_run.sbatch
#   sbatch --export=ALL,N_GROUPS=3,ITERATIVE=0 scripts/slurm/lms_run.sbatch   # committee mode
#   sbatch --export=ALL,N_GROUPS=3,ITERATIVE=0,RESET_EACH_GENERATION=1 scripts/slurm/lms_run.sbatch  # R9 arm
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
# 1 = pass --reset-each-generation: the foundation goes back to its seed before
# every generation after the first (ADR 0003's R9 arm). 0 or 1, nothing else.
RESET_EACH_GENERATION="${RESET_EACH_GENERATION:-0}"
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
# Seconds each start attempt gets to list its model; "Start vLLM" below says
# why there can be more than one. A healthy start answered in 70-80 s on the
# box and a cold one in 195-212 s. READY_TIMEOUT is this limit's name from
# when there was one attempt: a caller that sets it and not START_TIMEOUT gets
# it as the limit for each attempt.
START_TIMEOUT="${START_TIMEOUT:-${READY_TIMEOUT:-360}}"
# The start attempts to make, in order, until one answers. START_LADDER=base
# is the single attempt this script used to make.
START_LADDER="${START_LADDER:-base no-custom-ar no-p2p}"
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
echo "reset_each_generation=$RESET_EACH_GENERATION"
echo "start_ladder=$START_LADDER start_timeout=${START_TIMEOUT}s"

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

# Same reason, though nothing downstream would refuse it: anything but 1 would
# quietly run the carried arm under the reset arm's name.
if [ "$RESET_EACH_GENERATION" != "0" ] && [ "$RESET_EACH_GENERATION" != "1" ]; then
  echo "FATAL: RESET_EACH_GENERATION=$RESET_EACH_GENERATION must be 0 or 1"
  exit 2
fi

# lms.run refuses --groups with --iterative, but only after vLLM has loaded.
if [ "$N_GROUPS" -gt 0 ] && [ "$ITERATIVE" = "1" ]; then
  echo "FATAL: committee mode (N_GROUPS=$N_GROUPS) needs ITERATIVE=0"
  exit 2
fi

# A misspelt rung or limit would otherwise show only once the server is loading.
if ! [[ "$START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: START_TIMEOUT=$START_TIMEOUT (or READY_TIMEOUT, when that is unset) is not a positive whole number of seconds"
  exit 2
fi
read -ra LADDER <<< "$START_LADDER"
if [ -z "${START_LADDER//[[:space:]]/}" ]; then
  echo "FATAL: START_LADDER is empty; give it one or more of: base no-custom-ar no-p2p"
  exit 2
fi
for rung in "${LADDER[@]}"; do
  case "$rung" in
    base|no-custom-ar|no-p2p) ;;
    *) echo "FATAL: START_LADDER has '$rung'; the rungs are base, no-custom-ar, no-p2p"; exit 2 ;;
  esac
done
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
command -v setsid >/dev/null 2>&1 || { echo "FATAL: setsid not on PATH; it stops a hung server's workers"; exit 1; }

# Refuse to load onto a GPU that is already busy.
#
# A server started outside Slurm is invisible to the scheduler, so Slurm can
# allocate a GPU it is already holding. Without this check the failure is a
# confusing mid-load CUDA OOM rather than "something else is on this GPU".
#
# Only the GPUs Slurm allocated are checked. Other people's jobs share this
# node; a GPU busy with theirs is not ours to refuse over.
#
# busy_gpus prints those of them holding >1 GiB, e.g. "0 2 ". A start that
# hangs is retried only once its GPUs read free again, by the same test.
busy_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v vis="${CUDA_VISIBLE_DEVICES:-}" '
        BEGIN { n = split(vis, ids, ","); for (i = 1; i <= n; i++) mine[ids[i]] = 1 }
        (n == 0 || ($1 in mine)) && $2 > 1024 { printf "%s ", $1 }'
}
echo "--- GPU state (allocated: ${CUDA_VISIBLE_DEVICES:-all}) ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
BUSY=$(busy_gpus)
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

# --- Which GPUs, and how they are linked ------------------------------------
# The gpus= line above does not say which physical cards the job got, and a
# start that hangs (below) may depend on it: H100 NVL cards are NVLink-bridged
# in pairs. Slurm's IDX field names the cards; `nvidia-smi topo -m` shows the
# link between them (NV12 for a bridged pair). Neither may fail the job.
echo "--- GPUs this job holds, and how they are linked ---"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  GRES_IDX=$(scontrol show job -d "$SLURM_JOB_ID" 2>/dev/null | grep -o 'GRES=[^ ]*IDX:[^ ]*' | sort -u) || true
  echo "${GRES_IDX:-scontrol gave no GRES IDX for job $SLURM_JOB_ID}"
fi
nvidia-smi topo -m 2>&1 || true

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

# A start can hang. In batch q1, 4 of 18 TP=2 starts logged "vLLM is using
# nccl==2.30.7" and then nothing until vLLM's own 600 s engine timeout, where a
# healthy start logs its all-reduce backends a second later and answers in
# 70-80 s. Each hang cost the whole job. The one hung job that logged its GPUs
# had cards 0 and 2; the starts that served ran on 0-1, a bridged pair. The
# guess, not yet tested, is that custom all-reduce or NCCL peer-to-peer hangs
# over PCIe on an unbridged pair. So a start that is alive but silent after
# START_TIMEOUT is stopped and the next rung is tried:
#   base          today's command and environment, unchanged
#   no-custom-ar  base + --disable-custom-all-reduce
#   no-p2p        no-custom-ar + NCCL_P2P_DISABLE=1, for the server only
# A server that dies is not retried: that is a different failure. The rung
# that served is printed as SERVED_BY=<rung>. A rung below base may be slower,
# so report it next to the run's results.
#
# Each attempt logs to vllm.attempt-<rung>.log, and vllm.log is a second name
# for the current attempt's log, so it ends as the log of the one that served.
# A resumed job reuses this directory; an earlier job's attempts go first, as
# its vllm.log always did.
rm -f "$LOG_DIR"/vllm.attempt-*.log
VLLM_PID=""
ATTEMPT_LOG="$VLLM_LOG"

# Each attempt runs in its own process group, because vLLM starts an engine
# process and a worker per GPU, and a worker left behind keeps its GPU's
# memory. A script's background job is not a group leader, so setsid runs the
# server in place and $! is the new group's id.
start_server() {
  local rung=$1
  local -a rung_args=() rung_env=()
  case "$rung" in
    no-custom-ar) rung_args=(--disable-custom-all-reduce) ;;
    no-p2p) rung_args=(--disable-custom-all-reduce); rung_env=(NCCL_P2P_DISABLE=1) ;;
  esac
  ATTEMPT_LOG="$LOG_DIR/vllm.attempt-$rung.log"
  : > "$ATTEMPT_LOG"
  ln -f "$ATTEMPT_LOG" "$VLLM_LOG" 2>/dev/null || ln -sf "$(basename "$ATTEMPT_LOG")" "$VLLM_LOG"
  echo "--- starting vLLM ($rung) on GPUs ${CUDA_VISIBLE_DEVICES:-all} (log: $ATTEMPT_LOG) ---"
  env ${rung_env[@]+"${rung_env[@]}"} setsid "$SCRATCH/vllm-env/bin/vllm" serve "$MODEL_REPO" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$SERVED_NAME" \
    --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
    $LMS_EXTRA_VLLM_ARGS \
    ${rung_args[@]+"${rung_args[@]}"} \
    > "$ATTEMPT_LOG" 2>&1 &
  VLLM_PID=$!
}

# Stop an attempt's whole process group: TERM, then KILL whatever is left after
# STOP_GRACE seconds. A process stuck inside NCCL never runs its TERM handler.
STOP_GRACE="${STOP_GRACE:-20}"
stop_server() {
  local pid=$1 t0=$SECONDS
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  while kill -0 -- "-$pid" 2>/dev/null && [ $((SECONDS - t0)) -lt "$STOP_GRACE" ]; do
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  # KILL is not instant; give the group a moment to be gone.
  t0=$SECONDS
  while kill -0 -- "-$pid" 2>/dev/null && [ $((SECONDS - t0)) -lt 10 ]; do
    sleep 1
  done
}

# Kill the server on any exit path, or a failed job strands the H100s until the
# walltime expires.
cleanup() {
  [ -n "$VLLM_PID" ] || return 0
  echo "--- tearing down vLLM (pid $VLLM_PID and its process group) ---"
  stop_server "$VLLM_PID"
  VLLM_PID=""
}
trap cleanup EXIT INT TERM

# --- Wait for readiness -----------------------------------------------------
# Poll rather than sleep: load time varies with page cache and is ~2-3 min cold.
# vLLM binds the port long before it can answer, so a `curl` during load waits
# out its own limit: time comes from the clock, not from counting loops.
# Returns 0 once the server answers and 1 if it is alive but silent at
# START_TIMEOUT; a server that dies ends the job.
wait_for_server() {
  local rung=$1
  ATTEMPT_START=$SECONDS
  while [ $((SECONDS - ATTEMPT_START)) -lt "$START_TIMEOUT" ]; do
    kill -0 "$VLLM_PID" 2>/dev/null || break
    if curl -sS --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "FATAL: vLLM died during startup ($rung). Tail of $ATTEMPT_LOG:"
    tail -40 "$ATTEMPT_LOG"
    exit 1
  fi
  return 1
}

# A stopped attempt's GPUs must read free before the next one loads, or it
# fails for that reason instead. Bounded: GPU_FREE_TIMEOUT seconds.
GPU_FREE_TIMEOUT="${GPU_FREE_TIMEOUT:-60}"
wait_for_free_gpus() {
  local t0=$SECONDS busy
  busy=$(busy_gpus)
  while [ -n "$busy" ] && [ $((SECONDS - t0)) -lt "$GPU_FREE_TIMEOUT" ]; do
    sleep 5
    busy=$(busy_gpus)
  done
  [ -n "$busy" ] || return 0
  echo "FATAL: GPU(s) $busy still hold >1 GiB ${GPU_FREE_TIMEOUT}s after the hung start was"
  echo "       stopped; something outside its process group holds them."
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  return 1
}

SERVED_BY=""
TRIED=""
LADDER_START=$SECONDS
for rung in "${LADDER[@]}"; do
  if [ -n "$TRIED" ]; then
    wait_for_free_gpus || exit 1
  fi
  start_server "$rung"
  echo "--- waiting for vLLM on port $PORT (up to ${START_TIMEOUT}s) ---"
  if wait_for_server "$rung"; then
    SERVED_BY=$rung
    break
  fi
  TRIED="${TRIED:+$TRIED }$rung"
  echo "WARN: vLLM ($rung) is alive but not answering after ${START_TIMEOUT}s. Its last lines:"
  tail -3 "$ATTEMPT_LOG"
  echo "--- stopping it (pid $VLLM_PID and its process group) ---"
  stop_server "$VLLM_PID"
  VLLM_PID=""
done
if [ -z "$SERVED_BY" ]; then
  echo "FATAL: vLLM not ready after ${START_TIMEOUT}s on each rung tried: $TRIED"
  tail -40 "$ATTEMPT_LOG"
  exit 1
fi
READY_AFTER=$((SECONDS - ATTEMPT_START))
echo "SERVED_BY=$SERVED_BY after ~${READY_AFTER}s"
echo "vLLM ready after ~${READY_AFTER}s"
if [ -n "$TRIED" ]; then
  echo "NOTE: $TRIED hung; $SERVED_BY served ~$((SECONDS - LADDER_START))s after the first start."
  echo "      It may be slower than base: report the rung with this run's results."
fi

grep -m1 "GPU KV cache size" "$VLLM_LOG" || true

# --- Run --------------------------------------------------------------------
ITER_FLAG=""
[ "$ITERATIVE" = "1" ] && ITER_FLAG="--iterative"
MODE_FLAGS=""
[ "$N_GROUPS" -gt 0 ] && MODE_FLAGS="--groups --n-groups $N_GROUPS --repair-attempts $REPAIR_ATTEMPTS"
BUDGET_ARGS=()
[ -n "$MAX_TOKENS" ] && BUDGET_ARGS=(--max-tokens "$MAX_TOKENS")
RESET_ARGS=()
[ "$RESET_EACH_GENERATION" = "1" ] && RESET_ARGS=(--reset-each-generation)
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
  ${RESET_ARGS[@]+"${RESET_ARGS[@]}"} \
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
