#!/usr/bin/env bash
# LMS queue kit: a pack job. One GPU and one vLLM server for up to three runs at once, each in its
# own worktree (a slot of the job's lane), so no two runs ever share a foundation or a Lean tree.
# Batches q3 and q4 measured why: a run peaks at 3 requests and at most ~10% of one GPU's KV
# cache, and a TP=1 server gave 55 output tokens/s per request at 9 requests against 58 at 3.
#
# ~/lms_q_job.sbatch runs this (LMS_Q_RUN=~/lms_q_pack.sh, or a round script that execs it) in slot
# 1 of the lane, after it has taken the lane's lock and checked out the pin there. Then:
#   1. read the pack file (LMS_Q_PACK): one line per run, SLOT <TAB> TAG [<TAB> VAR=VALUE]...;
#   2. ready slots 2 and 3 as lms_q_job.sbatch readied slot 1: the tracked foundation put back, the
#      pin checked out, and Lake asked whether the library they share with lane 1 is up to date;
#   3. check each run as scripts/slurm/lms_run.sbatch checks before it loads a model. A run whose
#      slot or settings fail is left out with a FATAL line; the others still run;
#   4. start one vLLM server on this job's GPU, with lms_run.sbatch's command and settings;
#   5. start every run at once, each in its slot, doing what lms_run.sbatch's "Run" section does,
#      with its log at <slot>/logs/<job name>-<job id>.<tag>.out;
#   6. wait for all of them, print one "PACK <tag>:" line each, and stop the server.
# Exit code 0 only if every run in the pack started and exited 0.
#
# The job's environment holds the server's settings (PORT, MODEL_REPO, MAX_MODEL_LEN,
# LMS_EXTRA_VLLM_ARGS, VLLM_CACHE_ROOT) and the knobs the runs share; a pack line's VAR=VALUE
# pairs are that run's own knobs (seed, agents, budget, ...) and override the shared ones. The
# same script, with LMS_Q_PACK_ONE=1, is one run (step 5).
set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"

# --- what lms_run.sbatch sets for every job ------------------------------------------------------
export SCRATCH="${SCRATCH:-/scratch/users/${USER:-$(id -un)}}"
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/cache"
export ELAN_HOME="$SCRATCH/elan"
export PATH="$ELAN_HOME/bin:$HOME/.elan/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-lms-generalist}"
PORT="${PORT:-8000}"
FOUNDATION_FILES="lean/LMS/Foundation.lean lean/LMS/Foundation.json"

# ==================================================================================================
# One run (LMS_Q_PACK_ONE=1): lms_run.sbatch from "Experiment knobs" to the end, less the server.
# Started by the pack below in the run's slot, with the run's knobs in its environment.
# ==================================================================================================
if [[ "${LMS_Q_PACK_ONE:-}" == 1 ]]; then
    AGENTS="${AGENTS:-3}"
    GENERATIONS="${GENERATIONS:-3}"
    GOAL="${GOAL:-stacks-ch4-phase1}"
    ITERATIVE="${ITERATIVE:-1}"
    N_GROUPS="${N_GROUPS:-0}"
    REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-2}"
    RESET_EACH_GENERATION="${RESET_EACH_GENERATION:-0}"
    MAX_TOKENS="${MAX_TOKENS:-}"
    RUN_TAG="${RUN_TAG:?}"
    RESUME="${RESUME:-0}"
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
    REPO="${REPO:?}"
    export LMS_OPENAI_BASE_URL="http://localhost:${PORT}/v1"
    export LMS_OPENAI_MODEL="$SERVED_NAME"
    export OPENAI_API_KEY="EMPTY"
    LOG_DIR="$SCRATCH/lms-runs/$RUN_TAG"
    mkdir -p "$LOG_DIR"
    OUT_DIR="$REPO/experiments/$RUN_TAG"
    cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 1; }

    echo "=== LMS run $RUN_TAG ==="
    echo "node=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset-by-slurm}"
    echo "agents=$AGENTS generations=$GENERATIONS goal=$GOAL iterative=$ITERATIVE n_groups=$N_GROUPS"
    echo "resume=$RESUME checkpoint_interval=$CHECKPOINT_INTERVAL out=$OUT_DIR"
    echo "max_tokens=${MAX_TOKENS:-unlimited}"
    echo "reset_each_generation=$RESET_EACH_GENERATION"
    echo "pack=${LMS_Q_PACK_NAME:-?} slot=${LMS_Q_PACK_SLOT:-?}: one vLLM server on port $PORT for ${LMS_Q_PACK_TAGS:-?}"
    # The same refusals as lms_run.sbatch, in its order. The pack made them before it loaded the
    # model; they are made again here so this half stands alone.
    if [ "$RESUME" = "1" ]; then
        for f in results.json metadata.json foundation.json; do
            [ -f "$OUT_DIR/$f" ] || { echo "FATAL: RESUME=1 but $OUT_DIR has no $f"; exit 2; }
        done
    fi
    if [ "$RESUME" != "1" ] && [ -f "$OUT_DIR/results.json" ]; then
        echo "FATAL: $OUT_DIR already holds a run; pass RESUME=1 or a new RUN_TAG"
        exit 2
    fi
    if [ -n "$MAX_TOKENS" ] && ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: MAX_TOKENS=$MAX_TOKENS is not a positive whole number"
        exit 2
    fi
    if [ "$RESET_EACH_GENERATION" != "0" ] && [ "$RESET_EACH_GENERATION" != "1" ]; then
        echo "FATAL: RESET_EACH_GENERATION=$RESET_EACH_GENERATION must be 0 or 1"
        exit 2
    fi
    if [ "$N_GROUPS" -gt 0 ] && [ "$ITERATIVE" = "1" ]; then
        echo "FATAL: committee mode (N_GROUPS=$N_GROUPS) needs ITERATIVE=0"
        exit 2
    fi
    echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
    echo "logs=$LOG_DIR"

    for f in $FOUNDATION_FILES; do
        if [ -f "$f" ]; then
            cp "$f" "$LOG_DIR/$(basename "$f").bak"
            echo "backed up $f"
        fi
    done
    command -v lake > /dev/null 2>&1 || { echo "FATAL: lake not on PATH; check ELAN_HOME=$ELAN_HOME"; exit 1; }
    [ -d "lean/.lake" ] || { echo "FATAL: lean/.lake missing -- Mathlib not built on this node"; exit 1; }

    # What lms_run.sbatch prints about its GPUs and its server, from the pack's one server. vllm.log
    # names the pack's server log, so a collect reads the server this run used.
    [[ -f "${LMS_Q_PACK_GPUS:-}" ]] && cat "$LMS_Q_PACK_GPUS"
    [[ -n "${LMS_Q_PACK_VLLM_LOG:-}" ]] && ln -sfn "$LMS_Q_PACK_VLLM_LOG" "$LOG_DIR/vllm.log"
    echo "SERVED_BY=${LMS_Q_PACK_SERVED_BY:-?} after ~${LMS_Q_PACK_READY_AFTER:-?}s"
    echo "vLLM ready after ~${LMS_Q_PACK_READY_AFTER:-?}s"
    [[ -n "${LMS_Q_PACK_VLLM_LOG:-}" ]] && { grep -m1 "GPU KV cache size" "$LMS_Q_PACK_VLLM_LOG" || true; }

    # --- Run: lms_run.sbatch's own lines -------------------------------------------------------
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
        TEE_ARGS=(-a)
    fi

    echo "--- running experiment -> $OUT_DIR ---"
    START=$SECONDS
    # shellcheck disable=SC2086  # ITER_FLAG and MODE_FLAGS split into words, as in lms_run.sbatch
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
    print('elapsed_s (this job):', $ELAPSED, '| per artifact: n/a on a resumed job')
else:
    print('elapsed_s:', $ELAPSED, '| per artifact:', round($ELAPSED / max(len(d), 1), 1))
" | tee -a "$LOG_DIR/run.log"
    fi

    for f in $FOUNDATION_FILES; do
        bak="$LOG_DIR/$(basename "$f").bak"
        [ -f "$bak" ] || continue
        if ! diff -q "$bak" "$f" > /dev/null 2>&1; then
            echo "NOTE: the run modified $f; restoring from backup"
            cp "$bak" "$f"
        fi
    done
    if [ -n "$(git status --porcelain lean/ 2> /dev/null)" ]; then
        echo "WARN: lean/ still dirty after restore:"
        git status --short lean/
    else
        echo "lean/ clean after restore"
    fi
    echo "=== done: $LOG_DIR ==="
    exit "$RUN_RC"
fi

# ==================================================================================================
# The pack.
# ==================================================================================================
fatal() {
    echo "FATAL: $*"
    exit 1
}

: "${REPO:?REPO unset; submit through q_submit_pack}" "${LMS_Q_PIN:?LMS_Q_PIN unset; submit through q_submit_pack}"
: "${LMS_Q_PACK:?LMS_Q_PACK unset; submit through q_submit_pack}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TP_SIZE="${TP_SIZE:-1}"
START_TIMEOUT="${START_TIMEOUT:-${READY_TIMEOUT:-360}}"
# A start that is alive but silent at START_TIMEOUT is stopped and tried again, this many times in
# all. One GPU has no all-reduce to hang in, so every try is the same command (lms_run.sbatch's
# rung "base"); q3's 9 one-GPU starts all served on the first try.
START_TRIES="${START_TRIES:-2}"
GPU_FREE_TIMEOUT="${GPU_FREE_TIMEOUT:-60}"
STOP_GRACE="${STOP_GRACE:-20}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
LMS_EXTRA_VLLM_ARGS="${LMS_EXTRA_VLLM_ARGS:-}"
JOB_NAME="${SLURM_JOB_NAME:-lms-pack}"
JOB_ID="${SLURM_JOB_ID:-none}"
PACK_TAG="${RUN_TAG:-pack_$JOB_ID}"
PACK_DIR="$SCRATCH/lms-runs/$PACK_TAG"
mkdir -p "$PACK_DIR"
[[ "$TP_SIZE" == 1 ]] || fatal "TP_SIZE=$TP_SIZE: a pack serves its runs from one GPU"
[[ "$START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fatal "START_TIMEOUT=$START_TIMEOUT is not a positive whole number of seconds"
[[ "$START_TRIES" =~ ^[1-9]$ ]] || fatal "START_TRIES=$START_TRIES is not 1 to 9"

# Slot J's worktree: slot 1 is the lane's, slot J the lane's with -sJ after it (q_slot_wt).
slot_wt() { if [[ "$1" == 1 ]]; then echo "$REPO"; else echo "$REPO-s$1"; fi; }

# Knobs that belong to the server or the kit. A pack line may not set them: one would quietly
# point one run at another server or another worktree than the others.
SERVER_VARS=" REPO RUN_TAG PORT MODEL_REPO SERVED_NAME MAX_MODEL_LEN TP_SIZE LMS_EXTRA_VLLM_ARGS GDN_PREFILL_BACKEND VLLM_CACHE_ROOT VLLM_USE_DEEP_GEMM VLLM_ALLREDUCE_USE_FLASHINFER VLLM_USE_FLASHINFER_SAMPLER LMS_OPENAI_BASE_URL LMS_OPENAI_MODEL OPENAI_API_KEY SCRATCH HF_HOME ELAN_HOME LMS_Q_PIN LMS_Q_PACK LMS_Q_PACK_ONE LMS_Q_RUN LMS_Q_LANE LMS_Q_LOCK "

# --- 1. the pack file ----------------------------------------------------------------------------
[[ -f "$LMS_Q_PACK" ]] || fatal "no pack file at $LMS_Q_PACK"
SLOT=()
TAG=()
VARS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    IFS=$'\t' read -r -a f <<< "$line"
    ((${#f[@]} >= 2)) || fatal "pack line '$line' has no tag"
    [[ "${f[0]}" =~ ^[1-9]$ ]] || fatal "pack line '$line': slot '${f[0]}' is not 1 to 9"
    [[ "${f[1]}" =~ ^[A-Za-z0-9_.-]+$ ]] || fatal "pack line '$line': '${f[1]}' is not a run tag"
    for ((i = 0; i < ${#SLOT[@]}; i++)); do
        [[ "${SLOT[i]}" != "${f[0]}" ]] || fatal "slot ${f[0]} holds two runs (${TAG[i]} and ${f[1]})"
        [[ "${TAG[i]}" != "${f[1]}" ]] || fatal "run ${f[1]} is in the pack twice"
    done
    for kv in "${f[@]:2}"; do
        [[ "$kv" =~ ^[A-Z_][A-Z0-9_]*= ]] || fatal "pack line for ${f[1]}: '$kv' is not VAR=VALUE"
        [[ "$SERVER_VARS" != *" ${kv%%=*} "* ]] || fatal "pack line for ${f[1]} sets ${kv%%=*}, which is the pack's to set"
    done
    SLOT+=("${f[0]}")
    TAG+=("${f[1]}")
    VARS+=("$(printf '%s\t' "${f[@]:2}")")
done < "$LMS_Q_PACK"
N=${#TAG[@]}
((N)) || fatal "the pack file $LMS_Q_PACK lists no runs"
PEERS="$(printf '%s ' "${TAG[@]}")"
PEERS="${PEERS% }"
echo "=== pack $JOB_NAME: job $JOB_ID, pin ${LMS_Q_PIN:0:8}, $N run(s): $PEERS"

# run_vars I: the run's VAR=VALUE pairs, one per line.
run_vars() { tr '\t' '\n' <<< "${VARS[$1]}" | sed '/^$/d'; }

# --- 2. the slots, and 3. each run's own refusals -----------------------------------------------
# A slot of lane 1 links to lane 1's library, as lanes 2 to 4 do, so every slot but lane 1's own
# first one asks Lake whether that library is up to date (lms_q_job.sbatch asked for slot 1).
ready_slot() {
    local wt="$1" dirty head lake_out
    [[ -d "$wt" ]] || { echo "no worktree at $wt; q_slots makes it"; return 1; }
    [[ -L "$wt/lean/.lake" && -d "$wt/.venv" ]] || { echo "$wt has no Lean tree link or .venv; q_slots makes them"; return 1; }
    cd "$wt" || return 1
    for f in $FOUNDATION_FILES; do
        if ! git diff --quiet -- "$f" 2> /dev/null; then
            echo "NOTE: $f in $(basename "$wt") was left modified by an earlier job; putting the tracked version back"
            git checkout -q -- "$f" || { echo "could not restore $f"; return 1; }
        fi
    done
    find lean -maxdepth 1 -name 'tmp*.lean' -type f -delete 2> /dev/null
    dirty="$(git status --porcelain --untracked-files=no)"
    [[ -z "$dirty" ]] || { echo "$dirty"; echo "the worktree has changes to tracked files that are not the foundation"; return 1; }
    git checkout -q --detach "$LMS_Q_PIN" || { echo "could not check out $LMS_Q_PIN"; return 1; }
    head="$(git rev-parse HEAD)"
    [[ "$head" == "$LMS_Q_PIN" ]] || { echo "checked out $head, wanted $LMS_Q_PIN"; return 1; }
    lake_out="$( (cd lean && lake build --no-build Mathlib) 2>&1)" || {
        tail -n 5 <<< "$lake_out"
        echo "Lake would rebuild the library this slot shares with lane 1"
        return 1
    }
    echo "slot $(basename "$wt"): at ${head:0:8}, the shared library is up to date"
}

# check_run: lms_run.sbatch's refusals before it loads a model, in a subshell holding the run's
# environment. Prints the FATAL line it would print.
check_run() {
    local f
    RESUME="${RESUME:-0}"
    local out="$REPO/experiments/$RUN_TAG"
    if [ "$RESUME" = "1" ]; then
        for f in results.json metadata.json foundation.json; do
            [ -f "$out/$f" ] || { echo "RESUME=1 but experiments/$RUN_TAG has no $f"; return 2; }
        done
    fi
    if [ "$RESUME" != "1" ] && [ -f "$out/results.json" ]; then
        echo "experiments/$RUN_TAG already holds a run; pass RESUME=1 or a new RUN_TAG"
        return 2
    fi
    if [ -n "${MAX_TOKENS:-}" ] && ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_TOKENS=$MAX_TOKENS is not a positive whole number"
        return 2
    fi
    if [ "${RESET_EACH_GENERATION:-0}" != "0" ] && [ "${RESET_EACH_GENERATION:-0}" != "1" ]; then
        echo "RESET_EACH_GENERATION=$RESET_EACH_GENERATION must be 0 or 1"
        return 2
    fi
    if ! [[ "${N_GROUPS:-0}" =~ ^[0-9]+$ ]]; then
        echo "N_GROUPS=$N_GROUPS is not a whole number"
        return 2
    fi
    if [ "${N_GROUPS:-0}" -gt 0 ] && [ "${ITERATIVE:-1}" = "1" ]; then
        echo "committee mode (N_GROUPS=$N_GROUPS) needs ITERATIVE=0"
        return 2
    fi
    command -v lake > /dev/null 2>&1 || { echo "lake not on PATH; check ELAN_HOME=$ELAN_HOME"; return 1; }
    return 0
}

READY=()
WHY=()
for ((i = 0; i < N; i++)); do
    wt="$(slot_wt "${SLOT[i]}")"
    if [[ "${SLOT[i]}" == 1 ]]; then
        msg=""
    else
        msg="$(ready_slot "$wt" 2>&1)" || {
            sed 's/^/   /' <<< "$msg"
            READY+=(0)
            WHY+=("slot ${SLOT[i]} not ready: $(tail -n 1 <<< "$msg")")
            echo "FATAL: ${TAG[i]}: ${WHY[i]}; this run is left out"
            continue
        }
        echo "$msg"
    fi
    mapfile -t kv < <(run_vars "$i")
    msg="$(cd "$wt" && (
        export REPO="$wt" RUN_TAG="${TAG[i]}" ${kv[@]+"${kv[@]}"}
        check_run
    ) 2>&1)" || {
        READY+=(0)
        WHY+=("$msg")
        echo "FATAL: ${TAG[i]}: $msg; this run is left out"
        continue
    }
    READY+=(1)
    WHY+=("")
    echo "run ${TAG[i]}: slot ${SLOT[i]} ($(basename "$wt")), ${kv[*]:-no knobs of its own}"
done
cd "$REPO" || fatal "no worktree at $REPO"
n_ready=0
for r in "${READY[@]}"; do ((r)) && n_ready=$((n_ready + 1)); done
if ((n_ready == 0)); then
    for ((i = 0; i < N; i++)); do echo "PACK ${TAG[i]}: slot ${SLOT[i]}, NOT STARTED: ${WHY[i]}"; done
    fatal "no run of this pack can start; the model was not loaded"
fi

# --- 4. the server: lms_run.sbatch's preflight and start, on this job's one GPU ------------------
[ -x "$SCRATCH/vllm-env/bin/vllm" ] || fatal "no vLLM venv at $SCRATCH/vllm-env"
command -v setsid > /dev/null 2>&1 || fatal "setsid not on PATH; it stops a hung server's workers"
busy_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F', ' -v vis="${CUDA_VISIBLE_DEVICES:-}" '
        BEGIN { n = split(vis, ids, ","); for (i = 1; i <= n; i++) mine[ids[i]] = 1 }
        (n == 0 || ($1 in mine)) && $2 > 1024 { printf "%s ", $1 }'
}
echo "--- GPU state (allocated: ${CUDA_VISIBLE_DEVICES:-all}) ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
BUSY=$(busy_gpus)
[ -z "$BUSY" ] || fatal "GPU(s) $BUSY already hold >1 GiB; a vLLM started outside Slurm would collide. Free them, then resubmit"
if curl -sS --max-time 3 "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    fatal "something already answers on port $PORT. Stop it or set PORT=..."
fi
GPUS_FILE="$PACK_DIR/gpus.txt"
{
    echo "--- GPUs this job holds, and how they are linked ---"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        GRES_IDX=$(scontrol show job -d "$SLURM_JOB_ID" 2> /dev/null | grep -o 'GRES=[^ ]*IDX:[^ ]*' | sort -u) || true
        echo "${GRES_IDX:-scontrol gave no GRES IDX for job $SLURM_JOB_ID}"
    fi
    nvidia-smi topo -m 2>&1 || true
} > "$GPUS_FILE"
cat "$GPUS_FILE"
if [ -z "${SLURM_JOB_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

VLLM_PID=""
VLLM_LOG="$PACK_DIR/vllm.log"
rm -f "$PACK_DIR"/vllm.attempt-*.log
RUN_PIDS=()

start_server() {
    ATTEMPT_LOG="$PACK_DIR/vllm.attempt-$1.log"
    : > "$ATTEMPT_LOG"
    ln -f "$ATTEMPT_LOG" "$VLLM_LOG" 2> /dev/null || ln -sf "$(basename "$ATTEMPT_LOG")" "$VLLM_LOG"
    echo "--- starting vLLM (try $1 of $START_TRIES) on GPU ${CUDA_VISIBLE_DEVICES:-all} (log: $ATTEMPT_LOG) ---"
    # shellcheck disable=SC2086  # LMS_EXTRA_VLLM_ARGS splits on spaces, as in lms_run.sbatch
    setsid "$SCRATCH/vllm-env/bin/vllm" serve "$MODEL_REPO" \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --served-model-name "$SERVED_NAME" \
        --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
        $LMS_EXTRA_VLLM_ARGS \
        > "$ATTEMPT_LOG" 2>&1 &
    VLLM_PID=$!
}
stop_group() {
    local pid=$1 t0=$SECONDS
    kill -TERM -- "-$pid" 2> /dev/null || kill -TERM "$pid" 2> /dev/null
    while kill -0 -- "-$pid" 2> /dev/null && [ $((SECONDS - t0)) -lt "$STOP_GRACE" ]; do sleep 1; done
    kill -KILL -- "-$pid" 2> /dev/null
    wait "$pid" 2> /dev/null
    t0=$SECONDS
    while kill -0 -- "-$pid" 2> /dev/null && [ $((SECONDS - t0)) -lt 10 ]; do sleep 1; done
}
# On any exit, every run still going and the server are stopped together (each is its own
# process group): TERM to all, KILL to what is left after STOP_GRACE seconds, so a cancelled or
# timed-out pack is done well inside Slurm's own grace period, and a failed job never strands the
# GPU until its walltime.
cleanup() {
    local pid groups=() t0=$SECONDS left
    for pid in ${RUN_PIDS[@]+"${RUN_PIDS[@]}"} ${VLLM_PID:+"$VLLM_PID"}; do
        kill -0 -- "-$pid" 2> /dev/null && groups+=("$pid")
    done
    RUN_PIDS=()
    VLLM_PID=""
    ((${#groups[@]})) || return 0
    echo "--- stopping process group(s) ${groups[*]} (runs still going, then the server) ---"
    for pid in "${groups[@]}"; do kill -TERM -- "-$pid" 2> /dev/null; done
    while [ $((SECONDS - t0)) -lt "$STOP_GRACE" ]; do
        left=0
        for pid in "${groups[@]}"; do kill -0 -- "-$pid" 2> /dev/null && left=1; done
        ((left)) || break
        sleep 1
    done
    for pid in "${groups[@]}"; do kill -KILL -- "-$pid" 2> /dev/null; done
    for pid in "${groups[@]}"; do wait "$pid" 2> /dev/null; done
}
trap cleanup EXIT
trap 'echo "NOTE: signal received; stopping the runs and the server"; exit 143' INT TERM

served=""
for ((try = 1; try <= START_TRIES; try++)); do
    if ((try > 1)); then
        t0=$SECONDS
        while [ -n "$(busy_gpus)" ] && [ $((SECONDS - t0)) -lt "$GPU_FREE_TIMEOUT" ]; do sleep 5; done
        [ -z "$(busy_gpus)" ] || fatal "GPU(s) $(busy_gpus) still hold >1 GiB ${GPU_FREE_TIMEOUT}s after the hung start was stopped"
    fi
    start_server "$try"
    echo "--- waiting for vLLM on port $PORT (up to ${START_TIMEOUT}s) ---"
    ATTEMPT_START=$SECONDS
    while [ $((SECONDS - ATTEMPT_START)) -lt "$START_TIMEOUT" ]; do
        kill -0 "$VLLM_PID" 2> /dev/null || break
        if curl -sS --max-time 5 "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            served=1
            break
        fi
        sleep 5
    done
    [[ -n "$served" ]] && break
    if ! kill -0 "$VLLM_PID" 2> /dev/null; then
        echo "FATAL: vLLM died during startup (try $try). Tail of $ATTEMPT_LOG:"
        tail -40 "$ATTEMPT_LOG"
        VLLM_PID=""
        exit 1
    fi
    echo "WARN: vLLM (try $try) is alive but not answering after ${START_TIMEOUT}s. Its last lines:"
    tail -3 "$ATTEMPT_LOG"
    stop_group "$VLLM_PID"
    VLLM_PID=""
done
[[ -n "$served" ]] || fatal "vLLM not ready after ${START_TIMEOUT}s on each of $START_TRIES tries"
READY_AFTER=$((SECONDS - ATTEMPT_START))
echo "SERVED_BY=base after ~${READY_AFTER}s (try $try)"
grep -m1 "GPU KV cache size" "$VLLM_LOG" || true

# --- 5. the runs, all at once, each in its slot ----------------------------------------------------
declare -A LOG_OF=()
for ((i = 0; i < N; i++)); do
    ((READY[i])) || continue
    wt="$(slot_wt "${SLOT[i]}")"
    mkdir -p "$wt/logs"
    log="$wt/logs/$JOB_NAME-$JOB_ID.${TAG[i]}.out"
    LOG_OF[$i]="$log"
    mapfile -t kv < <(run_vars "$i")
    (
        cd "$wt" || exit 1
        export REPO="$wt" RUN_TAG="${TAG[i]}" LMS_Q_PACK_ONE=1 LMS_Q_PACK_NAME="$JOB_NAME" LMS_Q_PACK_SLOT="${SLOT[i]}" \
            LMS_Q_PACK_TAGS="$PEERS" LMS_Q_PACK_GPUS="$GPUS_FILE" LMS_Q_PACK_VLLM_LOG="$VLLM_LOG" \
            LMS_Q_PACK_SERVED_BY=base LMS_Q_PACK_READY_AFTER="$READY_AFTER" PORT SERVED_NAME ${kv[@]+"${kv[@]}"}
        exec setsid bash "$SELF"
    ) > "$log" 2>&1 < /dev/null &
    RUN_PIDS[i]=$!
    START_OF[i]=$SECONDS
    echo "started ${TAG[i]} in slot ${SLOT[i]}: pid ${RUN_PIDS[i]}, log $log"
done

# --- 6. wait for every run ------------------------------------------------------------------------
RC=()
bad=0
for ((i = 0; i < N; i++)); do
    if ! ((READY[i])); then
        bad=$((bad + 1))
        continue
    fi
    wait "${RUN_PIDS[i]}"
    RC[i]=$?
    ((RC[i] == 0)) || bad=$((bad + 1))
    echo "--- ${TAG[i]} ended: exit ${RC[i]} after $((SECONDS - START_OF[i]))s"
    unset 'RUN_PIDS[i]'
done
echo "=== pack $JOB_NAME done"
for ((i = 0; i < N; i++)); do
    if ((READY[i])); then
        line="$(grep -E -- '^--- experiment exited|^FATAL' "${LOG_OF[$i]}" 2> /dev/null | tail -n 1)"
        echo "PACK ${TAG[i]}: slot ${SLOT[i]}, exit ${RC[i]}${line:+ ($line)}"
    else
        echo "PACK ${TAG[i]}: slot ${SLOT[i]}, NOT STARTED: ${WHY[i]}"
    fi
done
((bad == 0))
