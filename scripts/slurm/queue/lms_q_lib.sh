#!/bin/bash
# LMS queue kit: what every queued round's go and collect scripts share. Source it, don't run it:
#   source "$HOME/lms_q_lib.sh"; q_init <round>
#
# A run rewrites its worktree's foundation and builds in its Lean tree, so no two LMS jobs may run
# in one worktree at once. Jobs therefore run in lanes: lane 1 is ~/code/lms-box, and lanes 2 to 4
# are ~/code/lms-box-l2 to -l4, each with its own Lean tree that links to lane 1's built library.
# Three rules keep one job per lane however many rounds are queued:
#   1. every job is submitted through q_submit, which makes it wait for the last LMS job queued in
#      its lane (sbatch --dependency=afterany), so each lane's jobs form one line;
#   2. every job runs ~/lms_q_job.sbatch, which takes its lane's lock and checks out that job's own
#      commit when it starts;
#   3. a job asks for one GPU (Q_GPU) or all four (Q_GPU_ALL), never two: a two-GPU job tends to
#      leave the other two idle, and a pair without an NVLink bridge hangs vLLM's start.
# So rounds pinned to different commits can sit in the queue together, and up to four lanes run
# side by side on one GPU each.
#
# go side:      q_init, q_require_box, q_require_files, q_round_guard, q_pin, q_pin_has, q_lanes,
#               q_lane, q_tag_free, q_submit, q_submit_free, q_show_queue
# collect side: q_init, q_collect_begin, q_jobs, q_field, q_job_wt, q_state, q_scrub, q_section,
#               q_key_lines, q_server_load, q_copy_run
# shellcheck shell=bash

# q_stop MESSAGE: print it and exit 1. If this go has already queued jobs, they are cancelled
# first, so a round is queued whole or not at all whatever stops it.
q_stop() {
    [[ -z "${Q_ROUND:-}" ]] || q_undo
    echo "STOP: $*"
    exit 1
}

# q_init ROUND: names and paths. Never touches the server.
q_init() {
    Q_ROUND="${1:?q_init needs a round name}"
    ME="${USER:-$(id -un)}"
    MAIN="$HOME/code/lms"
    WT="$HOME/code/lms-box"
    SCRATCH="${SCRATCH:-/scratch/users/$ME}"
    MODEL="Qwen/Qwen3.6-27B-FP8"
    Q_JOBS="$HOME/lms_${Q_ROUND}.jobs"
    Q_OUT="$HOME/lms_${Q_ROUND}"
    Q_KIT_JOB="$HOME/lms_q_job.sbatch"
    # One GPU, served at TP=1 (q_submit passes TP_SIZE=1), so no NCCL start can hang. Q_GPU_ALL is
    # the whole box, for a job that sets TP_SIZE=4 among its variables.
    Q_GPU=(--gres=gpu:nvidia_h100_nvl:1 --mem=96G --cpus-per-task=16)
    Q_GPU_ALL=(--gres=gpu:nvidia_h100_nvl:4 --mem=128G --cpus-per-task=16)
    Q_TP=1
    Q_LANES_MAX=4
    Q_LANE=1
    Q_PIN=""
    Q_TAIL=""
    Q_TAIL_KNOWN=""
    Q_LAST=""
    Q_SUBMITTED=()
    Q_PENDING=()
    export GIT_TERMINAL_PROMPT=0
}

# q_require_box [gpu]: the setup earlier rounds made. With "gpu", also the vLLM environment and weights.
q_require_box() {
    [[ -e "$MAIN/.git" && -e "$WT" ]] || q_stop "no ~/code/lms-box; round R4 sets it up"
    [[ -L "$WT/lean/.lake" && -d "$WT/.venv" ]] || q_stop "lms-box has no library link or .venv; run round R4 first"
    [[ -f "$Q_KIT_JOB" && -f "$HOME/lms_q_lib.sh" ]] || q_stop "no ~/lms_q_job.sbatch; copy the kit files first"
    if [[ "${1:-}" == gpu ]]; then
        [[ -x "$SCRATCH/vllm-env/bin/vllm" && -x "$SCRATCH/vllm-env/bin/python" ]] ||
            q_stop "no vLLM environment; round R5b's setup job makes it"
        [[ -d "$SCRATCH/hf/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots" ]] ||
            q_stop "no weights; round R5b's setup job downloads them"
    fi
}

# q_require_files NAME...: files this round copied to ~/.
q_require_files() {
    local f
    for f in "$@"; do
        [[ -f "$HOME/$f" ]] || q_stop "no ~/$f; copy this round's files first"
    done
}

# q_round_guard: a round queues once. Running its go again changes nothing.
q_round_guard() {
    [[ ! -e "$Q_JOBS" ]] ||
        q_stop "this round was already queued ($(grep -c '^job=' "$Q_JOBS") jobs in lms_${Q_ROUND}.jobs); nothing new is queued"
}

# q_pin SHA: sets Q_PIN to the full commit, or stops unless a branch on origin holds it.
q_pin() {
    local want="${1:?q_pin needs a commit}" got
    git -C "$MAIN" fetch -q origin '+refs/heads/*:refs/remotes/origin/*' || q_stop "could not fetch from origin"
    got="$(git -C "$MAIN" rev-parse --verify -q "$want^{commit}")" || q_stop "$want is not in this clone, so it is not on origin"
    [[ -n "$(git -C "$MAIN" branch -r --contains "$got" 2> /dev/null)" ]] ||
        q_stop "$want is not on any branch of origin"
    Q_PIN="$got"
    echo "== pin ${Q_PIN:0:8} is on origin"
}

# q_pin_has PATH PATTERN WHAT: stop unless the file at the pin matches, so a commit that lacks
# what the round needs stops here and not hours into a job.
# The file is read whole before it is searched. Piping git into `grep -q` fails under pipefail
# whenever grep finds its match early: it exits, and git dies of SIGPIPE on its next write.
q_pin_has() {
    local body
    body="$(git -C "$MAIN" show "$Q_PIN:$1" 2> /dev/null)" || body=""
    grep -qE -- "$2" <<< "$body" || q_stop "${Q_PIN:0:8} lacks $3; nothing is queued"
}

# --- lanes ---------------------------------------------------------------------------------------

# q_lane_wt K / q_lane_lock K: lane K's worktree and lock file. Lane 1 keeps the names it had
# before there were lanes.
q_lane_wt() {
    if [[ "$1" == 1 ]]; then echo "$HOME/code/lms-box"; else echo "$HOME/code/lms-box-l$1"; fi
}
q_lane_lock() {
    if [[ "$1" == 1 ]]; then echo "$HOME/lms_q.lock"; else echo "$HOME/lms_q-l$1.lock"; fi
}

# q_lanes N: makes lanes 2..N where they are missing, and checks every one. A new lane is a
# detached worktree of ~/code/lms whose lean/.lake is its own directory under $SCRATCH/lms-lanes,
# holding a link to lane 1's packages: each lane builds its own foundation, and all of them read
# the one built library. Running it again changes nothing.
q_lanes() {
    local n="${1:?q_lanes needs a lane count}" k wt lake pkgs lane_lake
    [[ "$n" =~ ^[1-9]$ ]] && ((n <= Q_LANES_MAX)) || q_stop "q_lanes $n: there are 1 to $Q_LANES_MAX lanes"
    lake="$(readlink -f "$(q_lane_wt 1)/lean/.lake")" || q_stop "lane 1 has no Lean tree"
    pkgs="$lake/packages"
    [[ -d "$pkgs" ]] || q_stop "lane 1's Lean tree has no packages at $pkgs; round R4 builds it"
    for ((k = 2; k <= n; k++)); do
        wt="$(q_lane_wt "$k")"
        lane_lake="$SCRATCH/lms-lanes/l$k/lake"
        if [[ ! -e "$wt" ]]; then
            git -C "$MAIN" worktree add -q --detach "$wt" "$(git -C "$MAIN" rev-parse HEAD)" ||
                q_stop "lane $k: could not make the worktree $wt"
            mkdir -p "$lane_lake" "$wt/logs" "$wt/experiments"
            [[ -e "$lane_lake/packages" ]] || ln -s "$pkgs" "$lane_lake/packages"
            ln -s "$lane_lake" "$wt/lean/.lake" || q_stop "lane $k: could not link its Lean tree"
            echo "== lane $k: made $wt, Lean tree $lane_lake, library linked from lane 1"
        fi
        [[ -L "$wt/lean/.lake" ]] || q_stop "lane $k: $wt/lean/.lake is not a link"
        [[ "$(readlink -f "$wt/lean/.lake")" != "$lake" ]] || q_stop "lane $k shares lane 1's Lean tree"
        [[ "$(readlink -f "$wt/lean/.lake/packages")" == "$(readlink -f "$pkgs")" ]] ||
            q_stop "lane $k: its library is not lane 1's ($pkgs)"
        if [[ ! -d "$wt/.venv" ]]; then
            (cd "$wt" && PATH="$HOME/.local/bin:$PATH" UV_FROZEN=1 uv sync -q) || q_stop "lane $k: uv sync failed"
            [[ -d "$wt/.venv" ]] || q_stop "lane $k: uv sync made no .venv"
        fi
    done
    echo "== lanes 1 to $n are ready"
    sinfo -h -N -o 'node %N: %m MB of memory, %c CPUs, %G' 2> /dev/null | sort -u || true
}

# q_lane K: the jobs submitted next go to lane K, each waiting for the last LMS job in that lane.
q_lane() {
    local k="${1:?q_lane needs a lane}" wt
    [[ "$k" =~ ^[1-9]$ ]] && ((k <= Q_LANES_MAX)) || q_stop "lane $k: there are 1 to $Q_LANES_MAX lanes"
    wt="$(q_lane_wt "$k")"
    [[ -L "$wt/lean/.lake" && -d "$wt/.venv" ]] || q_stop "lane $k has no worktree at $wt; q_lanes makes it"
    Q_LANE="$k"
    WT="$wt"
    Q_TAIL=""
    Q_TAIL_KNOWN=""
}

# q_tag_free TAG...: a run tag that already holds a run, in any lane, is never overwritten.
q_tag_free() {
    local tag k
    for tag in "$@"; do
        for ((k = 1; k <= Q_LANES_MAX; k++)); do
            [[ ! -e "$(q_lane_wt "$k")/experiments/$tag/results.json" ]] ||
                q_stop "experiments/$tag already exists in lane $k and holds a run; tell me before replacing it"
        done
    done
}

# The newest LMS job still queued or running in this lane, or nothing. New jobs wait for it. A
# job's lane is the directory it runs in, so jobs queued before lanes existed belong to lane 1.
q_tail() {
    local queue
    queue="$(squeue -h -u "$ME" -o '%i %j %T %Z')" || return 1
    awk -v wt="$WT" '$2 ~ /^lms-/ && $4 == wt { split($1, a, "_"); if (a[1] + 0 > m) m = a[1] + 0 } END { if (m) print m }' <<< "$queue"
}

# Cancels what this go queued so far and takes those jobs' lines out of the jobs file. Lines an
# earlier go wrote (a round's first jobs, when this go is a resume) stay. A jobs file left with no
# jobs is moved aside, so the round can be queued again. Safe to call twice.
q_undo() {
    ((${#Q_SUBMITTED[@]})) || return 0
    local id kept=""
    scancel "${Q_SUBMITTED[@]}" 2> /dev/null || true
    if [[ -f "$Q_JOBS" ]]; then
        while IFS= read -r line; do
            id="$(sed -n 's/^job=\([0-9]*\) .*/\1/p' <<< "$line")"
            [[ -n "$id" && " ${Q_SUBMITTED[*]} " == *" $id "* ]] || kept+="$line"$'\n'
        done < "$Q_JOBS"
        if [[ "$kept" == *job=* ]]; then
            cp -p "$Q_JOBS" "$Q_JOBS.failed-$(date +%Y%m%dT%H%M%S)"
            printf '%s' "$kept" > "$Q_JOBS"
        else
            mv "$Q_JOBS" "$Q_JOBS.failed-$(date +%Y%m%dT%H%M%S)"
        fi
    fi
    echo "cancelled the ${#Q_SUBMITTED[@]} job(s) this round had queued: ${Q_SUBMITTED[*]}"
    Q_SUBMITTED=()
}

q_record() {
    printf 'job=%s name=%s tag=%s pin=%s after=%s lane=%s\n' "$1" "$2" "$3" "$4" "$5" "${6:--}" >> "$Q_JOBS"
    Q_SUBMITTED+=("$1")
    Q_LAST="$1"
}

# q_submit NAME TAG [sbatch options...] -- [VAR=VALUE...]
# Queues ~/lms_q_job.sbatch at the current pin, behind the last LMS job in the current lane (q_lane;
# lane 1 unless set). The job is named lms-<round>-<NAME>, runs in its lane's worktree, and by
# default runs scripts/slurm/lms_run.sbatch from the pinned commit (set LMS_Q_RUN=<path> among the
# variables for another script). VAR=VALUE pairs become the job's environment; values may hold
# spaces and commas. TP_SIZE is 1 unless the variables set it.
q_submit() {
    local name="${1:?}" tag="${2:?}" opts=() vars=() dep=() id jobname
    shift 2
    while (($#)) && [[ "$1" != -- ]]; do
        opts+=("$1")
        shift
    done
    (($#)) && shift
    vars=("$@")
    [[ -n "$Q_PIN" ]] || q_stop "q_submit before q_pin"
    if [[ -z "$Q_TAIL_KNOWN" ]]; then
        Q_TAIL="$(q_tail)" || q_stop "squeue failed; nothing is queued"
        Q_TAIL_KNOWN=1
    fi
    [[ -n "$Q_TAIL" ]] && dep=(--dependency="afterany:$Q_TAIL")
    # Lanes run side by side on one node: each serves on its own port, and lanes 2 to 4 compile
    # into their own vLLM cache, so two servers starting at once never write the same files.
    local lane_env=()
    [[ "$Q_LANE" == 1 ]] || lane_env=("VLLM_CACHE_ROOT=$SCRATCH/cache/vllm-l$Q_LANE")
    jobname="lms-$Q_ROUND-$name"
    mkdir -p "$WT/logs"
    id="$(env "REPO=$WT" UV_FROZEN=1 "LMS_Q_PIN=$Q_PIN" "LMS_Q_ROUND=$Q_ROUND" "RUN_TAG=$tag" "MODEL_REPO=$MODEL" \
        "LMS_Q_LANE=$Q_LANE" "LMS_Q_LOCK=$(q_lane_lock "$Q_LANE")" "TP_SIZE=$Q_TP" "PORT=$((8050 + Q_LANE))" \
        ${lane_env[@]+"${lane_env[@]}"} ${vars[@]+"${vars[@]}"} \
        sbatch --parsable --export=ALL --job-name="$jobname" --chdir="$WT" \
        --output="$WT/logs/%x-%j.out" --error="$WT/logs/%x-%j.err" \
        ${dep[@]+"${dep[@]}"} ${opts[@]+"${opts[@]}"} "$Q_KIT_JOB")" || {
        q_undo
        q_stop "sbatch refused $jobname; nothing is queued for this round"
    }
    id="${id%%;*}"
    [[ "$id" =~ ^[0-9]+$ ]] || {
        q_undo
        q_stop "sbatch gave no job id for $jobname ('$id'); nothing is queued for this round"
    }
    q_record "$id" "$jobname" "$tag" "${Q_PIN:0:8}" "${Q_TAIL:--}" "$Q_LANE"
    echo "Queued $jobname: job $id, lane $Q_LANE, pin ${Q_PIN:0:8}${Q_TAIL:+, after job $Q_TAIL}"
    Q_TAIL="$id"
}

# q_submit_free NAME SCRIPT [sbatch options...] -- [VAR=VALUE...]
# For a job that touches neither the worktree nor the Lean tree: it waits for nothing and nothing
# waits for it. Named lmsfree-<round>-<NAME>, so the line of lms- jobs does not see it.
q_submit_free() {
    local name="${1:?}" script="${2:?}" opts=() vars=() id jobname
    shift 2
    while (($#)) && [[ "$1" != -- ]]; do
        opts+=("$1")
        shift
    done
    (($#)) && shift
    vars=("$@")
    jobname="lmsfree-$Q_ROUND-$name"
    mkdir -p "$Q_OUT.logs"
    id="$(env "LMS_Q_ROUND=$Q_ROUND" ${vars[@]+"${vars[@]}"} \
        sbatch --parsable --export=ALL --job-name="$jobname" --chdir="$HOME" \
        --output="$Q_OUT.logs/%x-%j.out" --error="$Q_OUT.logs/%x-%j.err" \
        ${opts[@]+"${opts[@]}"} "$script")" || {
        q_undo
        q_stop "sbatch refused $jobname; nothing is queued for this round"
    }
    id="${id%%;*}"
    [[ "$id" =~ ^[0-9]+$ ]] || {
        q_undo
        q_stop "sbatch gave no job id for $jobname ('$id'); nothing is queued for this round"
    }
    q_record "$id" "$jobname" "-" "-" "-"
    echo "Queued $jobname: job $id, waits for no other job"
}

q_show_queue() {
    echo "== your queue"
    squeue -u "$ME" -o '%.8i %.34j %.9T %.10M %.30R'
}

# --- collect side --------------------------------------------------------------------------------

# q_jobs: this round's "job=N name=... tag=... pin=... after=..." lines.
q_jobs() {
    grep '^job=' "$Q_JOBS" 2> /dev/null || true
}

# q_field LINE KEY: one value from a q_jobs line.
q_field() {
    sed -n "s/.*\b$2=\([^ ]*\).*/\1/p" <<< "$1"
}

# q_job_wt LINE: the worktree the job ran in. A line without a lane was written before lanes
# existed, when every job ran in lane 1.
q_job_wt() {
    local lane
    lane="$(q_field "$1" lane)"
    [[ "$lane" =~ ^[1-9]$ ]] || lane=1
    q_lane_wt "$lane"
}

# q_collect_begin [pending-ok]: stops while any of this round's jobs is queued or running, because
# a log copied mid-write looks complete. Then makes a fresh ~/lms_<round>/ (an older one is moved
# aside). With "pending-ok", a job that has not started does not stop the collect, since it has
# written nothing: only a job that is running (or in any state but PENDING) does. The ids of the
# jobs still waiting are left in Q_PENDING, and q_is_pending ID says whether a job is one of them,
# so the collect can report them as not started and read the rest.
q_collect_begin() {
    [[ -f "$Q_JOBS" ]] || q_stop "no lms_${Q_ROUND}.jobs, so this round queued nothing"
    local mode="${1:-}" queue ids id name state left=0
    Q_PENDING=()
    queue="$(squeue -h -u "$ME" -o '%i %j %T')" || q_stop "squeue failed"
    ids="$(q_jobs | sed 's/^job=\([0-9]*\) .*/\1/')"
    while read -r id name state; do
        [[ -n "$id" ]] || continue
        grep -qx -- "${id%%_*}" <<< "$ids" || continue
        if [[ "$mode" == pending-ok && "$state" == PENDING ]]; then
            Q_PENDING+=("${id%%_*}")
        else
            left=$((left + 1))
        fi
    done <<< "$queue"
    if [[ "$mode" == pending-ok ]]; then
        ((left == 0)) || q_stop "$left job(s) of this round still running; a log taken mid-write looks complete"
    else
        ((left == 0)) || q_stop "$left job(s) of this round still queued or running; a log taken mid-write looks complete"
    fi
    [[ -e "$Q_OUT" ]] && mv "$Q_OUT" "$Q_OUT.old-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$Q_OUT"
}

# q_is_pending ID: true if q_collect_begin pending-ok found this job still waiting to start.
q_is_pending() {
    local id
    for id in ${Q_PENDING[@]+"${Q_PENDING[@]}"}; do
        [[ "$id" == "$1" ]] && return 0
    done
    return 1
}

# q_state ID: the job's final state as one word, or "unknown".
q_state() {
    local state
    # A wide column: at the default width a cancelled job reads "CANCELLED+".
    state="$(sacct -n -X -j "$1" -o State%24 2> /dev/null | awk 'NR == 1 { print $1 }')"
    echo "${state:-unknown}"
}

# Paths print as $SCRATCH and $HOME, so nothing copied back names a user.
q_scrub() {
    sed -e "s#$SCRATCH#\$SCRATCH#g" -e "s#$HOME#\$HOME#g"
}

q_section() {
    printf '\n==== %s\n' "$*"
}

# q_key_lines LOG: the lines of a run job's log that say what ran and how it ended.
q_key_lines() {
    if [[ -f "$1" ]]; then
        grep -E '^(=== queued job|agents=|resume=|max_tokens=|commit=|FATAL|WARN|NOTE|vLLM ready|artifacts:|by status:|elapsed_s|lean/ clean|=== done)|GPU KV cache size|--- experiment exited|Token budget' "$1" | cut -c1-220 | q_scrub
    else
        echo "no log $(basename "$1")"
    fi
}

# q_server_load TAG: one line on how busy the model server was during a run, from vLLM's own
# 10-second reports. Requests waiting means the server, not the agents, set the pace. Prints
# nothing for a run that served no model.
q_server_load() {
    local log="$SCRATCH/lms-runs/$1/vllm.log"
    [[ -f "$log" ]] || return 0
    awk -v tag="$1" '
        /Avg generation throughput:/ {
            n++
            gen = run = wait = kv = hit = ""
            for (i = 1; i <= NF; i++) {
                if ($i == "generation" && $(i + 1) == "throughput:") gen = $(i + 2) + 0
                if ($i == "Running:") run = $(i + 1) + 0
                if ($i == "Waiting:") wait = $(i + 1) + 0
                if ($i == "usage:") kv = $(i + 1) + 0
                if ($i == "rate:") hit = $(i + 1) + 0
            }
            if (run > max_run) max_run = run
            if (wait > max_wait) max_wait = wait
            if (wait > 0) waited++
            if (kv > max_kv) max_kv = kv
            if (gen > max_gen) max_gen = gen
            if (gen > 0) { busy++; sum_gen += gen }
            last_hit = hit
        }
        END {
            if (!n) { printf "SERVER %s: the vLLM log has no load reports\n", tag; exit }
            printf "SERVER %s: %d reports, peak %d running and %d waiting (waiting in %d reports), KV cache peak %.1f%%, prefix cache hit %.1f%% at the end, generation mean %.0f peak %.0f tokens/s\n",
                tag, n, max_run, max_wait, waited, max_kv, last_hit, (busy ? sum_gen / busy : 0), max_gen
        }' "$log"
}

Q_RUN_FILES=(results.json artifacts.json attempts.json metadata.json conversations.json goal.json foundation.json textbook.json agent_stats.json)

# q_copy_run LINE: copies one run job's logs and result files into ~/lms_<round>/<tag>/ and prints
# "COPIED <tag>: <state>, N of M run files". Carries on past anything missing.
q_copy_run() {
    local line="$1" id name tag wt run_dir log_dir dest got=0 f
    id="$(q_field "$line" job)"
    name="$(q_field "$line" name)"
    tag="$(q_field "$line" tag)"
    wt="$(q_job_wt "$line")"
    run_dir="$wt/experiments/$tag"
    log_dir="$SCRATCH/lms-runs/$tag"
    dest="$Q_OUT/$tag"
    mkdir -p "$dest/logs" "$dest/run"
    for f in "$wt/logs/$name-$id.out" "$wt/logs/$name-$id.err"; do
        [[ -f "$f" ]] && q_scrub < "$f" > "$dest/logs/$(basename "$f")"
    done
    [[ -f "$log_dir/run.log" ]] && q_scrub < "$log_dir/run.log" > "$dest/logs/run.log"
    [[ -f "$log_dir/vllm.log" ]] && tail -n 400 "$log_dir/vllm.log" | q_scrub > "$dest/logs/vllm.tail.log"
    for f in "${Q_RUN_FILES[@]}"; do
        [[ -f "$run_dir/$f" ]] || continue
        q_scrub < "$run_dir/$f" > "$dest/run/$f"
        got=$((got + 1))
    done
    q_server_load "$tag"
    echo "COPIED $tag: $(q_state "$id"), $got of ${#Q_RUN_FILES[@]} run files"
}
