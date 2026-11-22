#!/bin/bash
# Tests the kit's packs (q_slots, q_submit_pack, lms_q_pack.sh, and the collect helpers that read a
# pack's runs) on a fake server whose commits hold this checkout's real scripts/slurm/lms_run.sbatch,
# so a pack's runs can be compared with that script's, call for call.
#   bash test_pack.sh
# The checks use "cond && ok || bad", which is safe here: ok never fails.
# shellcheck disable=SC2015,SC2016
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=fakebox.sh
source "$HERE/fakebox.sh"

seed_run_script() {
    cp "$ROOT/scripts/slurm/lms_run.sbatch" "$1/scripts/slurm/lms_run.sbatch"
    echo '# RUN SCRIPT main' >> "$1/scripts/slurm/lms_run.sbatch"
}
FB_SEED_HOOK=seed_run_script

stubs() {
    mkdir -p "$D/calls"
    cat > "$D/bin/curl" << STUB
#!/bin/bash
[[ " \$* " == *" --max-time 3 "* ]] && exit 7
for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -e "$D/vllm.up" ]] && exit 0; sleep 0.1; done
exit 7
STUB
    cat > "$D/scratch/vllm-env/bin/vllm" << STUB
#!/bin/bash
echo "start \$*" >> "$D/vllm.calls"
echo "INFO GPU KV cache size: 850,000 tokens"
[[ -e "$D/vllm.dies" ]] && exit 3
touch "$D/vllm.up"
exec -a "$D/scratch/fake-vllm" sleep 60
STUB
    cat > "$D/bin/uv" << STUB
#!/bin/bash
[[ "\$1" == sync ]] && { mkdir -p .venv; exit 0; }
args=("\$@")
while ((\${#args[@]})) && [[ "\${args[0]}" != python ]]; do args=("\${args[@]:1}"); done
args=("\${args[@]:1}")
if [[ "\${args[0]:-}" == -m && "\${args[1]:-}" == lms.run ]]; then
    out=""
    for ((i = 0; i < \${#args[@]}; i++)); do [[ "\${args[i]}" == --output ]] && out="\${args[i + 1]}"; done
    tag="\$(basename "\$out")"
    { echo "ARGV \${args[*]}"; echo "PWD \$PWD"; env | grep -E '^(LMS_|OPENAI_|PYTHONUNBUFFERED|VLLM_|AGENTS=)' | grep -v '^LMS_Q_' | sort; } > "$D/calls/\$tag"
    mkdir -p "\$out"
    echo '{"artifacts": []}' > "\$out/artifacts.json"
    echo '{"generations": []}' > "\$out/results.json"
    echo "-- written by \$tag" >> lean/LMS/Foundation.lean
    sleep "\${FAKE_RUN_SECS:-0.3}"
    exit "\${FAKE_RUN_RC:-0}"
fi
exec python3 "\${args[@]}"
STUB
    chmod +x "$D/bin/"* "$D/scratch/vllm-env/bin/vllm"
}

cat > "$FB_ROOT/lms_pk_go.sh" << 'GO'
#!/bin/bash
set -euo pipefail
source "$HOME/lms_q_lib.sh"
q_init pk
q_require_box gpu
q_round_guard
q_pin "${1:?usage}"
q_lanes "${LANES:-2}"
q_slots "${LANES:-2}" "${SLOTS:-3}"
q_tag_free pk_a pk_b pk_c
mkdir -p "$HOME/lms_pk.packs"
f="$HOME/lms_pk.packs/one.tsv"
if [[ -n "${PACK_TEXT:-}" ]]; then
    printf '%b' "$PACK_TEXT" > "$f"
else
    printf '1\tpk_a\tAGENTS=3\tLMS_SEED=1\n2\tpk_b\tAGENTS=3\tRESET_EACH_GENERATION=1\tLMS_SEED=1\n3\tpk_c\tAGENTS=6\tMAX_TOKENS=500\tLMS_SEED=2\n' > "$f"
fi
q_lane "${LANE:-1}"
q_submit_pack one "$f" "${Q_GPU_PACK[@]}" --time=02:00:00 -- N_GROUPS=3 ITERATIVE=0 GENERATIONS=2 MAX_MODEL_LEN=1000 "LMS_EXTRA_VLLM_ARGS=--x --y"
q_show_queue
GO
cat > "$FB_ROOT/lms_pk_collect.sh" << 'COLLECT'
#!/bin/bash
set -uo pipefail
source "$HOME/lms_q_lib.sh"
q_init pk
q_collect_begin
while read -r line; do
    echo "WT $(q_field "$line" tag) $(q_job_wt "$line")"
    echo "LOG $(q_field "$line" tag) $(q_job_log "$line")"
    echo "PACKLOG $(q_field "$line" tag) $(q_pack_log "$line")"
    q_copy_run "$line"
done < <(q_jobs)
COLLECT

box() {
    fb_new "$1"
    fb_install "$FB_ROOT/lms_pk_go.sh" "$FB_ROOT/lms_pk_collect.sh"
    stubs
    PIN="$(cat "$D/sha_main")"
    S2="$H/code/lms-box-s2"
    S3="$H/code/lms-box-s3"
}
go() { (cd "$H" && fb_env "$@" bash "$H/lms_pk_go.sh" "$PIN" > "$D/out" 2>&1); echo $? > "$D/rc"; }
job_rc_is() { if [[ "$(cat "$D/job.$1.rc")" == "$2" ]]; then ok "job $1 exit $2"; else bad "job $1 exit $(cat "$D/job.$1.rc"), wanted $2"; tail -20 "$J" | sed 's/^/      /'; fi; }
runlog() { echo "$1/logs/lms-pk-one-$2.$3.out"; }

echo "== 1. q_slots makes slots 2 and 3 in each lane, each with its own Lean tree and .venv; again changes nothing"
box slots
go LANES=2
rc_is 0
has "$D/out" "== lane 1 slot 2: made $S2, Lean tree $D/scratch/lms-lanes/l1s2/lake, library linked from lane 1"
has "$D/out" "== lane 2 slot 3: made $H/code/lms-box-l2-s3"
has "$D/out" "== slots 1 to 3 of lanes 1 to 2 are ready"
for wt in "$S2" "$S3" "$H/code/lms-box-l2-s2" "$H/code/lms-box-l2-s3"; do
    [[ -L "$wt/lean/.lake" && -d "$wt/.venv" && "$(readlink -f "$wt/lean/.lake/packages")" == "$(readlink -f "$D/lake/packages")" ]] || bad "slot $wt"
done
ok "four slots, each linking lane 1's library (a FAIL above names any that does not)"
[[ "$(readlink -f "$S2/lean/.lake")" != "$(readlink -f "$S3/lean/.lake")" ]] && ok "slots have their own Lean trees" || bad "two slots share a Lean tree"
mv "$H/lms_pk.jobs" "$H/lms_pk.jobs.first"
go LANES=2
rc_is 0
hasnt "$D/out" "made "
has "$D/out" "== slots 1 to 3 of lanes 1 to 2 are ready"
mv "$H/lms_pk.jobs" "$H/lms_pk.jobs.second"
go LANES=2 SLOTS=4
rc_is 1
has "$D/out" "STOP: q_slots: there are 1 to 3 slots in a lane"

echo "== 2. q_submit_pack queues one job, behind the lane's last, and writes a line per run with its slot"
box submit
go
rc_is 0
count_is "$D/slurm/sbatch.log" "--job-name=" 1
has "$D/slurm/sbatch.log" "--gres=gpu:nvidia_h100_nvl:1 --mem=160G --cpus-per-task=24"
has "$D/out" "Queued lms-pk-one: job 701, lane 1, pin ${PIN:0:8}: pk_a (slot 1), pk_b (slot 2), pk_c (slot 3)"
has "$H/lms_pk.jobs" "job=701 name=lms-pk-one tag=pk_b pin=${PIN:0:8} after=- lane=1 slot=2"
count_is "$H/lms_pk.jobs" "job=701 " 3
has "$D/slurm/env.701" "declare -x LMS_Q_RUN=\"$H/lms_q_pack.sh\""
has "$D/slurm/env.701" "declare -x LMS_Q_PACK=\"$H/lms_pk.packs/one.tsv\""
has "$D/slurm/env.701" 'declare -x RUN_TAG="pk_one"'
(cd "$H" && fb_env bash "$H/lms_pk_go.sh" "$PIN" > "$D/out" 2>&1); has "$D/out" "STOP: this round was already queued (1 jobs in lms_pk.jobs)"
box submit-bad
go "PACK_TEXT=1\tpk_a\n1\tpk_b\n"; rc_is 1; has "$D/out" "STOP: pack one: slot 1 holds two runs"
box submit-bad2
go "PACK_TEXT=1\tpk_a\tlower=1\n"; rc_is 1; has "$D/out" "STOP: pack one: 'lower=1' is not VAR=VALUE"
box submit-bad3
go SLOTS=2; rc_is 1; has "$D/out" "STOP: pack one: lane 1 has no slot 3; q_slots makes it"
[[ ! -s "$D/slurm/sbatch.log" ]] && ok "a refused pack queues nothing" || bad "a refused pack queued something"
box submit-fail
go FAKE_SBATCH_FAIL_AT=1; rc_is 1
has "$D/out" "STOP: sbatch refused lms-pk-one; nothing is queued for this round"
[[ ! -e "$H/lms_pk.jobs" ]] && ok "no jobs file left" || bad "jobs file left after a refused sbatch"

echo "== 3. the pack job: one server, three runs at once, each in its slot, as lms_run.sbatch runs them"
box run
go
rc_is 0
fb_run_job 701
job_rc_is 701 0
count_is "$D/vllm.calls" "start serve" 1
has "$D/vllm.calls" "start serve Qwen/Qwen3.6-27B-FP8 --port 8051 --tensor-parallel-size 1 --max-model-len 1000 --served-model-name lms-generalist --gdn-prefill-backend triton --x --y"
has "$J" "=== pack lms-pk-one: job 701, pin ${PIN:0:8}, 3 run(s): pk_a pk_b pk_c"
has "$J" "slot lms-box-s2: at ${PIN:0:8}, the shared library is up to date"
has "$J" "PACK pk_a: slot 1, exit 0 (--- experiment exited rc=0"
has "$J" "PACK pk_c: slot 3, exit 0"
has "$D/calls/pk_a" "PWD $W"; has "$D/calls/pk_b" "PWD $S2"; has "$D/calls/pk_c" "PWD $S3"
has "$D/calls/pk_b" "--reset-each-generation"
hasnt "$D/calls/pk_a" "--reset-each-generation"
has "$D/calls/pk_c" "--agents 6 --generations 2 --goal stacks-ch4-phase1 --groups --n-groups 3 --repair-attempts 2 --max-tokens 500"
has "$D/calls/pk_a" "LMS_OPENAI_BASE_URL=http://localhost:8051/v1"
has "$(runlog "$S2" 701 pk_b)" "pack=lms-pk-one slot=2: one vLLM server on port 8051 for pk_a pk_b pk_c"
has "$(runlog "$S2" 701 pk_b)" "lean/ clean after restore"
for wt in "$W" "$S2" "$S3"; do git -C "$wt" diff --quiet -- lean/LMS && ok "foundation restored in $(basename "$wt")" || bad "foundation left modified in $(basename "$wt")"; done
pgrep -f "$D/scratch/fake-vllm" > /dev/null && bad "the server outlived its pack" || ok "the server is gone"

echo "== 4. each run's lms.run call is lms_run.sbatch's, for the same knobs"
RUN_D="$D"
box equiv
go
mkdir -p "$D/sb"
for t in a b c; do
    case $t in a) wt="$W" ;; b) wt="$S2" ;; c) wt="$S3" ;; esac
    kv=()
    mapfile -t kv < <(grep -P "\tpk_$t(\t|\$)" "$H/lms_pk.packs/one.tsv" | cut -f3- | tr '\t' '\n' | sed '/^$/d')
    (
        # shellcheck disable=SC1091
        source "$D/slurm/env.701"
        cd "$wt" || exit 1
        export REPO="$wt" RUN_TAG="pk_$t" "${kv[@]}"
        SLURM_JOB_ID=999 bash "$wt/scripts/slurm/lms_run.sbatch"
    ) > "$D/sb/$t.out" 2>&1
    a_="$(sed "s#$D#/FAKE#g" "$D/calls/pk_$t" 2> /dev/null)"
    b_="$(sed "s#$RUN_D#/FAKE#g" "$RUN_D/calls/pk_$t")"
    if [[ -n "$a_" && "$a_" == "$b_" ]]; then ok "pk_$t: same lms.run call, directory and environment as lms_run.sbatch"; else bad "pk_$t differs:"; diff <(echo "$a_") <(echo "$b_") | head -8 | sed 's/^/      /'; tail -5 "$D/sb/$t.out" | sed 's/^/      /'; fi
done

echo "== 5. a run that cannot start is left out; the others run; a slot's left-over foundation is put back"
box leftout
go
mkdir -p "$S2/experiments/pk_b" && echo '{}' > "$S2/experiments/pk_b/results.json"
echo "-- left by a killed job" >> "$S3/lean/LMS/Foundation.lean"
fb_run_job 701
job_rc_is 701 1
has "$J" "FATAL: pk_b: experiments/pk_b already holds a run; pass RESUME=1 or a new RUN_TAG; this run is left out"
has "$J" "NOTE: lean/LMS/Foundation.lean in lms-box-s3 was left modified by an earlier job; putting the tracked version back"
has "$J" "PACK pk_b: slot 2, NOT STARTED: experiments/pk_b already holds a run"
has "$J" "PACK pk_a: slot 1, exit 0"
[[ ! -e "$D/calls/pk_b" ]] && ok "the refused run never ran" || bad "the refused run ran"
box stale
go
printf '#!/bin/bash\necho "lake $* in $PWD"\n[[ "$PWD" != %s/lean ]]\n' "$S3" > "$D/bin/lake"
fb_run_job 701
job_rc_is 701 1
has "$J" "FATAL: pk_c: slot 3 not ready: Lake would rebuild the library this slot shares with lane 1; this run is left out"
has "$J" "PACK pk_a: slot 1, exit 0"

echo "== 6. refusals before the model loads: a pack line that sets the server's port, a pack with nothing to run, a server that dies"
box port
go "PACK_TEXT=1\tpk_a\tPORT=9999\n"
fb_run_job 701; job_rc_is 701 1
has "$J" "FATAL: pack line for pk_a sets PORT, which is the pack's to set"
[[ ! -e "$D/vllm.calls" ]] && ok "no server started" || bad "a server started"
box nothing
go "PACK_TEXT=2\tpk_b\tRESUME=1\n"
fb_run_job 701; job_rc_is 701 1
has "$J" "FATAL: pk_b: RESUME=1 but experiments/pk_b has no results.json; this run is left out"
has "$J" "FATAL: no run of this pack can start; the model was not loaded"
[[ ! -e "$D/vllm.calls" ]] && ok "no server started" || bad "a server started"
box dies
go
touch "$D/vllm.dies"
fb_run_job 701; job_rc_is 701 1
has "$J" "FATAL: vLLM died during startup (try 1)"
[[ ! -d "$D/calls" || -z "$(ls "$D/calls")" ]] && ok "no run started" || bad "a run started without a server"

echo "== 7. TERM stops every run and the server together"
box term
go
(
    # shellcheck disable=SC1090
    source "$D/slurm/env.701"
    cd "$W" || exit 97
    FAKE_RUN_SECS=60 STOP_GRACE=3 SLURM_JOB_ID=701 SLURM_JOB_NAME=lms-pk-one exec bash "$H/lms_q_job.sbatch"
) > "$D/term.out" 2>&1 &
pid=$!
for _ in $(seq 1 100); do [[ "$(ls "$D/calls" 2> /dev/null | wc -l)" == 3 ]] && break; sleep 0.1; done
[[ "$(ls "$D/calls" | wc -l)" == 3 ]] && ok "three runs going" || bad "the runs did not start"
kill -TERM "$pid"
wait "$pid"
rc=$?
[[ "$rc" == 143 ]] && ok "the pack exits 143 on TERM" || bad "pack exit $rc on TERM"
has "$D/term.out" "NOTE: signal received; stopping the runs and the server"
sleep 0.5
pgrep -f "$D/" > /dev/null && { bad "something of the pack is still running"; pgrep -af "$D/" | sed 's/^/      /'; } || ok "every run and the server are gone"

echo "== 8. collect reads each run of a pack from its slot"
box collect
go
fb_run_job 701
fb_finish 701
(cd "$H" && fb_env bash "$H/lms_pk_collect.sh" > "$D/out" 2>&1)
has "$D/out" "WT pk_b $S2"
has "$D/out" "LOG pk_c $S3/logs/lms-pk-one-701.pk_c.out"
has "$D/out" "PACKLOG pk_c $W/logs/lms-pk-one-701.out"
has "$D/out" "COPIED pk_b: COMPLETED, 2 of 9 run files"
[[ -f "$H/lms_pk/pk_b/logs/lms-pk-one-701.out" && -f "$H/lms_pk/pk_b/logs/pack-lms-pk-one-701.log" ]] && ok "a run's own log and the pack's log are copied" || bad "logs not copied: $(ls "$H/lms_pk/pk_b/logs")"
has "$H/lms_pk/pk_b/logs/lms-pk-one-701.out" "=== LMS run pk_b ==="
mv "$H/lms_pk.jobs" "$H/lms_pk.jobs.done"
rm -r "$W/experiments/pk_a"
go; rc_is 1; has "$D/out" "STOP: experiments/pk_b already exists in lane 1 slot 2 and holds a run"

fb_done
