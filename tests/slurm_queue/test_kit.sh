#!/bin/bash
# Tests the queue kit (lms_q_lib.sh, lms_q_job.sbatch) on a fake server, through a small demo round.
#   bash test_kit.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=fakebox.sh
source "$HERE/fakebox.sh"

# A demo round: two chained jobs at one pin, an optional third at a second pin, an optional free job.
cat > "$FB_ROOT/lms_demo_go.sh" <<'GO'
#!/bin/bash
set -euo pipefail
source "$HOME/lms_q_lib.sh"
q_init demo
q_require_box gpu
q_round_guard
q_pin "${1:?usage}"
[[ -z "${NEED:-}" ]] || q_pin_has feature.txt feature "the feature file"
q_tag_free demo_a demo_b
q_submit a demo_a "${Q_GPU[@]}" --time=01:00:00 -- AGENTS=3 "EXTRA=--x 1,2 --y"
q_submit b demo_b "${Q_GPU[@]}" --time=02:00:00 -- AGENTS=9
if [[ -n "${2:-}" ]]; then
    q_pin "$2"
    q_submit c demo_c --time=00:30:00 --cpus-per-task=4 -- LMS_Q_RUN="${RUN3:-scripts/slurm/lms_run.sbatch}"
fi
[[ -z "${FREE:-}" ]] || q_submit_free probe "$HOME/lms_demo_free.sbatch" --gres=gpu:nvidia_h100_nvl:2 -- PORT=8011
q_show_queue
GO
cat > "$FB_ROOT/lms_demo_collect.sh" <<'COLLECT'
#!/bin/bash
set -uo pipefail
source "$HOME/lms_q_lib.sh"
q_init demo
q_collect_begin
while read -r line; do
    q_section "$(q_field "$line" name)"
    q_key_lines "$WT/logs/$(q_field "$line" name)-$(q_field "$line" job).out"
    q_copy_run "$line"
done < <(q_jobs)
COLLECT
printf '#!/bin/bash\necho "FREE JOB port=$PORT"\n' > "$FB_ROOT/lms_demo_free.sbatch"
install_demo() { fb_install "$FB_ROOT/lms_demo_go.sh" "$FB_ROOT/lms_demo_collect.sh" "$FB_ROOT/lms_demo_free.sbatch"; }
go() { fb_run lms_demo_go.sh "$@"; }
log() { echo "$D/slurm/sbatch.log"; }

echo "== 1. a round queues its jobs in one line, at the pin"
fb_new one; install_demo
go "$(cut -c1-8 "$D/sha_main")"
rc_is 0
has "$D/out" "== pin $(cut -c1-8 "$D/sha_main") is on origin"
has "$D/out" "Queued lms-demo-a: job 701, pin $(cut -c1-8 "$D/sha_main")"
has "$D/out" "Queued lms-demo-b: job 702, pin $(cut -c1-8 "$D/sha_main"), after job 701"
count_is "$(log)" "--job-name=lms-demo-" 2
if grep '^701 ' "$(log)" | grep -q -- '--dependency'; then bad "first job has a dependency with an empty queue"; else ok "first job waits for nothing"; fi
has "$(log)" "--dependency=afterany:701"
has "$(log)" "--gres=gpu:nvidia_h100_nvl:2 --mem=128G --cpus-per-task=16 --time=01:00:00"
has "$(log)" "--chdir=$W --output=$W/logs/%x-%j.out"
has "$D/slurm/env.701" "LMS_Q_PIN=\"$(cat "$D/sha_main")\""
has "$D/slurm/env.701" 'RUN_TAG="demo_a"'
has "$D/slurm/env.701" "REPO=\"$W\""
has "$D/slurm/env.701" 'UV_FROZEN="1"'
has "$D/slurm/env.701" 'MODEL_REPO="Qwen/Qwen3.6-27B-FP8"'
has "$D/slurm/env.701" 'EXTRA="--x 1,2 --y"'
has "$D/slurm/env.702" 'AGENTS="9"'
has "$H/lms_demo.jobs" "job=701 name=lms-demo-a tag=demo_a pin=$(cut -c1-8 "$D/sha_main") after=-"
has "$H/lms_demo.jobs" "job=702 name=lms-demo-b tag=demo_b pin=$(cut -c1-8 "$D/sha_main") after=701"
[[ "$(git -C "$W" rev-parse HEAD)" == "$(cat "$D/sha_main")" ]] && ok "go did not move the worktree" || bad "go moved the worktree"

echo "== 2. a second go queues nothing"
go "$(cat "$D/sha_main")"
rc_is 1
has "$D/out" "STOP: this round was already queued (2 jobs in lms_demo.jobs)"
count_is "$(log)" "--job-name=" 2

echo "== 3. new jobs wait for the newest lms- job already in the queue, and only for lms- jobs"
fb_new tail; install_demo
fb_queue_add 640 lms-r6-run RUNNING; fb_queue_add 655 lms-arms-s1 PENDING; fb_queue_add 660 btm-rebuild PENDING
fb_queue_add 662 lmsfree-thr-ladder PENDING; fb_queue_add 650_3 lms-array-x PENDING
go "$(cat "$D/sha_main")"
rc_is 0
has "$(log)" "--dependency=afterany:655"
has "$D/out" "Queued lms-demo-a: job 701, pin $(cut -c1-8 "$D/sha_main"), after job 655"
has "$(log)" "--dependency=afterany:701"

echo "== 4. a commit that is not on origin is refused"
fb_new pins; install_demo
go "$(cat "$D/sha_local")"; rc_is 1; has "$D/out" "is not on any branch of origin"
go deadbeef; rc_is 1; has "$D/out" "STOP: deadbeef is not in this clone"
[[ ! -s "$(log)" ]] && ok "nothing queued" || bad "something was queued"
[[ ! -e "$H/lms_demo.jobs" ]] && ok "no jobs file" || bad "a jobs file was left"

echo "== 5. a commit on another origin branch is accepted, and a second pin in the same round"
go "$(cat "$D/sha_main")" "$(cat "$D/sha_feat")"
rc_is 0
has "$D/slurm/env.703" "LMS_Q_PIN=\"$(cat "$D/sha_feat")\""
has "$H/lms_demo.jobs" "job=703 name=lms-demo-c tag=demo_c pin=$(cut -c1-8 "$D/sha_feat") after=702"
has "$(log)" "--dependency=afterany:702 --time=00:30:00 --cpus-per-task=4"

echo "== 6. the pin must hold what the round needs"
fb_new needs; install_demo
(export NEED=1; cd "$H" && fb_env NEED=1 bash "$H/lms_demo_go.sh" "$(cat "$D/sha_main")" > "$D/out" 2>&1; echo $? > "$D/rc")
rc_is 1; has "$D/out" "lacks the feature file; nothing is queued"
(cd "$H" && fb_env NEED=1 bash "$H/lms_demo_go.sh" "$(cat "$D/sha_feat")" > "$D/out" 2>&1; echo $? > "$D/rc")
rc_is 0

echo "== 7. a refused sbatch leaves nothing of the round queued, and the round can be queued again"
fb_new refuse; install_demo
FAKE_SBATCH_FAIL_AT=2 go "$(cat "$D/sha_main")"
rc_is 1
has "$D/out" "cancelled the 1 job(s) this round had queued: 701"
has "$D/out" "STOP: sbatch refused lms-demo-b; nothing is queued for this round"
has "$D/slurm/scancel.log" "701"
[[ ! -s "$D/slurm/queue" ]] && ok "queue is empty" || bad "a job is still queued"
[[ ! -e "$H/lms_demo.jobs" ]] && ok "jobs file moved aside" || bad "jobs file still there"
go "$(cat "$D/sha_main")"; rc_is 0; count_is "$H/lms_demo.jobs" "job=" 2

echo "== 8. an existing run is never overwritten; missing setup stops with what to do"
fb_new guard; install_demo
mkdir -p "$W/experiments/demo_b"; echo '{}' > "$W/experiments/demo_b/results.json"
go "$(cat "$D/sha_main")"; rc_is 1; has "$D/out" "STOP: experiments/demo_b already exists and holds a run"
rm -rf "$W/experiments/demo_b"; rm -rf "$D/scratch/hf"
go "$(cat "$D/sha_main")"; rc_is 1; has "$D/out" "STOP: no weights"
mkdir -p "$D/scratch/hf/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots"; rm "$H/lms_q_job.sbatch"
go "$(cat "$D/sha_main")"; rc_is 1; has "$D/out" "STOP: no ~/lms_q_job.sbatch; copy the kit files first"
cp "$FB_KIT/lms_q_job.sbatch" "$H/"
FAKE_SQUEUE_FAIL=1 go "$(cat "$D/sha_main")"; rc_is 1; has "$D/out" "STOP: squeue failed; nothing is queued"
[[ ! -s "$(log)" ]] && ok "nothing queued" || bad "something was queued"

echo "== 9. a free job waits for nothing and is not part of the line"
fb_new free; install_demo
(cd "$H" && fb_env FREE=1 bash "$H/lms_demo_go.sh" "$(cat "$D/sha_main")" > "$D/out" 2>&1; echo $? > "$D/rc")
rc_is 0
has "$D/out" "Queued lmsfree-demo-probe: job 703, waits for no other job"
if grep '^703 ' "$(log)" | grep -q -- '--dependency'; then bad "the free job has a dependency"; else ok "the free job has no dependency"; fi
has "$H/lms_demo.jobs" "job=703 name=lmsfree-demo-probe tag=- pin=- after=-"
fb_run_job 703; has "$J" "FREE JOB port=8011"

echo "== 10. each job checks out its own commit when it starts, and runs that commit's script"
fb_new jobs; install_demo
go "$(cat "$D/sha_feat")" "$(cat "$D/sha_main")"
rc_is 0
fb_run_job 701
has "$J" "=== queued job: round=demo job=701 pin=$(cut -c1-8 "$D/sha_feat") run=lms_run.sbatch tag=demo_a ==="
has "$J" "RUN SCRIPT feat tag=demo_a agents=3 extra=--x 1,2 --y"
[[ "$(cat "$D/job.701.rc")" == 0 ]] && ok "job 701 exit 0" || bad "job 701 exit $(cat "$D/job.701.rc")"
[[ "$(git -C "$W" rev-parse HEAD)" == "$(cat "$D/sha_feat")" ]] && ok "worktree at feat" || bad "worktree not at feat"
fb_run_job 703
has "$J" "RUN SCRIPT main tag=demo_c"
[[ "$(git -C "$W" rev-parse HEAD)" == "$(cat "$D/sha_main")" ]] && ok "worktree moved to main for the third job" || bad "worktree not at main"

echo "== 11. a job after a killed run puts the foundation back, clears probe leftovers, and runs"
echo "-- left by a killed run" >> "$W/lean/LMS/Foundation.lean"; echo '{"entries": [1]}' > "$W/lean/LMS/Foundation.json"
: > "$W/lean/tmpabc123.lean"; mkdir -p "$W/experiments/demo_a"; echo keep > "$W/experiments/demo_a/results.json"
fb_run_job 702
has "$J" "NOTE: lean/LMS/Foundation.lean was left modified by an earlier job; putting the tracked version back"
has "$J" "NOTE: lean/LMS/Foundation.json was left modified"
has "$J" "RUN SCRIPT feat tag=demo_b agents=9"
[[ ! -e "$W/lean/tmpabc123.lean" ]] && ok "probe leftover removed" || bad "probe leftover still there"
[[ -z "$(git -C "$W" status --porcelain --untracked-files=no)" ]] && ok "tracked files clean" || bad "tracked files dirty"
[[ "$(cat "$W/experiments/demo_a/results.json")" == keep ]] && ok "the earlier run's files are untouched" || bad "run files were touched"

echo "== 12. other tracked changes stop the job before any checkout"
echo "edited by hand" >> "$W/feature.txt"
fb_run_job 702
has "$J" "FATAL: the worktree has changes to tracked files that are not the foundation"
hasnt "$J" "RUN SCRIPT"
[[ "$(cat "$D/job.702.rc")" == 1 ]] && ok "exit 1" || bad "exit $(cat "$D/job.702.rc")"
git -C "$W" checkout -q -- feature.txt

echo "== 13. a script outside the repo runs by absolute path; a missing one is fatal"
fb_new abs; install_demo
printf '#!/bin/bash\necho "OWN SCRIPT in $(pwd | sed "s#.*/##") tag=$RUN_TAG"\n' > "$H/lms_demo_own.sh"
(cd "$H" && fb_env RUN3="$H/lms_demo_own.sh" bash "$H/lms_demo_go.sh" "$(cat "$D/sha_main")" "$(cat "$D/sha_feat")" > "$D/out" 2>&1; echo $? > "$D/rc")
rc_is 0
fb_run_job 703; has "$J" "OWN SCRIPT in lms-box tag=demo_c"; has "$J" "run=lms_demo_own.sh"
rm "$H/lms_demo_own.sh"; fb_run_job 703; has "$J" "FATAL: nothing to run at"

echo "== 14. the lock: a second job waits, and gives up after its limit"
( exec 9> "$H/lms_q.lock"; flock 9; sleep 3 ) &
sleep 0.5
sed -i 's/^declare -x LMS_Q_PIN=/declare -x LMS_Q_LOCK_WAIT="1"\ndeclare -x LMS_Q_PIN=/' "$D/slurm/env.701"
fb_run_job 701
has "$J" "NOTE: another LMS job holds the worktree; waiting up to 1s"
has "$J" "FATAL: the worktree was still held after 1s"
hasnt "$J" "RUN SCRIPT"
wait
fb_run_job 701; has "$J" "RUN SCRIPT main tag=demo_a"; hasnt "$J" "another LMS job holds"

echo "== 15. collect waits for this round's jobs only, then copies and scrubs"
fb_new collect; install_demo
go "$(cat "$D/sha_main")"
fb_queue_add 800 lms-other-run RUNNING
fb_run lms_demo_collect.sh; rc_is 1; has "$D/out" "STOP: 2 job(s) of this round still queued or running"
[[ ! -e "$H/lms_demo" ]] && ok "nothing copied while running" || bad "an output folder was made"
fb_run_job 701; fb_finish 701 COMPLETED; fb_finish 702 TIMEOUT
mkdir -p "$W/experiments/demo_a" "$D/scratch/lms-runs/demo_a"
echo "{\"path\": \"$H/code/lms-box/x\", \"scratch\": \"$D/scratch/y\"}" > "$W/experiments/demo_a/results.json"
echo '{}' > "$W/experiments/demo_a/attempts.json"; echo "run log $H" > "$D/scratch/lms-runs/demo_a/run.log"
seq 1 500 > "$D/scratch/lms-runs/demo_a/vllm.log"
fb_run lms_demo_collect.sh; rc_is 0
has "$D/out" "==== lms-demo-a"
has "$D/out" "=== queued job: round=demo job=701"
has "$D/out" "COPIED demo_a: COMPLETED, 2 of 9 run files"
has "$D/out" "COPIED demo_b: TIMEOUT, 0 of 9 run files"
has "$D/out" "no log lms-demo-b-702.out"
has "$H/lms_demo/demo_a/run/results.json" '"path": "$HOME/code/lms-box/x", "scratch": "$SCRATCH/y"'
hasnt "$H/lms_demo/demo_a/run/results.json" "$H"
has "$H/lms_demo/demo_a/logs/run.log" 'run log $HOME'
[[ "$(wc -l < "$H/lms_demo/demo_a/logs/vllm.tail.log")" == 400 ]] && ok "vLLM log tail is 400 lines" || bad "vLLM tail wrong length"
[[ -f "$H/lms_demo/demo_a/logs/lms-demo-a-701.out" ]] && ok "job log copied" || bad "job log not copied"
fb_run lms_demo_collect.sh; rc_is 0
compgen -G "$H/lms_demo.old-*" > /dev/null && ok "the earlier output folder was moved aside" || bad "no .old folder"
rm "$H/lms_demo.jobs"; fb_run lms_demo_collect.sh; rc_is 1; has "$D/out" "STOP: no lms_demo.jobs, so this round queued nothing"

echo "== 16. pending-ok: a job that has not started does not hold up the rest; a running one does"
cat > "$FB_ROOT/lms_demo_collect2.sh" <<'COLLECT'
#!/bin/bash
set -uo pipefail
source "$HOME/lms_q_lib.sh"
q_init demo
q_collect_begin pending-ok
while read -r line; do
    id="$(q_field "$line" job)"
    if q_is_pending "$id"; then echo "NOT STARTED: $(q_field "$line" name)"; continue; fi
    q_copy_run "$line"
done < <(q_jobs)
echo "pending: ${#Q_PENDING[@]}"
COLLECT
fb_new pending; install_demo; fb_install "$FB_ROOT/lms_demo_collect2.sh"
go "$(cat "$D/sha_main")"
fb_finish 701 COMPLETED
fb_run lms_demo_collect2.sh; rc_is 0
has "$D/out" "COPIED demo_a: COMPLETED"
has "$D/out" "NOT STARTED: lms-demo-b"
has "$D/out" "pending: 1"
sed -i 's/^702 lms-demo-b PENDING$/702 lms-demo-b RUNNING/' "$D/slurm/queue"
fb_run lms_demo_collect2.sh; rc_is 1
has "$D/out" "STOP: 1 job(s) of this round still running; a log taken mid-write looks complete"
fb_run lms_demo_collect.sh; rc_is 1
has "$D/out" "STOP: 1 job(s) of this round still queued or running"
fb_finish 702 CANCELLED
fb_run lms_demo_collect2.sh; rc_is 0
has "$D/out" "COPIED demo_b: CANCELLED"
has "$D/out" "pending: 0"

echo "== 17. any STOP after jobs were queued cancels them; an earlier go's lines survive a failed later one"
cat > "$FB_ROOT/lms_demo_late.sh" <<'GO'
#!/bin/bash
# A go that queues one job and then stops for another reason; with "more" it adds to a jobs file
# an earlier go wrote, as a resume does.
set -euo pipefail
source "$HOME/lms_q_lib.sh"
q_init demo
[[ "${2:-}" == more ]] || q_round_guard
q_pin "$1"
q_submit late demo_late --time=00:10:00 -- AGENTS=1
q_pin_has nope.txt x "a file no commit has"
GO
fb_new late; install_demo; fb_install "$FB_ROOT/lms_demo_late.sh"
fb_run lms_demo_late.sh "$(cat "$D/sha_main")"
rc_is 1
has "$D/out" "cancelled the 1 job(s) this round had queued: 701"
has "$D/out" "lacks a file no commit has; nothing is queued"
[[ ! -s "$D/slurm/queue" ]] && ok "queue is empty" || bad "a job is still queued"
[[ ! -e "$H/lms_demo.jobs" ]] && ok "no jobs file left" || bad "jobs file still there"
go "$(cat "$D/sha_main")"; rc_is 0
fb_run lms_demo_late.sh "$(cat "$D/sha_main")" more
rc_is 1
has "$D/out" "cancelled the 1 job(s) this round had queued: 704"
count_is "$H/lms_demo.jobs" "job=" 2
has "$H/lms_demo.jobs" "job=702 name=lms-demo-a"
has "$H/lms_demo.jobs" "job=703 name=lms-demo-b"
hasnt "$H/lms_demo.jobs" "job=704"
count_is "$D/slurm/queue" "lms-demo-" 2

echo "== 18. collect says how busy the model server was, for a run that served one"
fb_new load; install_demo
go "$(cat "$D/sha_main")"; fb_finish 701; fb_finish 702
mkdir -p "$D/scratch/lms-runs/demo_a"
{
    echo "(APIServer pid=1) INFO 11-09 18:12:24 [loggers.py:335] Engine 000: Avg prompt throughput: 267.1 tokens/s, Avg generation throughput: 50.0 tokens/s, Running: 3 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%"
    echo "(APIServer pid=1) INFO 11-09 18:12:34 [loggers.py:335] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 150.0 tokens/s, Running: 9 reqs, Waiting: 4 reqs, GPU KV cache usage: 12.5%, Prefix cache hit rate: 40.0%"
    echo "(APIServer pid=1) INFO 11-09 18:12:44 [loggers.py:335] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 61.5%"
} > "$D/scratch/lms-runs/demo_a/vllm.log"
fb_run lms_demo_collect.sh; rc_is 0
has "$D/out" "SERVER demo_a: 3 reports, peak 9 running and 4 waiting (waiting in 1 reports), KV cache peak 12.5%, prefix cache hit 61.5% at the end, generation mean 100 peak 150 tokens/s"
hasnt "$D/out" "SERVER demo_b"

fb_done
