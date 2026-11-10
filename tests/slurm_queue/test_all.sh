#!/bin/bash
# Tests the master runner (lms_all_run.sh) on the fake server with three small rounds:
# one that queues, one whose go stops, one already queued.
#   bash test_all.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=fakebox.sh
source "${FB_HARNESS:-$HERE/fakebox.sh}"
ALL="${LMS_ALL_RUN:-$HERE/../../scripts/slurm/queue/lms_all_run.sh}"

mk_round() { # name, extra line for the go
    cat > "$FB_ROOT/lms_$1_go.sh" <<GO
#!/bin/bash
set -euo pipefail
source "\$HOME/lms_q_lib.sh"
q_init $1
q_require_box
q_round_guard
q_pin "\$1"
$2
q_submit run $1_tag --time=01:00:00 -- AGENTS=1
GO
    cat > "$FB_ROOT/lms_$1_collect.sh" <<COLLECT
#!/bin/bash
set -uo pipefail
source "\$HOME/lms_q_lib.sh"
q_init $1
q_collect_begin
while read -r line; do q_copy_run "\$line"; done < <(q_jobs)
echo "RESULT: $1 read"
cat > /dev/null   # a collect that reads stdin must not swallow the plan
COLLECT
}
mk_round one ""
mk_round two 'q_pin_has nope.txt x "a file no commit has"'
mk_round three ""

plan() {
    printf '# round<TAB>phase<TAB>command\n' > "$H/lms_all_plan.txt"
    local r
    for r in one two three; do
        printf '%s\tgo\tbash ~/lms_%s_go.sh %s 2>&1 | tee -a ~/lms_%s_go.out\n' "$r" "$r" "$(cut -c1-8 "$D/sha_main")" "$r" >> "$H/lms_all_plan.txt"
        printf '%s\tcollect\tbash ~/lms_%s_collect.sh 2>&1 | tee ~/lms_%s.out\n' "$r" "$r" "$r" >> "$H/lms_all_plan.txt"
    done
}

echo "== 1. go: each round is tried, a stopped round does not stop the next"
fb_new all
fb_install "$ALL" "$FB_ROOT"/lms_one_* "$FB_ROOT"/lms_two_* "$FB_ROOT"/lms_three_*
plan
fb_run lms_all_run.sh go
rc_is 0
has "$D/out" "======== one"
has "$D/out" "Queued lms-one-run: job 701"
has "$D/out" "STOP: $(cut -c1-8 "$D/sha_main") lacks a file no commit has; nothing is queued"
has "$D/out" "Queued lms-three-run: job 702, pin $(cut -c1-8 "$D/sha_main"), after job 701"
has "$D/out" "======== summary (go)"
has "$D/out" "one        QUEUED"
has "$D/out" "two        GO STOPPED"
has "$D/out" "three      QUEUED"
has "$D/out" "rounds: 3, queued: 2, not queued: 1"
has "$H/lms_one_go.out" "Queued lms-one-run"

echo "== 2. go again: rounds already queued refuse, nothing new is queued"
fb_run lms_all_run.sh go
rc_is 0
count_is "$D/slurm/sbatch.log" "--job-name=" 2
has "$D/out" "STOP: this round was already queued (1 jobs in lms_one.jobs)"
has "$D/out" "rounds: 3, queued: 0, not queued: 3"

echo "== 3. collect: finished rounds are read, running ones are skipped"
fb_finish 701 COMPLETED
fb_run lms_all_run.sh collect
rc_is 0
has "$D/out" "RESULT: one read"
has "$D/out" "one        COLLECTED"
has "$D/out" "two        NOT COLLECTED"
has "$D/out" "three      NOT COLLECTED"
has "$D/out" "STOP: 1 job(s) of this round still queued or running"
has "$D/out" "rounds: 3, collected: 1, not collected: 2"
fb_finish 702 FAILED
fb_run lms_all_run.sh collect
has "$D/out" "three      COLLECTED"
has "$D/out" "COPIED three_tag: FAILED"

echo "== 4. plan mode, a missing plan and a bad phase"
fb_run lms_all_run.sh plan; rc_is 0; has "$D/out" "bash ~/lms_one_go.sh"
fb_run lms_all_run.sh; rc_is 2; has "$D/out" "usage:"
printf 'one\tlaunch\techo hi\n' > "$H/lms_all_plan.txt"
fb_run lms_all_run.sh go; rc_is 1; has "$D/out" "STOP: the plan has an unknown phase 'launch' for one; nothing was run"
rm "$H/lms_all_plan.txt"
fb_run lms_all_run.sh go; rc_is 1; has "$D/out" "STOP: no plan file at ~/lms_all_plan.txt"

fb_done
