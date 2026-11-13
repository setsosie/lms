#!/bin/bash
# Run every round's setup and go commands, or every round's collect command, from one plan file.
#   bash ~/lms_all_run.sh go        queue every round in the plan, in order
#   bash ~/lms_all_run.sh collect   collect every round that has finished
#   bash ~/lms_all_run.sh plan      print the plan without running anything
# The plan is ~/lms_all_plan.txt: one command per line as  round <TAB> phase <TAB> command,
# with phase one of setup, go, collect. Each command is a round's own script, called as its own
# RUN_THIS.txt calls it. A round that stops never stops the rounds after it: if a round's setup
# fails its go is skipped, and the last lines say what each round did.
set -uo pipefail

PLAN="${LMS_ALL_PLAN:-$HOME/lms_all_plan.txt}"
MODE="${1:-}"
case "$MODE" in
    go | collect | plan) ;;
    *)
        echo "usage: bash $0 {go|collect|plan}"
        exit 2
        ;;
esac
[[ -f "$PLAN" ]] || {
    echo "STOP: no plan file at ~/$(basename "$PLAN"); copy it over with the scripts"
    exit 1
}

rounds=()
declare -A seen=() result=()
while IFS=$'\t' read -r round phase command; do
    [[ -z "$round" || "$round" == \#* ]] && continue
    case "$phase" in
        setup | go | collect) ;;
        *)
            echo "STOP: the plan has an unknown phase '$phase' for $round; nothing was run"
            exit 1
            ;;
    esac
    [[ -n "$command" ]] || {
        echo "STOP: the plan has an empty command for $round $phase; nothing was run"
        exit 1
    }
    if [[ -z "${seen[$round]:-}" ]]; then
        seen[$round]=1
        rounds+=("$round")
    fi
done < "$PLAN"
((${#rounds[@]})) || {
    echo "STOP: the plan lists no rounds"
    exit 1
}

if [[ "$MODE" == plan ]]; then
    cat "$PLAN"
    exit 0
fi

# Run one phase of one round. Returns 0 if every command of that phase exited 0, 1 if one
# failed (later commands of the phase are then skipped), 2 if the round has no such phase.
run_phase() {
    local want_round="$1" want_phase="$2" round phase command status found=0
    while IFS=$'\t' read -r round phase command <&3; do
        [[ "$round" == "$want_round" && "$phase" == "$want_phase" ]] || continue
        found=1
        echo
        echo "---- $round $phase: $command"
        # stdin from /dev/null: a script that reads it must not swallow the rest of the plan.
        bash -o pipefail -c "$command" < /dev/null
        status=$?
        if ((status != 0)); then
            echo "---- $round $phase exited $status"
            return 1
        fi
    done 3< "$PLAN"
    ((found)) || return 2
    return 0
}

for round in "${rounds[@]}"; do
    echo
    echo "======== $round"
    if [[ "$MODE" == go ]]; then
        run_phase "$round" setup
        case $? in
            1)
                result[$round]="SETUP STOPPED, nothing queued for this round"
                continue
                ;;
        esac
        run_phase "$round" go
        case $? in
            0) result[$round]="QUEUED" ;;
            1) result[$round]="GO STOPPED (read its STOP line above; 'already' means it was queued before)" ;;
            2) result[$round]="no go command in the plan" ;;
        esac
    else
        run_phase "$round" collect
        case $? in
            0) result[$round]="COLLECTED" ;;
            1) result[$round]="NOT COLLECTED (still running, or its collect stopped; see above)" ;;
            2) result[$round]="no collect command in the plan" ;;
        esac
    fi
done

echo
echo "======== summary ($MODE)"
bad=0
for round in "${rounds[@]}"; do
    printf '%-10s %s\n' "$round" "${result[$round]}"
    [[ "${result[$round]}" == QUEUED || "${result[$round]}" == COLLECTED ]] || bad=$((bad + 1))
done
done_word=queued
[[ "$MODE" == collect ]] && done_word=collected
echo "rounds: ${#rounds[@]}, $done_word: $((${#rounds[@]} - bad)), not $done_word: $bad"
exit 0
