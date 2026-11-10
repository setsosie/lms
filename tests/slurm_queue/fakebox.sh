#!/bin/bash
# A fake server for testing queued rounds: a fake HOME and scratch, a small origin with three
# commits, the shared worktree, and stubs for Slurm that keep a queue. Source it from a test script.
#
#   source fakebox.sh
#   fb_new NAME            build a fresh fake server; sets D (its root), H (HOME), W (the worktree)
#   fb_install FILE...     copy files into the fake ~/ (the kit is already there)
#   fb_run SCRIPT ARGS...  run ~/SCRIPT as the user would; output in $D/out, exit code in $D/rc
#   fb_run_job ID          run a queued job as Slurm would (its script, directory and environment);
#                          sets J to its output file (where --output pointed), exit code in $D/job.<id>.rc
#   fb_finish ID [STATE]   take a job out of the queue (default COMPLETED)
#   ok / bad / has FILE TEXT / hasnt FILE TEXT / count_is FILE TEXT N / rc_is N
#   fb_done                print the totals; returns non-zero if any check failed
#
# Commits: $(cat $D/sha_main) is origin's main; sha_feat is one commit further, on origin's branch
# feat; sha_local exists only in ~/code/lms, on no origin branch. Each commit's run script prints
# "RUN SCRIPT <main|feat|local>", so a test can tell which commit a job ran.
# FB_SEED_HOOK, if set to a function name, is called with the seed directory before the first
# commit, to add files a round needs at the pin (e.g. the real lms_run.sbatch).
# Fake Slurm: sbatch gives ids from 701 and fails on its Nth call when FAKE_SBATCH_FAIL_AT=N.
# shellcheck shell=bash
FB_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FB_KIT="${FB_KIT:-$FB_HERE/../../scripts/slurm/queue}"
FB_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lms-fakebox.XXXXXX")"
trap 'rm -rf "$FB_ROOT"' EXIT
fb_fails=0
fb_checks=0
ok() { fb_checks=$((fb_checks + 1)); echo "  ok   $1"; }
bad() { fb_checks=$((fb_checks + 1)); fb_fails=$((fb_fails + 1)); echo "  FAIL $1"; }
has() { if grep -qF -- "$2" "$1" 2> /dev/null; then ok "has: $2"; else bad "missing: $2  (in $(basename "$1"))"; sed 's/^/      /' "$1" 2> /dev/null | tail -12; fi; }
hasnt() { if grep -qF -- "$2" "$1" 2> /dev/null; then bad "should not contain: $2  (in $(basename "$1"))"; else ok "absent: $2"; fi; }
count_is() { local n; n="$(grep -cF -- "$2" "$1" 2> /dev/null)"; n="${n:-0}"; if [[ "$n" == "$3" ]]; then ok "$3 x $2"; else bad "$n x $2, wanted $3  (in $(basename "$1"))"; fi; }
rc_is() { if [[ "$(cat "$D/rc")" == "$1" ]]; then ok "exit code $1"; else bad "exit code $(cat "$D/rc"), wanted $1"; sed 's/^/      /' "$D/out" | tail -12; fi; }
fb_done() { echo; echo "checks: $fb_checks, failed: $fb_fails"; ((fb_fails == 0)); }

fb_new() {
    D="$FB_ROOT/$1"
    H="$D/home"
    local s="$D/seed"
    mkdir -p "$H/code" "$D/bin" "$D/slurm" "$s/scripts/slurm" "$s/lean/LMS" "$s/logs" \
        "$D/scratch/vllm-env/bin" "$D/scratch/hf/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots" "$D/lake"
    echo 701 > "$D/slurm/next"
    : > "$D/slurm/queue"
    : > "$D/slurm/done"
    echo toolchain > "$s/lean/lean-toolchain"
    echo "-- seed foundation" > "$s/lean/LMS/Foundation.lean"
    echo '{"entries": []}' > "$s/lean/LMS/Foundation.json"
    : > "$s/logs/.gitkeep"
    printf 'logs/*.out\nlogs/*.err\nexperiments/\n.venv/\n.lake\n' > "$s/.gitignore"
    printf '#!/bin/bash\necho "RUN SCRIPT main tag=${RUN_TAG:-} agents=${AGENTS:-} extra=${EXTRA:-}"\n' > "$s/scripts/slurm/lms_run.sbatch"
    git init -q -b main "$s"
    git -C "$s" config user.email t@example.com
    git -C "$s" config user.name T
    [[ -n "${FB_SEED_HOOK:-}" ]] && "$FB_SEED_HOOK" "$s"
    git -C "$s" add -A
    git -C "$s" commit -q -m main
    git -C "$s" rev-parse HEAD > "$D/sha_main"
    git -C "$s" checkout -q -b feat
    sed -i 's/RUN SCRIPT main/RUN SCRIPT feat/' "$s/scripts/slurm/lms_run.sbatch"
    echo feature > "$s/feature.txt"
    git -C "$s" add -A
    git -C "$s" commit -q -m feat
    git -C "$s" rev-parse HEAD > "$D/sha_feat"
    git -C "$s" checkout -q main
    git clone -q --bare "$s" "$D/origin.git"
    git clone -q "$D/origin.git" "$H/code/lms"
    git -C "$H/code/lms" config user.email t@example.com
    git -C "$H/code/lms" config user.name T
    # A commit that never reached origin.
    git -C "$H/code/lms" checkout -q -b local-only
    sed -i 's/RUN SCRIPT main/RUN SCRIPT local/' "$H/code/lms/scripts/slurm/lms_run.sbatch"
    git -C "$H/code/lms" commit -q -am local
    git -C "$H/code/lms" rev-parse HEAD > "$D/sha_local"
    git -C "$H/code/lms" checkout -q main
    git -C "$H/code/lms" worktree add -q --detach "$H/code/lms-box" "$(cat "$D/sha_main")"
    W="$H/code/lms-box"
    mkdir -p "$W/.venv" "$W/experiments" "$W/logs"
    ln -s "$D/lake" "$W/lean/.lake"
    printf '#!/bin/bash\nexec tail -f /dev/null\n' > "$D/scratch/vllm-env/bin/vllm"
    printf '#!/bin/bash\nexec python3 "$@"\n' > "$D/scratch/vllm-env/bin/python"

    cat > "$D/bin/sbatch" <<STUB
#!/bin/bash
n="\$(cat "$D/slurm/calls" 2> /dev/null || echo 0)"; n=\$((n + 1)); echo "\$n" > "$D/slurm/calls"
if [[ -n "\${FAKE_SBATCH_FAIL_AT:-}" && "\$n" == "\$FAKE_SBATCH_FAIL_AT" ]]; then echo "sbatch: error: refused" >&2; exit 1; fi
id="\$(cat "$D/slurm/next")"; echo \$((id + 1)) > "$D/slurm/next"
name=job; dir="\$PWD"; out=""
for a in "\$@"; do
    case "\$a" in --job-name=*) name="\${a#--job-name=}" ;; --chdir=*) dir="\${a#--chdir=}" ;; --output=*) out="\${a#--output=}" ;; esac
done
[[ -n "\$out" ]] || out="\$dir/logs/%x-%j.out"
out="\${out//%x/\$name}"; out="\${out//%j/\$id}"; echo "\$out" > "$D/slurm/out.\$id"
echo "\$id \$*" >> "$D/slurm/sbatch.log"
export -p > "$D/slurm/env.\$id"
echo "\${@: -1}" > "$D/slurm/script.\$id"; echo "\$dir" > "$D/slurm/dir.\$id"; echo "\$name" > "$D/slurm/name.\$id"
echo "\$id \$name PENDING" >> "$D/slurm/queue"
echo "\$id;cluster"
STUB
    cat > "$D/bin/squeue" <<STUB
#!/bin/bash
[[ -n "\${FAKE_SQUEUE_FAIL:-}" ]] && exit 1
[[ " \$* " == *" -h "* ]] || echo "   JOBID NAME STATE"
cat "$D/slurm/queue"
STUB
    cat > "$D/bin/scancel" <<STUB
#!/bin/bash
echo "\$*" >> "$D/slurm/scancel.log"
for id in "\$@"; do
    [[ "\$id" =~ ^[0-9]+\$ ]] || continue
    grep -v "^\$id " "$D/slurm/queue" > "$D/slurm/queue.new"; mv "$D/slurm/queue.new" "$D/slurm/queue"
    echo "\$id CANCELLED" >> "$D/slurm/done"
done
STUB
    cat > "$D/bin/sacct" <<STUB
#!/bin/bash
# Like the real one: -o State prints the state alone; a list with JobID prints the whole row.
all="\$*"; id=""; while [[ \$# -gt 0 ]]; do [[ "\$1" == -j ]] && id="\$2"; shift; done
state="\$(awk -v id="\$id" '\$1 == id { s = \$2 } END { print s }' "$D/slurm/done")"
[[ -n "\$state" ]] || exit 0
if [[ "\$all" == *JobID* ]]; then echo "\$id \$state 0:0 00:10:00"; else echo "\$state"; fi
STUB
    printf '#!/bin/bash\necho "uv $*"\n' > "$D/bin/uv"
    printf '#!/bin/bash\n' > "$D/bin/nvidia-smi"
    chmod +x "$D/bin/"* "$D/scratch/vllm-env/bin/"*
    cp "$FB_KIT/lms_q_lib.sh" "$FB_KIT/lms_q_job.sbatch" "$H/"
}

fb_install() { cp "$@" "$H/"; }

fb_env() {
    env -i HOME="$H" USER=tester PATH="$D/bin:/usr/bin:/bin" SCRATCH="$D/scratch" \
        FAKE_SBATCH_FAIL_AT="${FAKE_SBATCH_FAIL_AT:-}" FAKE_SQUEUE_FAIL="${FAKE_SQUEUE_FAIL:-}" "$@"
}

fb_run() {
    local script="$1"
    shift
    (cd "$H" && fb_env bash "$H/$script" "$@" > "$D/out" 2>&1)
    echo $? > "$D/rc"
}

fb_queue_add() { echo "$1 $2 ${3:-RUNNING}" >> "$D/slurm/queue"; }

fb_finish() {
    grep -v "^$1 " "$D/slurm/queue" > "$D/slurm/queue.new"
    mv "$D/slurm/queue.new" "$D/slurm/queue"
    echo "$1 ${2:-COMPLETED}" >> "$D/slurm/done"
}

# Runs job ID with the environment sbatch saw, in its directory, writing where Slurm would.
fb_run_job() {
    local id="$1" dir script
    dir="$(cat "$D/slurm/dir.$id")"
    script="$(cat "$D/slurm/script.$id")"
    J="$(cat "$D/slurm/out.$id")"
    mkdir -p "$(dirname "$J")"
    (
        # shellcheck disable=SC1090
        source "$D/slurm/env.$id"
        cd "$dir" || exit 97
        SLURM_JOB_ID="$id" bash "$script"
    ) > "$J" 2> "${J%.out}.err"
    echo $? > "$D/job.$id.rc"
}
