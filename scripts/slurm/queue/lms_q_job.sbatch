#!/usr/bin/env bash
# LMS queue kit: the job every queued round submits (through q_submit in ~/lms_q_lib.sh).
#
# The rounds share one worktree, and a job's commit is chosen when it is queued, possibly hours
# before it starts. So the checkout happens here, when the job starts:
#   1. take the worktree lock (no two LMS jobs may use the Lean tree at once);
#   2. put back the tracked foundation files if an earlier job was killed before it could;
#   3. check out LMS_Q_PIN and confirm that is what is checked out;
#   4. run LMS_Q_RUN from that commit (default scripts/slurm/lms_run.sbatch), with this job's
#      environment, in this same process.
# Resources come from the sbatch command line, not from directives here or in the script it runs.
#SBATCH --job-name=lms-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -uo pipefail

fatal() {
    echo "FATAL: $*"
    exit 1
}

: "${REPO:?REPO unset; submit through q_submit}" "${LMS_Q_PIN:?LMS_Q_PIN unset; submit through q_submit}"
LMS_Q_ROUND="${LMS_Q_ROUND:-unknown}"
LMS_Q_RUN="${LMS_Q_RUN:-scripts/slurm/lms_run.sbatch}"
LMS_Q_LOCK_WAIT="${LMS_Q_LOCK_WAIT:-43200}"
# A batch job does not source ~/.bashrc. uv lives in ~/.local/bin on the box.
export PATH="$HOME/.local/bin:$PATH"
export UV_FROZEN="${UV_FROZEN:-1}"
export GIT_TERMINAL_PROMPT=0

cd "$REPO" || fatal "no worktree at $REPO"

# Held until this process and everything it starts has exited. Jobs queued through q_submit never
# overlap, so this only waits if something was queued another way.
exec 9> "$HOME/lms_q.lock" || fatal "cannot open the lock file"
if ! flock -n 9; then
    echo "NOTE: another LMS job holds the worktree; waiting up to ${LMS_Q_LOCK_WAIT}s"
    flock -w "$LMS_Q_LOCK_WAIT" 9 || fatal "the worktree was still held after ${LMS_Q_LOCK_WAIT}s"
fi

# A run rewrites these two tracked files and restores them when it ends. A job killed at its
# walltime never gets that far. The run's own copy is in its experiments/<tag>/ checkpoint.
for f in lean/LMS/Foundation.lean lean/LMS/Foundation.json; do
    if ! git diff --quiet -- "$f" 2> /dev/null; then
        echo "NOTE: $f was left modified by an earlier job; putting the tracked version back"
        git checkout -q -- "$f" || fatal "could not restore $f"
    fi
done
# An exact? probe that was killed leaves its scratch file behind.
find lean -maxdepth 1 -name 'tmp*.lean' -type f -delete 2> /dev/null

dirty="$(git status --porcelain --untracked-files=no)"
[[ -z "$dirty" ]] || {
    echo "$dirty"
    fatal "the worktree has changes to tracked files that are not the foundation; not checking out over them"
}

git cat-file -e "$LMS_Q_PIN^{commit}" 2> /dev/null || git fetch -q origin '+refs/heads/*:refs/remotes/origin/*' ||
    fatal "commit $LMS_Q_PIN is not here and origin could not be fetched"
git checkout -q --detach "$LMS_Q_PIN" || fatal "could not check out $LMS_Q_PIN"
head="$(git rev-parse HEAD)"
[[ "$head" == "$LMS_Q_PIN" ]] || fatal "checked out $head, wanted $LMS_Q_PIN"

run="$LMS_Q_RUN"
[[ "$run" == /* ]] || run="$REPO/$run"
[[ -f "$run" ]] || fatal "nothing to run at $LMS_Q_RUN in commit ${head:0:8}"

mkdir -p logs
echo "=== queued job: round=$LMS_Q_ROUND job=${SLURM_JOB_ID:-none} pin=${head:0:8} run=$(basename "$run") tag=${RUN_TAG:-none} ==="
exec bash "$run"
