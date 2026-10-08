#!/usr/bin/env bash
# N1-density measurement (Sprint 3 DoD items 1 and 3): the Gate A control,
# then both ANT arcs, through scripts/measure_n1_density.py.
#
# No GPU. The work is Lean elaboration (one `import Mathlib` per exact? probe,
# ~5 GB resident each, run one at a time) plus rate-limited HTTPS to LeanSearch
# and loogle. Runbook: docs/runbooks/2026-10-08-box-restore.md, Step R4.
#
#   mkdir -p logs && sbatch -p <partition> scripts/slurm/n1_density.sbatch
#
# No --partition, --account or --gres here: the restored box's site config is
# unresolved, so pass them at submit time from Step R0's `sinfo` output.
#SBATCH --job-name=lms-n1-density
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

REPO="${REPO:-$HOME/code/lms}"

# A batch job does not source ~/.bashrc; route everything explicitly.
export SCRATCH="${SCRATCH:-/scratch/users/$USER}"
export XDG_CACHE_HOME="$SCRATCH/cache"
export ELAN_HOME="$SCRATCH/elan"
export PATH="$ELAN_HOME/bin:$HOME/.elan/bin:$HOME/.local/bin:$PATH"

cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 1; }
OUT=experiments/n1_density
mkdir -p "$OUT"

echo "=== N1 density: commit=$(git rev-parse --short HEAD) node=$(hostname) job=${SLURM_JOB_ID:-none} ==="

# --- Preflight: each of these produced a silently wrong measurement before ---
command -v uv >/dev/null 2>&1 || { echo "FATAL: uv not on PATH"; exit 1; }
command -v lake >/dev/null 2>&1 || { echo "FATAL: lake not on PATH; check ELAN_HOME=$ELAN_HOME"; exit 1; }
# Without the checkout, name search and the exact? probe report themselves
# unavailable and every verdict degrades to semantic-only.
[ -d lean/.lake/packages/mathlib/Mathlib ] || { echo "FATAL: no library checkout under lean/.lake/packages"; exit 1; }
# The pre-fix classifier counted stages that could not search as empty
# searches. Its absence is the signal this checkout predates that fix.
grep -q "def read_output" lms/novelty/mathlib_search.py || { echo "FATAL: checkout predates the stage-availability fix; git pull main"; exit 1; }
# Compute nodes may have no outbound route even when the login node does.
for url in https://leansearch.net https://loogle.lean-lang.org; do
  curl -sS --max-time 15 -o /dev/null "$url" || { echo "FATAL: no route to $url from $(hostname)"; exit 1; }
done

# --- Measure ---------------------------------------------------------------
# Control first: if it reads decisive-N1-heavy, the arc densities are void.
rc=0
for arc in data/novelty_control/gate_a_control_arc.json data/ant_arcs/core_arc.json data/ant_arcs/ramification_arc.json; do
  name=$(basename "$arc" .json)
  echo "--- $name ($(date -Is)) ---"
  uv run python scripts/measure_n1_density.py "$arc" --json-out "$OUT/$name.report.json" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
done

echo "=== done rc=$rc ($(date -Is)); reports in $REPO/$OUT ==="
exit "$rc"
