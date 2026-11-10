# Queued rounds: several experiments in the Slurm queue at once

**Use this when** more than one LMS experiment should wait in the box's queue,
and they are not all at the same commit.

**Scope**: the single-node Slurm box. The person running the experiment runs
every command; nothing here is run by an agent on the server.

## Why rounds could not be queued together before

A *round* is a folder of scripts plus a `RUN_THIS.txt`: copy the files to the
server, run a `go` script that queues jobs, later run a `collect` script and
copy the results back. Until now a round's `go` checked out its commit in the
shared worktree (`~/code/lms-box`) and refused to run while any other LMS job
was queued, so rounds ran strictly one at a time, with a person in between.

Two things forced that:

- **One Lean tree.** A run rewrites `lean/LMS/Foundation.lean` inside the
  worktree and builds there (ADR 0003: "No two Phase C runs execute
  concurrently against the same `lean/` tree"). Two LMS jobs at once would
  share a foundation.
- **One checkout.** A job runs whatever commit the worktree holds when the job
  *starts*, which can be hours after it was queued.

## The two rules

1. **Every job waits for the LMS job queued before it.** `q_submit` adds
   `--dependency=afterany:<id>`, where `<id>` is the newest job in the queue
   whose name starts with `lms-`. The jobs form one line, in the order they
   were queued. `afterany` means a job that fails, is cancelled or hits its
   time limit still lets the next one start.
2. **Every job checks out its own commit when it starts.** Jobs run
   `lms_q_job.sbatch`, which takes a lock, puts the tracked foundation files
   back if a killed job left them modified, checks out the job's commit,
   confirms it, and then runs that commit's `scripts/slurm/lms_run.sbatch` (or
   another script named by `LMS_Q_RUN`) in the same process.

A job that uses neither the worktree nor the Lean tree (a serving benchmark,
say) is submitted with `q_submit_free`. It is named `lmsfree-…`, waits for
nothing, and nothing waits for it.

## The files

| File | Where it runs | What it is |
|---|---|---|
| `scripts/slurm/queue/lms_q_lib.sh` | sourced by every `go` and `collect` | pin check, chaining, the jobs file, collecting a run |
| `scripts/slurm/queue/lms_q_job.sbatch` | the job itself | lock, restore, checkout, run |
| `scripts/slurm/queue/lms_all_run.sh` | once, on the server | runs every round's `go`, or every round's `collect`, from a plan file |
| `tests/slurm_queue/fakebox.sh` | locally | a fake server for testing a round: fake `HOME`, origin, worktree, Slurm stubs with a queue |

The first three are copied to the server's `~/` with each batch of rounds. They
are not read from the server's checkout, because the job that needs them runs
before any checkout.

## Writing a round

```bash
#!/bin/bash
set -euo pipefail
source "$HOME/lms_q_lib.sh"
q_init myround
q_require_box gpu                 # drop "gpu" for a CPU-only round
q_round_guard                     # a second go queues nothing
q_pin "$1"                        # stops unless a branch on origin holds the commit
q_pin_has scripts/slurm/lms_run.sbatch 'MAX_TOKENS' "the budget knob"
q_tag_free my_run                 # never overwrite a run
q_submit a my_run "${Q_GPU[@]}" --time=04:00:00 -- TP_SIZE=2 AGENTS=1 MAX_TOKENS=2000000
q_show_queue
```

`q_submit NAME TAG [sbatch options] -- [VAR=VALUE…]` names the job
`lms-<round>-<NAME>` and appends
`job=<id> name=<jobname> tag=<tag> pin=<sha8> after=<id|->` to
`~/lms_<round>.jobs`. If `sbatch` refuses a job, or anything else stops the
script through `q_stop` after a submit, everything that `go` queued is
cancelled: a round is queued whole or not at all. `q_pin` can be called again
to queue later jobs at another commit.

The `collect` side: `q_collect_begin` stops while any job *of this round* is
queued or running (a log copied mid-write looks complete), then
`q_copy_run LINE` copies one run's logs and result files into
`~/lms_<round>/<tag>/` with `$HOME` and `$SCRATCH` scrubbed. Before its
`COPIED` line it prints a `SERVER <tag>:` line from vLLM's own load reports:
peak requests running and waiting, KV-cache use and generation speed. Requests
waiting means the server set the pace, not the agents.

`q_collect_begin pending-ok` lets a collect read the jobs that have finished
while others of the round still wait to start. A job that is running still
stops it. `q_is_pending ID` says which jobs to report as not started.

One trap for a round's own scripts: under `set -o pipefail`, never pipe a
producer into `grep -q` (or `head`, or an `awk` that exits early). The reader
exits at its first match, the producer dies of SIGPIPE, and the pipeline
"fails" exactly when the thing was found. Read into a variable and search a
here-string, as `q_pin_has` does.

Test a round on the fake server before handing it over:

```bash
source tests/slurm_queue/fakebox.sh   # fb_new, fb_install, fb_run, fb_run_job, fb_finish, has, rc_is, fb_done
```

`tests/slurm_queue/test_kit.sh` shows every helper in use.

## Queueing several rounds

`~/lms_all_plan.txt` has one command per line as `round<TAB>phase<TAB>command`,
with phase `go` or `collect`. The order of rounds in the plan is the order the
jobs run.

```bash
bash ~/lms_all_run.sh plan      # print the plan, run nothing
bash ~/lms_all_run.sh go        # every round's go, in order
bash ~/lms_all_run.sh collect   # every round's collect; unfinished rounds are skipped
```

A round whose `go` stops does not stop the rounds after it. The last lines
printed say what each round did.

## Operating

- **Status**: `squeue -u $USER -o '%.8i %.34j %.9T %.10M %.30R'`. All but the
  first LMS job show `PENDING (Dependency)`.
- **Cancel one round**: `sed -n 's/^job=\([0-9]*\) .*/\1/p' ~/lms_<round>.jobs | xargs -r scancel`.
  The jobs behind it start as soon as the cancelled ones are gone.
- **Cancel every LMS job**: `squeue -h -u $USER -o '%i %j' | awk '$2 ~ /^lms/ {print $1}' | xargs -r scancel`.
- **A job hit its time limit**: its run is checkpointed every generation and
  can be continued with `RESUME=1` and the same knobs (restore runbook, step
  R6). The next job in line restores the tracked foundation files before it
  checks out; the killed run's own copy is in its `experiments/<tag>/`.
- **Queue a round again**: move `~/lms_<round>.jobs` aside first. Its run tags
  must be free; `go` refuses to overwrite `experiments/<tag>/`.

## What a job refuses

- Tracked changes in the worktree other than the two foundation files: it
  stops before checking out, so nothing is overwritten.
- A commit it cannot find after fetching origin.
- A lock still held after `LMS_Q_LOCK_WAIT` seconds (12 h). Jobs queued through
  `q_submit` never overlap, so the lock only matters if an LMS job was
  submitted some other way.

## Limits

- LMS jobs run one at a time, on two of the four GPUs. Running two at once
  needs a `lean/` tree per run (a Lake build directory each), which ADR 0003
  deferred as untested on this box.
- The chain is built from the queue as it is when `go` runs. Two `go` scripts
  started at the same moment could each pick the same job to wait for.
  `lms_all_run.sh` runs them one after another.
- The fake server's Slurm keeps a queue but does not model scheduling. The
  first real use of each `sbatch` option is on the box.
