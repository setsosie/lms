"""The batch script serves the model with kernels this node can run.

vLLM 0.33.0 died at load on `find_cuda_toolkit`, because it compiles its DeepGEMM
and FlashInfer kernels and the node has no CUDA toolkit on its default paths. R6,
R7 and every Phase C arm use this script, so they would fail the same way.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "slurm" / "lms_run.sbatch"


def text() -> str:
    return SCRIPT.read_text()


def test_it_parses() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_kernels_needing_a_toolkit_default_off_and_can_be_overridden() -> None:
    assert 'export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"' in text()
    assert (
        'export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"'
        in text()
    )
    assert (
        'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"'
        in text()
    )


def test_the_prefill_backend_reaches_the_server() -> None:
    assert 'GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"' in text()
    serve = text().split('vllm" serve', 1)[1].split('> "$VLLM_LOG"', 1)[0]
    assert '--gdn-prefill-backend "$GDN_PREFILL_BACKEND"' in serve


def test_the_exports_come_before_the_server_starts() -> None:
    assert text().index("VLLM_USE_DEEP_GEMM=") < text().index('vllm" serve')


# --- Behaviour, with every external program replaced by a stub ----------------

STUBS = {
    # The server: records what it was given, starts a worker in its own process
    # group (vLLM starts engine and GPU workers), and answers only if the
    # attempt's flags match STUB_SERVE_WHEN. Then it stays up until it is killed.
    "vllm": """#!/bin/bash
printf '%s\\n' "$*" > "$STUB_DIR/vllm.args"
env | grep '^VLLM_' | sort > "$STUB_DIR/vllm.env"
printf '%s | NCCL_P2P_DISABLE=%s\\n' "$*" "${NCCL_P2P_DISABLE:-}" >> "$STUB_DIR/vllm.attempts"
[ -z "${STUB_DIES:-}" ] || exit 1
# An earlier attempt still running here would still hold its GPUs.
for p in $(cat "$STUB_DIR/vllm.pids" 2>/dev/null); do
  state=$(cut -d' ' -f3 "/proc/$p/stat" 2>/dev/null)
  [ -z "$state" ] || [ "$state" = Z ] || echo "$p" >> "$STUB_DIR/overlap"
done
[ -z "${STUB_IGNORES_TERM:-}" ] || trap '' TERM
tail -f /dev/null &
echo "$$ $!" >> "$STUB_DIR/vllm.pids"
case "${STUB_SERVE_WHEN:-always}" in
  always) touch "$STUB_DIR/serving" ;;
  custom-ar-off) [[ " $* " != *" --disable-custom-all-reduce "* ]] || touch "$STUB_DIR/serving" ;;
  p2p-off) [ "${NCCL_P2P_DISABLE:-}" != 1 ] || touch "$STUB_DIR/serving" ;;
esac
[ -z "${STUB_HOLDS_GPU:-}" ] || touch "$STUB_DIR/gpu.held"
exec tail -f /dev/null
""",
    # Answers only once the server stub is up, so the port check before it passes.
    "curl": """#!/bin/bash
[ -e "$STUB_DIR/serving" ]
""",
    "uv": """#!/bin/bash
printf '%s\\n' "$*" >> "$STUB_DIR/uv.args"
printf 'NCCL_P2P_DISABLE=%s\\n' "${NCCL_P2P_DISABLE:-}" >> "$STUB_DIR/uv.env"
""",
    "lake": "#!/bin/bash\n",
    # GPU 0 reads as busy once a server stub has left its memory held.
    "nvidia-smi": """#!/bin/bash
if [ "$1" = topo ]; then
  [ -z "${STUB_TOPO_FAILS:-}" ] || { echo "topo: no NVML" >&2; exit 9; }
  printf '\\tGPU0\\tGPU1\\nGPU0\\t X \\tNV12\\nGPU1\\tNV12\\t X \\n'
  exit 0
fi
[ ! -e "$STUB_DIR/gpu.held" ] || echo "0, 40000"
""",
    "scontrol": """#!/bin/bash
[ -z "${STUB_SCONTROL_FAILS:-}" ] || { echo "scontrol: error" >&2; exit 1; }
echo "JobId=$4 JobName=lms-run"
echo "   JOB_GRES=gpu:nvidia_h100_nvl:2(IDX:0,2)"
echo "     Nodes=fla1 CPU_IDs=0-15 Mem=0 GRES=gpu:nvidia_h100_nvl:2(IDX:0,2)"
""",
    "sleep": "#!/bin/bash\n",
}


@pytest.fixture
def box(tmp_path: Path) -> Iterator[dict[str, str]]:
    """A fake checkout and scratch area, with stubs first on PATH."""
    bin_dir = tmp_path / "bin"
    vllm_bin = tmp_path / "scratch" / "vllm-env" / "bin"
    repo = tmp_path / "repo"
    for d in (bin_dir, vllm_bin, repo / "lean" / ".lake", repo / "logs"):
        d.mkdir(parents=True)
    for name, body in STUBS.items():
        target = (vllm_bin if name == "vllm" else bin_dir) / name
        target.write_text(body)
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "USER": "tester",
        "SCRATCH": str(tmp_path / "scratch"),
        "REPO": str(repo),
        "STUB_DIR": str(tmp_path),
        "RUN_TAG": "t",
    }
    yield env
    # A stub the script failed to stop would run forever.
    for pid in stub_pids(env):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(env: dict[str, str], **knobs: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={**env, **knobs},
        capture_output=True,
        text=True,
        timeout=60,
    )


def recorded(env: dict[str, str], name: str) -> str:
    path = Path(env["STUB_DIR"]) / name
    return path.read_text() if path.exists() else ""


def attempts(env: dict[str, str]) -> list[str]:
    """One line per server start: its arguments, then its NCCL_P2P_DISABLE."""
    return recorded(env, "vllm.attempts").splitlines()


def stub_pids(env: dict[str, str]) -> list[int]:
    return [int(p) for p in recorded(env, "vllm.pids").split()]


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


def survivors(env: dict[str, str]) -> list[int]:
    """Server stub processes still running, after a moment for the reaper."""
    deadline = time.monotonic() + 2
    while True:
        left = [pid for pid in stub_pids(env) if alive(pid)]
        if not left or time.monotonic() > deadline:
            return left
        time.sleep(0.05)


def test_a_budget_reaches_the_harness(box: dict[str, str]) -> None:
    result = run(box, MAX_TOKENS="250000")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--max-tokens 250000" in recorded(box, "uv.args")
    assert "max_tokens=250000" in result.stdout


def test_no_budget_means_no_flag(box: dict[str, str]) -> None:
    result = run(box)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lms.run" in recorded(box, "uv.args")
    assert "--max-tokens" not in recorded(box, "uv.args")
    assert "max_tokens=unlimited" in result.stdout


@pytest.mark.parametrize("bad", ["2M", "0", "-5", "1e6", "12 34"])
def test_a_bad_budget_stops_before_the_server_loads(
    box: dict[str, str], bad: str
) -> None:
    result = run(box, MAX_TOKENS=bad)
    assert result.returncode == 2
    assert "is not a positive whole number" in result.stdout
    assert recorded(box, "vllm.args") == ""


def test_the_server_gets_the_toolkit_free_kernels(box: dict[str, str]) -> None:
    run(box)
    assert recorded(box, "vllm.env").splitlines() == [
        "VLLM_ALLREDUCE_USE_FLASHINFER=0",
        "VLLM_USE_DEEP_GEMM=0",
        "VLLM_USE_FLASHINFER_SAMPLER=0",
    ]
    assert "--gdn-prefill-backend triton" in recorded(box, "vllm.args")


def test_a_kernel_setting_can_be_overridden(box: dict[str, str]) -> None:
    run(box, VLLM_USE_DEEP_GEMM="1", GDN_PREFILL_BACKEND="flashinfer")
    assert "VLLM_USE_DEEP_GEMM=1" in recorded(box, "vllm.env")
    assert "--gdn-prefill-backend flashinfer" in recorded(box, "vllm.args")


def test_extra_server_flags_are_passed_as_separate_words(box: dict[str, str]) -> None:
    run(box, LMS_EXTRA_VLLM_ARGS="--enforce-eager --seed 7")
    assert recorded(box, "vllm.args").strip().endswith("--enforce-eager --seed 7")


def test_a_server_that_never_answers_is_reported_with_the_real_limit(
    box: dict[str, str],
) -> None:
    result = run(box, STUB_SERVE_WHEN="never", READY_TIMEOUT="1")
    assert result.returncode == 1
    assert "FATAL: vLLM not ready after 1s" in result.stdout
    assert recorded(box, "uv.args") == ""


def test_the_gpu_request_uses_the_name_this_box_accepts() -> None:
    assert "#SBATCH --gres=gpu:nvidia_h100_nvl:4" in text()
    assert "gpu:h100:" not in text()


# --- The R9 arm's knob: RESET_EACH_GENERATION=1 -------------------------------


def test_the_reset_knob_reaches_the_harness(box: dict[str, str]) -> None:
    result = run(box, RESET_EACH_GENERATION="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--reset-each-generation" in recorded(box, "uv.args")
    assert "reset_each_generation=1" in result.stdout


@pytest.mark.parametrize("knobs", [{}, {"RESET_EACH_GENERATION": "0"}])
def test_no_reset_unless_the_knob_is_1(
    box: dict[str, str], knobs: dict[str, str]
) -> None:
    result = run(box, **knobs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lms.run" in recorded(box, "uv.args")
    assert "--reset-each-generation" not in recorded(box, "uv.args")
    assert "reset_each_generation=0" in result.stdout


def test_a_resumed_job_keeps_the_reset(box: dict[str, str]) -> None:
    """The checkpoint does not keep the flag; the resubmission passes it again."""
    out = Path(box["REPO"]) / "experiments" / box["RUN_TAG"]
    out.mkdir(parents=True)
    for name in ("results.json", "metadata.json", "foundation.json"):
        (out / name).write_text("{}")

    result = run(box, RESUME="1", RESET_EACH_GENERATION="1")

    assert result.returncode == 0, result.stdout + result.stderr
    args = recorded(box, "uv.args")
    assert "--resume" in args
    assert "--reset-each-generation" in args


@pytest.mark.parametrize("bad", ["true", "yes", "2", "on"])
def test_a_reset_knob_that_is_not_0_or_1_stops_before_the_server_loads(
    box: dict[str, str], bad: str
) -> None:
    """`true` silently running the carried arm would cost a whole R9 run."""
    result = run(box, RESET_EACH_GENERATION=bad)
    assert result.returncode == 2
    assert "RESET_EACH_GENERATION" in result.stdout
    assert recorded(box, "vllm.args") == ""


# --- A start that hangs: the ladder of start attempts -------------------------
#
# In batch q1, 4 of 18 TP=2 starts stopped after vLLM logged its NCCL version
# and never answered. A start that hangs is stopped, process group and all, and
# retried without custom all-reduce, then without NCCL peer-to-peer as well.

TODAYS_SERVE = (
    "serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000 --tensor-parallel-size 4"
    " --max-model-len 131072 --served-model-name lms-generalist"
    " --gdn-prefill-backend triton"
)


def test_a_healthy_server_starts_once_with_todays_command(box: dict[str, str]) -> None:
    result = run(box)
    assert result.returncode == 0, result.stdout + result.stderr
    assert attempts(box) == [f"{TODAYS_SERVE} | NCCL_P2P_DISABLE="]
    assert "SERVED_BY=base after ~" in result.stdout
    assert "vLLM ready after ~" in result.stdout
    assert survivors(box) == []


def test_a_start_that_hangs_is_retried_without_custom_all_reduce(
    box: dict[str, str],
) -> None:
    result = run(box, STUB_SERVE_WHEN="custom-ar-off", START_TIMEOUT="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert attempts(box) == [
        f"{TODAYS_SERVE} | NCCL_P2P_DISABLE=",
        f"{TODAYS_SERVE} --disable-custom-all-reduce | NCCL_P2P_DISABLE=",
    ]
    assert "SERVED_BY=no-custom-ar after ~" in result.stdout
    assert "lms.run" in recorded(box, "uv.args")
    # The hung attempt's worker was gone before the next attempt started.
    assert recorded(box, "overlap") == ""
    assert survivors(box) == []


def test_peer_to_peer_is_turned_off_only_on_the_third_rung(
    box: dict[str, str],
) -> None:
    result = run(box, STUB_SERVE_WHEN="p2p-off", START_TIMEOUT="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [line.rsplit("| ", 1)[1] for line in attempts(box)] == [
        "NCCL_P2P_DISABLE=",
        "NCCL_P2P_DISABLE=",
        "NCCL_P2P_DISABLE=1",
    ]
    assert attempts(box)[2].startswith(f"{TODAYS_SERVE} --disable-custom-all-reduce ")
    assert "SERVED_BY=no-p2p after ~" in result.stdout
    # The setting is the server's alone; the harness never sees it.
    assert recorded(box, "uv.env").splitlines() == ["NCCL_P2P_DISABLE="]
    assert recorded(box, "overlap") == ""
    assert survivors(box) == []


def test_every_rung_hanging_is_fatal_and_names_the_rungs(box: dict[str, str]) -> None:
    result = run(box, STUB_SERVE_WHEN="never", START_TIMEOUT="1")

    assert result.returncode == 1
    fatal = [line for line in result.stdout.splitlines() if line.startswith("FATAL")]
    assert len(fatal) == 1, result.stdout
    assert "after 1s" in fatal[0]
    assert "base no-custom-ar no-p2p" in fatal[0]
    assert len(attempts(box)) == 3
    assert recorded(box, "uv.args") == ""
    assert survivors(box) == []


def test_a_ladder_of_one_rung_is_todays_single_attempt(box: dict[str, str]) -> None:
    result = run(box, STUB_SERVE_WHEN="never", START_TIMEOUT="1", START_LADDER="base")

    assert result.returncode == 1
    assert attempts(box) == [f"{TODAYS_SERVE} | NCCL_P2P_DISABLE="]
    assert "FATAL: vLLM not ready after 1s" in result.stdout
    assert recorded(box, "uv.args") == ""


def test_a_server_that_dies_is_fatal_at_once(box: dict[str, str]) -> None:
    """A crash is a different failure from a hang; another rung would not help."""
    result = run(box, STUB_DIES="1", START_TIMEOUT="5")

    assert result.returncode == 1
    assert "FATAL: vLLM died during startup" in result.stdout
    assert len(attempts(box)) == 1
    assert recorded(box, "uv.args") == ""


def test_ready_timeout_still_sets_the_limit_when_start_timeout_is_unset(
    box: dict[str, str],
) -> None:
    result = run(box, STUB_SERVE_WHEN="never", READY_TIMEOUT="1", START_LADDER="base")
    assert "FATAL: vLLM not ready after 1s" in result.stdout


def test_start_timeout_wins_over_ready_timeout(box: dict[str, str]) -> None:
    result = run(
        box,
        STUB_SERVE_WHEN="never",
        READY_TIMEOUT="900",
        START_TIMEOUT="1",
        START_LADDER="base",
    )
    assert "FATAL: vLLM not ready after 1s" in result.stdout


@pytest.mark.parametrize(
    "knobs",
    [
        {"START_LADDER": "base fast"},
        {"START_LADDER": " "},
        {"START_TIMEOUT": "6m"},
        {"START_TIMEOUT": "0"},
    ],
)
def test_a_bad_ladder_setting_stops_before_the_server_loads(
    box: dict[str, str], knobs: dict[str, str]
) -> None:
    result = run(box, **knobs)
    assert result.returncode == 2
    assert next(iter(knobs)) in result.stdout
    assert attempts(box) == []


def test_each_attempt_has_its_own_log_and_vllm_log_is_the_one_that_served(
    box: dict[str, str],
) -> None:
    result = run(box, STUB_SERVE_WHEN="custom-ar-off", START_TIMEOUT="1")

    assert result.returncode == 0, result.stdout + result.stderr
    log_dir = Path(box["SCRATCH"]) / "lms-runs" / box["RUN_TAG"]
    assert (log_dir / "vllm.attempt-base.log").exists()
    served = log_dir / "vllm.attempt-no-custom-ar.log"
    assert served.exists()
    assert (log_dir / "vllm.log").exists()
    assert (log_dir / "vllm.log").samefile(served)


def test_a_worker_that_ignores_term_is_killed(box: dict[str, str]) -> None:
    """A process stuck inside NCCL does not run its signal handler."""
    result = run(
        box,
        STUB_SERVE_WHEN="custom-ar-off",
        STUB_IGNORES_TERM="1",
        START_TIMEOUT="1",
        STOP_GRACE="1",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SERVED_BY=no-custom-ar" in result.stdout
    assert recorded(box, "overlap") == ""
    assert survivors(box) == []


def test_gpus_still_held_after_a_hang_stop_the_ladder(box: dict[str, str]) -> None:
    """Starting the next rung on top of a leftover would fail for another reason."""
    result = run(
        box,
        STUB_SERVE_WHEN="never",
        STUB_HOLDS_GPU="1",
        START_TIMEOUT="1",
        GPU_FREE_TIMEOUT="1",
    )

    assert result.returncode == 1
    assert "FATAL: GPU(s) 0" in result.stdout
    assert len(attempts(box)) == 1
    assert recorded(box, "uv.args") == ""


# --- Which GPUs the job holds, and how they are linked ------------------------


def test_the_job_logs_its_physical_gpus_and_their_links(box: dict[str, str]) -> None:
    result = run(box, SLURM_JOB_ID="427")

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "GRES=gpu:nvidia_h100_nvl:2(IDX:0,2)" in out
    assert "NV12" in out
    assert out.index("IDX:0,2") < out.index("--- starting vLLM")
    assert out.index("NV12") < out.index("--- starting vLLM")


def test_the_gpu_logging_cannot_fail_the_job(box: dict[str, str]) -> None:
    result = run(box, SLURM_JOB_ID="427", STUB_SCONTROL_FAILS="1", STUB_TOPO_FAILS="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "lms.run" in recorded(box, "uv.args")
