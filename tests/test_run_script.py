"""The batch script serves the model with kernels this node can run.

vLLM 0.33.0 died at load on `find_cuda_toolkit`, because it compiles its DeepGEMM
and FlashInfer kernels and the node has no CUDA toolkit on its default paths. R6,
R7 and every Phase C arm use this script, so they would fail the same way.
"""

from __future__ import annotations

import os
import stat
import subprocess
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
    # The server: records what it was given, then stays up until it is killed.
    "vllm": """#!/bin/bash
printf '%s\\n' "$*" > "$STUB_DIR/vllm.args"
env | grep '^VLLM_' | sort > "$STUB_DIR/vllm.env"
[ -n "${STUB_NEVER_READY:-}" ] || touch "$STUB_DIR/serving"
exec tail -f /dev/null
""",
    # Answers only once the server stub is up, so the port check before it passes.
    "curl": """#!/bin/bash
[ -e "$STUB_DIR/serving" ]
""",
    "uv": """#!/bin/bash
printf '%s\\n' "$*" >> "$STUB_DIR/uv.args"
""",
    "lake": "#!/bin/bash\n",
    "nvidia-smi": "#!/bin/bash\n",
    "sleep": "#!/bin/bash\n",
}


@pytest.fixture
def box(tmp_path: Path) -> dict[str, str]:
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
    return env


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
    result = run(box, STUB_NEVER_READY="1", READY_TIMEOUT="1")
    assert result.returncode == 1
    assert "FATAL: vLLM not ready after 1s" in result.stdout
    assert recorded(box, "uv.args") == ""


def test_the_gpu_request_uses_the_name_this_box_accepts() -> None:
    assert "#SBATCH --gres=gpu:nvidia_h100_nvl:4" in text()
    assert "gpu:h100:" not in text()
