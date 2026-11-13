"""The queue kit, run on a fake server.

`scripts/slurm/queue/` lets rounds pinned to different commits wait in the Slurm
queue together: every job waits for the LMS job before it, and checks out its own
commit when it starts. The checks live in bash, beside the fake server they need
(`tests/slurm_queue/`); this runs them so CI does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
KIT = ROOT / "scripts" / "slurm" / "queue"
HARNESS = ROOT / "tests" / "slurm_queue"

pytestmark = pytest.mark.skipif(
    shutil.which("flock") is None or shutil.which("git") is None,
    reason="needs flock and git",
)


def run(script: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "FB_KIT": str(KIT),
        "FB_HARNESS": str(HARNESS / "fakebox.sh"),
        "LMS_ALL_RUN": str(KIT / "lms_all_run.sh"),
    }
    return subprocess.run(
        ["bash", str(HARNESS / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize(
    "script", ["lms_q_lib.sh", "lms_q_job.sbatch", "lms_all_run.sh"]
)
def test_the_kit_parses(script: str) -> None:
    subprocess.run(["bash", "-n", str(KIT / script)], check=True)


def test_a_round_queues_in_one_line_and_each_job_runs_its_own_commit() -> None:
    result = run("test_kit.sh")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "failed: 0" in result.stdout


def test_a_pin_check_finds_what_is_there_under_pipefail() -> None:
    result = run("test_pin_has.sh")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "failed: 0" in result.stdout


def test_the_master_runner_tries_every_round_and_reports_each() -> None:
    result = run("test_all.sh")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "failed: 0" in result.stdout
