"""Tests for crash recovery logging."""

import json
import tempfile
from pathlib import Path

from lms.crash_log import CrashRecoveryLog


class TestCrashRecoveryLog:
    """Tests for CrashRecoveryLog."""

    def test_log_event(self, tmp_path: Path):
        """Test basic event logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_event("test", {"key": "value"})

        events = log.read_events()
        assert len(events) == 1
        assert events[0]["event"] == "test"
        assert events[0]["data"]["key"] == "value"
        assert "timestamp" in events[0]

    def test_log_generation_start(self, tmp_path: Path):
        """Test generation start logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_generation_start(gen=0, mode="standard")

        events = log.read_events()
        assert events[0]["event"] == "generation_start"
        assert events[0]["data"]["generation"] == 0
        assert events[0]["data"]["mode"] == "standard"

    def test_log_generation_end(self, tmp_path: Path):
        """Test generation end logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_generation_end(
            gen=0, artifacts_created=5, artifacts_verified=2, tokens_used=1000
        )

        events = log.read_events()
        assert events[0]["event"] == "generation_end"
        assert events[0]["data"]["generation"] == 0
        assert events[0]["data"]["artifacts_created"] == 5
        assert events[0]["data"]["artifacts_verified"] == 2
        assert events[0]["data"]["tokens_used"] == 1000

    def test_log_checkpoint(self, tmp_path: Path):
        """Test checkpoint logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_checkpoint(gen=10)

        events = log.read_events()
        assert events[0]["event"] == "checkpoint"
        assert events[0]["data"]["generation"] == 10

    def test_log_error(self, tmp_path: Path):
        """Test error logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_error("budget_exceeded", "Token limit reached", gen=5)

        events = log.read_events()
        assert events[0]["event"] == "error"
        assert events[0]["data"]["error_type"] == "budget_exceeded"
        assert events[0]["data"]["message"] == "Token limit reached"
        assert events[0]["data"]["generation"] == 5

    def test_log_shutdown(self, tmp_path: Path):
        """Test shutdown logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_shutdown("signal", gen=15)

        events = log.read_events()
        assert events[0]["event"] == "shutdown"
        assert events[0]["data"]["reason"] == "signal"
        assert events[0]["data"]["last_generation"] == 15

    def test_append_only(self, tmp_path: Path):
        """Test that events are appended, not overwritten."""
        log = CrashRecoveryLog(tmp_path)
        log.log_generation_start(0, "standard")
        log.log_generation_end(0, 5, 2, 1000)
        log.log_checkpoint(0)

        events = log.read_events()
        assert len(events) == 3
        assert events[0]["event"] == "generation_start"
        assert events[1]["event"] == "generation_end"
        assert events[2]["event"] == "checkpoint"

    def test_get_last_generation(self, tmp_path: Path):
        """Test getting last completed generation."""
        log = CrashRecoveryLog(tmp_path)
        log.log_generation_start(0, "standard")
        log.log_generation_end(0, 5, 2, 1000)
        log.log_generation_start(1, "standard")
        log.log_generation_end(1, 3, 1, 800)

        assert log.get_last_generation() == 1

    def test_get_last_generation_empty(self, tmp_path: Path):
        """Test getting last generation from empty log."""
        log = CrashRecoveryLog(tmp_path)
        assert log.get_last_generation() is None

    def test_get_last_generation_no_ends(self, tmp_path: Path):
        """Test getting last generation when no generation_end events."""
        log = CrashRecoveryLog(tmp_path)
        log.log_generation_start(0, "standard")

        assert log.get_last_generation() is None

    def test_read_events_nonexistent_file(self, tmp_path: Path):
        """Test reading from nonexistent log file."""
        log = CrashRecoveryLog(tmp_path)
        events = log.read_events()
        assert events == []

    def test_log_working_group_start(self, tmp_path: Path):
        """Test working group start logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_working_group_start(group_id=1, task_tag="CH4-FUNCTOR")

        events = log.read_events()
        assert events[0]["event"] == "working_group_start"
        assert events[0]["data"]["group_id"] == 1
        assert events[0]["data"]["task"] == "CH4-FUNCTOR"

    def test_log_working_group_end(self, tmp_path: Path):
        """Test working group end logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_working_group_end(
            group_id=1, task_tag="CH4-FUNCTOR", success=True, turns=3
        )

        events = log.read_events()
        assert events[0]["event"] == "working_group_end"
        assert events[0]["data"]["group_id"] == 1
        assert events[0]["data"]["task"] == "CH4-FUNCTOR"
        assert events[0]["data"]["success"] is True
        assert events[0]["data"]["turns"] == 3

    def test_log_verification(self, tmp_path: Path):
        """Test verification logging."""
        log = CrashRecoveryLog(tmp_path)
        log.log_verification("artifact-123", success=False, error="Type mismatch")

        events = log.read_events()
        assert events[0]["event"] == "verification"
        assert events[0]["data"]["artifact"] == "artifact-123"
        assert events[0]["data"]["success"] is False
        assert events[0]["data"]["error"] == "Type mismatch"
