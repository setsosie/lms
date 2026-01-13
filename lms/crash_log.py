"""Crash recovery logging for LMS experiments.

Provides an append-only JSONL log for post-mortem analysis of experiment runs.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CrashRecoveryLog:
    """Append-only log for crash recovery analysis.

    Logs structured events to a JSONL file for post-mortem debugging.
    Events are appended, never overwritten, ensuring no data loss on crash.

    Usage:
        log = CrashRecoveryLog(output_dir)
        log.log_generation_start(gen=0, mode="standard")
        # ... run generation ...
        log.log_generation_end(gen=0, result={"verified": 3, "created": 5})
    """

    path: Path

    @property
    def log_file(self) -> Path:
        """Path to the JSONL log file."""
        return self.path / "crash_recovery.jsonl"

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a recovery-relevant event.

        Args:
            event_type: Type of event (e.g., "generation_start", "checkpoint")
            data: Event-specific data
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data,
        }
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_generation_start(self, gen: int, mode: str = "standard") -> None:
        """Log the start of a generation.

        Args:
            gen: Generation number
            mode: Run mode (standard, iterative, working_groups)
        """
        self.log_event("generation_start", {"generation": gen, "mode": mode})

    def log_generation_end(
        self,
        gen: int,
        artifacts_created: int = 0,
        artifacts_verified: int = 0,
        tokens_used: int = 0,
    ) -> None:
        """Log the end of a generation.

        Args:
            gen: Generation number
            artifacts_created: Number of artifacts created
            artifacts_verified: Number of artifacts verified
            tokens_used: Tokens consumed
        """
        self.log_event(
            "generation_end",
            {
                "generation": gen,
                "artifacts_created": artifacts_created,
                "artifacts_verified": artifacts_verified,
                "tokens_used": tokens_used,
            },
        )

    def log_working_group_start(self, group_id: int, task_tag: str) -> None:
        """Log the start of a working group session.

        Args:
            group_id: Working group ID
            task_tag: Task tag being worked on
        """
        self.log_event("working_group_start", {"group_id": group_id, "task": task_tag})

    def log_working_group_end(
        self,
        group_id: int,
        task_tag: str,
        success: bool,
        turns: int = 0,
    ) -> None:
        """Log the end of a working group session.

        Args:
            group_id: Working group ID
            task_tag: Task tag being worked on
            success: Whether the group produced a verified artifact
            turns: Number of discussion turns
        """
        self.log_event(
            "working_group_end",
            {
                "group_id": group_id,
                "task": task_tag,
                "success": success,
                "turns": turns,
            },
        )

    def log_verification(
        self,
        artifact_id: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Log a verification attempt.

        Args:
            artifact_id: ID of artifact being verified
            success: Whether verification succeeded
            error: Error message if verification failed
        """
        self.log_event(
            "verification",
            {"artifact": artifact_id, "success": success, "error": error},
        )

    def log_checkpoint(self, gen: int) -> None:
        """Log a checkpoint save.

        Args:
            gen: Generation number at checkpoint
        """
        self.log_event("checkpoint", {"generation": gen, "path": str(self.path)})

    def log_error(self, error_type: str, message: str, gen: int | None = None) -> None:
        """Log an error event.

        Args:
            error_type: Type of error (e.g., "budget_exceeded", "api_error")
            message: Error message
            gen: Optional generation number
        """
        data: dict[str, Any] = {"error_type": error_type, "message": message}
        if gen is not None:
            data["generation"] = gen
        self.log_event("error", data)

    def log_shutdown(self, reason: str, gen: int) -> None:
        """Log a shutdown event.

        Args:
            reason: Reason for shutdown (e.g., "signal", "complete", "error")
            gen: Last generation completed
        """
        self.log_event("shutdown", {"reason": reason, "last_generation": gen})

    def read_events(self) -> list[dict[str, Any]]:
        """Read all events from the log.

        Returns:
            List of event dictionaries
        """
        if not self.log_file.exists():
            return []

        events = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def get_last_generation(self) -> int | None:
        """Get the last completed generation from the log.

        Returns:
            Last completed generation number, or None if no generations completed
        """
        events = self.read_events()
        for event in reversed(events):
            if event["event"] == "generation_end":
                return event["data"]["generation"]
        return None
