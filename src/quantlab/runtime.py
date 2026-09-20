"""Process liveness primitives shared by the worker and scheduler containers.

Docker healthchecks need a cheap, dependency-free answer to "is this process
actually doing work?". A long-running worker that has deadlocked still has a live
PID, so PID-based checks are worthless. Instead each long-running process writes a
heartbeat file into the state directory and the healthcheck asserts its freshness.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Heartbeat", "HeartbeatStatus", "read_heartbeat"]


@dataclass(frozen=True, slots=True)
class HeartbeatStatus:
    name: str
    path: Path
    exists: bool
    age_seconds: float | None
    stale_after_seconds: float
    detail: dict[str, Any]

    @property
    def healthy(self) -> bool:
        return (
            self.exists
            and self.age_seconds is not None
            and self.age_seconds <= self.stale_after_seconds
        )

    def describe(self) -> str:
        if not self.exists or self.age_seconds is None:
            return f"{self.name}: no heartbeat at {self.path}"
        state = "ok" if self.healthy else "STALE"
        return f"{self.name}: {state} (last beat {self.age_seconds:.0f}s ago)"


class Heartbeat:
    """A file whose mtime and contents record that a process is alive and working.

    Written atomically (write to a temp file, then rename) so a healthcheck can
    never observe a half-written file.
    """

    def __init__(self, state_dir: Path, name: str, stale_after_seconds: float = 120.0) -> None:
        self.name = name
        self.path = state_dir / f"{name}.heartbeat.json"
        self.stale_after_seconds = stale_after_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(self, **detail: Any) -> None:
        payload = {
            "name": self.name,
            "pid": os.getpid(),
            "timestamp": time.time(),
            "stale_after_seconds": self.stale_after_seconds,
            "detail": detail,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.path)

    def status(self) -> HeartbeatStatus:
        return read_heartbeat(self.path, self.name, self.stale_after_seconds)


def read_heartbeat(
    path: Path, name: str | None = None, stale_after_seconds: float | None = None
) -> HeartbeatStatus:
    """Read a heartbeat file without importing anything the writer used."""
    resolved_name = name or path.stem.split(".")[0]
    if not path.exists():
        return HeartbeatStatus(
            name=resolved_name,
            path=path,
            exists=False,
            age_seconds=None,
            stale_after_seconds=stale_after_seconds or 120.0,
            detail={},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt heartbeat is treated as no heartbeat rather than crashing the
        # healthcheck, but it is never treated as healthy.
        payload = {}
    timestamp = float(payload.get("timestamp", 0.0))
    threshold = (
        stale_after_seconds
        if stale_after_seconds is not None
        else float(payload.get("stale_after_seconds", 120.0))
    )
    return HeartbeatStatus(
        name=resolved_name,
        path=path,
        exists=True,
        age_seconds=max(0.0, time.time() - timestamp),
        stale_after_seconds=threshold,
        detail=dict(payload.get("detail", {})),
    )
