from __future__ import annotations

import json
import time
from pathlib import Path

from quantlab.runtime import Heartbeat, read_heartbeat


def test_heartbeat_roundtrip(tmp_path: Path) -> None:
    hb = Heartbeat(tmp_path, "worker", stale_after_seconds=60)
    hb.beat(state="idle")
    status = hb.status()
    assert status.healthy
    assert status.detail == {"state": "idle"}
    assert status.age_seconds is not None and status.age_seconds < 5


def test_missing_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    status = read_heartbeat(tmp_path / "nothing.heartbeat.json", "worker")
    assert not status.healthy
    assert not status.exists


def test_stale_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    """A deadlocked process keeps its PID but stops beating; this is the case that
    makes a heartbeat healthcheck worth more than a PID check."""
    hb = Heartbeat(tmp_path, "worker", stale_after_seconds=1)
    payload = {
        "name": "worker",
        "pid": 1,
        "timestamp": time.time() - 3600,
        "stale_after_seconds": 1,
        "detail": {},
    }
    hb.path.write_text(json.dumps(payload), encoding="utf-8")
    status = hb.status()
    assert status.exists
    assert not status.healthy
    assert "STALE" in status.describe()


def test_corrupt_heartbeat_is_never_reported_healthy(tmp_path: Path) -> None:
    hb = Heartbeat(tmp_path, "worker", stale_after_seconds=60)
    hb.path.write_text("{not json", encoding="utf-8")
    assert not hb.status().healthy


def test_beat_is_atomic(tmp_path: Path) -> None:
    """The healthcheck must never read a half-written file."""
    hb = Heartbeat(tmp_path, "worker")
    for _ in range(20):
        hb.beat(n=1)
        json.loads(hb.path.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp"))
