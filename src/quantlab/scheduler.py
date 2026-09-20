"""Cron-equivalent for daily ingest and paper-trading runs (spec section 10).

APScheduler rather than a cron binary, for two reasons: it is multi-architecture by
construction (no per-arch static binary to fetch), and jobs run in-process where
structured logging and the heartbeat already work.

Jobs are declared in `configs/schedule.yaml` and executed as subprocesses of the
`quantlab` CLI, so a crashing job cannot take the scheduler down with it.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from quantlab.config import get_settings
from quantlab.logging import get_logger
from quantlab.runtime import Heartbeat

__all__ = ["ScheduledJob", "load_schedule", "run_scheduler"]

log = get_logger("quantlab.scheduler")


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    name: str
    cron: str
    command: list[str]
    enabled: bool = True
    timeout_seconds: int = 3600
    description: str = ""

    @property
    def trigger(self) -> CronTrigger:
        # Timezone is fixed to UTC: market-time conversions belong in the trading
        # calendars, not in the scheduler, where a DST shift would silently move a
        # job across a data release.
        return CronTrigger.from_crontab(self.cron, timezone="UTC")


def load_schedule(path: Path) -> list[ScheduledJob]:
    """Parse configs/schedule.yaml into jobs, failing loudly on a malformed entry."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("jobs", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: `jobs` must be a list, got {type(entries).__name__}")

    jobs: list[ScheduledJob] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: job #{index} must be a mapping")
        missing = {"name", "cron", "command"} - entry.keys()
        if missing:
            raise ValueError(f"{path}: job #{index} is missing {sorted(missing)}")
        command = entry["command"]
        if isinstance(command, str):
            command = command.split()
        job = ScheduledJob(
            name=str(entry["name"]),
            cron=str(entry["cron"]),
            command=[str(part) for part in command],
            enabled=bool(entry.get("enabled", True)),
            timeout_seconds=int(entry.get("timeout_seconds", 3600)),
            description=str(entry.get("description", "")),
        )
        job.trigger  # validate the cron expression now, not at 03:00  # noqa: B018
        jobs.append(job)
    return jobs


def _run_job(job: ScheduledJob, heartbeat: Heartbeat) -> None:
    argv = [sys.executable, "-m", "quantlab", *job.command]
    log.info("job.start", job=job.name, command=job.command)
    heartbeat.beat(running=job.name)
    try:
        # argv is built entirely from our own schedule config and sys.executable;
        # no shell is involved and nothing from the network reaches it.
        completed = subprocess.run(argv, timeout=job.timeout_seconds, check=False)  # noqa: S603
    except subprocess.TimeoutExpired:
        log.error("job.timeout", job=job.name, timeout_seconds=job.timeout_seconds)
        heartbeat.beat(last_job=job.name, last_result="timeout")
        return
    level = log.info if completed.returncode == 0 else log.error
    level("job.finish", job=job.name, returncode=completed.returncode)
    heartbeat.beat(last_job=job.name, last_result=completed.returncode)


def run_scheduler(config_path: Path | None = None) -> None:
    """Block forever, running the configured jobs. Entry point of the `scheduler` service."""
    settings = get_settings()
    path = config_path or (settings.layout.configs / "schedule.yaml")
    jobs = load_schedule(path)

    heartbeat = Heartbeat(settings.layout.state, "scheduler", stale_after_seconds=180.0)
    heartbeat.beat(jobs=[j.name for j in jobs if j.enabled])

    scheduler = BlockingScheduler(timezone="UTC")
    # A liveness tick independent of the jobs, so the healthcheck distinguishes
    # "idle between jobs" from "scheduler is wedged".
    scheduler.add_job(
        lambda: heartbeat.beat(state="idle"),
        "interval",
        seconds=60,
        id="_heartbeat",
        max_instances=1,
    )
    for job in jobs:
        if not job.enabled:
            log.info("job.disabled", job=job.name)
            continue
        scheduler.add_job(
            _run_job,
            trigger=job.trigger,
            args=(job, heartbeat),
            id=job.name,
            name=job.description or job.name,
            max_instances=1,
            coalesce=True,  # a missed window runs once, not N times
            misfire_grace_time=1800,
        )
        log.info("job.scheduled", job=job.name, cron=job.cron)

    # APScheduler does not handle SIGTERM itself, so without this the container is
    # killed outright and exits 143 on every `docker compose down`, losing the
    # chance to let a running job finish writing.
    def _stop(signum: int, _frame: FrameType | None) -> None:  # pragma: no cover
        log.info("scheduler.signal", signal=signal.Signals(signum).name)
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log.info("scheduler.start", config=str(path), jobs=len(scheduler.get_jobs()) - 1)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - signal path
        scheduler.shutdown(wait=False)
    log.info("scheduler.stop")
