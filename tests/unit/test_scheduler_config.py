from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.scheduler import load_schedule


def test_repo_schedule_parses(repo_root: Path) -> None:
    jobs = load_schedule(repo_root / "configs" / "schedule.yaml")
    assert jobs
    assert {"ingest-daily", "paper-trading", "doctor"} <= {j.name for j in jobs}


def test_every_scheduled_command_is_a_real_cli_command(repo_root: Path) -> None:
    """A typo in schedule.yaml would otherwise surface at 05:30 as a subprocess exit
    code nobody is watching."""
    import typer.testing

    from quantlab.cli import app

    runner = typer.testing.CliRunner()
    for job in load_schedule(repo_root / "configs" / "schedule.yaml"):
        result = runner.invoke(app, [*job.command, "--help"])
        assert result.exit_code == 0, f"{job.name}: `{' '.join(job.command)}` is not a command"


def test_only_implemented_jobs_are_enabled(repo_root: Path) -> None:
    """Enabled jobs must actually work today; a job that exits 2 every night trains
    the operator to ignore scheduler failures."""
    jobs = load_schedule(repo_root / "configs" / "schedule.yaml")
    enabled = {j.name for j in jobs if j.enabled}
    assert enabled == {"doctor", "ingest-daily", "ingest-crypto", "options-snapshot"}
    # Paper trading lands in Milestone 11.
    assert {j.name for j in jobs if not j.enabled} == {"paper-trading"}


def test_the_options_collector_is_enabled_because_it_cannot_be_backfilled(
    repo_root: Path,
) -> None:
    """Every other job here can be re-run from the beginning. No free source
    sells historical option chains, so this one's history starts the day it first
    runs and every day it is off is a day that can never be recovered. Leaving it
    disabled is not a neutral default -- it is a decision to have no options data
    for however long it stays off."""
    jobs = {j.name: j for j in load_schedule(repo_root / "configs" / "schedule.yaml")}
    snapshot = jobs["options-snapshot"]

    assert snapshot.enabled
    assert "options_snapshot.chain_snapshot" in snapshot.command
    assert snapshot.cron.split()[4] == "1-5", "chains only change on trading days"


def test_crypto_universe_snapshot_runs_often(repo_root: Path) -> None:
    """binance.instruments is the only record of which symbols existed when.
    Running it rarely loses universe history permanently."""
    jobs = {j.name: j for j in load_schedule(repo_root / "configs" / "schedule.yaml")}
    crypto = jobs["ingest-crypto"]
    assert crypto.enabled
    assert "binance.instruments" in crypto.command
    assert crypto.cron.split()[1] == "*", "must run at least hourly"


def test_invalid_cron_fails_at_load_not_at_runtime(tmp_path: Path) -> None:
    path = tmp_path / "schedule.yaml"
    path.write_text("jobs:\n  - name: bad\n    cron: 'not a cron'\n    command: ['doctor']\n")
    with pytest.raises(ValueError):
        load_schedule(path)


def test_missing_fields_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "schedule.yaml"
    path.write_text("jobs:\n  - name: incomplete\n")
    with pytest.raises(ValueError, match="missing"):
        load_schedule(path)


def test_string_command_is_split(tmp_path: Path) -> None:
    path = tmp_path / "schedule.yaml"
    path.write_text("jobs:\n  - name: j\n    cron: '0 6 * * *'\n    command: data ingest\n")
    assert load_schedule(path)[0].command == ["data", "ingest"]


def test_empty_schedule_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "schedule.yaml"
    path.write_text("jobs: []\n")
    assert load_schedule(path) == []


def test_job_execution_runs_the_cli_and_records_the_result(tmp_path: Path) -> None:
    """The scheduler shells out to `python -m quantlab`, so a crashing job cannot
    take the scheduler down with it. This exercises that path for real."""
    from quantlab.runtime import Heartbeat
    from quantlab.scheduler import ScheduledJob, _run_job

    heartbeat = Heartbeat(tmp_path, "scheduler", stale_after_seconds=60)

    ok = ScheduledJob(name="ok", cron="0 6 * * *", command=["version"])
    _run_job(ok, heartbeat)
    assert heartbeat.status().detail == {"last_job": "ok", "last_result": 0}

    # A failing job is recorded, not swallowed, and the scheduler survives it.
    # `backtest` exits 2 immediately (Milestone 4), which makes it a fast, offline
    # stand-in for any job that fails.
    failing = ScheduledJob(
        name="failing", cron="0 6 * * *", command=["backtest", "--config", "nope.yaml"]
    )
    _run_job(failing, heartbeat)
    assert heartbeat.status().detail == {"last_job": "failing", "last_result": 2}


def test_job_timeout_is_recorded(tmp_path: Path) -> None:
    from quantlab.runtime import Heartbeat
    from quantlab.scheduler import ScheduledJob, _run_job

    heartbeat = Heartbeat(tmp_path, "scheduler", stale_after_seconds=60)
    job = ScheduledJob(
        name="slow",
        cron="0 6 * * *",
        command=["worker", "run", "--interval", "60"],
        timeout_seconds=1,
    )
    _run_job(job, heartbeat)
    assert heartbeat.status().detail == {"last_job": "slow", "last_result": "timeout"}
