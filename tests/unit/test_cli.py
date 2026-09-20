from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantlab import __version__
from quantlab.cli import app
from quantlab.config import Settings

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_init_creates_the_lake(settings: Settings) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    for directory in settings.layout.all_data_dirs():
        assert directory.is_dir()


def test_doctor_succeeds_on_a_bare_install() -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout


def test_doctor_strict_reports_missing_keys() -> None:
    """Without keys there are warnings; --strict is how CI asks for a fully
    configured installation."""
    runner.invoke(app, ["init"])
    assert runner.invoke(app, ["doctor", "--strict"]).exit_code == 1


def test_catalogue_lists_sources() -> None:
    result = runner.invoke(app, ["data", "catalogue"])
    assert result.exit_code == 0
    assert "binance" in result.stdout
    assert "sec_edgar" in result.stdout


def test_catalogue_detail_shows_caveats() -> None:
    result = runner.invoke(app, ["data", "catalogue", "yfinance"])
    assert result.exit_code == 0
    assert "caveats" in result.stdout
    assert "SURVIVORSHIP" in result.stdout


@pytest.mark.parametrize(
    ("argv", "milestone"),
    [
        (["data", "ingest"], "Milestone 2"),
        (["data", "ingest", "--incremental"], "Milestone 2"),
        (["backtest", "--config", "configs/x.yaml"], "Milestone 4"),
        (["paper"], "Milestone 11"),
    ],
)
def test_unimplemented_commands_fail_loudly(argv: list[str], milestone: str) -> None:
    """Spec section 13: never quietly return an empty result. An unimplemented
    command exits non-zero and names the milestone that will implement it."""
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert milestone in result.output


def test_worker_healthcheck_fails_without_a_heartbeat() -> None:
    assert runner.invoke(app, ["worker", "healthcheck"]).exit_code == 1


def test_scheduler_list(repo_root: Path) -> None:
    result = runner.invoke(
        app, ["scheduler", "list", "--config", str(repo_root / "configs" / "schedule.yaml")]
    )
    assert result.exit_code == 0
    assert "doctor" in result.stdout
