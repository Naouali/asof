from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quantlab import __version__
from quantlab.cli import app
from quantlab.config import Settings

# A wide terminal, so Rich does not elide the table cells under assertion.
runner = CliRunner(env={"COLUMNS": "200"})


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
    result = runner.invoke(app, ["data", "catalogue", "yahoo"])
    assert result.exit_code == 0
    assert "caveats" in result.stdout
    assert "SURVIVORSHIP" in result.stdout


def test_worker_healthcheck_fails_without_a_heartbeat() -> None:
    assert runner.invoke(app, ["worker", "healthcheck"]).exit_code == 1


def test_scheduler_list(repo_root: Path) -> None:
    result = runner.invoke(
        app, ["scheduler", "list", "--config", str(repo_root / "configs" / "schedule.yaml")]
    )
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_data_fetchers_reports_readiness() -> None:
    result = runner.invoke(app, ["data", "fetchers"])
    assert result.exit_code == 0
    assert "binance.ohlcv_bars" in result.stdout
    assert "alfred.series_observations" in result.stdout
    # FRED needs a key and must be reported as not ready, not hidden.
    assert "QUANTLAB_FRED_API_KEY" in result.stdout
    # Stooq is blocked by an anti-bot challenge, which no credential fixes. A table
    # that called it "ready" would be lying.
    assert "anti-bot" in result.stdout


def test_data_status_on_an_empty_lake() -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["data", "status"])
    assert result.exit_code == 0
    assert "empty" in result.stdout


def test_data_status_reports_the_lake(populated_store: object) -> None:
    result = runner.invoke(app, ["data", "status"])
    assert result.exit_code == 0
    assert "ohlcv_daily" in result.stdout
    assert "knowable" in result.stdout


def test_ingest_dry_run_shows_windows_without_fetching(repo_root: Path) -> None:
    result = runner.invoke(
        app,
        ["data", "ingest", "--dry-run", "--config", str(repo_root / "configs" / "ingest.yaml")],
    )
    assert result.exit_code == 0
    assert "dry run" in result.stdout


def test_ingest_with_a_missing_plan_fails_loudly() -> None:
    result = runner.invoke(app, ["data", "ingest", "--config", "/nope/ingest.yaml"])
    assert result.exit_code == 2
    assert "no ingest plan" in result.output


def test_query_requires_datasets_when_point_in_time(populated_store: object) -> None:
    """The sandbox holds nothing by default, so a point-in-time query cannot
    accidentally reach a dataset the caller did not think about."""
    result = runner.invoke(app, ["data", "query", "--as-of", "2024-01-04", "select 1"])
    assert result.exit_code == 2
    assert "--dataset" in result.output


def test_query_point_in_time_hides_the_future(populated_store: object) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "query",
            "--as-of",
            "2024-01-04",
            "-d",
            "ohlcv_daily",
            "select max(close) as mx from ohlcv_daily where symbol = 'AAPL'",
        ],
    )
    assert result.exit_code == 0
    assert "101.0" in result.stdout
    assert "999" not in result.stdout


def test_query_without_as_of_warns_it_is_not_research_grade(populated_store: object) -> None:
    result = runner.invoke(app, ["data", "query", "select count(*) as n from ohlcv_daily"])
    assert result.exit_code == 0
    assert "not point-in-time" in result.stdout
