from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
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


@pytest.mark.parametrize(
    ("argv", "milestone"),
    [
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


def test_costs_estimate_decomposes_the_cost() -> None:
    result = runner.invoke(
        app,
        ["costs", "estimate", "-n", "2e8", "--adv", "8e9", "--vol", "0.018", "--spread", "1.2"],
    )
    assert result.exit_code == 0
    for component in ("spread", "impact paid", "temporary", "permanent", "commission", "total"):
        assert component in result.stdout
    assert "participation" in result.stdout


def test_costs_estimate_flags_extrapolation() -> None:
    """Beyond ~10% of ADV the square-root law understates; the CLI must say so
    rather than printing a confident number."""
    result = runner.invoke(
        app, ["costs", "estimate", "-n", "4e9", "--adv", "8e9", "--spread", "1.2"]
    )
    assert result.exit_code == 0
    assert "likely HIGHER" in result.stdout


def test_costs_estimate_can_show_what_a_flat_model_would_claim() -> None:
    result = runner.invoke(
        app,
        [
            "costs",
            "estimate",
            "-n",
            "2e9",
            "--adv",
            "8e9",
            "--spread",
            "1.2",
            "--compare-flat",
            "10",
        ],
    )
    assert result.exit_code == 0
    assert "understating" in result.stdout
    assert "unlimited capacity" in result.stdout


def test_costs_estimate_rejects_an_unknown_asset_class() -> None:
    result = runner.invoke(
        app, ["costs", "estimate", "-n", "1e6", "--adv", "1e9", "--asset-class", "tulips"]
    )
    assert result.exit_code == 2
    assert "unknown asset class" in result.output


def test_costs_capacity_reports_a_break_even() -> None:
    result = runner.invoke(
        app, ["costs", "capacity", "--alpha", "25", "--turnover", "0.4", "--names", "200"]
    )
    assert result.exit_code == 0
    assert "break-even AUM" in result.stdout
    assert "gross bp/yr" in result.stdout


def test_costs_capacity_fails_loudly_when_a_strategy_never_pays() -> None:
    """Not a small number: reporting one would imply it works if kept tiny."""
    result = runner.invoke(
        app, ["costs", "capacity", "--alpha", "0.2", "--turnover", "0.8", "--spread", "20"]
    )
    assert result.exit_code == 1
    assert "no viable capacity" in result.output


def test_backtest_without_a_config_fails_loudly() -> None:
    result = runner.invoke(app, ["backtest", "--config", "/nope/run.yaml"])
    assert result.exit_code == 2
    assert "no run config" in result.output


def test_backtest_without_weights_explains_what_is_missing(tmp_path: Path, repo_root: Path) -> None:
    """Until Milestone 6 generates weights from signals, they come from a file.
    The error has to say so rather than just reporting a missing path."""
    config = tmp_path / "run.yaml"
    config.write_text("name: t\nas_of: '2024-01-05'\nsymbols: [AAPL]\nweights: missing.parquet\n")
    result = runner.invoke(app, ["backtest", "--config", str(config)])
    assert result.exit_code == 2
    assert "Milestone 6" in result.output


def test_backtest_on_an_empty_lake_says_to_ingest(tmp_path: Path) -> None:
    runner.invoke(app, ["init"])
    weights = tmp_path / "w.parquet"
    pl.DataFrame(
        {"symbol": ["AAPL"], "as_of": [dt.datetime(2024, 1, 4, tzinfo=dt.UTC)], "weight": [1.0]}
    ).write_parquet(weights)
    config = tmp_path / "run.yaml"
    config.write_text(f"name: t\nas_of: '2024-01-05'\nsymbols: [AAPL]\nweights: {weights.name}\n")
    result = runner.invoke(app, ["backtest", "--config", str(config)])
    assert result.exit_code == 2
    assert "data ingest" in result.output


def test_backtest_runs_end_to_end(tmp_path: Path, store: object) -> None:
    """Lake to snapshot to panel to ledger, through the CLI.

    Builds its own 60-session lake rather than reusing the three-bar fixture: the
    trailing liquidity statistics need a warm-up, so a three-bar panel has nothing
    left to trade and the smoke test would prove only that nothing crashed.
    """
    import numpy as np
    from tests.conftest import bar

    rng = np.random.default_rng(3)
    sessions = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(60)]
    rows = []
    for symbol, drift in (("AAPL", 0.001), ("MSFT", -0.0005)):
        price = 100.0
        for session in sessions:
            price *= float(np.exp(rng.normal(drift, 0.01)))
            row = bar(symbol, session, price)
            row["volume"] = 5_000_000.0
            rows.append(row)
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(rows).drop("volume").with_columns(volume=pl.lit(5_000_000.0)),
        asset_class="equity",
    )

    weights = tmp_path / "w.parquet"
    pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"] * 2,
            "as_of": [
                dt.datetime(s.year, s.month, s.day, 21, tzinfo=dt.UTC)
                for s in (sessions[30], sessions[30], sessions[45], sessions[45])
            ],
            "weight": [0.5, -0.5, -0.5, 0.5],
        }
    ).write_parquet(weights)

    config = tmp_path / "run.yaml"
    config.write_text(
        "name: smoke\n"
        "as_of: '2024-04-01'\n"
        "start: '2024-01-01'\n"
        "symbols: [AAPL, MSFT]\n"
        f"weights: {weights.name}\n"
        "spread_bps: 5.0\n"
        "liquidity_window: 10\n"
        "engine:\n"
        "  execution: next_close\n"
    )
    result = runner.invoke(app, ["backtest", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "UNDEFLATED" in result.stdout
    assert "reconciled" in result.stdout
    assert "point-in-time as of 2024-04-01" in result.stdout
