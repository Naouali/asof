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
    """Weights come from a file. The error has to say what the file contains and
    where to get one, not just report a missing path."""
    config = tmp_path / "run.yaml"
    config.write_text("name: t\nas_of: '2024-01-05'\nsymbols: [AAPL]\nweights: missing.parquet\n")
    result = runner.invoke(app, ["backtest", "--config", str(config)])
    assert result.exit_code == 2
    assert "symbol, as_of, weight" in result.output
    assert "signal library" in result.output


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
    assert "deflated" in result.stdout, "every reported Sharpe carries its deflation"
    assert "reconciled" in result.stdout
    assert "point-in-time as of 2024-04-01" in result.stdout


def test_validate_trials_on_an_empty_registry() -> None:
    result = runner.invoke(app, ["validate", "trials"])
    assert result.exit_code == 0
    assert "no trials recorded" in result.stdout


def test_validate_trials_lists_families(settings: Settings) -> None:
    from quantlab.validation import Trial, TrialRegistry, config_fingerprint

    registry = TrialRegistry(settings.layout.state)
    for index in range(4):
        registry.record(
            Trial(
                family="momentum",
                config_hash=config_fingerprint(variant=index),
                name=f"mom-{index}",
                sharpe_annual=0.4 + index * 0.2,
                observations=1260,
            )
        )
    result = runner.invoke(app, ["validate", "trials"])
    assert result.exit_code == 0
    assert "momentum" in result.stdout
    assert "deflation bar" in result.stdout

    detail = runner.invoke(app, ["validate", "trials", "momentum"])
    assert detail.exit_code == 0
    assert "mom-3" in detail.stdout


def test_validate_trials_for_an_unknown_family_fails_loudly(settings: Settings) -> None:
    from quantlab.validation import Trial, TrialRegistry, config_fingerprint

    TrialRegistry(settings.layout.state).record(
        Trial(
            family="a",
            config_hash=config_fingerprint(x=1),
            name="a",
            sharpe_annual=1.0,
            observations=100,
        )
    )
    result = runner.invoke(app, ["validate", "trials", "nope"])
    assert result.exit_code == 2
    assert "known families" in result.output


def test_validate_sharpe_survives_with_one_trial() -> None:
    result = runner.invoke(app, ["validate", "sharpe", "1.8", "--years", "10"])
    assert result.exit_code == 0
    assert "survives deflation" in result.stdout


def test_validate_sharpe_fails_after_a_wide_search() -> None:
    """The same number, deflated by the search that found it."""
    result = runner.invoke(app, ["validate", "sharpe", "1.8", "--years", "10", "--trials", "300"])
    assert result.exit_code == 1
    assert "does not survive" in result.output
    assert "luckiest member" in result.output


def test_validate_library_needs_enough_families(settings: Settings) -> None:
    """Below five signals the adjustment says more about your sample size than
    about your library."""
    result = runner.invoke(app, ["validate", "library"])
    assert result.exit_code == 2
    assert "at least 5" in result.output


def test_validate_library_reports_the_shrinkage(settings: Settings) -> None:
    import numpy as np

    from quantlab.validation import Trial, TrialRegistry, config_fingerprint

    registry = TrialRegistry(settings.layout.state)
    rng = np.random.default_rng(0)
    for index in range(30):
        registry.record(
            Trial(
                family=f"family-{index}",
                config_hash=config_fingerprint(x=index),
                name=f"sig-{index}",
                # t = Sharpe * sqrt(years); 5 years, so these are t/sqrt(5).
                sharpe_annual=float(rng.normal(0, 1) / np.sqrt(5)),
                observations=1260,
            )
        )
    result = runner.invoke(app, ["validate", "library"])
    assert result.exit_code == 0
    assert "Var(t)" in result.stdout
    assert "shrunk t" in result.stdout


# ----------------------------------------------------------------------------------
# Portfolio construction and risk
# ----------------------------------------------------------------------------------
def _equity_lake(store: object, *, yield_annual: float = 0.0, sessions: int = 320) -> None:
    """A small correlated universe, optionally paying a dividend.

    ``yield_annual`` separates ``adj_close`` from ``close``: the total-return
    series compounds at the extra rate while the price series does not, which is
    what a real dividend-paying instrument looks like in the lake.
    """
    import numpy as np
    from tests.conftest import bar

    rng = np.random.default_rng(11)
    days = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(sessions)]
    common = rng.normal(0.0, 0.009, sessions)
    rows = []
    for index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD")):
        beta = 0.7 + 0.2 * index
        price, total = 100.0, 100.0
        for step, session in enumerate(days):
            move = beta * common[step] + rng.normal(0.0, 0.004)
            price *= 1.0 + move
            total *= 1.0 + move + yield_annual / 252.0
            row = bar(symbol, session, price)
            row["adj_close"] = total
            row["volume"] = 4_000_000.0
            rows.append(row)
    store.write(pl.DataFrame(rows), asset_class="equity")  # type: ignore[attr-defined]


def _factor_lake(store: object, *, sessions: int = 320) -> None:
    """Daily factor observations stamped at midnight, as a real series is.

    The bars are stamped at their 21:00 session close, so anything joining the
    two has to join on the calendar date rather than the instant.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    days = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(sessions)]
    rows = []
    for symbol, scale in (("KF_MKT_RF", 0.009), ("KF_SMB", 0.004), ("KF_RF", 0.0)):
        for session in days:
            as_of = dt.datetime(session.year, session.month, session.day, tzinfo=dt.UTC)
            rows.append(
                {
                    "source": "ken_french",
                    "dataset": "series_observations",
                    "symbol": symbol,
                    "as_of": as_of,
                    "known_at": as_of,
                    "ingested_at": as_of,
                    "value": float(rng.normal(0.0, scale)) if scale else 0.00008,
                    "units": "daily return",
                    "vintage": False,
                }
            )
    store.write(pl.DataFrame(rows), asset_class="factors")  # type: ignore[attr-defined]


def test_portfolio_covariance_compares_the_estimators(store: object) -> None:
    _equity_lake(store)
    result = runner.invoke(
        app,
        ["portfolio", "covariance", "-s", "AAA", "-s", "BBB", "-s", "CCC", "--as-of", "2024-10-01"],
    )
    assert result.exit_code == 0, result.output
    assert "ledoit-wolf(constant_correlation)" in result.output
    assert "sample" in result.output


@pytest.mark.parametrize("method", ["equal", "inverse-vol", "risk-parity", "mean-variance"])
def test_portfolio_build_runs_every_method(store: object, method: str) -> None:
    _equity_lake(store)
    result = runner.invoke(
        app,
        [
            "portfolio", "build",
            "-s", "AAA", "-s", "BBB", "-s", "CCC", "-s", "DDD",
            "--as-of", "2024-10-01", "-m", method,
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "effective positions" in result.output


def test_risk_parity_equalises_risk_not_weights(store: object) -> None:
    """The output that makes the method worth having: risk shares, not weights."""
    _equity_lake(store)
    result = runner.invoke(
        app,
        [
            "portfolio", "build",
            "-s", "AAA", "-s", "BBB", "-s", "CCC", "-s", "DDD",
            "--as-of", "2024-10-01", "-m", "risk-parity",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    shares = [
        float(token.rstrip("%"))
        for line in result.output.splitlines()
        if line.split() and line.split()[0] in {"AAA", "BBB", "CCC", "DDD"}
        for token in [line.split()[-1]]
    ]
    assert len(shares) == 4
    assert max(shares) - min(shares) < 1.0  # every position carries ~25% of the risk


def test_portfolio_build_rejects_an_unknown_method(store: object) -> None:
    _equity_lake(store)
    result = runner.invoke(
        app,
        ["portfolio", "build", "-s", "AAA", "-s", "BBB", "--as-of", "2024-10-01", "-m", "sharpe"],
    )
    assert result.exit_code == 2
    assert "unknown method" in result.output


def test_portfolio_build_names_an_infeasible_constraint_set(store: object) -> None:
    """Four names capped at 10% cannot reach a fully invested book. The command
    has to say so rather than print weights that miss the target."""
    _equity_lake(store)
    result = runner.invoke(
        app,
        [
            "portfolio", "build",
            "-s", "AAA", "-s", "BBB", "-s", "CCC", "-s", "DDD",
            "--as-of", "2024-10-01", "-m", "mean-variance", "--max-position", "0.10",
        ],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "unreachable" in result.output


def test_portfolio_commands_fail_loudly_on_an_empty_lake(settings: Settings) -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(
        app, ["portfolio", "covariance", "-s", "AAA", "-s", "BBB", "--as-of", "2024-10-01"]
    )
    assert result.exit_code == 2
    assert "no daily bars" in result.output


def test_returns_are_total_not_price(store: object) -> None:
    """The regression test for the most expensive bug in this milestone.

    Computing returns from ``close`` rather than ``adj_close`` drops the dividend
    yield from every observation, and an attribution reports the missing yield as
    negative alpha. Here the instruments pay 6% a year, so a price-return
    implementation would understate every mean return by that amount. The
    covariance command's reported volatility is unaffected by a constant drift,
    so the check has to be on the returns themselves.
    """
    from quantlab.cli import _returns_frame

    _equity_lake(store, yield_annual=0.06)
    frame = _returns_frame(["AAA"], "2024-10-01", 250)
    realised = float(frame["AAA"].mean()) * 252

    # The same window, priced the wrong way. Taking it from the full history
    # instead would compare two different windows and prove nothing.
    prices = (
        store.as_of("2024-10-01")  # type: ignore[attr-defined]
        .ohlcv_daily(symbols=["AAA"])
        .sort("as_of")["close"]
        .to_numpy()[-(frame.height + 1) :]
    )
    price_only = float(((prices[1:] / prices[:-1]) - 1.0).mean()) * 252

    assert realised - price_only == pytest.approx(0.06, abs=0.002)


def test_risk_attribute_joins_on_the_calendar_date(store: object) -> None:
    """Bars are stamped at 21:00 UTC and factor observations at midnight, so an
    exact-instant join matches nothing. Getting this wrong is silent: the command
    would either find no data or, worse, regress against a shifted series."""
    _equity_lake(store)
    _factor_lake(store)
    result = runner.invoke(
        app,
        [
            "risk", "attribute", "-s", "AAA", "--as-of", "2024-10-01",
            "-f", "KF_MKT_RF", "-f", "KF_SMB",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "common dates" in result.output
    assert "KF_MKT_RF" in result.output
    assert "R²" in result.output


def test_risk_attribute_fails_without_the_factors(store: object) -> None:
    _equity_lake(store)
    result = runner.invoke(app, ["risk", "attribute", "-s", "AAA", "--as-of", "2024-10-01"])
    assert result.exit_code == 2
    assert "ken_french" in result.output


def test_risk_attribute_refuses_without_a_risk_free_rate(store: object) -> None:
    """Without it the instrument cannot be put on the same excess-return footing
    as the factors, and the mismatch lands on the alpha being tested."""
    import numpy as np

    _equity_lake(store)
    rng = np.random.default_rng(2)
    rows = []
    for session in [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(320)]:
        as_of = dt.datetime(session.year, session.month, session.day, tzinfo=dt.UTC)
        rows.append(
            {
                "source": "ken_french",
                "dataset": "series_observations",
                "symbol": "KF_MKT_RF",
                "as_of": as_of,
                "known_at": as_of,
                "ingested_at": as_of,
                "value": float(rng.normal(0.0, 0.009)),
                "units": "daily return",
                "vintage": False,
            }
        )
    store.write(pl.DataFrame(rows), asset_class="factors")  # type: ignore[attr-defined]

    result = runner.invoke(
        app, ["risk", "attribute", "-s", "AAA", "--as-of", "2024-10-01", "-f", "KF_MKT_RF"]
    )
    assert result.exit_code == 2
    assert "KF_RF" in result.output


def test_risk_pca_extracts_the_common_factor(store: object) -> None:
    _equity_lake(store)
    result = runner.invoke(
        app,
        [
            "risk", "pca",
            "-s", "AAA", "-s", "BBB", "-s", "CCC", "-s", "DDD",
            "--as-of", "2024-10-01", "--factors", "2",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "PC1" in result.output
    assert "useless for attribution" in result.output
    # The universe is built from one common factor, so PC1 should dominate.
    assert "market bet" in result.output


def test_risk_pca_rejects_too_many_factors(store: object) -> None:
    _equity_lake(store)
    result = runner.invoke(
        app,
        ["risk", "pca", "-s", "AAA", "-s", "BBB", "--as-of", "2024-10-01", "--factors", "5"],
    )
    assert result.exit_code == 2
    assert "n_factors must be between" in result.output


# ----------------------------------------------------------------------------------
# Tearsheets
# ----------------------------------------------------------------------------------
def _smoke_run(tmp_path: Path, store: object, *, sessions_count: int = 120) -> Path:
    """A lake and a run config that produce a real, reportable backtest."""
    import numpy as np
    from tests.conftest import bar

    rng = np.random.default_rng(7)
    sessions = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(sessions_count)]
    rows = []
    for symbol, drift in (("AAPL", 0.0012), ("MSFT", 0.0004)):
        price = 100.0
        for session in sessions:
            price *= float(np.exp(rng.normal(drift, 0.011)))
            row = bar(symbol, session, price)
            row["volume"] = 5_000_000.0
            rows.append(row)
    store.write(pl.DataFrame(rows), asset_class="equity")  # type: ignore[attr-defined]

    weights_path = tmp_path / "w.parquet"
    rebalances = [sessions[i] for i in range(40, sessions_count, 10)]
    pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"] * len(rebalances),
            "as_of": [
                dt.datetime(s.year, s.month, s.day, 21, tzinfo=dt.UTC)
                for s in rebalances
                for _ in range(2)
            ],
            "weight": [
                w for i in range(len(rebalances)) for w in ((0.5, -0.5) if i % 2 else (-0.5, 0.5))
            ],
        }
    ).write_parquet(weights_path)

    config = tmp_path / "run.yaml"
    config.write_text(
        "name: tearsheet-smoke\n"
        f"as_of: '{sessions[-1] + dt.timedelta(days=1)}'\n"
        "start: '2024-01-01'\n"
        "symbols: [AAPL, MSFT]\n"
        f"weights: {weights_path.name}\n"
        "spread_bps: 5.0\n"
        "liquidity_window: 10\n"
        "engine:\n"
        "  execution: next_close\n"
    )
    return config


def test_report_renders_a_tearsheet(tmp_path: Path, store: object) -> None:
    config = _smoke_run(tmp_path, store)
    result = runner.invoke(app, ["report", "--config", str(config)])

    assert result.exit_code in {0, 1}, result.output  # 1 when the result is disqualified
    assert "PERFORMANCE" in result.output
    assert "CAPACITY" in result.output
    assert "Deflated Sharpe" in result.output


def test_a_disqualified_result_exits_non_zero(tmp_path: Path, store: object) -> None:
    """So a scripted sweep cannot treat a strategy that fails deflation as a
    success. The exit code is the only part of a tearsheet a script reads."""
    config = _smoke_run(tmp_path, store)
    result = runner.invoke(app, ["report", "--config", str(config)])

    disqualified = "NOT EVIDENCE" in result.output
    assert result.exit_code == (1 if disqualified else 0)


def test_report_writes_html(tmp_path: Path, store: object) -> None:
    config = _smoke_run(tmp_path, store)
    out = tmp_path / "sheet.html"
    runner.invoke(app, ["report", "--config", str(config), "-f", "html", "-o", str(out)])

    body = out.read_text()
    assert body.startswith("<!doctype html>")
    assert "<svg" in body
    assert "<script" not in body  # self-contained, and no scripts


def test_report_writes_markdown(tmp_path: Path, store: object) -> None:
    config = _smoke_run(tmp_path, store)
    out = tmp_path / "sheet.md"
    runner.invoke(app, ["report", "--config", str(config), "-f", "markdown", "-o", str(out)])
    assert out.read_text().startswith("# tearsheet-smoke")


def test_report_rejects_an_unknown_format(tmp_path: Path, store: object) -> None:
    config = _smoke_run(tmp_path, store)
    result = runner.invoke(app, ["report", "--config", str(config), "-f", "pdf"])
    assert result.exit_code == 2
    assert "unknown format" in result.output


def test_report_rejects_an_unknown_source(tmp_path: Path, store: object) -> None:
    config = _smoke_run(tmp_path, store)
    result = runner.invoke(app, ["report", "--config", str(config), "--source", "not-a-source"])
    assert result.exit_code == 2


def test_report_carries_the_source_caveats(tmp_path: Path, store: object) -> None:
    config = _smoke_run(tmp_path, store)
    result = runner.invoke(app, ["report", "--config", str(config), "--source", "yahoo"])
    assert "DATA QUALITY" in result.output
    assert "survivorship" in result.output.lower()


def test_report_needs_a_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "no run config" in result.output


def test_validate_anomalies_needs_the_catalogue(settings: Settings) -> None:
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["validate", "anomalies"])
    assert result.exit_code == 2
    assert "open_asset_pricing" in result.output


def test_validate_anomalies_places_a_t_stat_in_the_literature(store: object) -> None:
    """The point of the command: a t-statistic means nothing until you know what
    the published distribution looks like, and that distribution is truncated at
    the significance threshold because that is where journals stop."""
    rows = [
        {
            "source": "open_asset_pricing",
            "dataset": "anomaly_catalogue",
            "symbol": f"SIG{i}",
            "as_of": dt.datetime(2000, 12, 31, tzinfo=dt.UTC),
            "known_at": dt.datetime(2005, 12, 31, tzinfo=dt.UTC),
            "ingested_at": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            "name": f"signal {i}",
            "authors": "Someone",
            "journal": "JF",
            "category": "Predictor",
            "replication": "1_good",
            "evidence": "1_clear",
            "published_return": 0.5,
            "published_t_stat": t,
            "sign": 1.0,
        }
        for i, t in enumerate([2.1, 2.6, 3.2, 4.0, 5.5, 8.0])
    ]
    store.write(pl.DataFrame(rows), asset_class="reference")  # type: ignore[attr-defined]

    result = runner.invoke(
        app, ["validate", "anomalies", "--as-of", "2026-01-01", "--t-stat", "2.4"]
    )
    assert result.exit_code == 0, result.output
    assert "published t-statistics" in result.output
    assert "larger than" in result.output
    assert "Harvey" in result.output  # the |t| > 3.0 bar is named, not implied


def test_validate_anomalies_credits_a_strong_result_without_overclaiming(store: object) -> None:
    rows = [
        {
            "source": "open_asset_pricing",
            "dataset": "anomaly_catalogue",
            "symbol": f"SIG{i}",
            "as_of": dt.datetime(2000, 12, 31, tzinfo=dt.UTC),
            "known_at": dt.datetime(2005, 12, 31, tzinfo=dt.UTC),
            "ingested_at": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            "name": f"signal {i}",
            "authors": "Someone",
            "journal": "JF",
            "category": "Predictor",
            "replication": "1_good",
            "evidence": "1_clear",
            "published_return": 0.5,
            "published_t_stat": t,
            "sign": 1.0,
        }
        for i, t in enumerate([2.1, 2.6, 3.2, 4.0])
    ]
    store.write(pl.DataFrame(rows), asset_class="reference")  # type: ignore[attr-defined]

    result = runner.invoke(
        app, ["validate", "anomalies", "--as-of", "2026-01-01", "--t-stat", "6.0"]
    )
    assert "Clears" in result.output
    assert "not sufficient" in result.output, "a t-stat says nothing about trial count"
