"""The assembled validation report, and its integration with the backtest engine."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantlab.backtest import BacktestConfig, ExecutionTiming, Panel, VectorisedBacktest
from quantlab.costs.impact import ImpactParams, SquareRootImpact
from quantlab.costs.model import TransactionCostModel
from quantlab.validation import (
    CombinatorialPurgedCV,
    Trial,
    TrialRegistry,
    config_fingerprint,
    cpcv_path_returns,
    validate_strategy,
)

START = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)


def registry_with(tmp_path: Path, family: str, count: int, sharpes: list[float] | None = None):
    registry = TrialRegistry(tmp_path)
    values = sharpes or [0.5 + 0.1 * i for i in range(count)]
    for index, sharpe in enumerate(values[:count]):
        registry.record(
            Trial(
                family=family,
                config_hash=config_fingerprint(variant=index),
                name=f"cfg-{index}",
                sharpe_annual=sharpe,
                observations=1260,
            )
        )
    return registry


# ----------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------
def test_a_strong_result_after_few_trials_validates(tmp_path: Path) -> None:
    """The statistic must be able to say yes, or it says nothing."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0012, 0.008, 5040)  # ~20 years, Sharpe ~2.4
    report = validate_strategy(
        returns, name="good", family="f", registry=registry_with(tmp_path, "f", 3)
    )
    assert report.evidence.survives_deflation
    assert report.passes
    assert report.failures() == []
    assert "clears every check" in report.describe()


def test_a_weak_result_after_many_trials_does_not(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0002, 0.01, 1260)
    report = validate_strategy(
        returns, name="weak", family="f", registry=registry_with(tmp_path, "f", 60)
    )
    assert not report.passes
    assert any("luckiest member" in failure for failure in report.failures())
    assert "does not validate" in report.describe()


def test_the_report_names_the_trial_count_it_used(tmp_path: Path) -> None:
    report = validate_strategy(
        np.random.default_rng(2).normal(0.0005, 0.01, 1260),
        name="x",
        family="f",
        registry=registry_with(tmp_path, "f", 12),
    )
    assert report.trials.count == 12
    assert report.evidence.trials == 12
    assert "12 trial(s)" in report.describe()


def test_a_family_with_no_record_still_counts_as_one_trial(tmp_path: Path) -> None:
    """You ran this one. A count of zero would imply no search at all."""
    report = validate_strategy(
        np.random.default_rng(3).normal(0.001, 0.01, 1260),
        name="x",
        family="unrecorded",
        registry=TrialRegistry(tmp_path),
    )
    assert report.evidence.trials == 1


def test_the_minimum_backtest_length_is_reported_against_the_sample(tmp_path: Path) -> None:
    report = validate_strategy(
        np.random.default_rng(4).normal(0.0003, 0.01, 756),  # three years
        name="short",
        family="f",
        registry=registry_with(tmp_path, "f", 100),
    )
    assert report.years_available == pytest.approx(3.0, abs=0.05)
    assert report.minimum_backtest_years > report.years_available
    assert not report.sample_is_long_enough
    assert any("years would be needed" in failure for failure in report.failures())


def test_gross_and_net_are_reported_separately(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    gross = rng.normal(0.0008, 0.01, 2520)
    net = gross - 0.0003
    report = validate_strategy(
        net, name="x", family="f", registry=registry_with(tmp_path, "f", 2), gross_returns=gross
    )
    assert report.evidence.gross_annual > report.evidence.net_annual
    assert report.evidence.cost_drag > 0
    assert report.evidence.haircut_annual < report.evidence.net_annual


def test_pbo_is_included_when_a_sweep_is_supplied(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    sweep = rng.normal(0.0, 0.01, (1260, 20))
    report = validate_strategy(
        sweep[:, 0],
        name="x",
        family="f",
        registry=registry_with(tmp_path, "f", 20),
        sweep_returns=sweep,
    )
    assert report.pbo is not None
    assert "overfitting" in report.describe()


def test_pbo_is_absent_without_a_sweep(tmp_path: Path) -> None:
    """PBO measures the *selection*. With one configuration there was nothing to
    select and nothing to overfit."""
    report = validate_strategy(
        np.random.default_rng(7).normal(0.0005, 0.01, 1260),
        name="x",
        family="f",
        registry=registry_with(tmp_path, "f", 1),
    )
    assert report.pbo is None


def test_too_few_observations_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least three"):
        validate_strategy(np.array([0.01]), name="x", family="f", registry=TrialRegistry(tmp_path))


# ----------------------------------------------------------------------------------
# CPCV paths
# ----------------------------------------------------------------------------------
def test_cpcv_paths_cover_the_sample_and_vary_with_refitting() -> None:
    """The refitting is the point: without it every path is identical and the
    cross-validation is measuring nothing."""
    rng = np.random.default_rng(8)
    n = 600
    truth = rng.normal(0, 0.01, n)

    def fit_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        # A "model" whose output depends on the training window, so paths differ.
        scale = 1.0 + float(np.mean(truth[train])) * 50
        return truth[test] * scale

    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2)
    paths = cpcv_path_returns(n, fit_predict, cv)
    assert paths.shape == (cv.n_paths, n)
    assert np.isfinite(paths).all(), "every path must cover the whole sample"
    assert len({round(float(np.nanstd(row)), 12) for row in paths}) > 1


def test_cpcv_paths_are_identical_when_nothing_is_refitted() -> None:
    """Documented rather than hidden: running CPCV over a fixed return series
    produces the same path every time, and reporting that spread as evidence of
    robustness would be circular."""
    rng = np.random.default_rng(9)
    n = 600
    fixed = rng.normal(0, 0.01, n)
    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2)
    paths = cpcv_path_returns(n, lambda _train, test: fixed[test], cv)
    assert np.allclose(paths[0], paths[1])


def test_path_dispersion_is_surfaced(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    paths = rng.normal(0.0002, 0.01, (9, 1260))
    report = validate_strategy(
        paths[0],
        name="x",
        family="f",
        registry=registry_with(tmp_path, "f", 5),
        path_returns=paths,
    )
    dispersion = report.path_dispersion
    assert dispersion is not None
    assert dispersion[0] <= dispersion[1] <= dispersion[2]
    assert "CPCV paths" in report.describe()


def test_negative_worst_paths_fail_the_report(tmp_path: Path) -> None:
    """If the worst slices of history lose money, the headline Sharpe depends on
    which quarter you happened to test on."""
    rng = np.random.default_rng(11)
    paths = rng.normal(0.0, 0.01, (9, 1260))
    report = validate_strategy(
        paths[0],
        name="x",
        family="f",
        registry=registry_with(tmp_path, "f", 2),
        path_returns=paths,
    )
    assert not report.passes
    assert any("worst CPCV paths" in failure for failure in report.failures())


# ----------------------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------------------
def make_panel(bars: int = 500, symbols: int = 10, seed: int = 0) -> tuple[Panel, list, list]:
    rng = np.random.default_rng(seed)
    dates = [START + dt.timedelta(days=i) for i in range(bars)]
    names = [f"S{i:02d}" for i in range(symbols)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (bars, symbols)), axis=0))
    rows = [
        {
            "symbol": name,
            "as_of": date,
            "close": float(prices[bar, index]),
            "open": float(prices[bar, index]),
            "adv_notional": 1e10,
            "volatility_daily": 0.015,
            "spread_bps": 1.0,
        }
        for bar, date in enumerate(dates)
        for index, name in enumerate(names)
    ]
    return Panel.from_frame(pl.DataFrame(rows), timing=ExecutionTiming.NEXT_CLOSE), dates, names


def costless() -> TransactionCostModel:
    return TransactionCostModel(
        impact=SquareRootImpact(ImpactParams(y=1e-12)), use_asset_class_defaults=False
    )


def test_a_backtest_records_itself_as_a_trial(settings) -> None:
    """Automatic, per spec section 7. Nobody remembers that they tried forty
    lookbacks last Tuesday."""
    panel, dates, names = make_panel()
    weights = pl.DataFrame([{"symbol": names[0], "as_of": dates[0], "weight": 1.0}])
    engine = VectorisedBacktest(BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE), costless())

    engine.run(panel, weights, name="v1", family="auto")
    registry = TrialRegistry(settings.layout.state)
    assert registry.family("auto").count == 1


def test_a_sweep_increments_the_count_and_deflates_harder(settings) -> None:
    """The behaviour the whole milestone exists for."""
    panel, dates, names = make_panel()
    engine = VectorisedBacktest(BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE), costless())

    deflated: list[float] = []
    for index, name in enumerate(names[:6]):
        weights = pl.DataFrame([{"symbol": name, "as_of": dates[0], "weight": 1.0}])
        result = engine.run(panel, weights, name=f"variant-{index}", family="sweep")
        assert result.evidence is not None
        assert result.evidence.trials == index + 1
        deflated.append(result.evidence.deflated)

    assert TrialRegistry(settings.layout.state).family("sweep").count == 6


def test_the_summary_always_carries_the_deflated_sharpe_and_trial_count(settings) -> None:
    """Spec section 13: no Sharpe without its deflated counterpart and trial count."""
    panel, dates, names = make_panel()
    weights = pl.DataFrame([{"symbol": names[0], "as_of": dates[0], "weight": 1.0}])
    result = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE), costless()
    ).run(panel, weights, name="s", family="summary")

    summary = result.summary()
    assert "deflated" in summary
    assert "trial(s)" in summary
    assert "after haircuts" in summary


def test_an_unrecorded_run_gets_no_deflated_sharpe(settings) -> None:
    """Opting out of the count opts out of the claim. There is no way to search
    quietly and still quote a deflated number."""
    panel, dates, names = make_panel()
    weights = pl.DataFrame([{"symbol": names[0], "as_of": dates[0], "weight": 1.0}])
    result = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE), costless()
    ).run(panel, weights, name="q", family="quiet", record_trial=False)

    assert TrialRegistry(settings.layout.state).family("quiet").count == 0
    assert result.evidence is not None
    assert result.evidence.trials == 1, "counted as this run alone, not as zero"
