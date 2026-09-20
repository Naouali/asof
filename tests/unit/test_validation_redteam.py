"""The red team: deliberately broken strategies the framework must reject.

Spec section 7: *"If the framework passes a known-fake strategy, the framework is
broken."* Four fakes, each broken in a different way, each with a known answer.

The fourth one is the most important test in this file, and it does not pass. It
documents a limit rather than a capability: **survivorship bias is invisible to
every statistic in this module.** Deflation corrects for how hard you searched;
nothing here corrects for a universe that quietly excluded the failures. That is a
data problem, caught in the data layer or not at all, and pretending otherwise
would be the most dangerous thing this package could do.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.backtest import (
    BacktestConfig,
    ExecutionTiming,
    Panel,
    VectorisedBacktest,
)
from quantlab.costs.impact import ImpactParams, SquareRootImpact
from quantlab.costs.model import TransactionCostModel
from quantlab.validation import (
    TrialRegistry,
    probability_of_backtest_overfitting,
    validate_strategy,
)

START = dt.datetime(2019, 1, 1, tzinfo=dt.UTC)
N_BARS = 1260  # five years of daily data
N_SYMBOLS = 40


def random_walk_panel(
    *, seed: int = 0, timing: ExecutionTiming = ExecutionTiming.NEXT_CLOSE
) -> tuple[Panel, np.ndarray, list[dt.datetime], list[str]]:
    """A market with no predictability whatsoever."""
    rng = np.random.default_rng(seed)
    dates = [START + dt.timedelta(days=i) for i in range(N_BARS)]
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, (N_BARS, N_SYMBOLS)), axis=0))
    rows = [
        {
            "symbol": symbol,
            "as_of": date,
            "close": float(prices[bar, index]),
            "open": float(prices[bar, index]),
            "adv_notional": 5e8,
            "volatility_daily": 0.015,
            "spread_bps": 2.0,
        }
        for bar, date in enumerate(dates)
        for index, symbol in enumerate(symbols)
    ]
    return Panel.from_frame(pl.DataFrame(rows), timing=timing), prices, dates, symbols


def costless() -> TransactionCostModel:
    return TransactionCostModel(
        impact=SquareRootImpact(ImpactParams(y=1e-12)), use_asset_class_defaults=False
    )


# ==================================================================================
# Fake 1: look-ahead bias
# ==================================================================================
def autocorrelated_panel(
    *, timing: ExecutionTiming, seed: int = 1, rho: float = 0.6
) -> tuple[Panel, np.ndarray, list[dt.datetime], list[str]]:
    """A market where this bar's return genuinely predicts the next one.

    Needed because look-ahead only *pays* when there is something to know. On a
    pure random walk, seeing tomorrow is worth nothing and a look-ahead bug would
    hide rather than announce itself.
    """
    rng = np.random.default_rng(seed)
    dates = [START + dt.timedelta(days=i) for i in range(800)]
    symbols = [f"S{i:02d}" for i in range(20)]
    returns = np.zeros((800, 20))
    for bar in range(1, 800):
        returns[bar] = rho * returns[bar - 1] + rng.normal(0, 0.01, 20)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    rows = [
        {
            "symbol": symbol,
            "as_of": date,
            "close": float(prices[bar, index]),
            "open": float(prices[bar, index]),
            "adv_notional": 1e12,
            "volatility_daily": 0.01,
            "spread_bps": 0.0,
        }
        for bar, date in enumerate(dates)
        for index, symbol in enumerate(symbols)
    ]
    return Panel.from_frame(pl.DataFrame(rows), timing=timing), prices, dates, symbols


def _cross_sectional_weights(
    signal: np.ndarray, date: dt.datetime, symbols: list[str], k: int = 5
) -> list[dict[str, object]]:
    order = np.argsort(signal)
    rows: list[dict[str, object]] = []
    for index in order[-k:]:
        rows.append({"symbol": symbols[index], "as_of": date, "weight": 0.5 / k})
    for index in order[:k]:
        rows.append({"symbol": symbols[index], "as_of": date, "weight": -0.5 / k})
    return rows


def test_a_look_ahead_execution_mode_is_refused_outright() -> None:
    """The first line of defence is that you cannot select it by accident."""
    with pytest.raises(ValueError, match="look-ahead bias, not a convention"):
        BacktestConfig(execution=ExecutionTiming.SAME_CLOSE)


def test_same_close_execution_measurably_inflates_the_same_signal() -> None:
    """What the cheat is actually worth, measured.

    ``SAME_CLOSE`` removes one bar of lag: a target dated on bar T is filled at
    T's close and earns T+1's return, where the honest convention fills at T+1 and
    earns T+2's. On a market with one-bar predictability that extra bar is the
    entire edge, and it nearly doubles the Sharpe. The run is stamped
    ``contaminated`` so the inflated number can never be quoted as a finding.
    """
    results = {}
    for timing, acknowledge in (
        (ExecutionTiming.NEXT_CLOSE, False),
        (ExecutionTiming.SAME_CLOSE, True),
    ):
        panel, prices, dates, symbols = autocorrelated_panel(timing=timing)
        rows: list[dict[str, object]] = []
        for bar in range(1, len(dates)):
            # Known at the close of `bar`: no future information whatsoever.
            signal = prices[bar] / prices[bar - 1] - 1.0
            rows.extend(_cross_sectional_weights(signal, dates[bar], symbols))
        results[timing] = VectorisedBacktest(
            BacktestConfig(
                execution=timing,
                acknowledge_look_ahead=acknowledge,
                borrow_bps_annual=0.0,
                financing_bps_annual=0.0,
            ),
            costless(),
        ).run(
            panel,
            pl.DataFrame(rows),
            name=timing.value,
            family="redteam-timing",
            record_trial=False,
        )

    honest = results[ExecutionTiming.NEXT_CLOSE]
    cheating = results[ExecutionTiming.SAME_CLOSE]

    assert not honest.contaminated
    assert cheating.contaminated
    assert "CONTAMINATED" in cheating.summary()
    assert cheating.stats.sharpe_undeflated > 1.5 * honest.stats.sharpe_undeflated, (
        "if removing a bar of lag does not improve a one-bar-predictive signal, the "
        "engine is not applying the lag it claims to, and the honest convention is "
        "protecting nothing"
    )


def test_a_forward_looking_signal_is_absurd_under_the_honest_convention_too() -> None:
    """The execution convention does not protect you from a signal that peeked.

    Here the weights are built from the return of a bar that has not happened yet,
    and the *default, honest* execution still produces an impossible Sharpe. The
    lesson is where the defence actually lives: not in the backtest engine, but in
    the point-in-time layer, which makes such a signal impossible to obtain.
    """
    panel, prices, dates, symbols = autocorrelated_panel(timing=ExecutionTiming.NEXT_CLOSE, rho=0.0)
    rows: list[dict[str, object]] = []
    for bar in range(len(dates) - 2):
        # The cheat: the return of bar+2, which is what a target dated `bar` earns
        # under the honest one-bar lag.
        future = prices[bar + 2] / prices[bar + 1] - 1.0
        rows.extend(_cross_sectional_weights(future, dates[bar], symbols))

    result = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.NEXT_CLOSE,
            borrow_bps_annual=0.0,
            financing_bps_annual=0.0,
        ),
        costless(),
    ).run(panel, pl.DataFrame(rows), name="peeking", family="redteam-peek", record_trial=False)

    assert not result.contaminated, "the engine cannot tell; the weights look ordinary"
    assert result.stats.sharpe_undeflated > 10.0


def test_the_point_in_time_layer_is_what_makes_that_signal_unobtainable() -> None:
    """Where the real defence lives. A snapshot refuses to serve the future at all,
    so the signal in the previous test cannot be constructed from one."""
    from quantlab.data.pit import LookAheadError

    panel, _, dates, _ = autocorrelated_panel(timing=ExecutionTiming.NEXT_CLOSE)
    del panel, dates
    # Exercised exhaustively in tests/unit/test_pit_leakage.py; asserted here so the
    # red-team suite names the component that actually stops this class of bug.
    assert issubclass(LookAheadError, RuntimeError)


# ==================================================================================
# Fake 2: pure noise
# ==================================================================================
def test_pure_noise_does_not_survive_deflation(tmp_path) -> None:
    """A strategy with no edge, over a realistic sample, after a realistic search."""
    registry = TrialRegistry(tmp_path)
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, N_BARS)

    report = validate_strategy(returns, name="noise", family="redteam-noise", registry=registry)
    assert not report.evidence.survives_deflation
    assert not report.passes
    assert report.failures()


def test_noise_that_happens_to_look_good_is_still_rejected(tmp_path) -> None:
    """The dangerous case: sample 200 noise series and validate the luckiest.

    Its raw Sharpe will look respectable. The deflation, told that 200 things were
    tried, must reject it anyway -- that is the entire purpose of the trial count.
    """
    from quantlab.validation import Trial, config_fingerprint

    registry = TrialRegistry(tmp_path)
    rng = np.random.default_rng(3)
    series = rng.normal(0.0002, 0.01, (200, N_BARS))
    sharpes = series.mean(axis=1) / series.std(axis=1, ddof=1) * np.sqrt(252)

    for index, sharpe in enumerate(sharpes):
        registry.record(
            Trial(
                family="redteam-lucky",
                config_hash=config_fingerprint(variant=index),
                name=f"variant-{index}",
                sharpe_annual=float(sharpe),
                observations=N_BARS,
            )
        )

    best = int(np.argmax(sharpes))
    report = validate_strategy(
        series[best], name=f"variant-{best}", family="redteam-lucky", registry=registry
    )

    assert report.evidence.net_annual > 0.5, "the luckiest of 200 looks respectable"
    assert report.trials.count == 200
    assert not report.evidence.survives_deflation, (
        "the luckiest of 200 pure-noise series must not survive deflation; if it "
        "does, the trial count is not reaching the statistic"
    )


# ==================================================================================
# Fake 3: an overfit parameter sweep
# ==================================================================================
def test_an_overfit_sweep_is_caught_by_pbo() -> None:
    """PBO measures the selection rule, not the winner.

    Configurations that are pure noise have no persistent ranking, so whichever
    looked best in-sample lands below median out-of-sample about half the time.
    A PBO near or above 0.5 is the signature.
    """
    rng = np.random.default_rng(5)
    sweep = rng.normal(0.0, 0.01, (N_BARS, 50))
    pbo, logits = probability_of_backtest_overfitting(sweep, partitions=10)

    assert pbo > 0.35, f"PBO of {pbo:.2f} on pure noise is implausibly low"
    assert len(logits) > 0


def test_pbo_is_low_when_one_configuration_is_genuinely_better() -> None:
    """The control: PBO must not condemn a real effect, or it condemns everything."""
    rng = np.random.default_rng(6)
    sweep = rng.normal(0.0, 0.01, (N_BARS, 50))
    sweep[:, 0] += 0.002  # one configuration with a persistent edge
    pbo, _ = probability_of_backtest_overfitting(sweep, partitions=10)
    assert pbo < 0.1


def test_the_best_of_a_sweep_on_noise_fails_deflation(tmp_path) -> None:
    """End to end: sweep, select the winner, and have the framework reject it."""
    from quantlab.validation import Trial, config_fingerprint

    registry = TrialRegistry(tmp_path)
    rng = np.random.default_rng(9)
    sweep = rng.normal(0.0, 0.01, (N_BARS, 40))
    per_config = sweep.mean(axis=0) / sweep.std(axis=0, ddof=1) * np.sqrt(252)

    for index, sharpe in enumerate(per_config):
        registry.record(
            Trial(
                family="redteam-sweep",
                config_hash=config_fingerprint(param=index),
                name=f"cfg-{index}",
                sharpe_annual=float(sharpe),
                observations=N_BARS,
            )
        )

    winner = int(np.argmax(per_config))
    report = validate_strategy(
        sweep[:, winner],
        name=f"cfg-{winner}",
        family="redteam-sweep",
        registry=registry,
        sweep_returns=sweep,
    )

    assert not report.passes
    assert report.pbo is not None
    assert not report.evidence.survives_deflation
    assert any("luckiest member" in failure for failure in report.failures())


# ==================================================================================
# Fake 4: survivorship bias -- the one the statistics CANNOT catch
# ==================================================================================
def test_survivorship_bias_is_invisible_to_every_statistic_here(tmp_path) -> None:
    """The most important test in this file, and it documents a *limit*.

    A universe assembled from instruments that happened to survive produces real
    returns from a real edge -- the edge of having excluded the failures in
    advance. There is nothing statistically anomalous about the resulting series,
    so deflation, PBO and the probabilistic Sharpe all pass it.

    Asserted here so that nobody later mistakes a clean validation report for
    evidence that the universe was clean. Survivorship is caught in the data layer
    or it is not caught at all.
    """
    registry = TrialRegistry(tmp_path)
    rng = np.random.default_rng(13)

    # 200 instruments, pure random walks. Keep only the ones that ended up ahead --
    # a decision made, as it always is, with full knowledge of the outcome.
    paths = np.cumsum(rng.normal(0.0, 0.012, (200, N_BARS)), axis=1)
    survivors = paths[paths[:, -1] > 0.4]
    assert len(survivors) > 10, "the fixture needs enough survivors to average over"
    biased_returns = np.diff(survivors, axis=1).mean(axis=0)

    report = validate_strategy(
        biased_returns, name="survivors-only", family="redteam-survivorship", registry=registry
    )

    assert report.evidence.net_annual > 1.0, "selecting on the outcome looks excellent"
    assert report.evidence.survives_deflation, (
        "this is the point: the statistics see a strong, consistent return series "
        "and pass it. Nothing in quantlab.validation detects survivorship bias."
    )
    assert report.passes


def test_the_data_layer_is_where_survivorship_is_caught_instead() -> None:
    """Since the statistics cannot see it, the platform has to flag it upstream --
    and a backtest result carries the delisting count precisely so that a universe
    with suspiciously few of them is visible."""
    from quantlab.data.catalogue import PitQuality, get_source

    assert get_source("yahoo").pit_quality is PitQuality.SURVIVORSHIP_BIASED
    assert any("survivorship" in caveat.lower() for caveat in get_source("yahoo").caveats)

    panel, _, dates, symbols = random_walk_panel()
    rows = [{"symbol": symbols[0], "as_of": dates[0], "weight": 1.0}]
    result = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE), costless()
    ).run(panel, pl.DataFrame(rows), name="q", family="redteam-quality")
    assert "delisted_symbols" in result.data_quality
