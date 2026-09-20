"""The runner, the weight constructions and the academic benchmark."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.portfolio.weights import (
    cross_sectional_long_short,
    time_series_weights,
    volatility_target,
)
from quantlab.signals.base import SignalUnavailableError
from quantlab.signals.benchmark import (
    SAME_UNIVERSE_THRESHOLD,
    Comparability,
    benchmark_correlation,
)
from quantlab.signals.futures.trend import TimeSeriesMomentum
from quantlab.signals.runner import monthly_dates, run_signal

START = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)


def scores_frame(dates: list[dt.datetime], values: dict[str, list[float]]) -> pl.DataFrame:
    rows = [
        {"symbol": symbol, "as_of": date, "score": series[index]}
        for symbol, series in values.items()
        for index, date in enumerate(dates)
    ]
    return pl.DataFrame(rows)


# ==================================================================================
# The runner
# ==================================================================================
def test_the_runner_takes_one_snapshot_per_date(populated_store, monkeypatch) -> None:
    """One snapshot per rebalance is slower than computing the panel at once, and
    it is the entire point: a signal that sees a single snapshot cannot see past
    its own as-of instant, whatever its author intended."""
    from quantlab.data.store import Store

    seen: list[dt.datetime] = []
    original = Store.as_of

    def spy(self, moment):
        snapshot = original(self, moment)
        seen.append(snapshot.as_of)
        return snapshot

    monkeypatch.setattr(Store, "as_of", spy)
    # Dates where the lake fixture actually has bars, so the signal runs rather
    # than refusing.
    dates = [dt.datetime(2024, 1, d, 21, tzinfo=dt.UTC) for d in (3, 4, 5)]
    run_signal(TimeSeriesMomentum(), populated_store, symbols=["AAPL"], dates=dates)
    assert len(seen) >= len(dates)
    assert sorted(seen) == seen, "dates are walked forward, never revisited"


def test_a_signal_that_cannot_run_anywhere_raises(store) -> None:
    """A run that produced nothing should not look like a run that produced zeros."""
    dates = [START + dt.timedelta(days=i) for i in range(3)]
    with pytest.raises(SignalUnavailableError, match="could not run on any"):
        run_signal(
            TimeSeriesMomentum(),
            store,
            symbols=["AAPL"],
            dates=dates,
            skip_unavailable=True,
        )


def test_dates_inside_the_warm_up_produce_no_scores_and_are_counted(
    populated_store,
) -> None:
    """Three sessions is far inside the 315-bar warm-up, so the signal correctly
    produces nothing -- which is different from failing, and is recorded as such."""
    dates = [dt.datetime(2024, 1, d, 21, tzinfo=dt.UTC) for d in (3, 4, 5)]
    run = run_signal(TimeSeriesMomentum(), populated_store, symbols=["AAPL"], dates=dates)
    assert run.coverage == 0.0
    assert len(run.empty_dates) == len(dates)
    assert "rebalance dates" in run.describe()


def test_the_rebalance_calendar_is_drawn_from_bars_that_exist() -> None:
    """Counted in bars, so a market holiday cannot skip a rebalance and a signal is
    never asked for a date on which nothing traded."""
    available = [START + dt.timedelta(days=i) for i in range(100)]
    dates = monthly_dates(available, every=21, warmup_bars=10)
    assert dates[0] == available[10]
    assert dates[1] == available[31]
    assert all(d in available for d in dates)


def test_rebalance_spacing_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        monthly_dates([START], every=0)


# ==================================================================================
# Weight constructions
# ==================================================================================
def test_cross_sectional_weights_are_dollar_neutral() -> None:
    dates = [START]
    scores = scores_frame(dates, {f"S{i}": [float(i)] for i in range(8)})
    weights = cross_sectional_long_short(scores, quantile=0.25, gross=1.0)
    assert weights["weight"].sum() == pytest.approx(0.0, abs=1e-12)
    assert weights["weight"].abs().sum() == pytest.approx(1.0)


def test_cross_sectional_weights_go_long_the_best_and_short_the_worst() -> None:
    scores = scores_frame([START], {f"S{i}": [float(i)] for i in range(8)})
    weights = cross_sectional_long_short(scores, quantile=0.25)
    longs = set(weights.filter(pl.col("weight") > 0)["symbol"])
    shorts = set(weights.filter(pl.col("weight") < 0)["symbol"])
    assert longs == {"S6", "S7"}
    assert shorts == {"S0", "S1"}


def test_only_the_ranking_is_used_not_the_magnitude() -> None:
    """A cross-sectional score claims to carry a ranking and nothing else. Weighting
    by magnitude reads information into a number that does not have it."""
    modest = scores_frame([START], {f"S{i}": [float(i)] for i in range(8)})
    extreme = scores_frame([START], {f"S{i}": [float(i) ** 5] for i in range(8)})
    assert cross_sectional_long_short(modest).equals(cross_sectional_long_short(extreme))


def test_a_cross_section_too_small_to_split_is_skipped() -> None:
    scores = scores_frame([START], {"A": [1.0]})
    assert cross_sectional_long_short(scores, quantile=0.5).height == 0


def test_time_series_weights_keep_the_signal_sign() -> None:
    """An all-long book is a legitimate output, and is what trend following
    produces in a sustained move."""
    scores = scores_frame([START], {"A": [1.0], "B": [0.5], "C": [-2.0]})
    weights = time_series_weights(scores, max_gross=1.0)
    by_symbol = dict(zip(weights["symbol"], weights["weight"], strict=True))
    assert by_symbol["A"] > 0 and by_symbol["C"] < 0
    assert weights["weight"].abs().sum() == pytest.approx(1.0)


def test_time_series_weights_scale_down_but_never_up() -> None:
    """Scaling a weak signal up to hit a gross target turns 'no view' into a full
    position, which is how a flat month becomes a loss."""
    weak = scores_frame([START], {"A": [0.01], "B": [-0.01]})
    weights = time_series_weights(weak, max_gross=1.0)
    assert weights["weight"].abs().sum() == pytest.approx(0.02)


def test_an_empty_cross_section_yields_an_empty_frame_with_the_right_schema() -> None:
    empty = pl.DataFrame(
        schema={
            "symbol": pl.Utf8(),
            "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
            "score": pl.Float64(),
        }
    )
    for frame in (cross_sectional_long_short(empty), time_series_weights(empty)):
        assert frame.height == 0
        assert set(frame.columns) == {"symbol", "as_of", "weight"}


# ==================================================================================
# Volatility targeting -- risk management, not alpha
# ==================================================================================
def test_volatility_targeting_scales_the_book_toward_the_target() -> None:
    weights = pl.DataFrame({"symbol": ["A"], "as_of": [START], "weight": [1.0]})
    realised = pl.DataFrame({"as_of": [START], "volatility": [0.20]})
    scaled = volatility_target(weights, realised, target_annual=0.10)
    assert scaled["weight"].item() == pytest.approx(0.5)


def test_volatility_targeting_levers_up_in_quiet_markets_but_only_so_far() -> None:
    """A volatility target with no cap becomes a leverage machine in quiet markets
    -- which is precisely when quiet ends."""
    weights = pl.DataFrame({"symbol": ["A"], "as_of": [START], "weight": [1.0]})
    calm = pl.DataFrame({"as_of": [START], "volatility": [0.005]})
    scaled = volatility_target(weights, calm, target_annual=0.10, max_leverage=3.0)
    assert scaled["weight"].item() == pytest.approx(3.0)


def test_zero_volatility_flattens_rather_than_dividing_by_zero() -> None:
    weights = pl.DataFrame({"symbol": ["A"], "as_of": [START], "weight": [1.0]})
    frozen = pl.DataFrame({"as_of": [START], "volatility": [0.0]})
    assert volatility_target(weights, frozen)["weight"].item() == 0.0


def test_volatility_targeting_rejects_nonsense() -> None:
    weights = pl.DataFrame({"symbol": ["A"], "as_of": [START], "weight": [1.0]})
    realised = pl.DataFrame({"as_of": [START], "volatility": [0.2]})
    with pytest.raises(ValueError, match="target_annual"):
        volatility_target(weights, realised, target_annual=0.0)
    with pytest.raises(ValueError, match="max_leverage"):
        volatility_target(weights, realised, max_leverage=0.5)


# ==================================================================================
# The academic benchmark
# ==================================================================================
def paired(n: int, correlation: float, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, n)
    other = correlation * base + np.sqrt(max(0.0, 1 - correlation**2)) * rng.normal(0, 0.01, n)
    dates = [START + dt.timedelta(days=i) for i in range(n)]
    return (
        pl.DataFrame({"as_of": dates, "return": base}),
        pl.DataFrame({"as_of": dates, "value": other}),
    )


def test_a_faithful_reconstruction_clears_the_threshold() -> None:
    ours, theirs = paired(500, 0.97)
    result = benchmark_correlation(
        ours,
        theirs,
        name="ours",
        benchmark_name="UMD",
        comparability=Comparability.SAME_UNIVERSE,
    )
    assert result.correlation > SAME_UNIVERSE_THRESHOLD
    assert result.passes
    assert "as expected" in result.describe()


def test_a_broken_reconstruction_fails_the_threshold() -> None:
    """The cheapest bug detector available for factor code: it catches a sign flip
    that still produces plausible returns, or a rebalance off by a month."""
    ours, theirs = paired(500, 0.35)
    result = benchmark_correlation(
        ours,
        theirs,
        name="ours",
        benchmark_name="UMD",
        comparability=Comparability.SAME_UNIVERSE,
    )
    assert not result.passes
    assert "suspect a bug" in result.describe()


def test_a_sign_flip_is_caught() -> None:
    ours, theirs = paired(500, -0.95)
    result = benchmark_correlation(
        ours,
        theirs,
        name="ours",
        benchmark_name="UMD",
        comparability=Comparability.SAME_UNIVERSE,
    )
    assert not result.passes


def test_the_threshold_does_not_apply_to_a_different_universe() -> None:
    """A factor on fourteen ETFs is not a noisy version of a factor on three
    thousand stocks. Applying the 0.9 test there would be superstition."""
    ours, theirs = paired(500, 0.45)
    result = benchmark_correlation(
        ours, theirs, name="etf", benchmark_name="UMD", comparability=Comparability.RELATED
    )
    assert result.passes, "a positive correlation is all that is claimed here"
    assert "does not apply" in result.describe()


def test_a_negative_correlation_is_a_red_flag_even_across_universes() -> None:
    ours, theirs = paired(500, -0.4)
    result = benchmark_correlation(
        ours, theirs, name="etf", benchmark_name="UMD", comparability=Comparability.RELATED
    )
    assert not result.passes


def test_too_short_an_overlap_is_refused() -> None:
    """A correlation over forty days means nothing however high it is."""
    ours, theirs = paired(20, 0.95)
    with pytest.raises(ValueError, match="is noise"):
        benchmark_correlation(
            ours,
            theirs,
            name="ours",
            benchmark_name="UMD",
            comparability=Comparability.SAME_UNIVERSE,
        )
