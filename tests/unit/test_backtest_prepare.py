"""Deriving liquidity inputs without borrowing information from the future."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.backtest.prepare import (
    DEFAULT_SPREAD_BPS,
    DEFAULT_WINDOW,
    panel_frame,
    rebalance_dates,
)

START = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)


def bars(n: int = 200, symbols: tuple[str, ...] = ("A", "B"), *, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for index, symbol in enumerate(symbols):
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01 * (index + 1), n)))
        for bar in range(n):
            out.append(
                {
                    "symbol": symbol,
                    "as_of": START + dt.timedelta(days=bar),
                    "close": float(prices[bar]),
                    "open": float(prices[bar]) * 0.999,
                    "high": float(prices[bar]) * 1.01,
                    "low": float(prices[bar]) * 0.99,
                    "volume": 1_000_000.0 * (index + 1),
                }
            )
    return pl.DataFrame(out)


# ----------------------------------------------------------------------------------
# The property that matters
# ----------------------------------------------------------------------------------
def test_statistics_are_trailing_only() -> None:
    """A full-sample volatility estimate would tell a 2008 backtest how volatile
    2008 turned out to be. Here, truncating the future must not change the past.
    """
    full = panel_frame(bars(200), window=20, spread_bps=5.0)
    truncated = panel_frame(
        bars(200).filter(pl.col("as_of") <= START + dt.timedelta(days=99)),
        window=20,
        spread_bps=5.0,
    )
    overlap = truncated["as_of"].max()
    left = full.filter(pl.col("as_of") <= overlap).sort("as_of", "symbol")
    right = truncated.sort("as_of", "symbol")

    assert left.height == right.height
    for column in ("adv_notional", "volatility_daily"):
        assert left[column].to_numpy() == pytest.approx(right[column].to_numpy(), rel=1e-12)


def test_warmup_rows_are_dropped_not_backfilled() -> None:
    """The early bars have no trailing window. Filling them from the future is the
    convenient option and the wrong one."""
    frame = panel_frame(bars(100), window=60, min_bars=30)
    first = frame["as_of"].min()
    assert first > START + dt.timedelta(days=25)
    assert frame["adv_notional"].null_count() == 0
    assert frame["volatility_daily"].null_count() == 0


def test_adv_is_a_currency_amount_not_a_share_count() -> None:
    frame = panel_frame(bars(100, symbols=("A",)), window=20, spread_bps=1.0)
    source = bars(100, symbols=("A",))
    expected = float((source["close"] * source["volume"]).tail(20).mean())
    assert frame["adv_notional"][-1] == pytest.approx(expected, rel=1e-9)


def test_quote_volume_is_used_directly_when_present() -> None:
    """Crypto venues publish a currency volume already; multiplying it by price
    again would overstate depth by the price."""
    frame = bars(60, symbols=("A",)).with_columns(quote_volume=pl.lit(5e8))
    prepared = panel_frame(frame.drop("volume"), window=10, spread_bps=1.0)
    assert prepared["adv_notional"][-1] == pytest.approx(5e8)


def test_volatility_is_of_log_returns() -> None:
    source = bars(120, symbols=("A",))
    frame = panel_frame(source, window=30, spread_bps=1.0)
    returns = np.diff(np.log(source["close"].to_numpy()))
    assert frame["volatility_daily"][-1] == pytest.approx(
        float(np.std(returns[-30:], ddof=1)), rel=0.02
    )


# ----------------------------------------------------------------------------------
# Spreads
# ----------------------------------------------------------------------------------
def test_a_single_spread_applies_to_everything() -> None:
    frame = panel_frame(bars(80), window=20, spread_bps=7.5)
    assert frame["spread_bps"].unique().to_list() == [7.5]


def test_per_symbol_spreads_are_honoured() -> None:
    frame = panel_frame(bars(80), window=20, spread_bps={"A": 2.0, "B": 9.0})
    by_symbol = dict(frame.group_by("symbol").agg(pl.col("spread_bps").first()).iter_rows())
    assert by_symbol == {"A": 2.0, "B": 9.0}


def test_an_unknown_symbol_gets_the_wide_fallback_not_zero() -> None:
    """An instrument whose spread is unknown is not one whose spread is zero."""
    frame = panel_frame(bars(80), window=20, spread_bps={"A": 2.0})
    b_spread = frame.filter(pl.col("symbol") == "B")["spread_bps"][0]
    assert b_spread == DEFAULT_SPREAD_BPS
    assert DEFAULT_SPREAD_BPS >= 10.0, "the fallback must be punitive, not convenient"


def test_the_default_spread_is_wide_when_nothing_is_supplied() -> None:
    frame = panel_frame(bars(80), window=20)
    assert frame["spread_bps"].unique().to_list() == [DEFAULT_SPREAD_BPS]


def test_estimating_spreads_needs_highs_and_lows() -> None:
    with pytest.raises(ValueError, match="needs `high` and `low`"):
        panel_frame(bars(80).drop("high", "low"), window=20, estimate_spreads=True)


def test_estimation_falls_back_loudly_when_it_fails(caplog: pytest.LogCaptureFixture) -> None:
    """On liquid instruments these estimators measure daily volatility rather than
    spread, so a failure must be visible rather than smoothed over."""
    import logging

    # A strong trend with each close at the day's high makes the estimator's two
    # deviations take opposite signs, so it has no real root. Reproduces what
    # happened on seven of eight real crypto majors.
    n = 120
    base = 100 * np.exp(np.cumsum(np.full(n, 0.03)))
    trending = pl.DataFrame(
        {
            "symbol": ["A"] * n,
            "as_of": [START + dt.timedelta(days=i) for i in range(n)],
            "close": base,
            "open": base * 0.999,
            "high": base,
            "low": base * 0.99,
            "volume": np.full(n, 1e6),
        }
    )
    with caplog.at_level(logging.WARNING):
        frame = panel_frame(trending, window=20, estimate_spreads=True)
    assert frame["spread_bps"].min() == DEFAULT_SPREAD_BPS, "falls back, not to zero"
    assert "spread_estimate_failed" in caplog.text


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------
def test_missing_columns_are_named() -> None:
    with pytest.raises(ValueError, match=r"\['close'\]"):
        panel_frame(bars(50).drop("close"))


def test_no_volume_means_no_capacity_aware_cost() -> None:
    with pytest.raises(ValueError, match="no capacity-aware cost is possible"):
        panel_frame(bars(50).drop("volume"))


def test_output_carries_exactly_what_the_panel_needs() -> None:
    frame = panel_frame(bars(80), window=20, spread_bps=3.0)
    assert set(frame.columns) == {
        "symbol",
        "as_of",
        "close",
        "open",
        "adv_notional",
        "volatility_daily",
        "spread_bps",
    }


# ----------------------------------------------------------------------------------
# Rebalance calendar
# ----------------------------------------------------------------------------------
def test_rebalance_dates_count_bars_not_calendar_days() -> None:
    """Bar-count based, so a market holiday cannot silently skip a rebalance."""
    dates = [START + dt.timedelta(days=i) for i in range(100)]
    monthly = rebalance_dates(dates, every=21, warmup=10)
    assert monthly[0] == dates[10]
    assert monthly[1] == dates[31]
    assert all(d in dates for d in monthly)


def test_rebalance_every_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rebalance_dates([START], every=0)


def test_default_window_is_a_quarter() -> None:
    """Long enough for the estimates to be stable, short enough to react."""
    assert 40 <= DEFAULT_WINDOW <= 90
