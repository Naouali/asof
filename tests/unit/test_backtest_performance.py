"""The throughput target from spec section 6.1.

A 30-year, 3,000-name cross-sectional backtest with full costs, in under ten
seconds. The number matters because this engine is what screens signals and sweeps
parameters: it runs thousands of times, so its speed sets how much research is
possible in a day.

Marked ``slow`` -- it allocates about a gigabyte and takes several seconds even
when it passes -- so it is excluded from the fast suite and run in CI.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import polars as pl
import pytest

from quantlab.backtest import BacktestConfig, ExecutionTiming, Panel, VectorisedBacktest

pytestmark = pytest.mark.slow

N_BARS = 7_560  # ~30 years of trading days
N_SYMBOLS = 3_000
BUDGET_SECONDS = 10.0


def build_frame() -> tuple[pl.DataFrame, list[dt.datetime], list[str]]:
    rng = np.random.default_rng(0)
    dates = [
        dt.datetime(1995, 1, 2, tzinfo=dt.UTC) + dt.timedelta(days=i * 365 // 252)
        for i in range(N_BARS)
    ]
    symbols = [f"S{i:04d}" for i in range(N_SYMBOLS)]
    closes = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0002, 0.018, (N_BARS, N_SYMBOLS)).astype(np.float32), axis=0)
    )
    flat = closes.ravel().astype(np.float64)
    frame = pl.DataFrame(
        {
            "symbol": np.tile(np.array(symbols), N_BARS),
            "as_of": np.repeat(np.array(dates, dtype="datetime64[us]"), N_SYMBOLS),
            "close": flat,
            "open": flat * 1.0005,
            "adv_notional": np.full(N_BARS * N_SYMBOLS, 2e8),
            "volatility_daily": np.full(N_BARS * N_SYMBOLS, 0.018),
            "spread_bps": np.full(N_BARS * N_SYMBOLS, 4.0),
        }
    ).with_columns(pl.col("as_of").dt.replace_time_zone("UTC"))
    return frame, dates, symbols


@pytest.mark.timeout(600)
def test_thirty_years_three_thousand_names_under_ten_seconds() -> None:
    frame, dates, symbols = build_frame()
    rng = np.random.default_rng(1)

    started = time.perf_counter()
    panel = Panel.from_frame(frame, timing=ExecutionTiming.NEXT_OPEN)
    panel_seconds = time.perf_counter() - started

    rebalance_bars = list(range(0, N_BARS, 21))
    picks = [rng.choice(N_SYMBOLS, 200, replace=False) for _ in rebalance_bars]
    weights = pl.DataFrame(
        {
            "symbol": [symbols[j] for pick in picks for j in pick],
            "as_of": [
                dates[bar] for bar, pick in zip(rebalance_bars, picks, strict=True) for _ in pick
            ],
            "weight": [1 / 200.0] * (len(rebalance_bars) * 200),
        }
    )

    started = time.perf_counter()
    result = VectorisedBacktest(BacktestConfig()).run(panel, weights, name="throughput")
    run_seconds = time.perf_counter() - started

    total = panel_seconds + run_seconds
    assert total < BUDGET_SECONDS, (
        f"panel {panel_seconds:.2f}s + backtest {run_seconds:.2f}s = {total:.2f}s, "
        f"over the {BUDGET_SECONDS}s budget in spec section 6.1"
    )
    # The run has to be a real one, not an empty loop that finished quickly.
    assert result.reconciliation.bars_checked == N_BARS - 1
    assert result.stats.turnover_annual > 0
    assert result.curve["trade_cost"].sum() > 0


@pytest.mark.timeout(600)
def test_the_books_still_balance_at_full_scale() -> None:
    """Floating-point error accumulates with bars and with book size. A tolerance
    that holds on a toy panel and fails on a real one would be worthless."""
    frame, dates, symbols = build_frame()
    rng = np.random.default_rng(2)
    panel = Panel.from_frame(frame, timing=ExecutionTiming.NEXT_OPEN)

    rebalance_bars = list(range(0, N_BARS, 63))
    picks = [rng.choice(N_SYMBOLS, 500, replace=False) for _ in rebalance_bars]
    weights = pl.DataFrame(
        {
            "symbol": [symbols[j] for pick in picks for j in pick],
            "as_of": [
                dates[bar] for bar, pick in zip(rebalance_bars, picks, strict=True) for _ in pick
            ],
            # Long/short, so cash goes both ways and the ledger is properly exercised.
            "weight": [(0.002 if i % 2 else -0.002) for _ in rebalance_bars for i in range(500)],
        }
    )
    result = VectorisedBacktest(BacktestConfig()).run(panel, weights, name="scale")
    assert abs(result.reconciliation.worst_absolute_gap) < 1.0
    assert result.reconciliation.bars_checked == N_BARS - 1
