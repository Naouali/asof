"""Panel construction: staleness, delisting, alignment and the errors that matter."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.backtest.conventions import ExecutionTiming, StalenessPolicy
from quantlab.backtest.panel import Panel

START = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)


def rows(
    n_bars: int = 20,
    symbols: tuple[str, ...] = ("A", "B"),
    *,
    skip: set[tuple[int, str]] | None = None,
    truncate: dict[str, int] | None = None,
) -> pl.DataFrame:
    skip = skip or set()
    truncate = truncate or {}
    out = []
    for bar in range(n_bars):
        for index, symbol in enumerate(symbols):
            if (bar, symbol) in skip or bar >= truncate.get(symbol, n_bars):
                continue
            price = 100.0 + bar + index
            out.append(
                {
                    "symbol": symbol,
                    "as_of": START + dt.timedelta(days=bar),
                    "close": price,
                    "open": price - 0.5,
                    "adv_notional": 1e8,
                    "volatility_daily": 0.02,
                    "spread_bps": 3.0,
                }
            )
    return pl.DataFrame(out)


# ----------------------------------------------------------------------------------
# Shape and alignment
# ----------------------------------------------------------------------------------
def test_panel_is_dense_and_sorted() -> None:
    panel = Panel.from_frame(rows())
    assert panel.shape == (20, 2)
    assert panel.symbols == ("A", "B")
    assert list(panel.dates) == sorted(panel.dates)
    assert panel.tradable.all()


def test_missing_columns_are_named() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        Panel.from_frame(rows().drop("adv_notional"))


def test_empty_panel_is_rejected() -> None:
    with pytest.raises(ValueError, match="nothing to backtest"):
        Panel.from_frame(rows().head(0))


def test_duplicate_observations_are_rejected() -> None:
    """A scatter silently keeps the last writer, so duplicates must be caught
    before the reindex, not discovered in the results."""
    doubled = pl.concat([rows(), rows().head(1)])
    with pytest.raises(ValueError, match="appear more than once"):
        Panel.from_frame(doubled)


def test_optional_columns_default_rather_than_failing() -> None:
    panel = Panel.from_frame(rows().drop("open"), timing=ExecutionTiming.NEXT_CLOSE)
    assert (panel.dividend == 0.0).all()
    assert np.array_equal(panel.exec_price, panel.close)


# ----------------------------------------------------------------------------------
# Staleness
# ----------------------------------------------------------------------------------
def test_short_gaps_are_forward_filled_within_the_limit() -> None:
    panel = Panel.from_frame(rows(skip={(5, "A"), (6, "A")}), staleness=StalenessPolicy(max_bars=5))
    assert panel.close[5, 0] == panel.close[4, 0]
    assert panel.close[6, 0] == panel.close[4, 0]
    assert panel.tradable[6, 0]
    assert panel.forward_filled_cells == 2


def test_long_gaps_breach_the_limit_and_become_untradable() -> None:
    """Forward-filling without a limit turns a halted instrument into a position
    with no volatility and no drawdown -- and a risk-parity weighting will then
    allocate *more* to it, precisely because it looks calm."""
    gap = {(bar, "A") for bar in range(5, 12)}
    panel = Panel.from_frame(rows(skip=gap), staleness=StalenessPolicy(max_bars=3))
    assert panel.tradable[7, 0], "within the limit"
    assert not panel.tradable[10, 0], "beyond it"
    assert np.isnan(panel.close[10, 0])


def test_strict_policy_forbids_forward_filling_entirely() -> None:
    panel = Panel.from_frame(rows(skip={(5, "A")}), staleness=StalenessPolicy.strict())
    assert not panel.tradable[5, 0]
    assert panel.forward_filled_cells == 0


def test_staleness_policy_describes_itself() -> None:
    assert "no forward-filling" in StalenessPolicy.strict().describe()
    assert "5 bars" in StalenessPolicy(max_bars=5).describe()


def test_negative_staleness_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StalenessPolicy(max_bars=-1)


# ----------------------------------------------------------------------------------
# Delisting
# ----------------------------------------------------------------------------------
def test_an_instrument_that_stops_is_marked_delisted() -> None:
    panel = Panel.from_frame(rows(truncate={"A": 12}))
    assert panel.delisted[0]
    assert not panel.delisted[1]
    assert panel.last_observed[0] == 11


def test_an_instrument_that_runs_to_the_end_is_not_delisted() -> None:
    panel = Panel.from_frame(rows())
    assert not panel.delisted.any()


# ----------------------------------------------------------------------------------
# Execution timing
# ----------------------------------------------------------------------------------
def test_next_open_executes_at_the_open() -> None:
    panel = Panel.from_frame(rows(), timing=ExecutionTiming.NEXT_OPEN)
    assert panel.exec_price[5, 0] == pytest.approx(panel.close[5, 0] - 0.5)


def test_next_close_executes_at_the_close() -> None:
    panel = Panel.from_frame(rows(), timing=ExecutionTiming.NEXT_CLOSE)
    assert np.array_equal(panel.exec_price, panel.close)


def test_open_execution_without_opens_fails_loudly() -> None:
    """Silently falling back to the close would turn an open-execution backtest
    into a close-execution one without saying so."""
    with pytest.raises(ValueError, match="needs an `open` column"):
        Panel.from_frame(rows().drop("open"), timing=ExecutionTiming.NEXT_OPEN)


def test_a_missing_open_falls_back_to_that_bar_close() -> None:
    frame = rows().with_columns(
        open=pl.when(pl.col("symbol") == "A").then(None).otherwise(pl.col("open"))
    )
    panel = Panel.from_frame(frame, timing=ExecutionTiming.NEXT_OPEN)
    assert panel.exec_price[5, 0] == panel.close[5, 0]
    assert panel.exec_price[5, 1] == pytest.approx(panel.close[5, 1] - 0.5)


# ----------------------------------------------------------------------------------
# Weight alignment
# ----------------------------------------------------------------------------------
def weights_for(bars: list[int], symbols: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"symbol": s, "as_of": START + dt.timedelta(days=b), "weight": 0.5}
            for b in bars
            for s in symbols
        ]
    )


def test_weights_align_to_the_panel_grid() -> None:
    panel = Panel.from_frame(rows())
    matrix, is_rebalance = panel.align_weights(weights_for([0, 10]))
    assert matrix.shape == panel.shape
    assert is_rebalance[0] and is_rebalance[10]
    assert not is_rebalance[5]
    assert matrix[10].tolist() == [0.5, 0.5]
    assert matrix[5].tolist() == [0.0, 0.0]


def test_weights_in_unknown_symbols_are_rejected() -> None:
    """A target in an instrument with no price data would never be filled, and the
    portfolio would silently run at less than its intended exposure."""
    panel = Panel.from_frame(rows())
    bad = weights_for([0], symbols=("A", "GHOST"))
    with pytest.raises(ValueError, match="absent from the"):
        panel.align_weights(bad)


def test_weights_on_non_trading_days_are_rejected() -> None:
    panel = Panel.from_frame(rows())
    bad = pl.DataFrame([{"symbol": "A", "as_of": START + dt.timedelta(days=999), "weight": 1.0}])
    with pytest.raises(ValueError, match="Align the rebalance"):
        panel.align_weights(bad)


def test_weights_missing_a_column_are_rejected() -> None:
    panel = Panel.from_frame(rows())
    with pytest.raises(ValueError, match="missing `weight`"):
        panel.align_weights(weights_for([0]).drop("weight"))
