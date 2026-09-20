"""Turning a long-format price panel into dense matrices the engine can march over.

The engine's inner loop is a recursion -- equity at bar ``t`` depends on equity at
``t-1`` -- so it cannot be expressed as a single polars operation. The split is:
polars does the joining, alignment and staleness logic on the long frame; this
module pivots the result into ``(bars x symbols)`` numpy arrays; and the engine runs
a tight numpy loop over bars with the whole cross-section vectorised inside it.

That keeps the cross-section vectorised, which is where the size is, while leaving
the sequential part explicit and exact rather than approximated to make it fit a
vectorised mould.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from quantlab.backtest.conventions import ExecutionTiming, StalenessPolicy
from quantlab.logging import get_logger

__all__ = ["REQUIRED_PANEL_COLUMNS", "Panel"]

log = get_logger("quantlab.backtest.panel")

REQUIRED_PANEL_COLUMNS = (
    "symbol",
    "as_of",
    "close",
    "adv_notional",
    "volatility_daily",
    "spread_bps",
)

#: Optional columns, with the value used when a panel omits them.
_OPTIONAL_DEFAULTS: dict[str, float] = {"open": float("nan"), "dividend": 0.0}


@dataclass(frozen=True, slots=True)
class Panel:
    """Dense, aligned market data for one backtest.

    Every array is ``(n_bars, n_symbols)``. ``NaN`` means "no data", and the engine
    is required to treat that as untradable rather than as a zero.
    """

    symbols: tuple[str, ...]
    dates: tuple[dt.datetime, ...]
    #: Mark-to-market price at each bar's close.
    close: np.ndarray
    #: Price at which trades execute, per the configured timing.
    exec_price: np.ndarray
    adv_notional: np.ndarray
    volatility_daily: np.ndarray
    spread_bps: np.ndarray
    #: Cash dividend per share, on its ex-date.
    dividend: np.ndarray
    #: True where the instrument has data fresh enough to trade.
    tradable: np.ndarray
    #: Bar index of each symbol's last observation, or ``-1`` if it never traded.
    last_observed: np.ndarray
    #: True for symbols whose data stops before the end of the sample.
    delisted: np.ndarray
    #: Number of (bar, symbol) cells filled forward from an earlier bar.
    forward_filled_cells: int
    staleness: StalenessPolicy
    #: The convention this panel's ``exec_price`` was built for. The engine checks
    #: it against its own config: a panel priced at the close driving a run that
    #: reports open execution would misstate the convention in the result, and for
    #: SAME_CLOSE it would mislabel a contaminated run as a clean one.
    timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN

    @property
    def n_bars(self) -> int:
        return len(self.dates)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_bars, self.n_symbols)

    def symbol_index(self) -> dict[str, int]:
        return {symbol: index for index, symbol in enumerate(self.symbols)}

    def date_index(self) -> dict[dt.datetime, int]:
        return {date: index for index, date in enumerate(self.dates)}

    # ------------------------------------------------------------------ build --
    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN,
        staleness: StalenessPolicy | None = None,
    ) -> Panel:
        """Build a panel from a long-format frame.

        Required columns: ``symbol``, ``as_of``, ``close``, ``adv_notional``,
        ``volatility_daily``, ``spread_bps``. Optional: ``open`` (needed for
        ``NEXT_OPEN`` execution) and ``dividend``.
        """
        policy = staleness or StalenessPolicy()
        missing = [c for c in REQUIRED_PANEL_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"panel is missing required columns {missing}")
        if frame.height == 0:
            raise ValueError("panel is empty; there is nothing to backtest")

        for column, default in _OPTIONAL_DEFAULTS.items():
            if column not in frame.columns:
                frame = frame.with_columns(pl.lit(default).alias(column))

        # Reindex onto the dense (bar x symbol) grid by integer position and a
        # direct scatter. Two earlier approaches were much slower on the spec's
        # 30-year, 3,000-name target: pivoting each value column separately took
        # 111 seconds, and a composite (date, symbol) join took 8 seconds of an
        # 11-second budget, because hashing 23 million string keys dominates. Two
        # small joins onto lookup tables plus a scatter does the same work in
        # about a second.
        frame = frame.sort("as_of", "symbol")
        dates = tuple(frame["as_of"].unique().sort().to_list())
        symbols = tuple(frame["symbol"].unique().sort().to_list())
        n_bars, n_symbols = len(dates), len(symbols)
        shape = (n_bars, n_symbols)

        indexed = frame.join(
            pl.DataFrame(
                {"symbol": list(symbols), "_symbol_ix": np.arange(n_symbols, dtype=np.int64)}
            ),
            on="symbol",
            how="inner",
        ).join(
            pl.DataFrame({"as_of": list(dates), "_date_ix": np.arange(n_bars, dtype=np.int64)}),
            on="as_of",
            how="inner",
        )
        flat_index = indexed["_date_ix"].to_numpy() * n_symbols + indexed["_symbol_ix"].to_numpy()

        # A scatter silently keeps the last writer, so duplicates must be caught
        # before it happens. bincount is O(n) where np.unique would sort.
        occupancy = np.bincount(flat_index, minlength=n_bars * n_symbols)
        if occupancy.max() > 1:
            raise ValueError(
                f"{int((occupancy > 1).sum())} (symbol, as_of) pairs appear more than "
                "once. The panel must carry exactly one observation per instrument "
                "per bar; deduplicate before backtesting rather than letting the "
                "reindex keep whichever row happened to come last."
            )

        def matrix(column: str, fill: float = float("nan")) -> np.ndarray:
            values = np.full(n_bars * n_symbols, fill, dtype=np.float64)
            values[flat_index] = indexed[column].to_numpy().astype(np.float64)
            return values.reshape(shape)

        close = matrix("close")
        open_ = matrix("open")
        dividend = matrix("dividend", fill=0.0)

        observed = ~np.isnan(close)
        filled_close, staleness_bars = _forward_fill(close, policy.max_bars)
        fresh = observed | (staleness_bars <= policy.max_bars)
        forward_filled = int((fresh & ~observed).sum())

        if forward_filled and policy.max_bars > 0:
            log.info(
                "backtest.panel.forward_filled",
                cells=forward_filled,
                max_bars=policy.max_bars,
                policy=policy.describe(),
            )

        if timing is ExecutionTiming.NEXT_OPEN:
            if np.isnan(open_).all():
                raise ValueError(
                    "NEXT_OPEN execution needs an `open` column, and the panel has "
                    "none. Supply opens, or choose NEXT_CLOSE -- do not let trades "
                    "silently execute at the close that generated the signal."
                )
            exec_price, _ = _forward_fill(open_, policy.max_bars)
            # An instrument with a close but no open is tradable at its close; the
            # alternative is to drop a bar of real data over a missing field.
            exec_price = np.where(np.isnan(exec_price), filled_close, exec_price)
        else:
            exec_price = filled_close

        tradable = fresh & np.isfinite(filled_close) & np.isfinite(exec_price) & (filled_close > 0)

        last_observed = np.where(
            observed.any(axis=0), observed.shape[0] - 1 - observed[::-1].argmax(axis=0), -1
        )
        delisted = (last_observed >= 0) & (last_observed < len(dates) - 1)
        if delisted.any():
            log.info(
                "backtest.panel.delistings",
                symbols=int(delisted.sum()),
                note="a delisting return is applied on each one's final bar",
            )

        return cls(
            symbols=symbols,
            dates=dates,
            close=filled_close,
            exec_price=exec_price,
            adv_notional=matrix("adv_notional", fill=0.0),
            volatility_daily=matrix("volatility_daily", fill=0.0),
            spread_bps=matrix("spread_bps", fill=0.0),
            dividend=dividend,
            tradable=tradable,
            last_observed=last_observed,
            delisted=delisted,
            forward_filled_cells=forward_filled,
            staleness=policy,
            timing=timing,
        )

    def align_weights(self, weights: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Map a long-format target weight frame onto the panel's grid.

        Returns ``(weights, is_rebalance)``: the target weight matrix, and a
        per-bar flag marking the bars on which targets were actually supplied.
        Between those bars the engine holds *shares* constant and lets weights
        drift, which is what a real portfolio does.
        """
        for column in ("symbol", "as_of", "weight"):
            if column not in weights.columns:
                raise ValueError(f"weights frame is missing `{column}`")

        symbol_index = self.symbol_index()
        date_index = self.date_index()

        unknown_symbols = set(weights["symbol"].unique().to_list()) - set(self.symbols)
        if unknown_symbols:
            raise ValueError(
                f"weights reference {len(unknown_symbols)} symbols absent from the "
                f"panel, e.g. {sorted(unknown_symbols)[:5]}. A target in an "
                "instrument with no price data would silently never be filled."
            )
        unknown_dates = set(weights["as_of"].unique().to_list()) - set(self.dates)
        if unknown_dates:
            raise ValueError(
                f"weights are dated on {len(unknown_dates)} bars absent from the "
                f"panel, e.g. {sorted(unknown_dates)[:3]}. Align the rebalance "
                "calendar to the trading calendar before backtesting."
            )

        # Pivoted, not looped. A 30-year monthly rebalance over 3,000 names is a
        # million weight rows, and a Python loop over those costs more than the
        # entire backtest it is preparing for.
        del symbol_index, date_index
        wide = weights.pivot(
            index="as_of", on="symbol", values="weight", aggregate_function="first"
        )
        rebalance_dates = set(wide["as_of"].to_list())

        grid = pl.DataFrame({"as_of": list(self.dates)})
        aligned = (
            grid.join(wide, on="as_of", how="left")
            .sort("as_of")
            .with_columns(
                [
                    pl.col(symbol).fill_null(0.0)
                    if symbol in wide.columns
                    else pl.lit(0.0).alias(symbol)
                    for symbol in self.symbols
                ]
            )
        )
        matrix = aligned.select(list(self.symbols)).to_numpy().astype(np.float64)
        is_rebalance = np.array([date in rebalance_dates for date in self.dates], dtype=bool)
        # Off-rebalance rows carry zeros from the fill; the engine only reads the
        # matrix where is_rebalance is set, but zeroing them keeps the array honest.
        matrix[~is_rebalance] = 0.0
        return matrix, is_rebalance


def _forward_fill(values: np.ndarray, max_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill down each column, and count how stale each cell is.

    The staleness count is returned rather than applied so the caller decides what
    breaching the limit means. Filling without limit is how a halted instrument
    becomes a position with no volatility and no drawdown -- and a risk-parity
    weighting will then allocate *more* to it, precisely because it looks calm.
    """
    filled = values.copy()
    staleness = np.zeros(values.shape, dtype=np.int32)
    n_bars = values.shape[0]

    for bar in range(1, n_bars):
        missing = np.isnan(filled[bar])
        if not missing.any():
            continue
        carried = missing & ~np.isnan(filled[bar - 1])
        filled[bar, carried] = filled[bar - 1, carried]
        staleness[bar, missing] = staleness[bar - 1, missing] + 1

    if max_bars >= 0:
        too_stale = staleness > max_bars
        filled[too_stale] = np.nan

    return filled, staleness
