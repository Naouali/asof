"""Double-entry accounting, reconciled every bar.

Spec section 6.4: position and cash accounting reconciled to the cent every bar,
with an assertion. The assertion is only worth anything if the two sides are
computed by genuinely different routes, so they are:

* **Balance sheet.** ``equity = cash + Σ shares · price``. Follows the cash ledger:
  every trade, dividend and fee moves cash by an explicit amount.
* **Income statement.** ``equity = previous equity + mark-to-market P&L +
  dividends − trade costs − holding costs``. Follows the P&L, and never looks at
  cash at all.

In exact arithmetic these are identically equal. Any gap is either floating-point
drift -- which the tolerance absorbs -- or a bug, and a bug in this layer produces
a plausible-looking equity curve that is simply wrong. So the gap is checked rather
than assumed, and a breach stops the run.

The mark-to-market term accounts for trading *within* the bar::

    pnl = Σ shares_before · (exec_price − previous_close)
        + Σ shares_after  · (close − exec_price)

With ``exec_price == close`` this collapses to the familiar
``Σ shares_before · (close − previous_close)``. With execution at the open it
correctly splits the bar at the moment the position actually changed, which is
what makes an open-execution backtest differ from a close-execution one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantlab.logging import get_logger

__all__ = ["AccountingError", "Ledger", "ReconciliationReport"]

log = get_logger("quantlab.backtest.accounting")

#: Absolute currency tolerance -- one cent, per the spec.
CENT = 0.01

#: Relative tolerance, because float64 carries about 15 significant digits and a
#: cent of a billion-dollar book is below that. Without this, the check would fail
#: on arithmetic noise at large AUM and the natural fix would be to delete it.
RELATIVE_TOLERANCE = 1e-9


class AccountingError(RuntimeError):
    """The balance sheet and the income statement disagree.

    Never downgraded to a warning. A backtest whose books do not balance is not
    slightly wrong; the number it reports is unrelated to the strategy it claims
    to measure.
    """


@dataclass(slots=True)
class ReconciliationReport:
    """Evidence that the books balanced, kept so the claim can be audited."""

    bars_checked: int = 0
    worst_absolute_gap: float = 0.0
    worst_bar: int = -1
    tolerance_used: float = CENT

    def record(self, bar: int, gap: float, tolerance: float) -> None:
        self.bars_checked += 1
        if abs(gap) > abs(self.worst_absolute_gap):
            self.worst_absolute_gap = gap
            self.worst_bar = bar
            self.tolerance_used = tolerance

    def describe(self) -> str:
        if self.bars_checked == 0:
            return "no bars reconciled"
        return (
            f"{self.bars_checked:,} bars reconciled; worst discrepancy "
            f"{self.worst_absolute_gap:+.6f} at bar {self.worst_bar} "
            f"(tolerance {self.tolerance_used:.6f})"
        )


@dataclass(slots=True)
class Ledger:
    """Cash and share state, advanced one bar at a time.

    Shares are held in *split-adjusted* units, matching the price series the data
    layer produces. That is why there is no split event here: an adjusted price
    series has already absorbed them, and a dividend series on the same adjustment
    basis stays consistent with it. Mixing bases -- unadjusted dividends against
    adjusted prices -- would silently misstate every total return, which is why
    :mod:`quantlab.backtest.vectorised` documents the requirement rather than
    trying to detect it.
    """

    n_symbols: int
    initial_equity: float
    shares: np.ndarray = field(init=False)
    cash: float = field(init=False)
    equity: float = field(init=False)
    report: ReconciliationReport = field(default_factory=ReconciliationReport)

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self.shares = np.zeros(self.n_symbols, dtype=np.float64)
        self.cash = float(self.initial_equity)
        self.equity = float(self.initial_equity)

    # ------------------------------------------------------------------ step ---
    def settle(
        self,
        *,
        bar: int,
        shares_after: np.ndarray,
        exec_price: np.ndarray,
        previous_close: np.ndarray,
        close: np.ndarray,
        dividend_per_share: np.ndarray,
        trade_costs: float,
        holding_costs: float,
    ) -> dict[str, float]:
        """Advance one bar and reconcile. Returns the bar's P&L decomposition.

        ``shares_after`` is the position held out of this bar; the ledger's own
        ``shares`` is the position held into it.
        """
        shares_before = self.shares
        traded = shares_after - shares_before

        # Prices are NaN where an instrument has no data. A position cannot exist
        # there -- the engine liquidates first -- so treating NaN as zero here is
        # safe and keeps one missing cell from poisoning the whole sum.
        exec_safe = np.nan_to_num(exec_price, nan=0.0)
        prev_safe = np.nan_to_num(previous_close, nan=0.0)
        close_safe = np.nan_to_num(close, nan=0.0)

        dividends = float(np.dot(shares_before, np.nan_to_num(dividend_per_share, nan=0.0)))
        mark_pnl = float(
            np.dot(shares_before, exec_safe - prev_safe)
            + np.dot(shares_after, close_safe - exec_safe)
        )
        trade_cash = float(np.dot(traded, exec_safe))

        previous_equity = self.equity
        self.cash = self.cash + dividends - trade_cash - trade_costs - holding_costs
        self.shares = shares_after.copy()

        balance_sheet = self.cash + float(np.dot(shares_after, close_safe))
        income_statement = previous_equity + mark_pnl + dividends - trade_costs - holding_costs

        self._reconcile(bar, balance_sheet, income_statement)
        self.equity = balance_sheet

        return {
            "mark_pnl": mark_pnl,
            "dividends": dividends,
            "trade_costs": trade_costs,
            "holding_costs": holding_costs,
            "trade_cash": trade_cash,
            "gross_pnl": mark_pnl + dividends,
            "net_pnl": self.equity - previous_equity,
        }

    def _reconcile(self, bar: int, balance_sheet: float, income_statement: float) -> None:
        gap = balance_sheet - income_statement
        tolerance = max(CENT, RELATIVE_TOLERANCE * max(abs(balance_sheet), abs(income_statement)))
        self.report.record(bar, gap, tolerance)

        if not np.isfinite(balance_sheet) or not np.isfinite(income_statement):
            raise AccountingError(
                f"bar {bar}: equity is not finite (balance sheet {balance_sheet}, "
                f"income statement {income_statement}). Almost always a NaN price "
                "reaching a live position."
            )
        if abs(gap) > tolerance:
            raise AccountingError(
                f"bar {bar}: the books do not balance. Balance sheet says "
                f"{balance_sheet:,.6f}, income statement says {income_statement:,.6f}, "
                f"a gap of {gap:+,.6f} against a tolerance of {tolerance:,.6f}. "
                "This is a bug in QuantLab's accounting, not in your strategy; the "
                "equity curve it would have produced is meaningless."
            )

    # ------------------------------------------------------------------ views --
    def position_values(self, close: np.ndarray) -> np.ndarray:
        values: np.ndarray = self.shares * np.nan_to_num(close, nan=0.0)
        return values

    def gross_exposure(self, close: np.ndarray) -> float:
        return float(np.abs(self.position_values(close)).sum())

    def net_exposure(self, close: np.ndarray) -> float:
        return float(self.position_values(close).sum())

    def leverage(self, close: np.ndarray) -> float:
        return self.gross_exposure(close) / self.equity if self.equity else float("inf")
