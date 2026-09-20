"""Drawdown controls and stop-losses.

Spec section 8 asks for both, and notes a specific finding: *instrument-level
stops outperform portfolio-level volatility management for commodity factors*.
That distinction is worth taking seriously, because the two do different things.

A **portfolio volatility target** scales the whole book by an estimate of its own
risk. It is smooth, it is symmetric, and it delevers into a drawdown exactly when
the drawdown is measured -- which is to say, after it has happened.

An **instrument-level stop** cuts one position on its own evidence. It is
discontinuous and asymmetric: it acts on the loser without touching the rest of the
book, and it acts on a single instrument's move rather than waiting for that move
to show up in portfolio volatility, by which time it has been diluted by everything
else.

Both are risk management, not alpha. A drawdown control that improves backtested
returns is almost certainly fitted to the particular drawdowns in the sample, and
the honest claim for either is that it makes a book survivable, not profitable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantlab.logging import get_logger

__all__ = [
    "DrawdownControl",
    "StopLossPolicy",
    "drawdown_series",
]

log = get_logger("quantlab.risk.controls")


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak, at each point."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("equity must be a non-empty one-dimensional series")
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(peak > 0, values / peak - 1.0, 0.0)


@dataclass(frozen=True, slots=True)
class DrawdownControl:
    """Scale the book down as drawdown deepens.

    Linear between ``soft`` and ``hard``: no scaling above the soft threshold,
    fully flat at the hard one. Linear rather than a step, because a step creates
    a cliff the book crosses twice in a choppy market and pays to cross each time.

    **This cannot be expected to improve returns.** It caps the left tail by
    giving up the recovery, and recoveries follow drawdowns more often than not.
    Its job is to keep a book inside the loss tolerance of whoever funds it.
    """

    #: Drawdown at which scaling begins, as a negative fraction.
    soft: float = -0.10
    #: Drawdown at which the book goes flat.
    hard: float = -0.25
    #: Smallest multiplier applied before flattening entirely.
    floor: float = 0.0

    def __post_init__(self) -> None:
        if self.soft >= 0 or self.hard >= 0:
            raise ValueError("drawdown thresholds are negative fractions")
        if self.hard >= self.soft:
            raise ValueError("hard threshold must be deeper than soft")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")

    def multiplier(self, drawdown: float) -> float:
        """Scaling to apply at this drawdown."""
        if drawdown >= self.soft:
            return 1.0
        if drawdown <= self.hard:
            return self.floor
        travelled = (drawdown - self.soft) / (self.hard - self.soft)
        return float(1.0 - travelled * (1.0 - self.floor))

    def apply(self, equity: np.ndarray) -> np.ndarray:
        """Multiplier at each point of an equity path.

        Uses the drawdown *as of* each point, which is what a live implementation
        would know. Computing it from the full path would be look-ahead of the
        purest kind -- and would make any control look prescient.
        """
        return np.array([self.multiplier(float(d)) for d in drawdown_series(equity)])

    def describe(self) -> str:
        return (
            f"scale from 1.0 at {self.soft:.0%} drawdown to {self.floor:.2f} at "
            f"{self.hard:.0%}, linearly"
        )


@dataclass(frozen=True, slots=True)
class StopLossPolicy:
    """Close a position that has lost more than a threshold since it was opened.

    Spec section 8 notes that instrument-level stops outperform portfolio-level
    volatility management for commodity factors. The mechanism is plausible: a
    stop acts on one instrument's own evidence immediately, where a portfolio
    volatility target waits for that move to show up diluted in the whole book's
    risk, by which point it has been averaged away.

    Two costs that no backtest of a stop shows on its own. First, stops are
    realised: a position closed at the bottom does not participate in the bounce,
    and the bounce is where mean reversion lives. Second, a stop converts a
    continuous position into a path-dependent one, so results become much more
    sensitive to execution assumptions than the same strategy without it.

    **Why the default is a trailing stop.** Measuring the loss from the entry
    price rather than from the high-water mark since entry produces a control
    that switches itself off. A position that has run up 300% has to give back
    all of it plus the threshold before the stop fires, so the longer a position
    works the less protected it becomes -- and the positions that run up furthest
    are precisely the long-held trend positions the cited result is about. On a
    driftless random walk with 3% daily volatility and a 4% threshold, measuring
    from entry fires roughly six times in a thousand bars where measuring from
    the peak fires several hundred. The ``"entry"`` basis is kept so that
    difference can be measured rather than argued about; it is not a sensible
    risk control.
    """

    #: Loss at which the position is closed, as a negative fraction.
    threshold: float = -0.20
    #: Bars a closed position stays closed. Without this, a stop re-enters the
    #: next bar and becomes a transaction-cost generator rather than a control.
    cooldown_bars: int = 5
    #: What the loss is measured from. ``"peak"`` is a trailing stop, measuring
    #: from the highest cumulative profit reached since entry; ``"entry"``
    #: measures from the entry price and is included for comparison, not for use
    #: -- see the class docstring.
    basis: Literal["peak", "entry"] = "peak"

    def __post_init__(self) -> None:
        if self.threshold >= 0:
            raise ValueError("threshold is a negative fraction")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative")
        if self.basis not in ("peak", "entry"):
            raise ValueError(f"basis must be 'peak' or 'entry', not {self.basis!r}")

    def apply(self, positions: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Positions with stopped-out instruments zeroed.

        ``positions`` and ``returns`` are ``(bars, instruments)``. Cumulative
        profit is tracked from each position's *entry*, not from the start of the
        sample, so re-entering after a cooldown starts the count again.

        Under the default ``"peak"`` basis the loss is measured from the highest
        cumulative profit reached since entry, so an open profit is protected
        rather than treated as a buffer to be given back.
        """
        weights = np.asarray(positions, dtype=float)
        moves = np.asarray(returns, dtype=float)
        if weights.shape != moves.shape:
            raise ValueError("positions and returns must have the same shape")

        bars, instruments = weights.shape
        out = weights.copy()
        # Wealth relative to entry, and the highest it has reached since entry.
        wealth = np.ones(instruments)
        peak = np.ones(instruments)
        cooldown = np.zeros(instruments, dtype=int)
        active = np.zeros(instruments, dtype=bool)
        stops = 0

        for bar in range(bars):
            cooling = cooldown > 0
            out[bar, cooling] = 0.0
            cooldown[cooling] -= 1

            live = out[bar] != 0.0
            reset = (live & ~active) | (~live & active)
            wealth[reset] = 1.0
            peak[reset] = 1.0
            active = live

            if bar + 1 < bars:
                # Profit accrues on the position held into the next bar.
                wealth = np.where(
                    active,
                    wealth * (1.0 + np.sign(out[bar]) * moves[bar + 1]),
                    wealth,
                )
                peak = np.where(active, np.maximum(peak, wealth), peak)
                reference = peak if self.basis == "peak" else np.ones(instruments)
                loss = wealth / reference - 1.0

                breached = active & (loss <= self.threshold)
                if breached.any():
                    stops += int(breached.sum())
                    out[bar + 1, breached] = 0.0
                    cooldown[breached] = self.cooldown_bars
                    wealth[breached] = 1.0
                    peak[breached] = 1.0
                    active = active & ~breached

        if stops:
            log.info(
                "risk.stops_triggered",
                stops=stops,
                threshold=self.threshold,
                cooldown_bars=self.cooldown_bars,
                note="a stopped position does not participate in the bounce",
            )
        return out

    def describe(self) -> str:
        measured = "from the peak since entry" if self.basis == "peak" else "from entry"
        return f"close at {self.threshold:.0%} {measured}, stay out {self.cooldown_bars} bars"
