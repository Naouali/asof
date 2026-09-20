"""Sweeping a signal across instruments, and not believing the winners.

The workflow this exists for: *"does this idea work everywhere, or only on some
names?"* It is the most natural question to ask and the most dangerous one to
answer by eye, because the answer always looks like yes. Test anything across
fourteen instruments and the dispersion alone hands you a best one; test it
across five hundred and you are guaranteed something at ``t`` above 4.

So the sweep does two things that reading a results table does not.

**Every cell is a trial.** Sweeping fourteen instruments records fourteen trials,
not one, and the deflated Sharpe of the best cell is deflated against all
fourteen. There is no way to run a search here and then quote the winner as
though it were the only thing tried.

**The dispersion is judged against the null.** Under the hypothesis that the
signal has no edge anywhere, the cross-sectional variance of the cells'
t-statistics is exactly 1. Anything beyond that is real heterogeneity and is kept;
anything at or below it is shrunk to nothing. On the fourteen-ETF trend sweep
that variance came out at **0.84** -- less dispersion than chance produces -- so
the apparent winner at ``t = +3.03`` shrank to zero. It was the maximum of
fourteen draws from a null, and nothing about the Nasdaq.

That is the outcome to expect. A sweep that finds survivors is unusual, which is
the point of running one rather than trusting the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from quantlab.logging import get_logger
from quantlab.validation.luck import LuckAdjustment, luck_adjust, t_statistic

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["SweepCell", "SweepResult", "sweep_returns"]

log = get_logger("quantlab.validation.sweep")

#: A cell with less history than this is reported but never counted as a
#: survivor: a t-statistic over two years of daily data is mostly noise, and
#: letting one into the library inflates the variance the others are judged by.
MIN_YEARS = 3.0


def _article(word: str) -> str:
    """`a instrument` reads as carelessness, and a reader who notices it starts
    wondering what else was not checked."""
    return f"{'an' if word[:1].lower() in 'aeiou' else 'a'} {word}"


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One cell of the sweep -- an instrument, a sector, a regime."""

    name: str
    sharpe: float
    t_statistic: float
    observations: int
    years: float
    #: The t-statistic after the library's own dispersion is taken out.
    shrunk_t: float
    #: Whether this cell survives at |t| >= 2 after shrinkage.
    survives: bool
    #: True where the cell had too little history to be judged.
    thin: bool = False


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What a sweep found, and what remains of it after the correction."""

    signal: str
    dimension: str
    cells: tuple[SweepCell, ...]
    adjustment: LuckAdjustment

    @property
    def trials(self) -> int:
        """Cells tested. This is the trial count the winner deflates against."""
        return len(self.cells)

    @property
    def survivors(self) -> tuple[SweepCell, ...]:
        return tuple(cell for cell in self.cells if cell.survives)

    @property
    def best_raw(self) -> SweepCell | None:
        judged = [c for c in self.cells if not c.thin]
        return max(judged, key=lambda c: abs(c.t_statistic)) if judged else None

    def verdict(self) -> str:
        best = self.best_raw
        if best is None:
            return f"{self.signal}: no cell had enough history to judge."

        head = (
            f"{self.signal} swept across {self.trials} {self.dimension}(s). "
            f"Best raw result: {best.name} at Sharpe {best.sharpe:+.2f}, "
            f"t = {best.t_statistic:+.2f}."
        )
        if self.survivors:
            names = ", ".join(c.name for c in self.survivors)
            return (
                f"{head}\n{len(self.survivors)} cell(s) survive the correction: "
                f"{names}. That is unusual. Before acting on it, check that the "
                "surviving cells are not the same bet under different names, and "
                "run them through `quantlab report` for capacity and costs."
            )
        return (
            f"{head}\nNothing survives. The spread across "
            f"{self.dimension}s is Var(t) = {self.adjustment.variance:.2f}, against "
            "1.00 under the null that the signal has no edge anywhere, so "
            f"{best.name} is the maximum of {self.trials} noisy draws rather than "
            f"{_article(self.dimension)} the signal suits."
        )

    def describe(self) -> str:
        lines = [self.verdict(), ""]
        for cell in sorted(self.cells, key=lambda c: -abs(c.t_statistic)):
            flag = "  thin" if cell.thin else ("  SURVIVES" if cell.survives else "")
            lines.append(
                f"  {cell.name:<14} Sharpe {cell.sharpe:+6.2f}   "
                f"t {cell.t_statistic:+6.2f} -> {cell.shrunk_t:+6.2f}{flag}"
            )
        lines += ["", self.adjustment.describe()]
        return "\n".join(lines)

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "cell": c.name,
                    "sharpe": c.sharpe,
                    "t_statistic": c.t_statistic,
                    "shrunk_t": c.shrunk_t,
                    "years": c.years,
                    "observations": c.observations,
                    "survives": c.survives,
                    "thin": c.thin,
                }
                for c in self.cells
            ]
        )


def sweep_returns(
    returns_by_cell: dict[str, np.ndarray],
    *,
    signal: str,
    dimension: str = "instrument",
    periods_per_year: float = 252.0,
) -> SweepResult:
    """Judge a set of per-cell return series against their own dispersion.

    ``returns_by_cell`` maps a cell name to that cell's realised strategy
    returns. Cells with too little history are reported and excluded from the
    dispersion estimate, because a noisy t-statistic inflates the variance every
    other cell is then judged against -- which would make the correction
    *weaker* precisely when the evidence is thinnest.
    """
    if len(returns_by_cell) < 2:
        raise ValueError(
            "a sweep needs at least two cells. With one there is no dispersion to "
            "judge against, and the result is just a backtest."
        )

    measured: dict[str, tuple[float, float, int]] = {}
    thin: dict[str, tuple[float, float, int]] = {}
    for name, series in returns_by_cell.items():
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values)]
        years = len(values) / periods_per_year
        if len(values) < 2 or values.std(ddof=1) <= 0:
            thin[name] = (0.0, 0.0, len(values))
            continue
        sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year))
        stat = t_statistic(sharpe, years)
        (thin if years < MIN_YEARS else measured)[name] = (sharpe, stat, len(values))

    if len(measured) < 2:
        raise ValueError(
            f"only {len(measured)} cell(s) have at least {MIN_YEARS:g} years of "
            "history. A sweep judged on fewer is judged against a variance that is "
            "mostly noise."
        )

    adjustment = luck_adjust({name: stat for name, (_, stat, _) in measured.items()})
    shrunk = dict(zip(adjustment.names, adjustment.shrunk_t, strict=True))

    cells = [
        SweepCell(
            name=name,
            sharpe=sharpe,
            t_statistic=stat,
            observations=n,
            years=n / periods_per_year,
            shrunk_t=float(shrunk[name]),
            survives=abs(float(shrunk[name])) >= 2.0,
        )
        for name, (sharpe, stat, n) in measured.items()
    ]
    cells += [
        SweepCell(
            name=name,
            sharpe=sharpe,
            t_statistic=stat,
            observations=n,
            years=n / periods_per_year,
            shrunk_t=0.0,
            survives=False,
            thin=True,
        )
        for name, (sharpe, stat, n) in thin.items()
    ]

    result = SweepResult(
        signal=signal,
        dimension=dimension,
        cells=tuple(cells),
        adjustment=adjustment,
    )
    log.info(
        "validation.sweep",
        signal=signal,
        dimension=dimension,
        cells=len(cells),
        judged=len(measured),
        variance=round(adjustment.variance, 3),
        shrinkage=round(adjustment.shrinkage, 3),
        survivors=len(result.survivors),
    )
    return result


def time_series_positions(prices: np.ndarray, *, lookback: int = 252, lag: int = 1) -> np.ndarray:
    """Sign of the trailing return, lagged so the position is tradeable.

    The lag is not decoration. A position taken on the same close that produced
    the signal earns that day's return for free, and on autocorrelated data that
    single bar is worth most of the apparent edge.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least two bars")
    values = np.asarray(prices, dtype=float)
    position = np.full(len(values), np.nan)
    for t in range(lookback, len(values)):
        previous = values[t - lookback]
        if previous > 0 and np.isfinite(previous) and np.isfinite(values[t]):
            position[t] = np.sign(values[t] / previous - 1.0)
    shifted = np.roll(position, lag)
    shifted[: lookback + lag] = np.nan
    return shifted


def cell_returns_from_prices(
    prices: dict[str, np.ndarray],
    *,
    lookback: int = 252,
    cost_bps_per_turn: float = 2.0,
) -> dict[str, np.ndarray]:
    """Per-instrument strategy returns from a price series, net of a turn cost.

    The cost is deliberately present and deliberately crude. A sweep run gross
    ranks instruments by how much they trend *and* by how cheap they are to
    trade, and those are not the same thing -- an instrument that flips position
    twice a month can look better gross and be worse net.
    """
    out: dict[str, np.ndarray] = {}
    for name, series in prices.items():
        values = np.asarray(series, dtype=float)
        position = time_series_positions(values, lookback=lookback)
        moves = np.concatenate([[np.nan], np.diff(values) / values[:-1]])
        gross = position * moves
        turns = np.abs(np.concatenate([[0.0], np.diff(np.nan_to_num(position))]))
        net = gross - turns * cost_bps_per_turn * 1e-4
        out[name] = net[np.isfinite(net)]
    return out
