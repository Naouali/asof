"""Backtest output and its summary statistics.

One naming decision is load-bearing. The Sharpe ratio on this object is called
:attr:`PerformanceStats.sharpe_undeflated`, never ``sharpe``. Spec section 13
forbids reporting a Sharpe without its deflated counterpart and the trial count
that produced it, and the surest way to enforce that is to make the raw number
impossible to read as anything else. The deflation machinery arrives in Milestone
5; until then every summary says so in as many words.

The cost decomposition is kept separate rather than netted, because the
decomposition is the diagnostic: a strategy killed by spread needs a slower
rebalance, one killed by impact needs less size, and one killed by borrow needs a
different short book.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from quantlab.backtest.accounting import ReconciliationReport
from quantlab.costs.base import BPS
from quantlab.logging import get_logger
from quantlab.validation.statistics import SharpeEvidence

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.backtest.panel import Panel
    from quantlab.backtest.vectorised import BacktestConfig

__all__ = ["BacktestResult", "PerformanceStats"]

log = get_logger("quantlab.backtest.results")

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class PerformanceStats:
    """Summary statistics. Every one of them is in-sample until proven otherwise."""

    years: float
    total_return: float
    cagr: float
    volatility_annual: float
    #: Deliberately not called ``sharpe``. Without a trial count this number is
    #: an upper bound on what the strategy is worth, not an estimate of it.
    sharpe_undeflated: float
    #: The same ratio before any trading costs, so the cost of implementation is
    #: visible as a difference rather than hidden in a single net figure.
    sharpe_gross_undeflated: float
    max_drawdown: float
    max_drawdown_days: int
    turnover_annual: float
    cost_drag_bps_annual: float
    #: Higher moments, kept because the deflated Sharpe in Milestone 5 needs them:
    #: a strategy with negative skew and fat tails earns a larger haircut.
    skewness: float
    excess_kurtosis: float
    hit_rate: float
    average_gross_exposure: float
    average_net_exposure: float
    average_positions: float
    bars: int

    def as_dict(self) -> dict[str, float]:
        return {
            "years": self.years,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility_annual": self.volatility_annual,
            "sharpe_undeflated": self.sharpe_undeflated,
            "sharpe_gross_undeflated": self.sharpe_gross_undeflated,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_days": float(self.max_drawdown_days),
            "turnover_annual": self.turnover_annual,
            "cost_drag_bps_annual": self.cost_drag_bps_annual,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "hit_rate": self.hit_rate,
            "average_gross_exposure": self.average_gross_exposure,
            "average_net_exposure": self.average_net_exposure,
            "average_positions": self.average_positions,
            "bars": float(self.bars),
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything one run produced, and the evidence that its books balanced."""

    name: str
    #: Per-bar equity, P&L, cost decomposition, exposures and turnover.
    curve: pl.DataFrame
    #: Per-bar held weights, long format, non-zero only.
    positions: pl.DataFrame
    stats: PerformanceStats
    reconciliation: ReconciliationReport
    #: True when the run used look-ahead execution. Such a result is a diagnostic,
    #: never a finding.
    contaminated: bool
    #: Facts about the inputs that any reader of the numbers needs.
    data_quality: dict[str, Any]
    #: The Sharpe with its trial count and deflation attached. Spec section 13
    #: forbids reporting one without the other, so it travels with the result
    #: rather than being available on request.
    evidence: SharpeEvidence | None = None
    family: str = ""

    # ------------------------------------------------------------------ build --
    @classmethod
    def build(
        cls,
        *,
        name: str,
        panel: Panel,
        family: str = "",
        record_trial: bool = False,
        config: BacktestConfig,
        equity: np.ndarray,
        gross_pnl: np.ndarray,
        trade_cost: np.ndarray,
        holding_cost: np.ndarray,
        spread_cost: np.ndarray,
        impact_cost: np.ndarray,
        commission_cost: np.ndarray,
        borrow_cost: np.ndarray,
        financing_cost: np.ndarray,
        traded_notional: np.ndarray,
        gross_exposure: np.ndarray,
        net_exposure: np.ndarray,
        n_positions: np.ndarray,
        weights: np.ndarray,
        reconciliation: ReconciliationReport,
    ) -> BacktestResult:
        previous_equity = np.concatenate([[config.initial_equity], equity[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            net_return = np.where(previous_equity > 0, equity / previous_equity - 1.0, 0.0)
            gross_return = np.where(previous_equity > 0, gross_pnl / previous_equity, 0.0)
        net_return[0] = 0.0
        gross_return[0] = 0.0

        curve = pl.DataFrame(
            {
                "as_of": list(panel.dates),
                "equity": equity,
                "net_return": net_return,
                "gross_return": gross_return,
                "gross_pnl": gross_pnl,
                "trade_cost": trade_cost,
                "spread_cost": spread_cost,
                "impact_cost": impact_cost,
                "commission_cost": commission_cost,
                "holding_cost": holding_cost,
                "borrow_cost": borrow_cost,
                "financing_cost": financing_cost,
                "traded_notional": traded_notional,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "n_positions": n_positions,
            }
        ).with_columns(
            drawdown=pl.col("equity") / pl.col("equity").cum_max() - 1.0,
            leverage=pl.when(pl.col("equity") > 0)
            .then(pl.col("gross_exposure") / pl.col("equity"))
            .otherwise(None),
        )

        rows, cols = np.nonzero(weights)
        positions = pl.DataFrame(
            {
                "as_of": [panel.dates[r] for r in rows],
                "symbol": [panel.symbols[c] for c in cols],
                "weight": weights[rows, cols],
            }
        )

        stats = _compute_stats(panel, config, curve)
        data_quality = {
            "bars": panel.n_bars,
            "symbols": panel.n_symbols,
            "forward_filled_cells": panel.forward_filled_cells,
            "staleness_policy": panel.staleness.describe(),
            "delisted_symbols": int(panel.delisted.sum()),
            "delisting_return_applied": config.delisting_return,
            "execution": config.execution.value,
            "signal_lag_bars": config.signal_lag_bars,
        }

        if config.is_contaminated:
            log.error(
                "backtest.contaminated",
                name=name,
                reason="SAME_CLOSE execution trades at the close that produced the signal",
            )

        evidence = _record_and_deflate(
            name=name,
            family=family or name,
            config=config,
            stats=stats,
            curve=curve,
            panel=panel,
            record_trial=record_trial,
        )

        return cls(
            name=name,
            curve=curve,
            positions=positions,
            stats=stats,
            reconciliation=reconciliation,
            contaminated=config.is_contaminated,
            data_quality=data_quality,
            evidence=evidence,
            family=family or name,
        )

    # ------------------------------------------------------------------ views --
    @property
    def equity_curve(self) -> pl.DataFrame:
        return self.curve.select("as_of", "equity", "drawdown")

    def cost_decomposition_bps_annual(self) -> dict[str, float]:
        """Annualised cost, in basis points of average equity, by component."""
        average_equity = _scalar(self.curve["equity"].mean())
        if average_equity <= 0 or self.stats.years <= 0:
            return {}
        scale = 1.0 / average_equity / self.stats.years / BPS
        return {
            "spread": _scalar(self.curve["spread_cost"].sum()) * scale,
            "impact": _scalar(self.curve["impact_cost"].sum()) * scale,
            "commission": _scalar(self.curve["commission_cost"].sum()) * scale,
            "borrow": _scalar(self.curve["borrow_cost"].sum()) * scale,
            "financing": _scalar(self.curve["financing_cost"].sum()) * scale,
        }

    def summary(self) -> str:
        """A short, honest summary.

        It always says the Sharpe is undeflated. A Sharpe ratio without a trial
        count is the single most over-read number in quantitative finance, and the
        deflation that fixes it lands in Milestone 5.
        """
        stats = self.stats
        lines = [
            f"{self.name}: {stats.years:.1f}y, {stats.bars:,} bars",
            f"  CAGR {stats.cagr:+.2%}   vol {stats.volatility_annual:.2%}   "
            f"max DD {stats.max_drawdown:.2%} ({stats.max_drawdown_days}d)",
        ]
        if self.evidence is not None:
            verdict = "survives" if self.evidence.survives_deflation else "DOES NOT SURVIVE"
            lines.append(
                f"  Sharpe {stats.sharpe_undeflated:.2f} net / "
                f"{stats.sharpe_gross_undeflated:.2f} gross / "
                f"{self.evidence.haircut_annual:.2f} after haircuts"
            )
            lines.append(
                f"  deflated {self.evidence.deflated:.3f} over "
                f"{self.evidence.trials} trial(s) in family '{self.family}' "
                f"-- {verdict} deflation"
            )
        else:
            lines.append(
                f"  Sharpe {stats.sharpe_undeflated:.2f} net / "
                f"{stats.sharpe_gross_undeflated:.2f} gross  "
                f"[UNDEFLATED -- this run was not recorded as a trial]"
            )
        lines += [
            f"  turnover {stats.turnover_annual:.1%}/yr   "
            f"cost drag {stats.cost_drag_bps_annual:.0f} bp/yr",
            f"  {self.reconciliation.describe()}",
        ]
        if self.contaminated:
            lines.insert(
                1,
                "  *** CONTAMINATED: look-ahead execution. This is a diagnostic, "
                "not a finding. ***",
            )
        if self.data_quality["delisted_symbols"]:
            lines.append(
                f"  {self.data_quality['delisted_symbols']} instruments delisted; "
                f"{self.data_quality['delisting_return_applied']:+.0%} applied on exit"
            )
        return "\n".join(lines)


def _record_and_deflate(
    *,
    name: str,
    family: str,
    config: BacktestConfig,
    stats: PerformanceStats,
    curve: pl.DataFrame,
    panel: Panel,
    record_trial: bool,
) -> SharpeEvidence | None:
    """Record this run as a trial and deflate its Sharpe by the family's count.

    The fingerprint covers the engine settings, the sample window and the
    instrument set, so changing any of them is a new trial. It deliberately does
    *not* cover the weights file's contents: two different signals with the same
    engine settings are the same trial only if they are literally the same run,
    and the weight fingerprint is what distinguishes them.
    """
    from quantlab.config import get_settings
    from quantlab.validation.registry import Trial, TrialRegistry, config_fingerprint
    from quantlab.validation.statistics import (
        HaircutSchedule,
        SharpeEvidence,
        deflated_sharpe_ratio,
        expected_max_sharpe,
        probabilistic_sharpe_ratio,
    )

    settings = get_settings()
    registry = TrialRegistry(settings.layout.state)

    fingerprint = config_fingerprint(
        name=name,
        initial_equity=config.initial_equity,
        execution=config.execution.value,
        max_staleness_bars=config.staleness.max_bars,
        delisting_return=config.delisting_return,
        borrow_bps_annual=config.borrow_bps_annual,
        financing_bps_annual=config.financing_bps_annual,
        execution_horizon_days=config.execution_horizon_days,
        commission_bps=config.commission_bps,
        symbols=list(panel.symbols),
        first_bar=panel.dates[0],
        last_bar=panel.dates[-1],
    )

    if record_trial:
        registry.record(
            Trial(
                family=family,
                config_hash=fingerprint,
                name=name,
                sharpe_annual=stats.sharpe_undeflated,
                observations=stats.bars,
                skewness=stats.skewness,
                excess_kurtosis=stats.excess_kurtosis,
                metadata={"bars": stats.bars, "years": round(stats.years, 3)},
            )
        )

    trials = registry.family(family)
    count = max(1, trials.count)
    observations = max(2, stats.bars - 1)
    bars_per_year = observations / stats.years if stats.years > 0 else 252.0
    per_bar = stats.sharpe_undeflated / math.sqrt(bars_per_year) if bars_per_year > 0 else 0.0
    variance = trials.sharpe_variance(observations)
    schedule = HaircutSchedule()

    return SharpeEvidence(
        gross_annual=stats.sharpe_gross_undeflated,
        net_annual=stats.sharpe_undeflated,
        haircut_annual=schedule.apply(stats.sharpe_undeflated),
        observations=observations,
        trials=count,
        sharpe_variance=variance,
        skewness=stats.skewness,
        excess_kurtosis=stats.excess_kurtosis,
        probabilistic=probabilistic_sharpe_ratio(
            per_bar,
            observations=observations,
            skewness=stats.skewness,
            excess_kurtosis=stats.excess_kurtosis,
        ),
        deflated=deflated_sharpe_ratio(
            per_bar,
            observations=observations,
            trials=count,
            sharpe_variance=variance,
            skewness=stats.skewness,
            excess_kurtosis=stats.excess_kurtosis,
        ),
        expected_max_from_search=expected_max_sharpe(count, variance),
        haircuts=schedule,
    )


def _compute_stats(panel: Panel, config: BacktestConfig, curve: pl.DataFrame) -> PerformanceStats:
    equity = curve["equity"].to_numpy()
    net = curve["net_return"].to_numpy()[1:]
    gross = curve["gross_return"].to_numpy()[1:]

    span_days = (panel.dates[-1] - panel.dates[0]).total_seconds() / 86400.0
    years = max(span_days / DAYS_PER_YEAR, 1e-9)
    # Annualisation from the observed bar count rather than an assumed 252, so a
    # weekly or monthly panel is scaled correctly without being told.
    bars_per_year = max(len(net) / years, 1e-9)

    total_return = float(equity[-1] / config.initial_equity - 1.0)
    cagr = (
        float((equity[-1] / config.initial_equity) ** (1.0 / years) - 1.0)
        if equity[-1] > 0
        else -1.0
    )

    volatility = float(np.std(net, ddof=1) * np.sqrt(bars_per_year)) if len(net) > 1 else 0.0
    sharpe = _sharpe(net, bars_per_year)
    sharpe_gross = _sharpe(gross, bars_per_year)

    drawdown = curve["drawdown"].to_numpy()
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    max_drawdown_days = _longest_drawdown_days(panel.dates, drawdown)

    average_equity = float(np.mean(equity))
    turnover_annual = (
        _scalar(curve["traded_notional"].sum()) / average_equity / years
        if average_equity > 0
        else 0.0
    )
    total_costs = _scalar(curve["trade_cost"].sum()) + _scalar(curve["holding_cost"].sum())
    cost_drag = total_costs / average_equity / years / BPS if average_equity > 0 else 0.0

    return PerformanceStats(
        years=years,
        total_return=total_return,
        cagr=cagr,
        volatility_annual=volatility,
        sharpe_undeflated=sharpe,
        sharpe_gross_undeflated=sharpe_gross,
        max_drawdown=max_drawdown,
        max_drawdown_days=max_drawdown_days,
        turnover_annual=turnover_annual,
        cost_drag_bps_annual=cost_drag,
        skewness=_moment(net, 3),
        excess_kurtosis=_moment(net, 4) - 3.0,
        hit_rate=float((net > 0).mean()) if len(net) else 0.0,
        average_gross_exposure=_scalar(curve["gross_exposure"].mean()) / average_equity
        if average_equity > 0
        else 0.0,
        average_net_exposure=_scalar(curve["net_exposure"].mean()) / average_equity
        if average_equity > 0
        else 0.0,
        average_positions=_scalar(curve["n_positions"].mean()),
        bars=len(equity),
    )


def _scalar(value: object, default: float = 0.0) -> float:
    """Narrow a polars aggregate to a float.

    ``Series.sum()`` and ``.mean()`` are typed as a broad union covering dates and
    bytes, so every arithmetic use of one needs narrowing. Doing it here keeps the
    statistics readable.
    """
    return float(value) if isinstance(value, int | float) else default


def _sharpe(returns: np.ndarray, bars_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    if deviation == 0:
        return 0.0
    return float(np.mean(returns) / deviation * np.sqrt(bars_per_year))


def _moment(returns: np.ndarray, order: int) -> float:
    """Standardised central moment. Needed by the deflated Sharpe in Milestone 5."""
    if len(returns) < 3:
        return 0.0
    deviation = float(np.std(returns, ddof=0))
    if deviation == 0:
        return 0.0
    centred = returns - np.mean(returns)
    return float(np.mean(centred**order) / deviation**order)


def _longest_drawdown_days(dates: tuple, drawdown: np.ndarray) -> int:  # type: ignore[type-arg]
    """Calendar days in the longest stretch spent below a previous high.

    Duration matters as much as depth: a 20% drawdown recovered in a month is a bad
    quarter, and the same 20% taking four years is a career.
    """
    longest = 0
    start: int | None = None
    for index, value in enumerate(drawdown):
        if value < 0 and start is None:
            start = index
        elif value >= 0 and start is not None:
            longest = max(longest, (dates[index] - dates[start]).days)
            start = None
    if start is not None:
        longest = max(longest, (dates[-1] - dates[start]).days)
    return longest
