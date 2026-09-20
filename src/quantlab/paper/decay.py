"""Has the strategy decayed, or is it too early to tell?

The second question is the one that usually has an answer, and it is the one
people skip. A strategy backtested at Sharpe 1.2 that runs at 0.3 for six months
looks broken. It is not evidence of anything: the standard error of a Sharpe
estimate over 126 observations is about 0.09 *annualised at that sample length*,
which in practice means a six-month realised Sharpe is compatible with almost any
true value. Turning it off is as unjustified as leaving it on.

This module answers both questions, in that order. It reports the sample size
needed to distinguish the live result from the backtest, and only then whether
the observed difference is significant. Where it is not, it says so rather than
producing a number the reader will treat as a verdict anyway.

**The prior matters more than the test.** Spec section 7 and Milestone 5 put the
platform's working prior at live Sharpe being roughly half the backtest — 12%
in-sample selection, 40% post-publication decay, composing to ×0.53. A live
Sharpe of 0.6 against a backtest of 1.2 is therefore *exactly what was expected*
and is not decay at all. The comparison here is against the haircut expectation
by default, not against the raw backtest number, because comparing against the
raw number declares decay on every strategy that behaves as predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantlab.logging import get_logger

__all__ = [
    "DEFAULT_HAIRCUT",
    "DecayReport",
    "assess_decay",
    "sharpe_standard_error",
]

log = get_logger("quantlab.paper.decay")

#: The platform's working prior on how much of a backtest Sharpe survives
#: contact with reality. From Milestone 5's haircut schedule: 12% in-sample
#: selection and 40% post-publication decay compose to this.
DEFAULT_HAIRCUT = 0.53

#: Below this many observations nothing is concluded. Sixty trading days is
#: three months, and the standard error of a Sharpe over it is so wide that any
#: verdict would be noise.
MIN_OBSERVATIONS = 60


def sharpe_standard_error(sharpe: float, observations: int) -> float:
    """Standard error of a Sharpe estimate, per period.

    ``sqrt((1 + S²/2) / T)`` — the usual asymptotic form under IID returns. It
    is optimistic: real returns are autocorrelated and fat-tailed, both of which
    widen it, so a difference that is insignificant here is comfortably
    insignificant in reality.
    """
    if observations < 2:
        raise ValueError("a Sharpe standard error needs at least two observations")
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / observations)


@dataclass(frozen=True, slots=True)
class DecayReport:
    """What the live record does and does not say about the backtest."""

    strategy: str
    backtest_sharpe: float
    live_sharpe: float
    observations: int
    periods_per_year: float
    #: The backtest Sharpe after the platform's haircut -- what was actually
    #: expected, as opposed to what the backtest printed.
    expected_sharpe: float
    #: t-statistic of (live − expected), in units of the live estimate's own error.
    t_statistic: float
    #: Observations needed to detect a shortfall this size at |t| = 2.
    observations_needed: int

    @property
    def is_conclusive(self) -> bool:
        return self.observations >= MIN_OBSERVATIONS and abs(self.t_statistic) >= 2.0

    @property
    def has_decayed(self) -> bool:
        """Only true when the shortfall is both real and measurable."""
        return self.is_conclusive and self.t_statistic < 0

    @property
    def years_needed(self) -> float:
        return self.observations_needed / self.periods_per_year

    def verdict(self) -> str:
        if self.observations < MIN_OBSERVATIONS:
            return (
                f"{self.strategy}: {self.observations} observations is too few to "
                f"conclude anything. The standard error of a Sharpe over this "
                f"sample is {sharpe_standard_error(self.live_sharpe, max(2, self.observations)) * math.sqrt(self.periods_per_year):.2f} "
                "annualised, which is wider than most of the effects anyone is "
                "looking for."
            )
        if self.has_decayed:
            return (
                f"{self.strategy} has decayed: live Sharpe {self.live_sharpe:.2f} "
                f"against an expected {self.expected_sharpe:.2f} after haircuts "
                f"(t = {self.t_statistic:.2f} over {self.observations} "
                "observations). The shortfall is larger than sampling error "
                "explains."
            )
        if self.is_conclusive:
            return (
                f"{self.strategy} is running ahead of expectation: live Sharpe "
                f"{self.live_sharpe:.2f} against {self.expected_sharpe:.2f} "
                f"expected (t = +{self.t_statistic:.2f}). Treat that with the same "
                "suspicion as a good backtest."
            )
        return (
            f"{self.strategy}: no conclusion yet. Live Sharpe {self.live_sharpe:.2f} "
            f"against {self.expected_sharpe:.2f} expected is a t of "
            f"{self.t_statistic:.2f} over {self.observations} observations; "
            f"{self.observations_needed:,} ({self.years_needed:.1f} years) would be "
            "needed to call a shortfall this size. Turning the strategy off now "
            "would be as unjustified as leaving it on."
        )


def assess_decay(
    returns: np.ndarray,
    *,
    strategy: str,
    backtest_sharpe: float,
    periods_per_year: float = 252.0,
    haircut: float = DEFAULT_HAIRCUT,
) -> DecayReport:
    """Compare a live return series against what the backtest implied.

    ``haircut`` defaults to the platform's prior rather than to 1.0, so a
    strategy delivering half its backtest Sharpe is reported as behaving *as
    expected*. Comparing against the raw backtest number would declare decay on
    every strategy that did exactly what the literature says it would.
    """
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 2:
        raise ValueError("need at least two live observations to assess decay")

    deviation = float(series.std(ddof=1))
    live_per_period = float(series.mean()) / deviation if deviation > 0 else 0.0
    live_annual = live_per_period * math.sqrt(periods_per_year)
    expected_annual = backtest_sharpe * haircut

    # Both sides are put on a per-period footing before being compared, because
    # the standard error is a per-period quantity and annualising first would
    # scale the difference without scaling its uncertainty.
    expected_per_period = expected_annual / math.sqrt(periods_per_year)
    error = sharpe_standard_error(live_per_period, series.size)
    t_statistic = (live_per_period - expected_per_period) / error if error > 0 else 0.0

    # Observations to reach |t| = 2 on a gap this size. With no gap there is
    # nothing to detect, and no sample length would settle it.
    gap = abs(live_per_period - expected_per_period)
    needed = math.ceil(4.0 * (1.0 + 0.5 * live_per_period**2) / gap**2) if gap > 0 else 10**9

    report = DecayReport(
        strategy=strategy,
        backtest_sharpe=backtest_sharpe,
        live_sharpe=live_annual,
        observations=int(series.size),
        periods_per_year=periods_per_year,
        expected_sharpe=expected_annual,
        t_statistic=float(t_statistic),
        observations_needed=needed,
    )
    log.info(
        "paper.decay",
        strategy=strategy,
        live=round(live_annual, 3),
        expected=round(expected_annual, 3),
        t=round(float(t_statistic), 3),
        observations=int(series.size),
        conclusive=report.is_conclusive,
    )
    return report
