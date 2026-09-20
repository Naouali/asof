"""Empirical-Bayes luck adjustment across the whole signal library.

Every other statistic in this package judges one strategy at a time. This one
judges the *library*, and it is the most uncomfortable number the platform
produces.

The idea: if you test many signals and each t-statistic is an unbiased but noisy
estimate of that signal's true t, then under the null -- no signal has any edge --
the t-statistics are standard normal and their cross-sectional variance is **1**.
Any dispersion beyond 1 is real. So:

    shrinkage = 1 − 1/Var(t)

which is the posterior mean multiplier under a normal-normal model. Var(t) = 2
shrinks every t by half. Var(t) = 1 shrinks everything to zero, and the reading is
blunt: *the spread of your results is exactly what two hundred dart-throwing
monkeys would have produced, and you have found nothing.*

Spec section 7 asks for this to be displayed prominently, which is why it returns a
verdict in words rather than only a number. A shrinkage factor of 0.04 is easy to
glance past; "consistent with no signal in the library having any edge" is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantlab.logging import get_logger

__all__ = ["LuckAdjustment", "luck_adjust", "t_statistic"]

log = get_logger("quantlab.validation.luck")

#: Below this many signals the cross-sectional variance is itself so noisy that
#: the adjustment says more about the sample size than about the library.
MINIMUM_SIGNALS = 5

#: Below this, the estimate is usable but should be read with suspicion: the
#: sampling error of a variance from n observations is roughly sqrt(2/n).
RELIABLE_SIGNALS = 20


def t_statistic(sharpe_annual: float, years: float) -> float:
    """Convert an annualised Sharpe into a t-statistic.

    ``t = SR · sqrt(years)``. A Sharpe of 1.0 over four years is a t of 2.0 --
    which is the conventional significance threshold, and a useful reminder of how
    little four years of a good-looking strategy actually establishes.
    """
    if years <= 0:
        raise ValueError("years must be positive")
    return sharpe_annual * math.sqrt(years)


@dataclass(frozen=True, slots=True)
class LuckAdjustment:
    """What survives when the library is judged against its own dispersion."""

    names: tuple[str, ...]
    t_statistics: np.ndarray
    variance: float
    shrinkage: float
    shrunk_t: np.ndarray
    verdict: str
    reliable: bool

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def survivors(self) -> tuple[str, ...]:
        """Signals whose shrunk t-statistic still clears 2.0."""
        return tuple(
            name for name, value in zip(self.names, self.shrunk_t, strict=True) if abs(value) >= 2.0
        )

    def ranked(self) -> list[tuple[str, float, float]]:
        """``(name, raw t, shrunk t)``, strongest first."""
        rows = list(zip(self.names, self.t_statistics, self.shrunk_t, strict=True))
        return sorted(rows, key=lambda row: -abs(row[2]))

    def describe(self) -> str:
        lines = [
            f"Library of {self.count} signals: Var(t) = {self.variance:.2f}, "
            f"shrinkage factor {self.shrinkage:.2f}",
            f"  {self.verdict}",
        ]
        if self.survivors:
            lines.append(
                f"  {len(self.survivors)} signal(s) keep |t| >= 2 after shrinkage: "
                f"{', '.join(self.survivors[:5])}"
            )
        else:
            lines.append("  No signal keeps |t| >= 2 after shrinkage.")
        if not self.reliable:
            lines.append(
                f"  NOTE: only {self.count} signals. The cross-sectional variance is "
                "itself noisy at this size; treat the adjustment as indicative."
            )
        return "\n".join(lines)


def luck_adjust(
    t_statistics: dict[str, float] | np.ndarray,
    *,
    names: tuple[str, ...] | None = None,
) -> LuckAdjustment:
    """Shrink a library of t-statistics toward zero by its own dispersion.

    Pass a mapping of signal name to t-statistic, or an array with ``names``.
    """
    if isinstance(t_statistics, dict):
        names = tuple(t_statistics.keys())
        values = np.array(list(t_statistics.values()), dtype=float)
    else:
        values = np.asarray(t_statistics, dtype=float)
        names = names or tuple(f"signal_{i}" for i in range(len(values)))

    if len(values) != len(names):
        raise ValueError("names and t_statistics must be the same length")
    if len(values) < MINIMUM_SIGNALS:
        raise ValueError(
            f"the luck adjustment needs at least {MINIMUM_SIGNALS} signals to say "
            f"anything; {len(values)} tells you about your sample size, not your "
            "library. Build more signals before asking whether any of them are real."
        )
    if not np.isfinite(values).all():
        raise ValueError("t-statistics must all be finite")

    variance = float(np.var(values, ddof=1))
    shrinkage = max(0.0, 1.0 - 1.0 / variance) if variance > 0 else 0.0
    shrunk = values * shrinkage
    reliable = len(values) >= RELIABLE_SIGNALS

    if variance <= 1.05:
        verdict = (
            "The dispersion of your results is what pure chance produces. Under the "
            "null that no signal has an edge, Var(t) is exactly 1 -- this library is "
            "indistinguishable from that. The correct conclusion is that nothing has "
            "been found."
        )
    elif variance <= 1.5:
        verdict = (
            f"Var(t) = {variance:.2f} against 1.0 under pure chance. Most of the "
            f"spread is noise: only {shrinkage:.0%} of each t-statistic survives. "
            "Any individual result here needs far more evidence than its raw t suggests."
        )
    elif variance <= 4.0:
        verdict = (
            f"Var(t) = {variance:.2f}. There is real dispersion beyond chance, and "
            f"{shrinkage:.0%} of each t-statistic survives shrinkage. Treat the "
            "shrunk values as the honest ones."
        )
    else:
        verdict = (
            f"Var(t) = {variance:.2f}, well beyond chance. Either the library "
            "contains genuine and quite different effects, or the t-statistics are "
            "not comparable -- check that they cover similar samples before "
            "celebrating."
        )

    log.info(
        "validation.luck_adjustment",
        signals=len(values),
        variance=round(variance, 4),
        shrinkage=round(shrinkage, 4),
        survivors=int((np.abs(shrunk) >= 2.0).sum()),
    )

    return LuckAdjustment(
        names=names,
        t_statistics=values,
        variance=variance,
        shrinkage=shrinkage,
        shrunk_t=shrunk,
        verdict=verdict,
        reliable=reliable,
    )
