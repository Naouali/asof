"""Benchmarking a hand-built factor against the academic reference.

Spec section 3.3: *"If your hand-built momentum factor doesn't correlate >0.9 with
Ken French's UMD, your construction has a bug."* That test is the cheapest bug
detector available for factor code, and it catches the errors that unit tests
cannot -- a sign flip that still produces plausible returns, a rebalance off by a
month, a universe filter that quietly excludes half the cross-section.

**It only works against a comparable universe.** UMD is built from every NYSE,
AMEX and NASDAQ stock. A factor built on fourteen ETFs is not a noisy version of
it; it is a different portfolio, and a low correlation says nothing about whether
either is correct. :func:`benchmark_correlation` therefore reports the number and
:func:`assess` decides what it means given how comparable the universes are --
rather than applying a 0.9 threshold to comparisons it does not apply to.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl

from quantlab.logging import get_logger

__all__ = ["BenchmarkResult", "Comparability", "benchmark_correlation"]

log = get_logger("quantlab.signals.benchmark")

#: The threshold spec section 3.3 names, for a like-for-like universe.
SAME_UNIVERSE_THRESHOLD = 0.9


class Comparability(str, Enum):
    """How much the two constructions share."""

    SAME_UNIVERSE = "same_universe"
    """Same instruments, same construction. The 0.9 threshold applies and a miss
    is a bug."""

    RELATED = "related"
    """Overlapping exposures -- a US equity ETF factor against a US stock factor.
    A moderate correlation is expected; a *negative* one is still a red flag."""

    DIFFERENT = "different"
    """Different asset classes entirely. The correlation is a curiosity, not a
    test, and asserting on it would be superstition."""


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    benchmark: str
    correlation: float
    observations: int
    comparability: Comparability

    @property
    def passes(self) -> bool:
        """Whether the comparison is evidence the construction is sound.

        Only meaningful at ``SAME_UNIVERSE``. At ``RELATED`` the weaker claim is
        that the sign is right; at ``DIFFERENT`` there is nothing to pass.
        """
        if self.comparability is Comparability.SAME_UNIVERSE:
            return self.correlation >= SAME_UNIVERSE_THRESHOLD
        if self.comparability is Comparability.RELATED:
            return self.correlation > 0.0
        return True

    def describe(self) -> str:
        lines = [
            f"{self.name} vs {self.benchmark}: correlation {self.correlation:+.3f} "
            f"over {self.observations:,} overlapping observations"
        ]
        if self.comparability is Comparability.SAME_UNIVERSE:
            verdict = "as expected" if self.passes else "TOO LOW -- suspect a bug"
            lines.append(
                f"  same universe, so the >{SAME_UNIVERSE_THRESHOLD} threshold applies: {verdict}"
            )
        elif self.comparability is Comparability.RELATED:
            lines.append(
                "  related but not comparable universes. A moderate correlation is "
                "expected; the threshold in spec section 3.3 does not apply, and "
                "only a negative correlation would be a red flag."
            )
        else:
            lines.append("  different asset classes. This number is a curiosity, not a test.")
        return "\n".join(lines)


def benchmark_correlation(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    name: str,
    benchmark_name: str,
    comparability: Comparability,
    date_column: str = "as_of",
    value_column: str = "return",
    benchmark_value_column: str = "value",
) -> BenchmarkResult:
    """Correlate a factor's returns with a published benchmark's.

    Joins on date, so only overlapping observations count -- and reports how many,
    because a correlation over forty days means nothing however high it is.
    """
    left = returns.select(
        pl.col(date_column).dt.date().alias("_day"), pl.col(value_column).alias("_left")
    )
    right = benchmark.select(
        pl.col(date_column).dt.date().alias("_day"),
        pl.col(benchmark_value_column).alias("_right"),
    )
    joined = left.join(right, on="_day", how="inner").drop_nulls()

    if joined.height < 30:
        raise ValueError(
            f"only {joined.height} overlapping observations between {name} and "
            f"{benchmark_name}; a correlation on that is noise. Check the date "
            "alignment before concluding anything about the construction."
        )

    correlation = float(np.corrcoef(joined["_left"].to_numpy(), joined["_right"].to_numpy())[0, 1])
    result = BenchmarkResult(
        name=name,
        benchmark=benchmark_name,
        correlation=correlation,
        observations=joined.height,
        comparability=comparability,
    )
    log.info(
        "signals.benchmark",
        name=name,
        benchmark=benchmark_name,
        correlation=round(correlation, 4),
        observations=joined.height,
        comparability=comparability.value,
        passes=result.passes,
    )
    return result
