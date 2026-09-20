"""Bid-ask spread: observed where possible, estimated where not.

Order of preference, and it is not close:

1. **Observed quotes.** Crypto order books are free and complete, so crypto
   spreads are measured, not guessed. Use them.
2. **Low-frequency estimators** from daily OHLC, for everything else. These are
   the only option for free equity and futures data, and they are *biased*.

The bias is the point of this module. Roll and Corwin-Schultz were calibrated in
an era of wide spreads; post-decimalisation they are known to be materially
**upward**-biased for liquid names, and both routinely produce negative or
undefined estimates that implementations quietly clamp to zero -- which converts
an upward bias in the mean into a strange bimodal distribution. Spec section 5
therefore forbids using them uncritically.

The correction here is deliberately **not** a fabricated constant. A shrinkage
factor copied from a paper's sample period into your universe is a guess wearing a
citation. Instead:

* :class:`SpreadCorrection` carries an explicit factor, a tick floor, a cap, and a
  provenance string saying where the factor came from.
* The default is :meth:`SpreadCorrection.uncalibrated`, which applies **no**
  shrinkage -- the conservative direction, since overstating spread understates
  capacity -- and marks itself uncalibrated so every tearsheet says so.
* :func:`calibrate_correction` fits the factor from data. This platform can
  actually do that: Binance gives real quoted spreads alongside real OHLC, so the
  estimators can be scored against ground truth on the one asset class where
  ground truth is free, and the resulting factor is at least measured on
  *something* rather than assumed.

Nothing here silently improves a number. A corrected spread always reports the raw
estimate alongside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from quantlab.costs.base import BPS
from quantlab.logging import get_logger

__all__ = [
    "MIN_MEANINGFUL_BPS",
    "AbdiRanaldoSpread",
    "CorwinSchultzSpread",
    "ObservedSpread",
    "RollSpread",
    "SpreadCorrection",
    "SpreadEstimate",
    "SpreadEstimator",
    "calibrate_correction",
    "resolution_floor_bps",
]

log = get_logger("quantlab.costs.spread")

Method = Literal["observed", "roll", "corwin_schultz", "abdi_ranaldo"]

#: Below this, an estimate is numerical noise rather than a spread. No instrument
#: that anyone trades quotes a spread of a thousandth of a basis point, so a
#: result this small means the estimator found nothing -- which must be an error,
#: not a suspiciously attractive cost.
MIN_MEANINGFUL_BPS = 1e-3


@dataclass(frozen=True, slots=True)
class SpreadEstimate:
    """A spread estimate, with everything needed to distrust it appropriately."""

    #: Full quoted spread in basis points, after correction.
    spread_bps: float
    #: The estimator's own output, before any correction, floor or cap.
    raw_bps: float
    method: Method
    #: Fraction of the input observations the estimator could actually use. Roll
    #: and Corwin-Schultz are undefined for a large share of real windows; a
    #: coverage of 0.4 means most of the sample told us nothing.
    coverage: float = 1.0
    #: Whether the correction applied was fitted to data or is an assumption.
    calibrated: bool = False
    #: The smallest spread this sample could have distinguished from zero, in basis
    #: points. See :func:`resolution_floor_bps`. An estimate near or below its own
    #: floor is noise, whatever number came out.
    resolution_floor_bps: float | None = None
    note: str = ""

    @property
    def half_spread_bps(self) -> float:
        return self.spread_bps / 2.0

    @property
    def is_trustworthy(self) -> bool:
        """Whether this estimate should drive a capacity number unchallenged."""
        if self.method == "observed":
            return True
        if self.resolution_floor_bps is not None and self.raw_bps < 2 * self.resolution_floor_bps:
            return False
        return self.calibrated and self.coverage >= 0.5

    @property
    def above_resolution(self) -> bool:
        """Whether the estimate is large enough to be distinguishable from noise."""
        if self.resolution_floor_bps is None:
            return True
        return self.raw_bps >= 2 * self.resolution_floor_bps


@dataclass(frozen=True, slots=True)
class SpreadCorrection:
    """How a raw low-frequency estimate is turned into a usable spread.

    ``factor`` multiplies the raw estimate. Values below 1 shrink it, which is what
    the post-decimalisation literature implies and what makes the number less
    conservative -- hence the insistence on provenance.
    """

    factor: float = 1.0
    #: No spread can be narrower than one tick. Prevents an estimator's noise from
    #: implying a free round trip.
    tick_floor_bps: float = 0.0
    #: Estimates above this are treated as estimator failure rather than as a
    #: genuinely illiquid name. 1000 bp is 10%, which no continuously quoted
    #: instrument sustains.
    cap_bps: float = 1000.0
    calibrated: bool = False
    provenance: str = "uncalibrated: no shrinkage applied"

    @classmethod
    def uncalibrated(cls, tick_floor_bps: float = 0.0) -> SpreadCorrection:
        """The default. No shrinkage, and honest about it.

        Leaves the known upward bias in place, which overstates cost and therefore
        understates capacity. That is the error this platform prefers to make, but
        it is still an error: calibrate when you can.
        """
        return cls(
            factor=1.0,
            tick_floor_bps=tick_floor_bps,
            calibrated=False,
            provenance=(
                "uncalibrated: raw estimator output, known to be upward-biased "
                "post-decimalisation. Costs are overstated and capacity understated."
            ),
        )

    def apply(self, raw_bps: float) -> float:
        if not np.isfinite(raw_bps) or raw_bps <= 0:
            return self.tick_floor_bps
        return float(np.clip(raw_bps * self.factor, self.tick_floor_bps, self.cap_bps))


def resolution_floor_bps(volatility_daily: float, observations: int) -> float:
    """The smallest spread a daily-bar estimator could distinguish from zero.

    Every low-frequency estimator identifies the spread as a small perturbation of
    a much larger quantity -- the daily price variation. The standard error of that
    identification falls like ``σ/sqrt(n)``, so a spread below roughly that value
    is indistinguishable from sampling noise, and the estimator will happily return
    a number anyway.

    This matters enormously and is easy to miss. Measured on 993 days of real
    Binance daily bars (2026-09-20), with a true quoted BTCUSDT spread of
    **0.001 bp**: Roll returned 167 bp, Corwin-Schultz 106 bp, Abdi-Ranaldo 62 bp,
    and Abdi-Ranaldo was outright undefined for seven of eight majors. Wrong by
    five orders of magnitude -- not "upward-biased" but measuring something else
    entirely, namely daily volatility.

    Reporting this floor is what lets a caller see the problem *without* knowing
    the true spread: if the floor is 10 bp and the instrument you are trading has
    a spread anywhere near a basis point, the estimator cannot see it.
    """
    if observations < 2 or volatility_daily <= 0:
        return float("inf")
    return float(volatility_daily / np.sqrt(observations) / BPS)


class SpreadEstimator(ABC):
    """Estimate a full quoted spread, in basis points, from price history.

    Deliberately **not** given a uniform ``estimate()`` signature: Roll needs
    closes, Corwin-Schultz needs highs and lows, Abdi-Ranaldo needs all three, and
    forcing them behind one signature would hide which inputs each actually uses.
    :attr:`required_columns` lets a caller dispatch from a frame instead.
    """

    method: Method

    def __init__(self, correction: SpreadCorrection | None = None) -> None:
        self.correction = correction or SpreadCorrection.uncalibrated()

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Columns this estimator reads, so a caller can check before calling."""

    @abstractmethod
    def estimate_frame(self, frame: pl.DataFrame) -> SpreadEstimate:
        """Estimate from a frame carrying :attr:`required_columns`."""

    def _finish(
        self,
        raw_bps: float,
        coverage: float,
        note: str = "",
        floor_bps: float | None = None,
    ) -> SpreadEstimate:
        if floor_bps is not None and raw_bps < 2 * floor_bps:
            detail = (
                f"estimate {raw_bps:.2f} bp is within noise of this sample's "
                f"resolution floor ({floor_bps:.2f} bp); it measures daily "
                "volatility, not spread"
            )
            note = f"{note}. {detail}" if note else detail
            log.warning(
                "costs.spread_below_resolution",
                method=self.method,
                raw_bps=round(raw_bps, 4),
                floor_bps=round(floor_bps, 4),
            )
        return SpreadEstimate(
            spread_bps=self.correction.apply(raw_bps),
            raw_bps=raw_bps,
            method=self.method,
            coverage=coverage,
            calibrated=self.correction.calibrated,
            resolution_floor_bps=floor_bps,
            note=note or self.correction.provenance,
        )


class ObservedSpread(SpreadEstimator):
    """Spread measured from real quotes. The only kind worth trusting outright."""

    method: Method = "observed"

    def __init__(self) -> None:
        # Observed quotes need no correction; applying one would corrupt ground truth.
        super().__init__(
            SpreadCorrection(factor=1.0, calibrated=True, provenance="observed quotes")
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("bid", "ask")

    def estimate_frame(self, frame: pl.DataFrame) -> SpreadEstimate:
        return self.estimate(frame["bid"].to_numpy(), frame["ask"].to_numpy())

    def _raw_from(self, bid_in: np.ndarray, ask_in: np.ndarray) -> tuple[float, float]:
        bid = np.asarray(bid_in, dtype=float)
        ask = np.asarray(ask_in, dtype=float)
        mid = (bid + ask) / 2.0
        usable = np.isfinite(bid) & np.isfinite(ask) & (mid > 0) & (ask >= bid)
        if not usable.any():
            raise ValueError("no usable quotes: every pair was missing, zero, or crossed")
        spread = (ask[usable] - bid[usable]) / mid[usable]
        return float(np.median(spread) / BPS), float(usable.mean())

    def estimate(self, bid: np.ndarray, ask: np.ndarray) -> SpreadEstimate:
        raw, coverage = self._raw_from(bid, ask)
        return self._finish(raw, coverage, note="observed quotes; no correction needed")


class RollSpread(SpreadEstimator):
    """Roll (1984): ``S = 2·sqrt(-Cov(Δp_t, Δp_{t-1}))``.

    Bid-ask bounce makes consecutive price changes negatively autocorrelated, and
    the size of that autocovariance identifies the spread.

    Its weakness is structural, not incidental: whenever the serial covariance
    comes out **positive** -- which happens for a large share of real windows,
    because genuine momentum swamps the bounce -- the estimator has no real root
    and is simply undefined. Implementations that clamp those to zero turn an
    honest "no information" into a confident "spread is zero". This one reports
    coverage instead, so a series where the estimator mostly failed is visible.
    """

    method: Method = "roll"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("close",)

    def estimate_frame(self, frame: pl.DataFrame) -> SpreadEstimate:
        return self.estimate(frame["close"].to_numpy())

    def _raw_from(self, close: np.ndarray, window: int) -> tuple[float, float]:
        prices = np.asarray(close, dtype=float)
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 3:
            raise ValueError("Roll needs at least four prices")

        if window and window < len(log_returns):
            estimates: list[float] = []
            defined = 0
            total = 0
            for start in range(0, len(log_returns) - window + 1):
                chunk = log_returns[start : start + window]
                covariance = float(np.cov(chunk[:-1], chunk[1:], ddof=1)[0, 1])
                total += 1
                if covariance < 0 and 2.0 * np.sqrt(-covariance) / BPS >= MIN_MEANINGFUL_BPS:
                    defined += 1
                    estimates.append(2.0 * np.sqrt(-covariance))
            if not estimates:
                raise ValueError(
                    "Roll is undefined over every window: the serial covariance of "
                    "returns was positive throughout, so bid-ask bounce is not "
                    "identifiable from this series"
                )
            return float(np.median(estimates) / BPS), defined / total

        covariance = float(np.cov(log_returns[:-1], log_returns[1:], ddof=1)[0, 1])
        if covariance >= 0:
            raise ValueError(
                f"Roll is undefined: serial covariance is {covariance:.3e}, not "
                "negative. Returns show momentum rather than bid-ask bounce over this "
                "window. Use a different estimator rather than clamping this to zero."
            )
        spread_bps = float(2.0 * np.sqrt(-covariance) / BPS)
        if spread_bps < MIN_MEANINGFUL_BPS:
            # A covariance of ~1e-33 has a real square root, so the arithmetic
            # succeeds and hands back a spread of ~1e-12 bp. That is the
            # clamp-to-zero failure arrived at by floating-point accident rather
            # than by choice, and it is exactly the number that would make a
            # high-turnover strategy look free.
            raise ValueError(
                f"Roll implies a spread of {spread_bps:.2e} bp, which is numerical "
                "noise rather than a measurement. The series carries no identifiable "
                "bid-ask bounce."
            )
        return spread_bps, 1.0

    def estimate(self, close: np.ndarray, window: int = 0) -> SpreadEstimate:
        raw, coverage = self._raw_from(close, window)
        returns = np.diff(np.log(np.asarray(close, dtype=float)))
        floor = resolution_floor_bps(float(np.std(returns, ddof=1)), len(returns))
        return self._finish(raw, coverage, floor_bps=floor)


class CorwinSchultzSpread(SpreadEstimator):
    """Corwin & Schultz (2012), from daily highs and lows.

    The high-low range over two days reflects both volatility and the spread; over
    one day it reflects proportionally more spread. Differencing the two identifies
    the spread without needing intraday data.

    Known to be materially **upward**-biased for liquid post-decimalisation names,
    and it produces negative estimates often enough that the treatment of negatives
    changes the answer. The original paper sets negatives to zero; doing that
    silently is what spec section 5 warns against, so negatives are counted and
    surfaced through ``coverage``.

    Overnight returns are **not** adjusted for here. The paper's adjustment assumes
    the price gaps but the spread does not, and applying it to instruments that
    trade nearly continuously (crypto) or barely at all overstates the correction.
    """

    method: Method = "corwin_schultz"

    #: (3 - 2·sqrt(2)), which appears in both alpha terms.
    _K = 3.0 - 2.0 * np.sqrt(2.0)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("high", "low")

    def estimate_frame(self, frame: pl.DataFrame) -> SpreadEstimate:
        return self.estimate(frame["high"].to_numpy(), frame["low"].to_numpy())

    def _raw_from(self, high_in: np.ndarray, low_in: np.ndarray) -> tuple[float, float]:
        high = np.asarray(high_in, dtype=float)
        low = np.asarray(low_in, dtype=float)
        if len(high) < 2 or len(high) != len(low):
            raise ValueError("Corwin-Schultz needs at least two aligned high/low pairs")
        if np.any(low <= 0) or np.any(high < low):
            raise ValueError("highs and lows must be positive with high >= low")

        single = np.log(high / low) ** 2
        beta = single[:-1] + single[1:]
        two_day_high = np.maximum(high[:-1], high[1:])
        two_day_low = np.minimum(low[:-1], low[1:])
        gamma = np.log(two_day_high / two_day_low) ** 2

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / self._K - np.sqrt(gamma / self._K)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

        usable = np.isfinite(spread) & (spread > 0)
        if not usable.any():
            raise ValueError(
                "Corwin-Schultz produced no positive estimate over this window. The "
                "original paper sets negatives to zero; doing so here would report a "
                "free round trip, so the failure is raised instead."
            )
        return float(np.median(spread[usable]) / BPS), float(usable.mean())

    def estimate(self, high: np.ndarray, low: np.ndarray) -> SpreadEstimate:
        raw, coverage = self._raw_from(high, low)
        note = ""
        if coverage < 0.5:
            note = (
                f"only {coverage:.0%} of two-day windows gave a positive estimate; "
                "the median is taken over the survivors, which selects upward"
            )
        # The daily range is a volatility proxy, so it sets the resolution scale.
        ranges = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float))
        floor = resolution_floor_bps(float(np.mean(ranges) / 2.0), len(ranges))
        return self._finish(raw, coverage, note, floor_bps=floor)


class AbdiRanaldoSpread(SpreadEstimator):
    """Abdi & Ranaldo (2017), from close, high and low.

    Compares the close to the mid-range of the day it falls in and of the next day.
    Generally better behaved than Roll or Corwin-Schultz -- fewer undefined windows
    and a less severe bias -- which makes it the default low-frequency estimator
    here. It is still an estimator, and still needs its correction calibrated.
    """

    method: Method = "abdi_ranaldo"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("close", "high", "low")

    def estimate_frame(self, frame: pl.DataFrame) -> SpreadEstimate:
        return self.estimate(
            frame["close"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy()
        )

    def _raw_from(
        self, close_in: np.ndarray, high_in: np.ndarray, low_in: np.ndarray
    ) -> tuple[float, float]:
        close = np.asarray(close_in, dtype=float)
        high = np.asarray(high_in, dtype=float)
        low = np.asarray(low_in, dtype=float)
        if len({len(close), len(high), len(low)}) != 1:
            raise ValueError("close, high and low must be the same length")
        if len(close) < 3:
            raise ValueError("Abdi-Ranaldo needs at least three observations")
        if np.any(close <= 0) or np.any(low <= 0):
            raise ValueError("prices must be positive")

        log_close = np.log(close)
        mid_range = (np.log(high) + np.log(low)) / 2.0

        # 4·E[(c_t - η_t)(c_t - η_{t+1})]: the covariance of the close's deviation
        # from today's mid-range with its deviation from tomorrow's.
        terms = 4.0 * (log_close[:-1] - mid_range[:-1]) * (log_close[:-1] - mid_range[1:])
        usable = np.isfinite(terms)
        if not usable.any():
            raise ValueError("Abdi-Ranaldo produced no finite terms")

        mean_square = float(np.mean(terms[usable]))
        if mean_square <= 0:
            raise ValueError(
                f"Abdi-Ranaldo is undefined: the mean squared spread came out "
                f"{mean_square:.3e}, which has no real root. Reporting zero here would "
                "claim a free round trip."
            )
        positive = float((terms[usable] > 0).mean())
        return float(np.sqrt(mean_square) / BPS), positive

    def estimate(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> SpreadEstimate:
        raw, coverage = self._raw_from(close, high, low)
        returns = np.diff(np.log(np.asarray(close, dtype=float)))
        floor = resolution_floor_bps(float(np.std(returns, ddof=1)), len(returns))
        return self._finish(raw, coverage, floor_bps=floor)


def calibrate_correction(
    estimated_bps: np.ndarray,
    observed_bps: np.ndarray,
    *,
    provenance: str,
    tick_floor_bps: float = 0.0,
) -> SpreadCorrection:
    """Fit the shrinkage factor that maps estimates onto observed spreads.

    Least squares through the origin, which is the right functional form: an
    estimator that is unbiased should give a factor of 1, and a proportionally
    upward-biased one a factor below 1. An intercept would let the fit absorb bias
    that is genuinely proportional and hide it.

    The intended use is the one case where ground truth is free: crypto, where
    Binance publishes real quotes alongside real OHLC. A factor fitted there is
    still an extrapolation when applied to equities -- say so in ``provenance``.
    """
    estimated = np.asarray(estimated_bps, dtype=float)
    observed = np.asarray(observed_bps, dtype=float)
    if estimated.shape != observed.shape:
        raise ValueError("estimated and observed must have the same shape")

    usable = np.isfinite(estimated) & np.isfinite(observed) & (estimated > 0)
    if usable.sum() < 10:
        raise ValueError(
            f"only {int(usable.sum())} usable pairs; refusing to fit a correction "
            "factor that would be indistinguishable from noise"
        )

    x = estimated[usable]
    y = observed[usable]
    factor = float((x @ y) / (x @ x))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"fitted factor {factor} is not usable")

    log.info(
        "costs.spread_correction_calibrated",
        factor=round(factor, 4),
        pairs=int(usable.sum()),
        provenance=provenance,
    )
    return SpreadCorrection(
        factor=factor,
        tick_floor_bps=tick_floor_bps,
        calibrated=True,
        provenance=f"{provenance} (n={int(usable.sum())}, factor={factor:.3f})",
    )
