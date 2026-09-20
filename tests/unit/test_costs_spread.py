"""Spread estimators, tested against a simulated ground truth.

The point of simulating is that the true spread is *known*, so the tests can
measure each estimator's bias instead of asserting that it returns whatever it
happens to return. That is what makes the spec's warning checkable: the
low-frequency proxies really are badly biased, and this file proves it on data
where the answer is not in doubt.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quantlab.costs.spread import (
    AbdiRanaldoSpread,
    CorwinSchultzSpread,
    ObservedSpread,
    RollSpread,
    SpreadCorrection,
    calibrate_correction,
)

TRUE_SPREAD = 0.0020  # 20 bp
TRUE_SPREAD_BPS = 20.0


def simulate(n: int = 4000, sigma: float = 0.015, spread: float = TRUE_SPREAD, seed: int = 0):
    """An efficient price plus bid-ask bounce, with an intraday range.

    The close bounces to a random side of the spread, which is the structure Roll
    identifies. The range is widened by the spread, which is the structure
    Corwin-Schultz identifies. Both estimators therefore get the data-generating
    process they were designed for -- and still misbehave.
    """
    rng = np.random.default_rng(seed)
    efficient = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    side = rng.choice([-1.0, 1.0], n)
    close = efficient * (1 + side * spread / 2)
    half_range = np.abs(rng.normal(0, sigma, n)) * 1.3
    high = efficient * np.exp(half_range) * (1 + spread / 2)
    low = efficient * np.exp(-half_range) * (1 - spread / 2)
    bid = efficient * (1 - spread / 2)
    ask = efficient * (1 + spread / 2)
    return close, high, low, bid, ask


# ----------------------------------------------------------------------------------
# Observed quotes: the only ground truth
# ----------------------------------------------------------------------------------
def test_observed_spread_is_exact() -> None:
    _, _, _, bid, ask = simulate()
    estimate = ObservedSpread().estimate(bid, ask)
    assert estimate.raw_bps == pytest.approx(TRUE_SPREAD_BPS, rel=1e-6)
    assert estimate.method == "observed"
    assert estimate.is_trustworthy


def test_observed_spread_needs_usable_quotes() -> None:
    with pytest.raises(ValueError, match="no usable quotes"):
        ObservedSpread().estimate(np.array([np.nan, 0.0]), np.array([np.nan, 0.0]))


def test_crossed_quotes_are_discarded_not_negated() -> None:
    """A crossed book produces a negative spread, which would show up as a
    *subsidy* if it were averaged in."""
    bid = np.array([100.0, 101.0, 100.0])
    ask = np.array([100.2, 100.5, 100.2])  # the middle quote is crossed
    estimate = ObservedSpread().estimate(bid, ask)
    assert estimate.raw_bps > 0
    assert estimate.coverage == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------------
# The spec's warning, demonstrated
# ----------------------------------------------------------------------------------
def test_low_frequency_estimators_are_materially_biased() -> None:
    """Spec section 5: low-frequency spread proxies are upward-biased, so they may
    not be used uncritically. Measured here against a known 20 bp spread."""
    close, high, low, _, _ = simulate()
    roll = RollSpread().estimate(close, window=60).raw_bps
    corwin = CorwinSchultzSpread().estimate(high, low).raw_bps
    abdi = AbdiRanaldoSpread().estimate(close, high, low).raw_bps

    assert roll > 2 * TRUE_SPREAD_BPS, "Roll is severely upward-biased here"
    assert corwin > 2 * TRUE_SPREAD_BPS, "Corwin-Schultz is severely upward-biased here"
    # Abdi-Ranaldo is the best behaved of the three, which is why it is the default.
    assert abs(abdi / TRUE_SPREAD_BPS - 1) < 0.25
    assert abs(abdi - TRUE_SPREAD_BPS) < abs(corwin - TRUE_SPREAD_BPS)
    assert abs(abdi - TRUE_SPREAD_BPS) < abs(roll - TRUE_SPREAD_BPS)


def test_coverage_exposes_how_often_an_estimator_failed() -> None:
    """Roll and Corwin-Schultz are undefined for a large share of real windows.
    Implementations that clamp those to zero convert "no information" into a
    confident claim of a free round trip; this one reports the rate instead."""
    close, high, low, _, _ = simulate()
    assert RollSpread().estimate(close, window=60).coverage < 1.0
    assert CorwinSchultzSpread().estimate(high, low).coverage < 1.0


# ----------------------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------------------
def test_roll_is_undefined_when_returns_trend() -> None:
    """Positive serial covariance means momentum, not bid-ask bounce. There is no
    real root, and reporting zero would claim a free round trip."""
    trending = np.exp(np.cumsum(np.full(200, 0.01)))
    with pytest.raises(ValueError, match=r"numerical noise"):
        RollSpread().estimate(trending)


def test_roll_does_not_return_a_floating_point_zero_as_a_spread() -> None:
    """A perfectly trending series has a serial covariance of ~1e-33, whose square
    root is a spread of ~1e-12 bp. Returning that is the clamp-to-zero failure
    arrived at by accident; it must be an error instead."""
    trending = np.exp(np.cumsum(np.full(200, 0.01)))
    with pytest.raises(ValueError):
        RollSpread().estimate(trending)


def test_roll_needs_enough_prices() -> None:
    with pytest.raises(ValueError, match="at least four prices"):
        RollSpread().estimate(np.array([100.0, 101.0, 102.0]))


def test_corwin_schultz_rejects_impossible_bars() -> None:
    with pytest.raises(ValueError, match="high >= low"):
        CorwinSchultzSpread().estimate(np.array([100.0, 99.0]), np.array([101.0, 100.0]))


def test_abdi_ranaldo_is_undefined_when_the_estimate_has_no_real_root() -> None:
    """A strong trend with each close at the day's high makes the two deviations
    take opposite signs, so the mean squared spread comes out negative. Reporting
    zero there would claim a free round trip."""
    base = 100 * np.exp(np.cumsum(np.full(100, 0.03)))
    with pytest.raises(ValueError, match="no real root"):
        AbdiRanaldoSpread().estimate(close=base, high=base, low=base * 0.99)


def test_estimators_declare_what_they_read() -> None:
    """So a caller can dispatch from a frame without guessing."""
    assert RollSpread().required_columns == ("close",)
    assert CorwinSchultzSpread().required_columns == ("high", "low")
    assert set(AbdiRanaldoSpread().required_columns) == {"close", "high", "low"}


def test_frame_dispatch_matches_the_array_api() -> None:
    close, high, low, _, _ = simulate()
    frame = pl.DataFrame({"close": close, "high": high, "low": low})
    estimator = AbdiRanaldoSpread()
    assert estimator.estimate_frame(frame).raw_bps == pytest.approx(
        estimator.estimate(close, high, low).raw_bps
    )


# ----------------------------------------------------------------------------------
# Correction
# ----------------------------------------------------------------------------------
def test_default_correction_applies_no_shrinkage_and_says_so() -> None:
    """The conservative direction: leaving the upward bias in overstates cost and
    understates capacity. That is still an error, so it is labelled, not hidden."""
    correction = SpreadCorrection.uncalibrated()
    assert correction.factor == 1.0
    assert not correction.calibrated
    assert "upward-biased" in correction.provenance
    assert correction.apply(37.0) == pytest.approx(37.0)


def test_uncalibrated_estimates_are_not_trustworthy() -> None:
    close, high, low, _, _ = simulate()
    assert not AbdiRanaldoSpread().estimate(close, high, low).is_trustworthy


def test_correction_floors_at_a_tick_and_caps_at_the_absurd() -> None:
    correction = SpreadCorrection(factor=1.0, tick_floor_bps=2.0, cap_bps=500.0)
    assert correction.apply(0.5) == 2.0, "no spread can be narrower than a tick"
    assert correction.apply(9999.0) == 500.0, "a 100% spread is estimator failure"
    assert correction.apply(float("nan")) == 2.0
    assert correction.apply(-3.0) == 2.0


def test_calibration_recovers_a_known_bias() -> None:
    """Fit through the origin, so a proportionally biased estimator yields the
    reciprocal of its bias and an unbiased one yields 1."""
    rng = np.random.default_rng(1)
    observed = rng.uniform(5, 50, 200)
    estimated = observed / 0.6 + rng.normal(0, 0.5, 200)  # 1/0.6 upward bias
    correction = calibrate_correction(estimated, observed, provenance="unit test")
    assert correction.factor == pytest.approx(0.6, rel=0.05)
    assert correction.calibrated
    assert "n=200" in correction.provenance


def test_calibrated_estimates_become_trustworthy() -> None:
    close, high, low, _, _ = simulate()
    estimator = AbdiRanaldoSpread(
        SpreadCorrection(factor=0.9, calibrated=True, provenance="calibrated on crypto")
    )
    estimate = estimator.estimate(close, high, low)
    assert estimate.is_trustworthy
    assert estimate.spread_bps == pytest.approx(estimate.raw_bps * 0.9)


def test_calibration_refuses_a_sample_too_small_to_mean_anything() -> None:
    with pytest.raises(ValueError, match="indistinguishable from noise"):
        calibrate_correction(np.arange(1.0, 6.0), np.arange(1.0, 6.0), provenance="too few")


def test_calibration_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calibrate_correction(np.ones(10), np.ones(11), provenance="mismatched")


def test_corrected_estimate_always_reports_the_raw_value() -> None:
    """Nothing here silently improves a number: the shrunk estimate and the
    estimator's own output travel together."""
    close, high, low, _, _ = simulate()
    estimate = AbdiRanaldoSpread(
        SpreadCorrection(factor=0.5, calibrated=True, provenance="test")
    ).estimate(close, high, low)
    assert estimate.raw_bps > estimate.spread_bps
    assert estimate.spread_bps == pytest.approx(estimate.raw_bps * 0.5)


# ----------------------------------------------------------------------------------
# The measured failure on real data
# ----------------------------------------------------------------------------------
def test_resolution_floor_scales_like_sigma_over_root_n() -> None:
    """Every low-frequency estimator identifies the spread as a perturbation of a
    much larger quantity. Below ``σ/sqrt(n)`` it cannot tell the spread from noise,
    and it will return a number anyway."""
    from quantlab.costs.spread import resolution_floor_bps

    assert resolution_floor_bps(0.03, 900) == pytest.approx(0.03 / 30 / 1e-4)
    # More data lowers the floor; more volatility raises it.
    assert resolution_floor_bps(0.03, 3600) < resolution_floor_bps(0.03, 900)
    assert resolution_floor_bps(0.06, 900) > resolution_floor_bps(0.03, 900)
    assert resolution_floor_bps(0.03, 1) == float("inf")


def test_estimators_report_the_floor_they_could_not_see_below() -> None:
    close, high, low, _, _ = simulate()
    for estimate in (
        RollSpread().estimate(close, window=60),
        CorwinSchultzSpread().estimate(high, low),
        AbdiRanaldoSpread().estimate(close, high, low),
    ):
        assert estimate.resolution_floor_bps is not None
        assert estimate.resolution_floor_bps > 0


def test_an_estimate_inside_its_own_noise_is_rejected_as_untrustworthy() -> None:
    """A tight spread on a volatile instrument is invisible in daily bars.

    Constructed directly rather than simulated, because on such data the
    estimators usually raise instead of returning a small number -- which is the
    other half of the same failure, and is what happened on seven of eight real
    crypto majors.
    """
    from quantlab.costs.spread import SpreadEstimate

    estimate = SpreadEstimate(
        spread_bps=3.0,
        raw_bps=3.0,
        method="abdi_ranaldo",
        coverage=1.0,
        calibrated=True,
        resolution_floor_bps=8.0,
    )
    assert not estimate.above_resolution
    assert not estimate.is_trustworthy, "an estimate inside its own noise is not evidence"


def test_an_estimate_well_above_its_floor_can_be_trusted_once_calibrated() -> None:
    from quantlab.costs.spread import SpreadEstimate

    estimate = SpreadEstimate(
        spread_bps=40.0,
        raw_bps=40.0,
        method="abdi_ranaldo",
        coverage=0.9,
        calibrated=True,
        resolution_floor_bps=5.0,
    )
    assert estimate.above_resolution
    assert estimate.is_trustworthy


def test_a_tight_spread_on_a_volatile_instrument_is_wildly_overestimated() -> None:
    """The finding that motivates the whole provenance apparatus.

    Measured on 993 days of real Binance daily bars (2026-09-20), against a true
    quoted BTCUSDT spread of **0.001 bp**: Roll returned 167 bp, Corwin-Schultz
    106 bp, Abdi-Ranaldo 62 bp, and Abdi-Ranaldo was undefined outright for seven
    of eight majors. Wrong by five orders of magnitude -- these estimators are not
    biased here, they are measuring daily volatility instead of spread.

    Reproduced on simulation so it is checkable offline and cannot quietly regress.
    """
    close, high, low, _, _ = simulate(n=1000, sigma=0.03, spread=1e-7)
    true_bps = 1e-7 / 1e-4  # 0.001 bp
    overestimates = []
    for call in (
        lambda: RollSpread().estimate(close, window=60),
        lambda: CorwinSchultzSpread().estimate(high, low),
        lambda: AbdiRanaldoSpread().estimate(close, high, low),
    ):
        try:
            overestimates.append(call().raw_bps / true_bps)
        except ValueError:
            overestimates.append(float("inf"))  # undefined is its own kind of failure
    assert all(ratio > 1000 for ratio in overestimates), (
        "every low-frequency estimator should be off by orders of magnitude here; "
        "if this ever passes, re-check the simulation before believing the estimator"
    )
