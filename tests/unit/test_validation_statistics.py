"""Overfitting statistics: closed forms, monotonicity, and known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quantlab.validation.statistics import (
    EULER_MASCHERONI,
    HaircutSchedule,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)

BARS = 252
FIVE_YEARS = 5 * BARS


def per_bar(annual: float) -> float:
    return annual / math.sqrt(BARS)


# ----------------------------------------------------------------------------------
# Probabilistic Sharpe
# ----------------------------------------------------------------------------------
def test_psr_matches_its_closed_form_for_normal_returns() -> None:
    """With zero skew and zero excess kurtosis the variance term is
    ``1 + SR²/2``, so the whole statistic can be written out by hand."""
    sharpe = per_bar(1.0)
    expected_variance = 1.0 + sharpe**2 / 2.0
    expected = norm.cdf(sharpe * math.sqrt(FIVE_YEARS - 1) / math.sqrt(expected_variance))
    assert probabilistic_sharpe_ratio(sharpe, observations=FIVE_YEARS) == pytest.approx(expected)


def test_a_zero_sharpe_is_a_coin_flip() -> None:
    assert probabilistic_sharpe_ratio(0.0, observations=1000) == pytest.approx(0.5)


def test_psr_rises_with_sample_length() -> None:
    """The same point estimate is stronger evidence over ten years than over one."""
    sharpe = per_bar(1.0)
    values = [
        probabilistic_sharpe_ratio(sharpe, observations=n)
        for n in (BARS, 2 * BARS, 5 * BARS, 10 * BARS)
    ]
    assert values == sorted(values)


def test_psr_rises_with_the_point_estimate() -> None:
    values = [
        probabilistic_sharpe_ratio(per_bar(s), observations=FIVE_YEARS)
        for s in (0.0, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)


def test_higher_moments_weaken_evidence_against_a_zero_benchmark() -> None:
    """Negative skew and fat tails widen the estimator's standard error. Against a
    benchmark the strategy beats, that pulls the probability down -- and it is the
    return shape most strategies that "work" turn out to have."""
    sharpe = per_bar(1.5)
    normal = probabilistic_sharpe_ratio(sharpe, observations=FIVE_YEARS)
    ugly = probabilistic_sharpe_ratio(
        sharpe, observations=FIVE_YEARS, skewness=-1.5, excess_kurtosis=6.0
    )
    assert ugly < normal


def test_higher_moments_cut_both_ways() -> None:
    """Against a benchmark the strategy is already *below*, the same widening makes
    the result harder to rule out. Both directions are correct: more uncertainty
    means less is established either way."""
    sharpe = per_bar(1.5)
    benchmark = per_bar(2.5)
    normal = probabilistic_sharpe_ratio(sharpe, observations=FIVE_YEARS, benchmark=benchmark)
    ugly = probabilistic_sharpe_ratio(
        sharpe,
        observations=FIVE_YEARS,
        benchmark=benchmark,
        skewness=-1.5,
        excess_kurtosis=6.0,
    )
    assert ugly > normal


def test_psr_is_undefined_rather_than_fabricated_for_impossible_moments() -> None:
    """Extreme moment estimates on short samples can drive the Sharpe estimator's
    variance negative. Returning a probability from that would be fabricating
    precision, so it comes back NaN and the caller has to notice."""
    value = probabilistic_sharpe_ratio(0.5, observations=100, skewness=3.0, excess_kurtosis=-3.0)
    assert math.isnan(value)


def test_too_few_observations_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        probabilistic_sharpe_ratio(0.1, observations=1)


# ----------------------------------------------------------------------------------
# Expected maximum and deflation
# ----------------------------------------------------------------------------------
def test_expected_max_matches_its_closed_form() -> None:
    variance, trials = 0.25, 100
    expected = math.sqrt(variance) * (
        (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / trials)
        + EULER_MASCHERONI * norm.ppf(1 - 1 / (trials * math.e))
    )
    assert expected_max_sharpe(trials, variance) == pytest.approx(expected)


def test_a_single_trial_needs_no_deflation() -> None:
    """Nothing was selected, so there is no selection to correct for."""
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_the_bar_rises_with_the_breadth_of_the_search() -> None:
    values = [expected_max_sharpe(n, 0.25) for n in (2, 10, 100, 1000, 10000)]
    assert values == sorted(values)


def test_the_bar_rises_with_the_spread_of_the_search() -> None:
    """A wider search finds a luckier maximum, so it earns a harsher correction."""
    narrow = expected_max_sharpe(100, 0.01)
    wide = expected_max_sharpe(100, 1.00)
    assert wide > narrow


def test_deflation_falls_away_as_trials_multiply() -> None:
    """The headline behaviour: search hard enough and nothing survives."""
    sharpe = per_bar(1.5)
    values = [
        deflated_sharpe_ratio(sharpe, observations=FIVE_YEARS, trials=n, sharpe_variance=0.5 / BARS)
        for n in (1, 10, 100, 1000)
    ]
    assert values == sorted(values, reverse=True)
    assert values[0] > 0.99, "one trial is just the probabilistic Sharpe"
    assert values[-1] < 0.10, "a thousand tries makes a 1.5 Sharpe unremarkable"


def test_a_strong_result_over_a_long_sample_can_still_survive() -> None:
    """The statistic must be able to say yes, or it says nothing."""
    survived = deflated_sharpe_ratio(
        per_bar(2.5), observations=20 * BARS, trials=20, sharpe_variance=0.2 / BARS
    )
    assert survived > 0.95


# ----------------------------------------------------------------------------------
# Minimum backtest length
# ----------------------------------------------------------------------------------
def test_minimum_length_grows_with_trials_and_falls_with_the_target() -> None:
    assert minimum_backtest_length(1.0, 1000) > minimum_backtest_length(1.0, 10)
    assert minimum_backtest_length(0.5, 100) > minimum_backtest_length(2.0, 100)


def test_minimum_length_scales_as_the_inverse_square_of_the_target() -> None:
    """Halving the Sharpe you are willing to claim quadruples the data you need."""
    assert minimum_backtest_length(0.5, 100) == pytest.approx(4 * minimum_backtest_length(1.0, 100))


def test_a_single_trial_needs_no_minimum() -> None:
    assert minimum_backtest_length(1.0, 1) == 0.0


def test_a_non_positive_target_is_an_error() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        minimum_backtest_length(0.0, 10)


# ----------------------------------------------------------------------------------
# PBO
# ----------------------------------------------------------------------------------
def test_pbo_is_high_when_every_configuration_is_noise() -> None:
    rng = np.random.default_rng(0)
    pbo, logits = probability_of_backtest_overfitting(
        rng.normal(0, 0.01, (1000, 30)), partitions=10
    )
    assert 0.3 < pbo < 0.7, "with no persistent ranking, the winner is a coin flip"
    assert len(logits) == math.comb(10, 5)


def test_pbo_is_low_when_one_configuration_is_genuinely_best() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 0.01, (1000, 30))
    matrix[:, 7] += 0.003
    pbo, _ = probability_of_backtest_overfitting(matrix, partitions=10)
    assert pbo < 0.05


def test_pbo_needs_something_to_select_between() -> None:
    with pytest.raises(ValueError, match="at least two configurations"):
        probability_of_backtest_overfitting(np.zeros((100, 1)))


def test_pbo_needs_an_even_number_of_partitions() -> None:
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(np.zeros((100, 3)), partitions=7)


def test_pbo_needs_enough_observations_for_its_blocks() -> None:
    with pytest.raises(ValueError, match="cannot be cut"):
        probability_of_backtest_overfitting(np.zeros((10, 3)), partitions=16)


# ----------------------------------------------------------------------------------
# Haircuts
# ----------------------------------------------------------------------------------
def test_the_default_haircuts_compose_to_the_specs_working_prior() -> None:
    """Spec section 7: live Sharpe is about half the backtest Sharpe. The two
    defaults multiply to 0.53. If either moves, that prior should be re-examined
    rather than drifting silently."""
    schedule = HaircutSchedule()
    assert schedule.combined_multiplier == pytest.approx(0.88 * 0.60)
    assert 0.45 <= schedule.combined_multiplier <= 0.55


def test_haircuts_are_applied_multiplicatively() -> None:
    schedule = HaircutSchedule(in_sample_selection=0.2, post_publication_decay=0.5)
    assert schedule.apply(2.0) == pytest.approx(2.0 * 0.8 * 0.5)


def test_a_haircut_of_one_would_erase_everything_and_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        HaircutSchedule(post_publication_decay=1.0)


def test_haircuts_describe_themselves() -> None:
    assert "combined" in HaircutSchedule().describe()
