"""Factor attribution and statistical risk models.

The question the module exists to answer is "is this signal secretly just beta",
so the tests are built around cases where the true answer is known by
construction: a strategy that *is* beta, a strategy that has alpha on top of
beta, and a strategy that has neither.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlab.risk.factor_model import (
    ALPHA_THRESHOLD,
    PCARiskModel,
    attribute,
    newey_west_covariance,
    pca_risk_model,
)


def factors(n: int = 1500, seed: int = 11) -> np.ndarray:
    """Three weakly-correlated factor return series."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.010, n)
    value = rng.normal(0.0001, 0.006, n) + 0.15 * market
    momentum = rng.normal(0.0002, 0.007, n) - 0.10 * market
    return np.column_stack([market, value, momentum])


NAMES = ("market", "value", "momentum")


# ------------------------------------------------------------------ attribution --
def test_known_betas_are_recovered() -> None:
    x = factors()
    rng = np.random.default_rng(3)
    true_betas = np.array([0.80, -0.30, 0.45])
    returns = x @ true_betas + rng.normal(0.0, 0.002, len(x))

    result = attribute(returns, x, factor_names=NAMES)

    assert np.allclose(result.betas, true_betas, atol=0.02)
    assert result.r_squared > 0.9
    assert abs(result.alpha_t_statistic) < ALPHA_THRESHOLD


def test_a_pure_beta_strategy_is_called_a_factor_bet() -> None:
    """A levered market position dressed up as a signal. The model should say so."""
    x = factors()
    rng = np.random.default_rng(5)
    returns = 1.3 * x[:, 0] + rng.normal(0.0, 0.001, len(x))

    result = attribute(returns, x, factor_names=NAMES)

    assert result.is_mostly_factor
    assert not result.has_residual_alpha
    assert "factor bet" in result.verdict()
    assert "more cheaply" in result.verdict()
    assert [name for name, _, _ in result.significant_exposures()] == ["market"]


def test_alpha_on_top_of_beta_survives_the_factors() -> None:
    x = factors(n=3000, seed=71)
    rng = np.random.default_rng(7)
    # 20% a year of genuine alpha, on top of a half-unit market exposure. Large
    # enough that a single sample identifies it -- see the power test below for
    # what happens at a size anyone actually reports.
    returns = 0.20 / 252 + 0.5 * x[:, 0] + rng.normal(0.0, 0.003, len(x))

    result = attribute(returns, x, factor_names=NAMES)

    assert result.has_residual_alpha
    assert result.alpha_annual == pytest.approx(0.20, abs=0.03)
    assert "Residual alpha" in result.verdict()
    assert "deflated Sharpe" in result.verdict()


@pytest.mark.slow
def test_how_much_alpha_six_years_can_actually_resolve() -> None:
    """The arithmetic behind every attribution table, measured.

    Residual volatility here is about 4.8% a year over 1,500 bars, so the
    standard error of the intercept is roughly 2% a year. That fixes what the
    sample can resolve, and the answer is coarser than the numbers people report:

    * 8%/yr (residual IR 1.7, an outstanding strategy) -- found almost always
    * 4%/yr (residual IR 0.83, a strategy most desks would fund) -- a coin flip
    * 2%/yr (residual IR 0.42, a real and useful edge) -- usually missed

    The estimator is unbiased at every level; the misses are sampling noise. The
    point is that a single t-statistic near the threshold carries almost no
    information, which is why the platform reports a deflated Sharpe and a trial
    count alongside it rather than treating |t| > 2 as a verdict.
    """
    trials = 200

    def power_at(annual_alpha: float) -> tuple[float, float]:
        detected, shortfalls = 0, []
        for seed in range(trials):
            x = factors(n=1500, seed=1000 + seed)
            rng = np.random.default_rng(seed)
            returns = annual_alpha / 252 + 0.5 * x[:, 0] + rng.normal(0.0, 0.003, 1500)
            result = attribute(returns, x, factor_names=NAMES)
            detected += int(result.has_residual_alpha)
            shortfalls.append(result.alpha_annual - annual_alpha)
        return detected / trials, float(np.mean(shortfalls))

    strong, strong_bias = power_at(0.08)
    fundable, fundable_bias = power_at(0.04)
    real, real_bias = power_at(0.02)

    # Unbiased throughout: what the sample lacks is precision, not accuracy.
    for bias in (strong_bias, fundable_bias, real_bias):
        assert bias == pytest.approx(0.0, abs=0.01)

    assert real < fundable < strong
    assert strong > 0.90, f"IR 1.7 detected only {strong:.0%} of the time"
    assert 0.35 < fundable < 0.70, f"IR 0.83 detected {fundable:.0%} of the time"
    assert real < 0.35, f"IR 0.42 detected {real:.0%} of the time"


def test_noise_is_reported_as_noise() -> None:
    x = factors()
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.004, len(x))

    result = attribute(returns, x, factor_names=NAMES)

    assert not result.is_mostly_factor
    assert not result.has_residual_alpha
    assert "no evidence of skill" in result.verdict()


def test_idiosyncratic_volatility_is_annualised_residual_risk() -> None:
    x = factors()
    rng = np.random.default_rng(13)
    returns = 0.6 * x[:, 0] + rng.normal(0.0, 0.005, len(x))

    result = attribute(returns, x, factor_names=NAMES)

    assert result.idiosyncratic_volatility == pytest.approx(0.005 * math.sqrt(252), rel=0.1)


def test_describe_marks_the_significant_exposures() -> None:
    x = factors()
    rng = np.random.default_rng(17)
    returns = x @ np.array([0.9, 0.0, 0.0]) + rng.normal(0.0, 0.002, len(x))

    text = attribute(returns, x, factor_names=NAMES).describe()

    assert "market" in text
    assert text.count(" *") == 1  # only the market loading is distinguishable from zero


# ------------------------------------------------------------------ Newey-West --
def test_newey_west_is_wider_than_ols_on_autocorrelated_residuals() -> None:
    """The module's central claim, measured rather than asserted.

    OLS standard errors assume independent residuals. Strategy returns are
    autocorrelated by construction when positions are held for more than a bar,
    and assuming otherwise makes every t-statistic too large in the direction of
    finding alpha that is not there.
    """
    rng = np.random.default_rng(23)
    n = 1200
    shocks = rng.normal(0.0, 0.004, n)
    residuals = np.empty(n)
    residuals[0] = shocks[0]
    for i in range(1, n):  # AR(1), rho = 0.6
        residuals[i] = 0.6 * residuals[i - 1] + shocks[i]

    design = np.column_stack([np.ones(n), rng.normal(0.0, 0.01, n)])
    hac = newey_west_covariance(design, residuals)

    variance = float(residuals @ residuals) / (n - design.shape[1])
    ols = variance * np.linalg.pinv(design.T @ design)

    hac_alpha_se = math.sqrt(hac[0, 0])
    ols_alpha_se = math.sqrt(ols[0, 0])
    # Roughly sqrt((1+rho)/(1-rho)) = 2.0 for rho = 0.6.
    assert hac_alpha_se > 1.5 * ols_alpha_se


def test_newey_west_matches_ols_when_residuals_are_independent() -> None:
    """The correction should cost nothing when there is nothing to correct."""
    rng = np.random.default_rng(29)
    n = 2000
    residuals = rng.normal(0.0, 0.004, n)
    design = np.column_stack([np.ones(n), rng.normal(0.0, 0.01, n)])

    hac = newey_west_covariance(design, residuals)
    variance = float(residuals @ residuals) / (n - design.shape[1])
    ols = variance * np.linalg.pinv(design.T @ design)

    assert math.sqrt(hac[0, 0]) == pytest.approx(math.sqrt(ols[0, 0]), rel=0.25)


def test_autocorrelation_does_not_inflate_the_alpha_t_statistic() -> None:
    """End to end: the same fake alpha, judged with and without the correction."""
    rng = np.random.default_rng(31)
    n = 1000
    x = rng.normal(0.0, 0.01, (n, 1))
    shocks = rng.normal(0.0, 0.004, n)
    returns = np.empty(n)
    returns[0] = shocks[0]
    for i in range(1, n):
        returns[i] = 0.75 * returns[i - 1] + shocks[i]

    hac_t = attribute(returns, x, factor_names=("market",), lags=20).alpha_t_statistic
    naive_t = attribute(returns, x, factor_names=("market",), lags=0).alpha_t_statistic

    assert abs(hac_t) < abs(naive_t)


# ------------------------------------------------------------------ validation --
def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        attribute(np.zeros(100), np.zeros((90, 1)), factor_names=("market",))


def test_mismatched_factor_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="factor_names"):
        attribute(np.zeros(100), np.zeros((100, 2)), factor_names=("market",))


def test_too_few_observations_to_identify_the_loadings() -> None:
    with pytest.raises(ValueError, match="cannot identify"):
        attribute(np.zeros(3), np.zeros((3, 3)), factor_names=NAMES)


def test_non_finite_inputs_are_rejected() -> None:
    returns = np.zeros(100)
    returns[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        attribute(returns, np.zeros((100, 1)), factor_names=("market",))


def test_a_single_factor_may_be_passed_as_a_vector() -> None:
    x = factors()[:, 0]
    rng = np.random.default_rng(37)
    result = attribute(0.7 * x + rng.normal(0.0, 0.002, len(x)), x, factor_names=("market",))
    assert result.betas.shape == (1,)
    assert result.betas[0] == pytest.approx(0.7, abs=0.02)


# ------------------------------------------------------------------------- PCA --
def one_factor_universe(n: int = 800, assets: int = 12, seed: int = 41) -> np.ndarray:
    """A universe with a dominant common factor, as real markets have."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.011, n)
    loadings = rng.uniform(0.7, 1.3, assets)
    return common[:, None] * loadings[None, :] + rng.normal(0.0, 0.005, (n, assets))


def test_the_first_component_finds_the_market_nobody_measured() -> None:
    model = pca_risk_model(one_factor_universe(), n_factors=3)

    assert model.first_factor_share > 0.6
    # And the loadings on it all share a sign: that is what "the market" looks like.
    first = model.loadings[:, 0]
    assert np.all(first > 0) or np.all(first < 0)


def test_the_factor_structure_reconstructs_the_covariance() -> None:
    returns = one_factor_universe()
    model = pca_risk_model(returns, n_factors=5)

    sample = np.cov(returns, rowvar=False, ddof=1)
    reconstructed = model.covariance()

    # Diagonals are exact by construction; off-diagonals are approximated.
    assert np.allclose(np.diag(reconstructed), np.diag(sample), rtol=1e-6)
    error = np.abs(reconstructed - sample).max() / np.abs(sample).max()
    assert error < 0.1


def test_specific_variance_is_what_the_factors_leave_behind() -> None:
    returns = one_factor_universe()
    sparse = pca_risk_model(returns, n_factors=1)
    dense = pca_risk_model(returns, n_factors=8)

    assert dense.specific_variance.sum() < sparse.specific_variance.sum()
    assert np.all(sparse.specific_variance > 0)


def test_explained_fractions_sum_toward_one_as_factors_are_added() -> None:
    returns = one_factor_universe()
    assert (
        pca_risk_model(returns, n_factors=1).explained_fraction.sum()
        < pca_risk_model(returns, n_factors=6).explained_fraction.sum()
    )


def test_n_factors_must_leave_something_unexplained() -> None:
    returns = one_factor_universe(assets=6)
    with pytest.raises(ValueError, match="n_factors must be between"):
        pca_risk_model(returns, n_factors=6)
    with pytest.raises(ValueError, match="n_factors must be between"):
        pca_risk_model(returns, n_factors=0)


def test_pca_rejects_a_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match=r"\(observations, assets\)"):
        pca_risk_model(np.zeros(100), n_factors=1)


def test_the_loudest_asset_does_not_get_to_be_the_market() -> None:
    """Why the decomposition standardises first.

    One asset with several times everyone else's volatility and no correlation to
    them will dominate the leading component of a covariance decomposition, because
    maximising explained variance is exactly what selects for it. The result looks
    like a market factor and is a single-asset factor.
    """
    rng = np.random.default_rng(97)
    n, assets = 1000, 8
    common = rng.normal(0.0, 0.010, n)
    returns = common[:, None] * np.ones(assets) + rng.normal(0.0, 0.004, (n, assets))
    # One loud, independent instrument -- an oil fund among equity ETFs.
    returns[:, -1] = rng.normal(0.0, 0.045, n)

    standardised = pca_risk_model(returns, n_factors=1, standardise=True)
    raw = pca_risk_model(returns, n_factors=1, standardise=False)

    def dominance(model: PCARiskModel) -> float:
        weights = np.abs(model.loadings[:, 0]) / np.abs(model.loadings[:, 0]).sum()
        return float(weights[-1])

    # The raw decomposition hands most of its first component to the loud asset.
    assert dominance(raw) > 0.6
    # Standardised, the first component is the factor the other seven share.
    assert dominance(standardised) < 0.15
    shared = standardised.loadings[:-1, 0]
    assert np.all(shared > 0) or np.all(shared < 0)


def test_a_constant_column_cannot_be_standardised() -> None:
    returns = one_factor_universe(assets=5)
    returns[:, 2] = 0.0
    with pytest.raises(ValueError, match="positive variance"):
        pca_risk_model(returns, n_factors=2)


def test_explained_and_specific_variance_account_for_the_whole_matrix() -> None:
    """They are in the same units, so together they are the total. If they were
    not, every explained fraction the model reports would be meaningless."""
    returns = one_factor_universe()
    model = pca_risk_model(returns, n_factors=4)

    total = np.trace(np.cov(returns, rowvar=False, ddof=1))
    assert model.explained_variance.sum() + model.specific_variance.sum() == pytest.approx(
        float(total), rel=1e-9
    )


def test_a_statistical_factor_has_no_name_and_the_description_says_so() -> None:
    text = pca_risk_model(one_factor_universe(), n_factors=3).describe()
    assert "no names" in text
    assert "useless for attribution" in text
