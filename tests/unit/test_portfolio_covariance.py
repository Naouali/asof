"""Covariance estimation: conditioning, shrinkage, and accuracy against truth."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.portfolio.covariance import (
    constant_correlation_target,
    correlation_from_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    matrix_condition_number,
    sample_covariance,
)


def factor_returns(
    n_assets: int, observations: int, *, rho: float = 0.6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """One common factor with dispersed volatilities, and the truth behind it."""
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.005, 0.02, n_assets)
    common = rng.normal(0, 1, (observations, 1))
    specific = rng.normal(0, 1, (observations, n_assets))
    returns = (np.sqrt(rho) * common + np.sqrt(1 - rho) * specific) * vols
    truth = (rho * np.ones((n_assets, n_assets)) + (1 - rho) * np.eye(n_assets)) * np.outer(
        vols, vols
    )
    return returns, truth


# ----------------------------------------------------------------------------------
# What shrinkage is for
# ----------------------------------------------------------------------------------
def test_shrinkage_beats_the_sample_estimate_on_a_short_wide_panel() -> None:
    """The case shrinkage exists for: more parameters than the data can identify."""
    errors = {"sample": [], "shrunk": []}
    for seed in range(25):
        returns, truth = factor_returns(40, 80, seed=seed)
        errors["sample"].append(np.linalg.norm(sample_covariance(returns).matrix - truth))
        errors["shrunk"].append(np.linalg.norm(ledoit_wolf_covariance(returns).matrix - truth))
    assert np.mean(errors["shrunk"]) < 0.9 * np.mean(errors["sample"])


def test_the_target_choice_matters() -> None:
    """A scaled-identity target is so wrong for correlated assets that the
    estimator correctly declines to use it -- and gains almost nothing."""
    gains = {"identity": [], "constant_correlation": []}
    for seed in range(20):
        returns, truth = factor_returns(40, 80, seed=seed)
        base = np.linalg.norm(sample_covariance(returns).matrix - truth)
        for target in gains:
            estimate = ledoit_wolf_covariance(returns, target=target)
            gains[target].append(np.linalg.norm(estimate.matrix - truth) / base)
    assert np.mean(gains["constant_correlation"]) < np.mean(gains["identity"])
    assert np.mean(gains["identity"]) > 0.9, "the identity target barely helps here"


def heterogeneous_returns(n_assets: int, observations: int, *, seed: int = 0) -> np.ndarray:
    """Three factors with different loadings, so pairwise correlations genuinely
    differ and the constant-correlation target is imperfect."""
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.005, 0.02, n_assets)
    factors = rng.normal(0, 1, (observations, 3))
    loadings = rng.uniform(-0.8, 0.9, (3, n_assets))
    return (factors @ loadings + rng.normal(0, 1, (observations, n_assets)) * 0.6) * vols


def test_shrinkage_falls_as_the_sample_grows() -> None:
    """With more data the sample estimate becomes trustworthy and the target is
    needed less. Monotone, on data whose structure the target only approximates."""
    shrinkages = [
        ledoit_wolf_covariance(heterogeneous_returns(30, t, seed=1)).shrinkage
        for t in (40, 100, 500, 2000)
    ]
    assert shrinkages == sorted(shrinkages, reverse=True)
    assert shrinkages[0] > 5 * shrinkages[-1]


def test_a_perfectly_specified_target_is_used_completely() -> None:
    """When the data really is constant-correlation, full shrinkage is optimal at
    any sample size -- and the estimator finds that. Worth pinning, because it
    looks like a bug until you notice the target is exactly right."""
    returns, _ = factor_returns(20, 5000, seed=1)
    assert ledoit_wolf_covariance(returns).shrinkage > 0.99


def test_shrinkage_stays_within_bounds() -> None:
    for observations in (10, 50, 500):
        estimate = ledoit_wolf_covariance(factor_returns(15, observations, seed=2)[0])
        assert 0.0 <= estimate.shrinkage <= 1.0


def test_shrinkage_improves_conditioning() -> None:
    """A badly conditioned matrix is not a nuisance: inverting it multiplies
    estimation error, which is how an optimiser produces a confident, enormous,
    meaningless position."""
    returns, _ = factor_returns(60, 70, seed=3)
    assert ledoit_wolf_covariance(returns).condition < sample_covariance(returns).condition / 10


# ----------------------------------------------------------------------------------
# The target itself
# ----------------------------------------------------------------------------------
def test_the_constant_correlation_target_keeps_the_variances() -> None:
    """Variances are estimated tolerably from a few hundred observations;
    correlations are not. The target shrinks only the latter."""
    returns, _ = factor_returns(10, 300, seed=4)
    sample = np.cov(returns, rowvar=False, ddof=0)
    target = constant_correlation_target(sample)
    assert np.diag(target) == pytest.approx(np.diag(sample))


def test_the_target_has_one_common_correlation() -> None:
    returns, _ = factor_returns(10, 300, seed=5)
    sample = np.cov(returns, rowvar=False, ddof=0)
    correlation = correlation_from_covariance(constant_correlation_target(sample))
    off_diagonal = correlation[~np.eye(10, dtype=bool)]
    assert np.ptp(off_diagonal) < 1e-12


# ----------------------------------------------------------------------------------
# EWMA
# ----------------------------------------------------------------------------------
def test_ewma_tracks_a_volatility_regime_change() -> None:
    rng = np.random.default_rng(6)
    calm = rng.normal(0, 0.005, (400, 3))
    stormy = rng.normal(0, 0.030, (100, 3))
    returns = np.vstack([calm, stormy])

    ewma = ewma_covariance(returns, halflife=20).volatilities()
    full = sample_covariance(returns).volatilities()
    assert np.all(ewma > full), "recent turbulence should dominate a short half-life"


def test_a_shorter_half_life_uses_less_data() -> None:
    returns, _ = factor_returns(5, 500, seed=7)
    assert (
        ewma_covariance(returns, halflife=10).observations
        < ewma_covariance(returns, halflife=200).observations
    )


def test_an_invalid_half_life_is_rejected() -> None:
    returns, _ = factor_returns(5, 100, seed=8)
    with pytest.raises(ValueError, match="halflife must be positive"):
        ewma_covariance(returns, halflife=0)


# ----------------------------------------------------------------------------------
# Diagnostics and validation
# ----------------------------------------------------------------------------------
def test_observations_per_parameter_is_reported() -> None:
    """Below about two, the sample estimate is noise wearing a matrix."""
    estimate = sample_covariance(factor_returns(50, 60, seed=9)[0])
    assert estimate.observations_per_parameter < 3
    assert "per parameter" in estimate.describe()


def test_a_singular_matrix_has_infinite_condition_number() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert matrix_condition_number(singular) == float("inf")


def test_non_finite_returns_are_rejected_not_propagated() -> None:
    """Covariance estimation spreads a single NaN into every element, so it is
    caught here rather than discovered as a NaN portfolio weight."""
    returns, _ = factor_returns(5, 100, seed=10)
    returns[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ledoit_wolf_covariance(returns)


def test_too_little_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        sample_covariance(np.zeros((1, 3)))


def test_an_unknown_target_is_rejected() -> None:
    returns, _ = factor_returns(5, 100, seed=11)
    with pytest.raises(ValueError, match="unknown target"):
        ledoit_wolf_covariance(returns, target="wishful")


def test_a_zero_variance_asset_has_no_correlation() -> None:
    with pytest.raises(ValueError, match="positive variance"):
        correlation_from_covariance(np.array([[1.0, 0.0], [0.0, 0.0]]))
