"""Covariance estimation.

The sample covariance matrix is unbiased and nearly useless for optimisation. With
``N`` assets it has ``N(N+1)/2`` parameters estimated from ``N·T`` numbers, so for
any realistic universe it is badly conditioned -- and a mean-variance optimiser
responds to a badly conditioned matrix by taking enormous offsetting positions in
whichever assets happened to look most correlated in-sample. The optimiser is not
misbehaving; it is doing exactly what the estimate told it to.

Shrinkage fixes this by pulling the sample estimate toward a structured target.
Ledoit-Wolf chooses *how far* analytically, by minimising expected squared error,
so the intensity is estimated rather than picked -- which matters, because picking
it is another parameter nobody counts.

Two targets, and the choice is not cosmetic:

``constant_correlation`` (the default)
    Keeps each asset's own sample variance and pulls every correlation toward
    their common average. This is the target that works on financial data:
    variances are estimated tolerably from a few hundred observations, correlations
    are not, and assets really do share a dominant common factor.

``identity``
    Uncorrelated assets with equal variance. So badly wrong for a real universe
    that the estimator correctly declines to shrink toward it -- which makes it a
    good demonstration of the derived intensity doing its job, and a poor default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.logging import get_logger

__all__ = [
    "CovarianceEstimate",
    "correlation_from_covariance",
    "ewma_covariance",
    "ledoit_wolf_covariance",
    "matrix_condition_number",
    "sample_covariance",
]

log = get_logger("quantlab.portfolio.covariance")

#: Above this a covariance matrix is effectively singular in float64, and an
#: optimiser inverting it is amplifying noise rather than using information.
ILL_CONDITIONED = 1e8


def matrix_condition_number(matrix: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue.

    The multiplier on estimation error that inverting this matrix applies. A
    condition number of 1e6 means a 0.01% error in the inputs becomes a 100% error
    in the inverse -- which is how an optimiser produces a confident, enormous,
    meaningless position.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    smallest = float(eigenvalues.min())
    if smallest <= 0:
        return float("inf")
    return float(eigenvalues.max() / smallest)


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    """A covariance matrix, and how much to trust it."""

    matrix: np.ndarray
    observations: int
    #: Fraction of the way from the sample estimate to the target, 0 to 1.
    shrinkage: float
    method: str

    @property
    def n_assets(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def condition(self) -> float:
        return matrix_condition_number(self.matrix)

    @property
    def observations_per_parameter(self) -> float:
        """Data points per free parameter. Below about two, the sample estimate is
        noise wearing a matrix."""
        parameters = self.n_assets * (self.n_assets + 1) / 2
        return self.observations * self.n_assets / parameters

    @property
    def is_usable(self) -> bool:
        return bool(np.isfinite(self.matrix).all()) and self.condition < ILL_CONDITIONED

    def volatilities(self, periods_per_year: float = 252.0) -> np.ndarray:
        return np.sqrt(np.diag(self.matrix) * periods_per_year)

    def describe(self) -> str:
        return (
            f"{self.method}: {self.n_assets} assets from {self.observations} "
            f"observations ({self.observations_per_parameter:.1f} per parameter), "
            f"shrinkage {self.shrinkage:.2f}, condition number {self.condition:,.0f}"
        )


def _validated(returns: np.ndarray) -> np.ndarray:
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be (observations, assets)")
    if matrix.shape[0] < 2:
        raise ValueError("need at least two observations to estimate a covariance")
    if not np.isfinite(matrix).all():
        raise ValueError(
            "returns contain non-finite values. Covariance estimation propagates "
            "them into every element, so they are rejected here rather than "
            "discovered later as a NaN portfolio weight."
        )
    return matrix


def sample_covariance(returns: np.ndarray) -> CovarianceEstimate:
    """The unbiased sample estimate. Correct, and usually not what you want."""
    matrix = _validated(returns)
    estimate = CovarianceEstimate(
        matrix=np.cov(matrix, rowvar=False, ddof=1),
        observations=matrix.shape[0],
        shrinkage=0.0,
        method="sample",
    )
    if estimate.observations_per_parameter < 2:
        log.warning(
            "portfolio.covariance.underdetermined",
            assets=estimate.n_assets,
            observations=estimate.observations,
            per_parameter=round(estimate.observations_per_parameter, 2),
            note="an optimiser inverting this amplifies noise rather than using information",
        )
    return estimate


def ewma_covariance(returns: np.ndarray, *, halflife: float = 63.0) -> CovarianceEstimate:
    """Exponentially weighted covariance.

    Adapts to regime change at the cost of a shorter effective sample: a 63-day
    half-life uses roughly ninety observations' worth of information however long
    the history is. That trade is usually worth making for volatility and usually
    not for correlation, which is noisier to begin with.
    """
    matrix = _validated(returns)
    if halflife <= 0:
        raise ValueError("halflife must be positive")

    observations = matrix.shape[0]
    decay = 0.5 ** (1.0 / halflife)
    weights = decay ** np.arange(observations - 1, -1, -1)
    weights /= weights.sum()

    centred = matrix - np.average(matrix, axis=0, weights=weights)
    covariance = centred.T @ (centred * weights[:, None])
    # Bias correction for weighted estimates: the analogue of ddof=1.
    covariance /= 1.0 - float((weights**2).sum())

    effective = 1.0 / float((weights**2).sum())
    return CovarianceEstimate(
        matrix=covariance,
        observations=int(effective),
        shrinkage=0.0,
        method=f"ewma(halflife={halflife:g})",
    )


def ledoit_wolf_covariance(
    returns: np.ndarray, *, target: str = "constant_correlation"
) -> CovarianceEstimate:
    """Ledoit-Wolf shrinkage, with the intensity derived rather than chosen.

    Short samples and wide universes shrink hard; long samples and narrow ones
    barely shrink at all. Both are correct, and neither is a setting.
    """
    matrix = _validated(returns)
    observations, assets = matrix.shape
    centred = matrix - matrix.mean(axis=0)
    sample = centred.T @ centred / observations

    if target == "identity":
        structure = float(np.trace(sample) / assets) * np.eye(assets)
    elif target == "constant_correlation":
        structure = constant_correlation_target(sample)
    else:
        raise ValueError(f"unknown target {target!r}; use 'constant_correlation' or 'identity'")

    # gamma: how far the target sits from the sample estimate.
    misspecification = float(np.sum((structure - sample) ** 2))
    if misspecification <= 0:
        return CovarianceEstimate(
            matrix=sample,
            observations=observations,
            shrinkage=0.0,
            method=f"ledoit-wolf({target})",
        )

    # pi: the sample estimate's own sampling variance, element by element.
    squared = centred**2
    pi_matrix = (squared.T @ squared) / observations - sample**2
    pi = float(pi_matrix.sum())

    # rho: covariance between the target's estimation error and the sample's. Zero
    # for a fixed target; not zero for constant correlation, because that target is
    # itself estimated from the same data. Ignoring it over-shrinks, which is the
    # usual way this estimator is got wrong.
    if target == "constant_correlation":
        rho = _constant_correlation_rho(centred, sample, pi_matrix)
    else:
        rho = float(np.trace(pi_matrix))

    intensity = float(np.clip((pi - rho) / misspecification / observations, 0.0, 1.0))
    shrunk = intensity * structure + (1.0 - intensity) * sample

    estimate = CovarianceEstimate(
        matrix=shrunk,
        observations=observations,
        shrinkage=intensity,
        method=f"ledoit-wolf({target})",
    )
    log.debug(
        "portfolio.covariance.shrunk",
        assets=assets,
        observations=observations,
        target=target,
        shrinkage=round(intensity, 4),
        condition=round(estimate.condition, 1),
    )
    return estimate


def constant_correlation_target(sample: np.ndarray) -> np.ndarray:
    """Sample variances kept; correlations replaced by their common average."""
    deviations = np.sqrt(np.diag(sample))
    outer = np.outer(deviations, deviations)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(outer > 0, sample / outer, 0.0)

    assets = sample.shape[0]
    off_diagonal = ~np.eye(assets, dtype=bool)
    average = float(correlation[off_diagonal].mean()) if assets > 1 else 0.0

    structure = average * outer
    np.fill_diagonal(structure, np.diag(sample))
    return structure


def _constant_correlation_rho(
    centred: np.ndarray, sample: np.ndarray, pi_matrix: np.ndarray
) -> float:
    assets = centred.shape[1]
    if assets < 2:
        return float(np.trace(pi_matrix))

    deviations = np.sqrt(np.diag(sample))
    outer = np.outer(deviations, deviations)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(outer > 0, sample / outer, 0.0)
    off_diagonal = ~np.eye(assets, dtype=bool)
    average = float(correlation[off_diagonal].mean())

    squared = centred**2
    # theta[i, j] = cov((x_i - mean)^2, (x_i - mean)(x_j - mean))
    theta = np.empty((assets, assets), dtype=float)
    for i in range(assets):
        cross = centred[:, i : i + 1] * centred
        theta[i] = (squared[:, i : i + 1] * cross).mean(axis=0) - sample[i, i] * sample[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(deviations[:, None] > 0, deviations[None, :] / deviations[:, None], 0.0)
    contribution = (average / 2.0) * (ratio * theta + ratio.T * theta.T)

    rho = float(np.trace(pi_matrix)) + float(contribution[off_diagonal].sum())
    return rho


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Correlation matrix implied by a covariance matrix."""
    matrix = np.asarray(covariance, dtype=float)
    deviations = np.sqrt(np.diag(matrix))
    if np.any(deviations <= 0):
        raise ValueError("every asset needs a positive variance to have a correlation")
    correlation: np.ndarray = matrix / np.outer(deviations, deviations)
    return correlation
