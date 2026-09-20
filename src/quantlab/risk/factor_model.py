"""Factor risk models and exposure attribution.

Spec section 8 states the purpose exactly: a factor risk model exists so that
exposure attribution can answer *"is this signal secretly just beta, or size, or a
sector bet?"* That question has a specific shape -- it is not "does this strategy
make money" but "would a passive position have made the same money more cheaply".

Two models:

:func:`attribute`
    Regress strategy returns on known factors -- Fama-French, momentum, whatever
    you have -- and report what is left. The residual alpha is the part the
    factors cannot explain, and its t-statistic is the honest measure of whether
    there is anything there.

:func:`pca_risk_model`
    When you have no factor returns, extract statistical factors from the
    covariance of the universe itself. Useless for attribution -- a statistical
    factor has no name and no economic meaning -- and genuinely useful for risk,
    because the first principal component of almost any asset universe is "the
    market" whether or not anyone measured it.

Standard errors are Newey-West, not OLS. Strategy returns are autocorrelated, and
OLS standard errors on autocorrelated data are too small -- which inflates every
t-statistic in the direction of finding alpha that is not there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantlab.logging import get_logger

__all__ = [
    "FactorAttribution",
    "PCARiskModel",
    "attribute",
    "newey_west_covariance",
    "pca_risk_model",
]

log = get_logger("quantlab.risk.factor_model")

#: Above this share of variance explained, a strategy is largely a factor bet.
MOSTLY_FACTOR = 0.70

#: Below this |t| the residual alpha is not distinguishable from zero.
ALPHA_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    """What the factors explain, and what they do not."""

    factor_names: tuple[str, ...]
    betas: np.ndarray
    beta_t_statistics: np.ndarray
    alpha_per_period: float
    alpha_t_statistic: float
    r_squared: float
    observations: int
    periods_per_year: float
    idiosyncratic_volatility: float

    @property
    def alpha_annual(self) -> float:
        return self.alpha_per_period * self.periods_per_year

    @property
    def is_mostly_factor(self) -> bool:
        return self.r_squared >= MOSTLY_FACTOR

    @property
    def has_residual_alpha(self) -> bool:
        return abs(self.alpha_t_statistic) >= ALPHA_THRESHOLD

    def significant_exposures(self) -> list[tuple[str, float, float]]:
        """``(factor, beta, t)`` for exposures distinguishable from zero."""
        return [
            (name, float(beta), float(t))
            for name, beta, t in zip(
                self.factor_names, self.betas, self.beta_t_statistics, strict=True
            )
            if abs(t) >= ALPHA_THRESHOLD
        ]

    def verdict(self) -> str:
        """The answer to the question the model exists for."""
        exposures = self.significant_exposures()
        if self.is_mostly_factor and not self.has_residual_alpha:
            names = ", ".join(name for name, _, _ in exposures) or "the factors supplied"
            return (
                f"Largely a factor bet: {self.r_squared:.0%} of the variance is "
                f"explained by {names}, and the residual alpha "
                f"({self.alpha_annual:+.2%} a year, t={self.alpha_t_statistic:.2f}) is "
                "not distinguishable from zero. A passive position in those factors "
                "would have done the same thing more cheaply."
            )
        if self.has_residual_alpha:
            return (
                f"Residual alpha of {self.alpha_annual:+.2%} a year "
                f"(t={self.alpha_t_statistic:.2f}) survives the factors, which "
                f"explain {self.r_squared:.0%} of the variance. That is what the "
                "deflated Sharpe should be computed on, not the raw return."
            )
        return (
            f"The factors explain {self.r_squared:.0%} of the variance and the "
            f"residual alpha (t={self.alpha_t_statistic:.2f}) is not significant. "
            "There is no evidence of skill here, factor-driven or otherwise."
        )

    def describe(self) -> str:
        lines = [
            f"R² {self.r_squared:.3f} over {self.observations:,} observations; "
            f"alpha {self.alpha_annual:+.2%}/yr (t={self.alpha_t_statistic:+.2f})",
        ]
        for name, beta, t in zip(
            self.factor_names, self.betas, self.beta_t_statistics, strict=True
        ):
            marker = " *" if abs(t) >= ALPHA_THRESHOLD else ""
            lines.append(f"  {name:<12} beta {beta:+.3f}  (t={t:+.2f}){marker}")
        lines.append(f"  {self.verdict()}")
        return "\n".join(lines)


def newey_west_covariance(
    design: np.ndarray, residuals: np.ndarray, *, lags: int | None = None
) -> np.ndarray:
    """HAC covariance of the OLS coefficients.

    Strategy returns are autocorrelated -- by construction when positions are held
    for more than a period, and in practice even when they are not. OLS standard
    errors assume they are not, come out too small, and inflate every t-statistic
    toward finding alpha that is not there.

    The lag length defaults to the usual ``4(T/100)^(2/9)`` rule.
    """
    observations = design.shape[0]
    if lags is None:
        lags = max(1, math.floor(4.0 * (observations / 100.0) ** (2.0 / 9.0)))

    weighted = design * residuals[:, None]
    meat = weighted.T @ weighted
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = weighted[lag:].T @ weighted[:-lag]
        meat += weight * (cross + cross.T)

    bread = np.linalg.pinv(design.T @ design)
    covariance: np.ndarray = bread @ meat @ bread
    return covariance


def attribute(
    returns: np.ndarray,
    factors: np.ndarray,
    *,
    factor_names: tuple[str, ...],
    periods_per_year: float = 252.0,
    lags: int | None = None,
) -> FactorAttribution:
    """Regress strategy returns on factor returns and report what survives.

    ``returns`` is ``(observations,)``; ``factors`` is ``(observations, k)``. Both
    should be excess returns over the same risk-free rate, or neither -- mixing
    them loads the mismatch onto the intercept, which is precisely the number
    being tested.
    """
    y = np.asarray(returns, dtype=float).ravel()
    x = np.asarray(factors, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(y) != x.shape[0]:
        raise ValueError("returns and factors have different lengths")
    if len(factor_names) != x.shape[1]:
        raise ValueError("factor_names does not match the number of factor columns")
    if len(y) <= x.shape[1] + 1:
        raise ValueError(
            f"{len(y)} observations cannot identify {x.shape[1]} factor loadings plus an intercept"
        )
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("returns and factors must be finite")

    design = np.column_stack([np.ones(len(y)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted

    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total if total > 0 else 0.0

    covariance = newey_west_covariance(design, residuals, lags=lags)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.where(errors > 0, coefficients / errors, 0.0)

    attribution = FactorAttribution(
        factor_names=tuple(factor_names),
        betas=coefficients[1:],
        beta_t_statistics=t_statistics[1:],
        alpha_per_period=float(coefficients[0]),
        alpha_t_statistic=float(t_statistics[0]),
        r_squared=float(r_squared),
        observations=len(y),
        periods_per_year=periods_per_year,
        idiosyncratic_volatility=float(np.std(residuals, ddof=1) * math.sqrt(periods_per_year)),
    )
    log.info(
        "risk.attribution",
        r_squared=round(attribution.r_squared, 4),
        alpha_annual=round(attribution.alpha_annual, 5),
        alpha_t=round(attribution.alpha_t_statistic, 3),
        mostly_factor=attribution.is_mostly_factor,
    )
    return attribution


@dataclass(frozen=True, slots=True)
class PCARiskModel:
    """Statistical factors extracted from the universe's own covariance."""

    #: ``(assets, n_factors)`` loadings.
    loadings: np.ndarray
    #: Variance explained by each factor, in return units. Components come in
    #: principal-component order, which is by explained *correlation*; once
    #: rescaled these shares need not be monotone, because a later component
    #: loading on high-volatility assets can carry more return variance than an
    #: earlier one loading on quiet ones.
    explained_variance: np.ndarray
    #: Asset-specific variance the factors do not explain.
    specific_variance: np.ndarray
    n_factors: int

    @property
    def explained_fraction(self) -> np.ndarray:
        total = float(self.explained_variance.sum() + self.specific_variance.sum())
        fraction: np.ndarray = (
            self.explained_variance / total if total > 0 else self.explained_variance
        )
        return fraction

    @property
    def first_factor_share(self) -> float:
        """Variance share of the leading factor.

        In almost any asset universe this is "the market", whether or not anyone
        measured it. A strategy whose returns load heavily on it is a market bet
        however its signal is described.
        """
        shares = self.explained_fraction
        return float(shares[0]) if len(shares) else 0.0

    def covariance(self) -> np.ndarray:
        """Reconstruct the covariance implied by the factor structure."""
        reconstructed: np.ndarray = self.loadings @ self.loadings.T + np.diag(
            self.specific_variance
        )
        return reconstructed

    def describe(self) -> str:
        shares = self.explained_fraction
        head = ", ".join(
            f"PC{k + 1} {share:.1%}" for k, share in enumerate(shares[: min(3, len(shares))])
        )
        return (
            f"{self.n_factors} statistical factors explaining "
            f"{shares.sum():.0%} of variance ({head}). "
            "These have no names: useful for risk, useless for attribution."
        )


def pca_risk_model(
    returns: np.ndarray, *, n_factors: int = 3, standardise: bool = True
) -> PCARiskModel:
    """Extract statistical risk factors by eigendecomposition.

    Use when no factor returns are available. It will always find structure --
    that is what PCA does -- so the temptation is to read meaning into the
    components. Resist it: a principal component is a direction of variance, not
    an economic exposure, and naming one is how a risk model becomes a story.

    **The decomposition is of the correlation matrix, not the covariance matrix.**
    Decomposing covariance directly lets the single highest-variance asset
    dominate the leading component, because that is literally what maximising
    explained variance selects for. On a twelve-ETF universe containing a 41%
    volatility oil fund, covariance PCA returns a first component loading +0.71
    on oil and under 0.35 on everything else -- an oil factor wearing the label
    "market". Standardising first gives the same universe a first component
    loading between -0.31 and -0.38 on every equity and credit name and 0.09 on
    oil, which is the common factor the model is supposed to find.

    Loadings and specific variances are rescaled back into return units
    afterwards, so :meth:`PCARiskModel.covariance` reconstructs the covariance
    matrix and not the correlation matrix. Pass ``standardise=False`` for the
    unscaled decomposition, which is occasionally what you want when the variance
    differences *are* the structure of interest.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be (observations, assets)")
    observations, assets = matrix.shape
    if n_factors < 1 or n_factors >= assets:
        raise ValueError(f"n_factors must be between 1 and {assets - 1}")
    if observations < assets:
        log.warning(
            "risk.pca.underdetermined",
            observations=observations,
            assets=assets,
            note="fewer observations than assets; the components are largely noise",
        )

    covariance = np.cov(matrix, rowvar=False, ddof=1)
    deviations = np.sqrt(np.diag(covariance))
    if standardise:
        if np.any(deviations <= 0):
            raise ValueError(
                "every asset needs a positive variance to be standardised; a constant "
                "column has no correlation with anything and cannot carry a loading"
            )
        target = covariance / np.outer(deviations, deviations)
        scale = deviations
    else:
        target = covariance
        scale = np.ones(assets)

    eigenvalues, eigenvectors = np.linalg.eigh(target)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    kept = np.clip(eigenvalues[:n_factors], 0.0, None)
    # Back into return units, so the reconstruction is of covariance throughout.
    loadings = scale[:, None] * eigenvectors[:, :n_factors] * np.sqrt(kept)
    specific = np.clip(np.diag(covariance) - np.sum(loadings**2, axis=1), 1e-12, None)

    return PCARiskModel(
        loadings=loadings,
        # Return-space variance carried by each factor, so that the explained and
        # specific pieces are in the same units and sum to the total.
        explained_variance=np.sum(loadings**2, axis=0),
        specific_variance=specific,
        n_factors=n_factors,
    )
