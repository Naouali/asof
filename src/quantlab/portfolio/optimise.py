"""Portfolio construction from a covariance estimate.

Four schemes, in increasing order of how much they trust the inputs:

:func:`equal_weight`
    Trusts nothing. Hard to beat out of sample, and the honest baseline every
    other scheme should be measured against.

:func:`inverse_volatility`
    Trusts volatility estimates, which are the most reliable thing a covariance
    matrix contains. Ignores correlation entirely.

:func:`risk_parity`
    Trusts the full covariance, but only to equalise *risk contributions* -- it
    never uses expected returns, so it cannot be wrong about them.

:func:`mean_variance`
    Trusts expected returns too, which are estimated far worse than covariances.
    This is where optimisers earn their reputation: the scheme is correct and the
    inputs are not, and it responds to bad inputs with enormous confident
    positions. Constraints here are not a nicety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quantlab.logging import get_logger
from quantlab.portfolio.constraints import Constraints

__all__ = [
    "Allocation",
    "equal_weight",
    "inverse_volatility",
    "mean_variance",
    "risk_contributions",
    "risk_parity",
]

log = get_logger("quantlab.portfolio.optimise")


@dataclass(frozen=True, slots=True)
class Allocation:
    """Weights, and what they imply."""

    weights: np.ndarray
    method: str
    #: Annualised portfolio volatility implied by the covariance used.
    volatility: float
    #: True when the optimiser reported success. False weights are still returned,
    #: because a failed optimisation that silently falls back to something else is
    #: worse than one you can see.
    converged: bool = True
    message: str = ""

    @property
    def gross(self) -> float:
        return float(np.abs(self.weights).sum())

    @property
    def net(self) -> float:
        return float(self.weights.sum())

    @property
    def effective_positions(self) -> float:
        """Inverse Herfindahl: how many positions this really is.

        A book of fifty names where two carry ninety per cent of the risk is a
        book of two names, and this is the number that says so.
        """
        gross = self.gross
        if gross <= 0:
            return 0.0
        shares = np.abs(self.weights) / gross
        return float(1.0 / np.sum(shares**2))


def _check(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance contains non-finite values")
    if np.any(np.diag(matrix) <= 0):
        raise ValueError(
            "every asset needs a positive variance. A zero-variance asset has "
            "infinite risk-adjusted appeal and will consume the whole book."
        )
    return matrix


def _volatility(weights: np.ndarray, covariance: np.ndarray, periods: float) -> float:
    return float(np.sqrt(max(weights @ covariance @ weights, 0.0) * periods))


def equal_weight(n_assets: int, *, gross: float = 1.0) -> Allocation:
    """The baseline. Every other scheme should be measured against it."""
    if n_assets < 1:
        raise ValueError("need at least one asset")
    weights = np.full(n_assets, gross / n_assets)
    return Allocation(weights=weights, method="equal-weight", volatility=float("nan"))


def inverse_volatility(
    covariance: np.ndarray, *, gross: float = 1.0, periods_per_year: float = 252.0
) -> Allocation:
    """Weight inversely to volatility, ignoring correlation.

    Cruder than risk parity and often barely worse, because the correlation half
    of the covariance matrix is where most of the estimation error lives.
    """
    matrix = _check(covariance)
    deviations = np.sqrt(np.diag(matrix))
    raw = 1.0 / deviations
    weights = gross * raw / raw.sum()
    return Allocation(
        weights=weights,
        method="inverse-volatility",
        volatility=_volatility(weights, matrix, periods_per_year),
    )


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Each position's share of total portfolio variance.

    ``w_i · (Σw)_i / (w'Σw)``. These sum to one, and their dispersion is the
    honest measure of concentration -- weights say nothing about it.
    """
    weights = np.asarray(weights, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    variance = float(weights @ matrix @ weights)
    if variance <= 0:
        return np.zeros_like(weights)
    contributions: np.ndarray = weights * (matrix @ weights) / variance
    return contributions


def risk_parity(
    covariance: np.ndarray,
    *,
    gross: float = 1.0,
    periods_per_year: float = 252.0,
    tolerance: float = 1e-10,
    max_iterations: int = 5000,
) -> Allocation:
    """Equalise each asset's contribution to portfolio risk.

    Solved by the multiplicative update ``w_i ← w_i · (1/N) / RC_i``, which pushes
    each contribution directly toward its target and converges for any
    positive-definite covariance without an optimiser.

    The obvious-looking alternative, ``w_i ← w_i / (Σw)_i``, is **not** risk
    parity and converges to a degenerate corner -- it was the first thing tried
    here and it produced a book with one position carrying 57% of the risk while
    reporting success.

    Long only by construction, which is a real restriction: risk parity has no way
    to express a short. Its appeal is that it never uses expected returns, so it
    cannot be wrong about the quantity that is hardest to estimate.
    """
    matrix = _check(covariance)
    assets = matrix.shape[0]
    if assets == 1:
        return Allocation(
            weights=np.array([gross]),
            method="risk-parity",
            volatility=_volatility(np.array([gross]), matrix, periods_per_year),
        )

    weights = np.full(assets, 1.0 / assets)
    target = 1.0 / assets
    iteration = 0

    for iteration in range(1, max_iterations + 1):  # noqa: B007 - read after the loop
        contributions = risk_contributions(weights, matrix)
        if np.any(contributions <= 0):
            raise ValueError(
                "covariance is not positive definite; risk parity is undefined. "
                "Shrink the estimate before optimising against it."
            )
        if float(np.abs(contributions - target).max()) < tolerance:
            break
        weights = weights * (target / contributions) ** 0.5
        weights /= weights.sum()
    else:
        log.warning(
            "portfolio.risk_parity.not_converged",
            iterations=max_iterations,
            worst_error=float(np.abs(risk_contributions(weights, matrix) - target).max()),
        )

    weights = weights * gross
    return Allocation(
        weights=weights,
        method="risk-parity",
        volatility=_volatility(weights, matrix, periods_per_year),
        converged=iteration < max_iterations,
        message=f"{iteration} iterations",
    )


def mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    risk_aversion: float = 1.0,
    constraints: Constraints | None = None,
    periods_per_year: float = 252.0,
) -> Allocation:
    """Maximise ``w'μ − (γ/2)·w'Σw`` subject to constraints.

    The unconstrained solution is ``w = (γΣ)⁻¹μ``, and on real inputs it is
    usually unusable: expected returns are estimated far worse than covariances,
    and inverting a covariance matrix amplifies whatever error is in it. The
    constraints are not a refinement of this scheme, they are what makes it
    survivable.

    Solved numerically because the constraints are inequalities. The unconstrained
    closed form is used as the starting point, which both speeds convergence and
    makes the constrained answer directly comparable to it.
    """
    mu = np.asarray(expected_returns, dtype=float)
    matrix = _check(covariance)
    if mu.shape[0] != matrix.shape[0]:
        raise ValueError("expected_returns and covariance disagree about the universe")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    limits = constraints or Constraints()
    # Fail on an impossible problem rather than returning the plausible-looking
    # weights SLSQP produces when it gives up.
    limits.check_feasible(len(mu))

    def objective(weights: np.ndarray) -> float:
        return float(-(weights @ mu) + 0.5 * risk_aversion * (weights @ matrix @ weights))

    def gradient(weights: np.ndarray) -> np.ndarray:
        derivative: np.ndarray = -mu + risk_aversion * (matrix @ weights)
        return derivative

    start: np.ndarray
    try:
        start = np.linalg.solve(risk_aversion * matrix, mu)
    except np.linalg.LinAlgError:
        start = np.zeros_like(mu)
    if not np.isfinite(start).all():
        start = np.zeros_like(mu)

    # scipy's overloads do not cover a separate `jac` callable alongside dict
    # constraints; the call is correct and the stubs are narrower than the API.
    result = minimize(  # type: ignore[call-overload]
        objective,
        x0=limits.clip(start),
        jac=gradient,
        method="SLSQP",
        bounds=limits.bounds(len(mu)),
        constraints=limits.scipy_constraints(len(mu)),
        options={"maxiter": 500, "ftol": 1e-12},
    )

    weights = np.asarray(result.x, dtype=float)
    if not result.success:
        log.warning(
            "portfolio.mean_variance.not_converged",
            message=str(result.message),
            note="weights are returned anyway; a silent fallback would be worse",
        )
    return Allocation(
        weights=weights,
        method="mean-variance",
        volatility=_volatility(weights, matrix, periods_per_year),
        converged=bool(result.success),
        message=str(result.message),
    )
