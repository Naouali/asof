"""Overfitting statistics.

The arithmetic that turns an impressive backtest into an honest one. Four ideas,
each correcting a different way a Sharpe ratio overstates what you found:

**Probabilistic Sharpe** corrects for the *sample*. A Sharpe of 1.5 over six
months is weaker evidence than the same 1.5 over ten years, and negative skew with
fat tails weakens it further -- which is exactly the return profile most strategies
that "work" turn out to have.

**Deflated Sharpe** corrects for the *search*. If you try two hundred variations,
the best one looks good by construction. The deflation asks whether the winner
beats what the best of two hundred coin flips would have produced.

**Probability of backtest overfitting** corrects for the *selection rule*. It asks
how often the configuration that looked best in-sample lands below median
out-of-sample. Above a half means your selection procedure is worse than random.

**Minimum backtest length** turns the question around: given how many things you
intend to try, how long a sample would you need for the winner to mean anything?

All Sharpe ratios in this module are **per-observation** unless a name says
``annual``. Mixing the two silently rescales every result by the square root of
252, which is a factor of sixteen and would be obvious -- except that it makes bad
strategies look good, which people tend not to question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from quantlab.logging import get_logger

__all__ = [
    "EULER_MASCHERONI",
    "HaircutSchedule",
    "SharpeEvidence",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "minimum_backtest_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
]

log = get_logger("quantlab.validation.statistics")

EULER_MASCHERONI = 0.5772156649015329

#: Trading days a year, used only to move between per-observation and annual
#: Sharpe ratios. Where an actual bar count is available it is preferred.
BARS_PER_YEAR = 252.0


# ======================================================================================
# Probabilistic and deflated Sharpe
# ======================================================================================
def probabilistic_sharpe_ratio(
    sharpe: float,
    *,
    observations: int,
    benchmark: float = 0.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark``.

    ``sharpe`` and ``benchmark`` are **per-observation**.

    Negative skew and fat tails widen the Sharpe estimator's standard error, which
    pulls this probability toward 0.5 -- *in both directions*. When the observed
    Sharpe beats the benchmark, as in the ordinary case of testing against zero,
    they weaken the evidence, and they are the return shape that most strategies
    which "work" turn out to have. When the observed Sharpe is already below the
    benchmark, as often happens under heavy deflation, the same widening makes the
    result harder to rule out and nudges the probability up. Both are correct: more
    uncertainty means less is established either way.

    Returns a probability in [0, 1]. Values below about 0.95 mean the sample does
    not establish the claim, whatever the point estimate says.
    """
    if observations < 2:
        raise ValueError("need at least two observations to estimate a standard error")

    kurtosis = excess_kurtosis + 3.0
    variance_term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0:
        # The estimator's variance has gone negative, which happens for extreme
        # moment estimates on short samples. Reporting a probability from it would
        # be fabricating precision.
        log.warning(
            "validation.psr_undefined",
            sharpe=sharpe,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            note="the Sharpe estimator's variance is non-positive for these moments",
        )
        return float("nan")

    statistic = (sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(variance_term)
    return float(norm.cdf(statistic))


def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across ``trials`` independent strategies with no edge.

    This is the bar a backtest has to clear to be evidence of anything. Search hard
    enough over strategies that are all worthless and the best of them will still
    look good; this says how good, so the comparison can be made explicitly.

    Grows like the square root of the log of the trial count, so it is brutal
    early -- going from one trial to ten costs far more than going from a hundred
    to a thousand.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be non-negative")
    if trials == 1:
        # Nothing was selected, so there is no selection to correct for.
        return 0.0

    upper = norm.ppf(1.0 - 1.0 / trials)
    lower = norm.ppf(1.0 - 1.0 / (trials * math.e))
    return float(
        math.sqrt(sharpe_variance) * ((1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * lower)
    )


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    observations: int,
    trials: int,
    sharpe_variance: float,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Probability the strategy has genuine skill, given how hard you looked.

    The probabilistic Sharpe measured against the expected best of ``trials``
    worthless strategies rather than against zero. A value below 0.95 means the
    result is consistent with having found the luckiest member of the search.

    ``sharpe_variance`` is the variance of the Sharpe ratios *across the trials*.
    A search over near-identical configurations has low variance and is punished
    less; a search over wildly different ones has high variance and is punished
    more, which is correct -- a wider search finds a luckier maximum.
    """
    benchmark = expected_max_sharpe(trials, sharpe_variance)
    return probabilistic_sharpe_ratio(
        sharpe,
        observations=observations,
        benchmark=benchmark,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )


def minimum_backtest_length(target_sharpe_annual: float, trials: int) -> float:
    """Years of data needed before a Sharpe of ``target_sharpe_annual`` means anything.

    Turns the deflation around: rather than asking whether this result survives, it
    asks how long a sample would have to be for the winner of ``trials`` to be
    distinguishable from the luckiest of ``trials`` coin flips.

        MinBTL (years) = ( E[max of N standard normals] / SR_annual )²

    The expected maximum is measured in units of the Sharpe estimator's own
    standard error, which is what makes the expression self-contained. Scaling it
    by the observed spread of the trials would be circular -- that spread depends
    on the sample length being solved for.

    The answer is usually longer than the data anyone has, and it is brutal at low
    Sharpe: a target of 0.5 after a hundred tries needs about thirty years, while a
    target of 2.0 needs about two. Most published anomalies live at the former end.
    """
    if target_sharpe_annual <= 0:
        raise ValueError("target_sharpe_annual must be positive")
    threshold = expected_max_sharpe(trials, 1.0)
    if threshold <= 0:
        return 0.0
    return float((threshold / target_sharpe_annual) ** 2)


# ======================================================================================
# Probability of backtest overfitting
# ======================================================================================
def probability_of_backtest_overfitting(
    returns: np.ndarray, *, partitions: int = 16, max_combinations: int | None = 20_000
) -> tuple[float, np.ndarray]:
    """PBO by combinatorially symmetric cross-validation.

    ``returns`` is ``(observations, configurations)`` -- one column per parameter
    setting tried. The procedure:

    1. Cut the sample into ``partitions`` equal blocks.
    2. For every way of choosing half the blocks as in-sample, pick the
       configuration that looked best there.
    3. Find where that configuration ranks out-of-sample, on the other half.

    PBO is the fraction of those splits where the in-sample winner landed **below
    median** out-of-sample. Above 0.5 means your selection rule does worse than
    picking at random, which is the signature of fitting noise.

    Unlike the deflated Sharpe, this needs the returns of *every* configuration
    tried, not just the winner -- so it measures the selection procedure itself
    rather than only its output.

    Returns ``(pbo, logits)``; the logit distribution is what a tearsheet plots.
    """
    from itertools import combinations

    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be (observations, configurations)")
    observations, configurations = matrix.shape
    if configurations < 2:
        raise ValueError(
            "PBO measures a *selection*, so it needs at least two configurations. "
            "With one there was nothing to select and nothing to overfit."
        )
    if partitions % 2 != 0:
        raise ValueError("partitions must be even so the sample splits in half")
    if observations < partitions * 2:
        raise ValueError(
            f"{observations} observations cannot be cut into {partitions} usable "
            "blocks; use fewer partitions or a longer sample"
        )

    block_size = observations // partitions
    blocks = [matrix[index * block_size : (index + 1) * block_size] for index in range(partitions)]

    half = partitions // 2
    logits: list[float] = []
    for count, chosen in enumerate(combinations(range(partitions), half)):
        if max_combinations is not None and count >= max_combinations:
            log.info(
                "validation.pbo_truncated",
                evaluated=count,
                note="sampling the first combinations; PBO is stable well before this",
            )
            break
        rest = [index for index in range(partitions) if index not in chosen]
        in_sample = np.vstack([blocks[index] for index in chosen])
        out_sample = np.vstack([blocks[index] for index in rest])

        best = int(np.argmax(_sharpe_columns(in_sample)))
        out_scores = _sharpe_columns(out_sample)
        # Rank of the in-sample winner among all configurations, out-of-sample.
        rank = float((out_scores <= out_scores[best]).sum()) / (configurations + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))

    array = np.array(logits)
    return float((array < 0).mean()), array


def _sharpe_columns(block: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe of each column, zero where it is undefined."""
    means = block.mean(axis=0)
    deviations = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(deviations > 0, means / deviations, 0.0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


# ======================================================================================
# Haircuts
# ======================================================================================
@dataclass(frozen=True, slots=True)
class HaircutSchedule:
    """Multiplicative haircuts applied to a net-of-cost Sharpe.

    Spec section 7 asks for three, shown separately: in-sample selection,
    post-publication decay, and transaction costs. Costs are not here because the
    backtest already charges them -- what remains is the part no backtest can
    measure.

    The two defaults compose to ``0.88 x 0.60 = 0.53``, which is the spec's working
    prior that **live Sharpe is about half the backtest Sharpe**. That is not a
    coincidence and is asserted by test: if either default moves, the composed
    prior should be re-examined rather than silently drifting.
    """

    #: Shrinkage for having chosen this strategy after seeing its results.
    in_sample_selection: float = 0.12
    #: Shrinkage for decay after publication, as others find and trade the same
    #: effect. The published range is roughly 35-50%; this is the midpoint.
    post_publication_decay: float = 0.40

    def __post_init__(self) -> None:
        for name, value in (
            ("in_sample_selection", self.in_sample_selection),
            ("post_publication_decay", self.post_publication_decay),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

    @property
    def combined_multiplier(self) -> float:
        return (1.0 - self.in_sample_selection) * (1.0 - self.post_publication_decay)

    def apply(self, net_sharpe: float) -> float:
        return net_sharpe * self.combined_multiplier

    def describe(self) -> str:
        return (
            f"in-sample selection -{self.in_sample_selection:.0%}, "
            f"post-publication decay -{self.post_publication_decay:.0%}, "
            f"combined x{self.combined_multiplier:.2f}"
        )


@dataclass(frozen=True, slots=True)
class SharpeEvidence:
    """A Sharpe ratio with everything needed to read it honestly.

    Constructing one of these is the only sanctioned way to report a Sharpe in
    this platform (spec section 13): the trial count and the deflated value travel
    with the number rather than being available on request.
    """

    gross_annual: float
    net_annual: float
    haircut_annual: float
    observations: int
    trials: int
    sharpe_variance: float
    skewness: float
    excess_kurtosis: float
    probabilistic: float
    deflated: float
    expected_max_from_search: float
    haircuts: HaircutSchedule

    @property
    def survives_deflation(self) -> bool:
        """Whether the result is distinguishable from the best of the search.

        0.95 is the conventional threshold and is arbitrary in the usual way. What
        is not arbitrary is that most backtests fail it.
        """
        return self.deflated >= 0.95

    @property
    def cost_drag(self) -> float:
        return self.gross_annual - self.net_annual

    def describe(self) -> str:
        verdict = "survives" if self.survives_deflation else "DOES NOT SURVIVE"
        return (
            f"Sharpe {self.gross_annual:.2f} gross / {self.net_annual:.2f} net / "
            f"{self.haircut_annual:.2f} after haircuts\n"
            f"  deflated {self.deflated:.3f} over {self.trials} trial(s) -- {verdict} "
            f"deflation at the 0.95 threshold\n"
            f"  probabilistic {self.probabilistic:.3f}; the search's expected best "
            f"was {self.expected_max_from_search * math.sqrt(BARS_PER_YEAR):.2f} annualised\n"
            f"  haircuts: {self.haircuts.describe()}"
        )
