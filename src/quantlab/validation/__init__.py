"""Validation: the module that saves the user money.

Everything here exists to answer one question a backtest cannot answer about
itself: **is this result distinguishable from the best of however many things you
tried?**

    from quantlab.validation import TrialRegistry, validate_strategy

    report = validate_strategy(net_returns, name="mom-250", family="momentum",
                               registry=TrialRegistry(layout.state))
    print(report.describe())

The pieces, and what each corrects for:

* :mod:`~quantlab.validation.statistics` -- probabilistic Sharpe (corrects for the
  sample), deflated Sharpe (for the search), PBO (for the selection rule), and
  minimum backtest length (how long a sample the search would need).
* :mod:`~quantlab.validation.splitting` -- purged k-fold, combinatorial purged
  cross-validation and walk-forward. Ordinary k-fold leaks whenever labels overlap
  in time, and the leak is silent.
* :mod:`~quantlab.validation.registry` -- the **automatic** trial counter. Spec
  section 7: manual honesty about trial counts does not work.
* :mod:`~quantlab.validation.luck` -- empirical-Bayes shrinkage across the whole
  signal library. The most uncomfortable number the platform produces.

The red-team suite lives in ``tests/unit/test_validation_redteam.py``: deliberately
broken strategies -- look-ahead, survivorship, pure noise, an overfit sweep -- that
this module is required to reject. If the framework passes a known-fake strategy,
the framework is broken.
"""

from __future__ import annotations

from quantlab.validation.luck import LuckAdjustment, luck_adjust, t_statistic
from quantlab.validation.registry import (
    Trial,
    TrialRegistry,
    TrialSet,
    config_fingerprint,
)
from quantlab.validation.report import (
    ValidationReport,
    cpcv_path_returns,
    validate_strategy,
)
from quantlab.validation.splitting import (
    CombinatorialPurgedCV,
    PathSegment,
    PurgedKFold,
    Split,
    WalkForward,
)
from quantlab.validation.statistics import (
    HaircutSchedule,
    SharpeEvidence,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)

__all__ = [
    "CombinatorialPurgedCV",
    "HaircutSchedule",
    "LuckAdjustment",
    "PathSegment",
    "PurgedKFold",
    "SharpeEvidence",
    "Split",
    "Trial",
    "TrialRegistry",
    "TrialSet",
    "ValidationReport",
    "WalkForward",
    "config_fingerprint",
    "cpcv_path_returns",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "luck_adjust",
    "minimum_backtest_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "t_statistic",
    "validate_strategy",
]
