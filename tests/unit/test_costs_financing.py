"""Financing, borrow, funding and roll.

These are the costs a backtest forgets. The tests below are mostly about the
*defaults*, because the defaults are what a strategy silently inherits.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.costs.base import Instrument
from quantlab.costs.financing import DAYS_PER_YEAR, FinancingModel, FinancingParams
from quantlab.data.catalogue import AssetClass


def stock(borrow_bps_annual: float | None = None) -> Instrument:
    return Instrument(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        adv_notional=8e9,
        volatility_daily=0.018,
        spread_bps=1.2,
        borrow_bps_annual=borrow_bps_annual,
    )


# ----------------------------------------------------------------------------------
# Borrow
# ----------------------------------------------------------------------------------
def test_unknown_borrow_is_charged_punitively_not_free() -> None:
    """Spec section 5: 10-20 bp/month is enough to erase most published anomaly
    alphas, so an unknown fee is charged as hard-to-borrow rather than assumed
    away. This is the single most consequential default in the package."""
    model = FinancingModel()
    cost, assumed = model.borrow_bps(stock(), days=DAYS_PER_YEAR, is_short=True)
    assert assumed, "an assumed rate must be flagged all the way to the tearsheet"
    assert cost == pytest.approx(15.0 * 12)
    assert cost >= 120.0, "at least 10 bp/month, per the spec"


def test_known_borrow_is_used_and_not_flagged() -> None:
    cost, assumed = FinancingModel().borrow_bps(stock(40.0), days=DAYS_PER_YEAR, is_short=True)
    assert not assumed
    assert cost == pytest.approx(40.0)


def test_longs_pay_no_borrow() -> None:
    cost, assumed = FinancingModel().borrow_bps(stock(), days=30, is_short=False)
    assert cost == 0.0
    assert not assumed


def test_borrow_accrues_pro_rata() -> None:
    model = FinancingModel()
    year, _ = model.borrow_bps(stock(365.0), days=DAYS_PER_YEAR, is_short=True)
    month, _ = model.borrow_bps(stock(365.0), days=30.0, is_short=True)
    assert month == pytest.approx(year * 30 / DAYS_PER_YEAR)


def test_act_365_is_used_not_act_360() -> None:
    """A 360-day convention would understate every financing accrual by 1.4%."""
    cost, _ = FinancingModel().borrow_bps(stock(365.0), days=365.0, is_short=True)
    assert cost == pytest.approx(365.0)


def test_hard_to_borrow_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """A name this expensive can be recalled, and a forced buy-in is a cost no
    model here captures."""
    import logging

    with caplog.at_level(logging.WARNING):
        FinancingModel().borrow_bps(stock(900.0), days=30, is_short=True)
    assert "costs.hard_to_borrow" in caplog.text


def test_negative_days_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FinancingModel().borrow_bps(stock(), days=-1, is_short=True)


# ----------------------------------------------------------------------------------
# Short rebate
# ----------------------------------------------------------------------------------
def test_no_short_rebate_is_assumed_by_default() -> None:
    """Most accounts earn none of the interest on short proceeds. Assuming
    otherwise manufactures a risk-free income stream that flatters every short
    book -- and the flattery is proportional to the rate environment, so it shows
    up exactly when rates are high and the strategy looks best."""
    assert FinancingModel().short_rebate_bps(days=365, base_rate_annual_bps=500) == 0.0


def test_configured_rebate_is_income_not_cost() -> None:
    model = FinancingModel(FinancingParams(short_rebate_fraction=0.8))
    rebate = model.short_rebate_bps(days=DAYS_PER_YEAR, base_rate_annual_bps=500)
    assert rebate == pytest.approx(-400.0), "income is a negative cost"


# ----------------------------------------------------------------------------------
# Margin financing
# ----------------------------------------------------------------------------------
def test_unlevered_book_pays_no_financing() -> None:
    model = FinancingModel()
    assert (
        model.financing_bps(
            gross_notional=1_000_000, equity=1_000_000, days=365, base_rate_annual_bps=400
        )
        == 0.0
    )


def test_leverage_is_charged_on_the_borrowed_portion_only() -> None:
    """Gross 3.0 finances two units of equity's worth, which is how leverage
    quietly consumes a carry trade's edge."""
    model = FinancingModel(FinancingParams(margin_spread_bps_annual=100.0))
    cost = model.financing_bps(
        gross_notional=3_000_000, equity=1_000_000, days=DAYS_PER_YEAR, base_rate_annual_bps=400
    )
    # 2m borrowed at 500 bp on 3m gross = 2/3 x 500.
    assert cost == pytest.approx(500.0 * 2 / 3)


def test_financing_scales_with_leverage() -> None:
    model = FinancingModel()
    costs = [
        model.financing_bps(gross_notional=g, equity=1e6, days=365, base_rate_annual_bps=400)
        for g in (1e6, 2e6, 4e6, 8e6)
    ]
    assert costs == sorted(costs)


def test_zero_equity_is_rejected() -> None:
    with pytest.raises(ValueError, match="equity must be positive"):
        FinancingModel().financing_bps(
            gross_notional=1e6, equity=0.0, days=1, base_rate_annual_bps=400
        )


# ----------------------------------------------------------------------------------
# Perpetual funding
# ----------------------------------------------------------------------------------
def test_longs_pay_positive_funding_and_shorts_receive_it() -> None:
    """Crypto carry *is* the decision to be on the receiving side, so the sign
    convention has to be exact."""
    model = FinancingModel()
    rates = np.array([0.0001, 0.0001, 0.0001])  # 1 bp per 8h funding
    assert model.funding_bps(rates, side=1) == pytest.approx(3.0)
    assert model.funding_bps(rates, side=-1) == pytest.approx(-3.0)


def test_negative_funding_pays_the_long() -> None:
    assert FinancingModel().funding_bps(np.array([-0.0002]), side=1) == pytest.approx(-2.0)


def test_missing_funding_observations_are_ignored_not_zeroed() -> None:
    model = FinancingModel()
    assert model.funding_bps(np.array([0.0001, np.nan, 0.0001]), side=1) == pytest.approx(2.0)
    assert model.funding_bps(np.array([]), side=1) == 0.0


def test_funding_side_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="side must be"):
        FinancingModel().funding_bps(np.array([0.0001]), side=0)


# ----------------------------------------------------------------------------------
# Futures roll
# ----------------------------------------------------------------------------------
def test_each_roll_is_two_crossings() -> None:
    """Out of the expiring contract and into the next. A quarterly roll is eight
    one-way crossings a year, which for trend following is often more than the
    signal's own turnover -- and is why paper trend-following beats the real thing."""
    cost = FinancingModel.roll_bps(
        half_spread_bps=1.0, impact_bps=2.0, rolls_per_year=4, days=DAYS_PER_YEAR
    )
    assert cost == pytest.approx(2 * 3.0 * 4)


def test_roll_cost_is_pro_rated_by_holding_period() -> None:
    annual = FinancingModel.roll_bps(
        half_spread_bps=1.0, impact_bps=1.0, rolls_per_year=4, days=DAYS_PER_YEAR
    )
    quarter = FinancingModel.roll_bps(
        half_spread_bps=1.0, impact_bps=1.0, rolls_per_year=4, days=DAYS_PER_YEAR / 4
    )
    assert quarter == pytest.approx(annual / 4)


def test_no_rolls_means_no_roll_cost() -> None:
    assert (
        FinancingModel.roll_bps(half_spread_bps=1.0, impact_bps=1.0, rolls_per_year=0, days=365)
        == 0.0
    )


# ----------------------------------------------------------------------------------
# Aggregate
# ----------------------------------------------------------------------------------
def test_holding_cost_aggregates_and_keeps_the_assumption_flag() -> None:
    model = FinancingModel()
    holding = model.holding_cost(
        stock(),
        days=30,
        is_short=True,
        gross_notional=2e6,
        equity=1e6,
        base_rate_annual_bps=400,
        funding_rates=None,
    )
    assert holding.assumed_borrow, "a strategy resting on an assumed borrow fee is untested"
    assert holding.borrow_bps > 0
    assert holding.financing_bps > 0
    assert holding.total_bps == pytest.approx(
        holding.borrow_bps + holding.financing_bps + holding.funding_bps
    )


def test_a_long_unlevered_position_costs_nothing_to_hold() -> None:
    holding = FinancingModel().holding_cost(
        stock(), days=30, is_short=False, gross_notional=1e6, equity=1e6
    )
    assert holding.total_bps == 0.0
