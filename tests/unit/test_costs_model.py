"""The composite cost model, and the prohibition on flat basis-point defaults."""

from __future__ import annotations

import polars as pl
import pytest

from quantlab.costs.base import CostBreakdown, FlatBpsCostModel, Instrument, Order
from quantlab.costs.model import ASSET_CLASS_DEFAULTS, TransactionCostModel
from quantlab.data.catalogue import AssetClass


def equity(**overrides: object) -> Instrument:
    base = {
        "symbol": "AAPL",
        "asset_class": AssetClass.EQUITY,
        "adv_notional": 8e9,
        "volatility_daily": 0.018,
        "spread_bps": 1.2,
    }
    return Instrument(**{**base, **overrides})  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# The prohibition
# ----------------------------------------------------------------------------------
def test_flat_costs_cannot_be_constructed_by_accident() -> None:
    """Spec section 13. Flat costs are roughly right for research-sized trades and
    wildly optimistic at deployment size, so they make every strategy look
    scalable and every capacity estimate infinite."""
    with pytest.raises(ValueError, match="prohibited as a default"):
        FlatBpsCostModel(10.0)


def test_flat_costs_require_explicit_acknowledgement(caplog: pytest.LogCaptureFixture) -> None:
    """They exist only to show, on a tearsheet, how much they understate."""
    import logging

    with caplog.at_level(logging.WARNING):
        model = FlatBpsCostModel(10.0, acknowledge_unrealistic=True)
    assert "costs.flat_model_constructed" in caplog.text
    small = model.estimate(Order("AAPL", 1e5), equity()).total_bps
    enormous = model.estimate(Order("AAPL", 1e10), equity()).total_bps
    assert small == enormous, "which is exactly the problem"


def test_flat_model_understates_real_size_dramatically() -> None:
    """The comparison the flat model exists to make."""
    flat = FlatBpsCostModel(10.0, acknowledge_unrealistic=True)
    real = TransactionCostModel.for_asset_class(AssetClass.EQUITY)
    order = Order("AAPL", 2e9)  # 25% of ADV
    assert real.estimate(order, equity()).total_bps > 3 * flat.estimate(order, equity()).total_bps


def test_the_default_model_is_size_aware() -> None:
    model = TransactionCostModel()
    small = model.estimate(Order("AAPL", 1e5), equity()).total_bps
    large = model.estimate(Order("AAPL", 1e9), equity()).total_bps
    assert large > 5 * small


# ----------------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------------
def test_breakdown_components_sum_to_the_total() -> None:
    breakdown = TransactionCostModel().estimate(Order("AAPL", 1e8), equity())
    assert breakdown.total_bps == pytest.approx(
        breakdown.spread_bps
        + breakdown.impact_bps
        + breakdown.commission_bps
        + breakdown.financing_bps
        + breakdown.borrow_bps
    )


def test_spread_component_is_half_the_quoted_spread() -> None:
    """Crossing costs half the spread, not all of it. Charging the full spread
    doubles the cost of every trade in a high-turnover strategy."""
    breakdown = TransactionCostModel().estimate(Order("AAPL", 1e6), equity(spread_bps=4.0))
    assert breakdown.spread_bps == pytest.approx(2.0)


def test_tiny_orders_cost_about_the_half_spread_plus_commission() -> None:
    """The floor: impact vanishes as size does, but crossing still costs."""
    model = TransactionCostModel()
    breakdown = model.estimate(Order("AAPL", 1.0), equity(spread_bps=4.0))
    assert breakdown.impact_bps < 0.01
    assert breakdown.total_bps == pytest.approx(2.0 + breakdown.commission_bps, abs=0.01)


def test_round_trip_is_twice_one_way() -> None:
    """Turnover is quoted one-way and the cost of a position is two-way;
    conflating them halves every cost estimate in a strategy summary."""
    model = TransactionCostModel()
    order, instrument = Order("AAPL", 1e7), equity()
    assert model.round_trip_bps(order, instrument) == pytest.approx(
        2 * model.estimate(order, instrument).total_bps
    )


def test_currency_conversion() -> None:
    breakdown = CostBreakdown(spread_bps=10.0)
    assert breakdown.currency(1e6) == pytest.approx(1000.0)


def test_breakdowns_add_and_scale() -> None:
    a = CostBreakdown(spread_bps=2.0, temporary_impact_bps=4.0)
    b = CostBreakdown(spread_bps=1.0, borrow_bps=3.0)
    assert (a + b).spread_bps == 3.0
    assert (a + b).borrow_bps == 3.0
    assert a.scaled(0.5).temporary_impact_bps == 2.0


def test_basket_cost_is_notional_weighted() -> None:
    """An equal-weighted average of per-order bps would let a tiny cheap order
    offset a large expensive one."""
    model = TransactionCostModel()
    instruments = {"AAPL": equity(), "TINY": equity(symbol="TINY", adv_notional=1e7)}
    orders = [Order("AAPL", 1e6), Order("TINY", 9e6)]
    aggregate = model.estimate_many(orders, instruments)
    individually = [model.estimate(o, instruments[o.symbol]).total_bps for o in orders]
    assert min(individually) < aggregate.total_bps < max(individually)
    assert aggregate.total_bps > sum(individually) / 2, "weighted toward the big, costly order"


def test_empty_basket_costs_nothing() -> None:
    assert TransactionCostModel().estimate_many([], {}).total_bps == 0.0


# ----------------------------------------------------------------------------------
# Asset classes
# ----------------------------------------------------------------------------------
def test_every_asset_class_default_explains_itself() -> None:
    for asset_class, defaults in ASSET_CLASS_DEFAULTS.items():
        assert len(defaults.rationale) > 60, f"{asset_class} default has no rationale"


def test_credit_is_the_most_punitive_calibration() -> None:
    """Corporate bonds trade by appointment, and reported prices are often stale
    or matrix-derived."""
    ys = {k: v.impact.y for k, v in ASSET_CLASS_DEFAULTS.items()}
    assert ys[AssetClass.CREDIT] == max(ys.values())
    assert ys[AssetClass.FX] == min(ys.values())


def test_unknown_asset_class_falls_back_to_the_harshest_not_the_gentlest() -> None:
    """An unclassified instrument is not evidence of a liquid one."""
    model = TransactionCostModel()
    exotic = equity(symbol="???", asset_class=AssetClass.OPTIONS)
    credit = equity(symbol="BOND", asset_class=AssetClass.CREDIT)
    order = Order("X", 1e8)
    assert model.estimate(order, exotic).impact_bps == pytest.approx(
        model.estimate(order, credit).impact_bps
    )


def test_crypto_commission_dominates_small_trades() -> None:
    """Taker fees around 4 bp are why crypto strategies die of turnover rather
    than of impact."""
    model = TransactionCostModel()
    crypto = Instrument("BTCUSDT", AssetClass.CRYPTO, 2e9, 0.03, spread_bps=1.0)
    breakdown = model.estimate(Order("BTCUSDT", 1e5), crypto)
    assert breakdown.commission_bps > breakdown.impact_bps + breakdown.spread_bps


def test_instrument_commission_overrides_the_asset_class_default() -> None:
    model = TransactionCostModel()
    assert model.estimate(Order("AAPL", 1e6), equity(commission_bps=7.0)).commission_bps == 7.0


# ----------------------------------------------------------------------------------
# Scalar and vectorised agreement
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("notional", [1e4, 1e6, 1e8, 1e9])
def test_vectorised_matches_scalar(notional: float) -> None:
    """A backtest that costs trades differently from the tests is untested."""
    model = TransactionCostModel(use_asset_class_defaults=False)
    instrument = equity()
    scalar = model.estimate(Order("AAPL", notional), instrument)

    frame = pl.DataFrame(
        {
            "notional": [notional],
            "adv": [instrument.adv_notional],
            "vol": [instrument.volatility_daily],
            "spread": [instrument.spread_bps],
        }
    ).with_columns(
        cost=model.cost_bps_expr(pl.col("notional"), pl.col("adv"), pl.col("vol"), pl.col("spread"))
    )
    # The scalar path adds the asset-class commission; supply it explicitly here.
    assert frame["cost"].item() + scalar.commission_bps == pytest.approx(scalar.total_bps)


def test_vectorised_currency_cost_grows_super_linearly() -> None:
    model = TransactionCostModel(use_asset_class_defaults=False)
    frame = pl.DataFrame(
        {"notional": [1e6, 4e6], "adv": [1e9, 1e9], "vol": [0.02, 0.02], "spread": [0.0, 0.0]}
    ).with_columns(
        cost=model.cost_currency_expr(
            pl.col("notional"), pl.col("adv"), pl.col("vol"), pl.col("spread")
        )
    )
    costs = frame["cost"].to_list()
    assert costs[1] / costs[0] == pytest.approx(4**1.5)


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------
def test_instrument_with_no_volume_is_rejected() -> None:
    with pytest.raises(ValueError, match="no measurable capacity"):
        equity(adv_notional=0.0)


def test_signed_notional_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute traded value"):
        Order("AAPL", -1e6)


def test_side_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="side must be"):
        Order("AAPL", 1e6, side=0)


def test_extrapolation_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        breakdown = TransactionCostModel().estimate(Order("AAPL", 4e9), equity())
    assert "costs.impact_extrapolated" in caplog.text
    assert breakdown.detail["extrapolating"] is True


def test_estimating_a_crypto_spread_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """Crypto books are free. Estimating there is a choice, and a 60 bp error on a
    weekly-turnover strategy is tens of percent a year of imaginary cost."""
    import logging

    from quantlab.costs.base import Instrument as Inst

    crypto = Inst(
        "BTCUSDT", AssetClass.CRYPTO, 2e9, 0.03, spread_bps=60.0, spread_source="estimated"
    )
    with caplog.at_level(logging.WARNING):
        TransactionCostModel().estimate(Order("BTCUSDT", 1e6), crypto)
    assert "costs.estimated_crypto_spread" in caplog.text
    assert "observe the spread" in caplog.text


def test_observed_crypto_spread_is_not_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from quantlab.costs.base import Instrument as Inst

    crypto = Inst("BTCUSDT", AssetClass.CRYPTO, 2e9, 0.03, spread_bps=0.1, spread_source="observed")
    with caplog.at_level(logging.WARNING):
        TransactionCostModel().estimate(Order("BTCUSDT", 1e6), crypto)
    assert "costs.estimated_crypto_spread" not in caplog.text


def test_spread_provenance_reaches_the_breakdown() -> None:
    """So a tearsheet can say how much of a cost is measurement and how much is
    assumption."""
    breakdown = TransactionCostModel().estimate(Order("AAPL", 1e6), equity())
    assert breakdown.detail["spread_source"] == "assumed"
