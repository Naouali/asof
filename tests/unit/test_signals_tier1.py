"""The Tier 1 signals themselves: arithmetic, sign conventions and failure paths."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.signals.base import SignalUnavailableError
from quantlab.signals.crypto.carry import PerpetualFundingCarry
from quantlab.signals.equity.profitability import (
    Fundamentals,
    ProfitabilityMeasure,
    profitability,
)
from quantlab.signals.futures.basis import annualised_basis
from quantlab.signals.futures.trend import TimeSeriesMomentum
from quantlab.signals.fx.carry import forward_implied_carry
from quantlab.signals.rates.carry import RatesCarry, carry_and_rolldown

START = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)


# ==================================================================================
# Time-series momentum
# ==================================================================================
def trending(slope: float, n: int = 400) -> np.ndarray:
    return 100 * np.exp(np.cumsum(np.full(n, slope)))


def test_a_steady_uptrend_scores_maximally_long() -> None:
    assert TimeSeriesMomentum().score_series(trending(0.001)) == pytest.approx(2.0)


def test_a_steady_downtrend_scores_maximally_short() -> None:
    assert TimeSeriesMomentum().score_series(trending(-0.001)) == pytest.approx(-2.0)


def test_the_score_is_signed_the_same_way_as_the_trend() -> None:
    signal = TimeSeriesMomentum()
    rng = np.random.default_rng(0)
    for seed_slope in (0.0005, -0.0005):
        prices = 100 * np.exp(np.cumsum(rng.normal(seed_slope, 0.008, 500)))
        score = signal.score_series(prices)
        assert score is not None
        assert np.sign(score) == np.sign(seed_slope)


def test_scores_are_clipped() -> None:
    """A trend three sigma strong is not three times as reliable as one sigma, and
    without the clip a single dislocated instrument dominates the whole book."""
    signal = TimeSeriesMomentum(clip=1.5)
    assert abs(signal.score_series(trending(0.01))) <= 1.5


def test_volatility_scaling_makes_instruments_comparable() -> None:
    """A quiet instrument's 5% move should count for more than a violent one's.
    Without this the signal is dominated by whatever is most volatile rather than
    by whatever is most trending."""
    rng = np.random.default_rng(1)
    signal = TimeSeriesMomentum(clip=100.0)
    quiet = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.002, 500)))
    loud = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.020, 500)))
    assert signal.score_series(quiet) > signal.score_series(loud)


def test_a_flat_series_has_no_trend_to_measure() -> None:
    assert TimeSeriesMomentum().score_series(np.full(400, 100.0)) is None


def test_too_little_history_returns_no_score_rather_than_a_guess() -> None:
    signal = TimeSeriesMomentum()
    assert signal.score_series(trending(0.001, n=signal.warmup_days - 1)) is None


def test_blending_lookbacks_is_steadier_than_any_single_one() -> None:
    """Picking one lookback is unstable, and it is the first step of a parameter
    search nobody counts."""
    rng = np.random.default_rng(2)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 600)))
    blended = TimeSeriesMomentum(clip=100.0)
    singles = [
        TimeSeriesMomentum(lookbacks=(lb,), clip=100.0).score_series(prices) for lb in (21, 63, 252)
    ]
    score = blended.score_series(prices)
    assert score is not None
    assert min(singles) <= score <= max(singles)


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one lookback"):
        TimeSeriesMomentum(lookbacks=())
    with pytest.raises(ValueError, match="at least two bars"):
        TimeSeriesMomentum(lookbacks=(1,))


# ==================================================================================
# Crypto funding carry
# ==================================================================================
def test_positive_funding_makes_a_long_expensive() -> None:
    """The sign convention that matters. Getting it backwards produces a strategy
    that systematically takes the losing side of every carry, and backtests as a
    consistent, confident, slow loss."""
    signal = PerpetualFundingCarry()
    score = signal.score_series(np.full(90, 0.0001), np.full(90, 8.0))
    assert score is not None and score < 0


def test_negative_funding_pays_the_long() -> None:
    signal = PerpetualFundingCarry()
    score = signal.score_series(np.full(90, -0.0001), np.full(90, 8.0))
    assert score is not None and score > 0


def test_carry_is_annualised_from_the_observed_interval() -> None:
    """1 bp every 8 hours is three payments a day: 0.0001 x 3 x 365 = 10.95%."""
    signal = PerpetualFundingCarry()
    score = signal.score_series(np.full(90, 0.0001), np.full(90, 8.0))
    assert score == pytest.approx(-0.0001 * 3 * 365, rel=1e-6)


def test_a_shorter_interval_doubles_the_annualised_carry() -> None:
    """The interval is measured, not assumed: it has changed over time and differs
    by contract, and annualising an eight-hour rate as daily overstates it."""
    signal = PerpetualFundingCarry()
    eight = signal.score_series(np.full(90, 0.0001), np.full(90, 8.0))
    four = signal.score_series(np.full(90, 0.0001), np.full(90, 4.0))
    assert four == pytest.approx(2 * eight, rel=1e-9)


def test_extreme_funding_is_clipped() -> None:
    """200% annualised is a dislocation, not a carry opportunity -- the market
    paying almost anything to stay positioned is the moment before it stops."""
    signal = PerpetualFundingCarry(clip_annual=2.0)
    assert signal.score_series(np.full(90, 0.05), np.full(90, 8.0)) == pytest.approx(-2.0)


def test_unusable_observations_yield_no_score() -> None:
    signal = PerpetualFundingCarry()
    assert signal.score_series(np.array([np.nan]), np.array([8.0])) is None
    assert signal.score_series(np.array([0.0001]), np.array([0.0])) is None


# ==================================================================================
# Rates carry
# ==================================================================================
def test_carry_and_rolldown_on_an_upward_sloping_curve() -> None:
    """10y at 4.5%, 2y at 4.0%: you earn the 4.5% yield plus the roll down a
    positively sloped curve."""
    value = carry_and_rolldown(0.045, 0.040, 10.0, shorter_years=2.0)
    assert value == pytest.approx(0.045 + 9.0 * 0.005)
    assert value > 0.045, "roll-down adds to carry when the curve slopes up"


def test_an_inverted_curve_makes_roll_down_negative() -> None:
    value = carry_and_rolldown(0.040, 0.045, 10.0, shorter_years=2.0)
    assert value < 0.040


def test_percentages_passed_as_decimals_are_caught() -> None:
    """A hundredfold error here reads as an extraordinary opportunity rather than
    as a unit mistake."""
    with pytest.raises(ValueError, match="look like percentages"):
        carry_and_rolldown(4.5, 4.0, 10.0, shorter_years=2.0)


def test_rolling_down_needs_a_shorter_tenor() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        carry_and_rolldown(0.045, 0.040, 2.0, shorter_years=10.0)


def test_an_unknown_tenor_is_rejected() -> None:
    with pytest.raises(ValueError, match="no FRED series"):
        RatesCarry(tenor_years=7.0)


# ==================================================================================
# FX and commodity carry
# ==================================================================================
def test_fx_carry_is_the_rate_differential() -> None:
    assert forward_implied_carry(0.08, 0.05) == pytest.approx(0.03)
    assert forward_implied_carry(0.02, 0.05) == pytest.approx(-0.03)


def test_fx_rates_must_be_decimals_too() -> None:
    with pytest.raises(ValueError, match="percentages"):
        forward_implied_carry(8.0, 5.0)


def test_backwardation_is_positive_roll_yield() -> None:
    """The near contract dearer than the far one: a long rolls into something
    cheaper and earns the difference."""
    assert annualised_basis(102.0, 100.0, 3.0) > 0
    assert annualised_basis(98.0, 100.0, 3.0) < 0


def test_basis_is_annualised_by_the_gap_between_contracts() -> None:
    quarterly = annualised_basis(102.0, 100.0, 3.0)
    semi_annual = annualised_basis(102.0, 100.0, 6.0)
    assert quarterly == pytest.approx(2 * semi_annual)


def test_basis_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="prices must be positive"):
        annualised_basis(0.0, 100.0, 3.0)
    with pytest.raises(ValueError, match="months_between"):
        annualised_basis(102.0, 100.0, 0.0)


# ==================================================================================
# Equity profitability
# ==================================================================================
FULL = Fundamentals(
    revenue=1000.0,
    cost_of_goods_sold=600.0,
    total_assets=2000.0,
    sg_and_a=150.0,
    interest_expense=20.0,
    book_equity=800.0,
    accruals=30.0,
)


def test_gross_profitability_matches_its_definition() -> None:
    assert profitability(FULL, ProfitabilityMeasure.GROSS) == pytest.approx((1000 - 600) / 2000)


def test_operating_profitability_matches_its_definition() -> None:
    assert profitability(FULL, ProfitabilityMeasure.OPERATING) == pytest.approx(
        (1000 - 600 - 150 - 20) / 800
    )


def test_cash_based_operating_profitability_removes_accruals() -> None:
    """The strongest of the three, and the reason is instructive: accruals are the
    part of earnings management controls."""
    assert profitability(FULL, ProfitabilityMeasure.CASH_BASED_OPERATING) == pytest.approx(
        (1000 - 600 - 150 - 20 - 30) / 800
    )


def test_a_missing_line_item_returns_none_rather_than_zero() -> None:
    """Treating a missing SG&A as zero makes a company look more profitable than
    it is -- and the filers with missing items are systematically the small, badly
    reported ones, so the error is a size tilt, not noise."""
    incomplete = Fundamentals(revenue=1000.0, cost_of_goods_sold=600.0, total_assets=2000.0)
    assert profitability(incomplete, ProfitabilityMeasure.OPERATING) is None
    assert profitability(incomplete, ProfitabilityMeasure.GROSS) is not None


def test_zero_is_a_real_value_and_missing_is_not() -> None:
    """A company really can report no interest expense."""
    from dataclasses import replace

    zero_interest = replace(FULL, interest_expense=0.0)
    assert profitability(zero_interest, ProfitabilityMeasure.OPERATING) == pytest.approx(
        (1000 - 600 - 150) / 800
    )


def test_a_company_with_no_assets_or_equity_yields_nothing() -> None:
    from dataclasses import replace

    assert profitability(replace(FULL, total_assets=0.0), ProfitabilityMeasure.GROSS) is None
    assert profitability(replace(FULL, book_equity=-100.0), ProfitabilityMeasure.OPERATING) is None


def test_nan_is_treated_as_missing() -> None:
    from dataclasses import replace

    assert (
        profitability(replace(FULL, sg_and_a=float("nan")), ProfitabilityMeasure.OPERATING) is None
    )


# ==================================================================================
# Signals that cannot run refuse to, loudly
# ==================================================================================
def test_profitability_refuses_without_fundamentals(store) -> None:
    from quantlab.signals.equity.profitability import EquityProfitability

    snapshot = store.as_of("2024-01-05")
    with pytest.raises(SignalUnavailableError, match="Milestone 9"):
        EquityProfitability().compute(snapshot, ["AAPL"])


def test_commodity_basis_explains_why_free_data_cannot_support_it(store) -> None:
    from quantlab.signals.futures.basis import CommodityBasis

    with pytest.raises(SignalUnavailableError, match=r"no second contract|section 3\.4"):
        CommodityBasis().compute(store.as_of("2024-01-05"), [])


def test_fx_carry_refuses_a_universe_of_two(store) -> None:
    from quantlab.signals.fx.carry import FxCarry

    with pytest.raises(SignalUnavailableError, match="universe of two"):
        FxCarry().compute(store.as_of("2024-01-05"), [])


def test_rates_carry_names_the_free_key_it_needs(store) -> None:
    with pytest.raises(SignalUnavailableError, match="QUANTLAB_FRED_API_KEY"):
        RatesCarry().compute(store.as_of("2024-01-05"), [])


def test_trend_refuses_an_empty_snapshot(store) -> None:
    """An empty score frame would look like a signal with no view rather than like
    missing data."""
    with pytest.raises(SignalUnavailableError, match="no daily bars"):
        TimeSeriesMomentum().compute(store.as_of("2024-01-05"), ["AAPL"])


def test_trend_produces_scores_from_a_real_snapshot(populated_store) -> None:
    """The lake fixture holds three sessions, far inside the warm-up, so the signal
    correctly produces nothing -- and says so rather than raising."""
    snapshot = populated_store.as_of("2024-01-05")
    scores = TimeSeriesMomentum().compute(snapshot, ["AAPL", "MSFT"])
    assert scores.height == 0
    assert list(scores.columns) == ["symbol", "as_of", "score"]


# ==================================================================================
# The compute paths, exercised against a synthetic lake
# ==================================================================================
def write_rows(store, rows: list[dict], asset_class: str) -> None:
    store.write(pl.DataFrame(rows), asset_class=asset_class)


def test_crypto_carry_ranks_a_real_cross_section(store) -> None:
    """Cheap funding scores above expensive funding, and the scores are annualised
    from the observed interval."""
    rows = []
    for symbol, rate in (("CHEAP", -0.0002), ("MID", 0.0000), ("DEAR", 0.0004)):
        for hour in range(0, 24 * 20, 8):
            stamp = START + dt.timedelta(hours=hour)
            rows.append(
                {
                    "source": "binance",
                    "dataset": "funding_rate",
                    "symbol": symbol,
                    "as_of": stamp,
                    "known_at": stamp,
                    "ingested_at": stamp,
                    "funding_rate": rate,
                    "mark_price": 100.0,
                    "interval_hours": 8.0,
                }
            )
    write_rows(store, rows, "crypto")

    snapshot = store.as_of(START + dt.timedelta(days=20))
    scores = PerpetualFundingCarry(lookback_days=30).compute(snapshot, ["CHEAP", "MID", "DEAR"])
    by_symbol = dict(zip(scores["symbol"], scores["score"], strict=True))
    assert by_symbol["CHEAP"] > by_symbol["MID"] > by_symbol["DEAR"]
    assert by_symbol["DEAR"] == pytest.approx(-0.0004 * 3 * 365, rel=1e-6)


def test_rates_carry_computes_from_a_real_curve(store) -> None:
    rows = []
    for series, level in (("DGS2", 4.0), ("DGS10", 4.5)):
        for day in range(10):
            stamp = START + dt.timedelta(days=day)
            rows.append(
                {
                    "source": "fred",
                    "dataset": "series_observations",
                    "symbol": series,
                    "as_of": stamp,
                    "known_at": stamp,
                    "ingested_at": stamp,
                    "value": level,
                    "units": "percent",
                    "vintage": False,
                }
            )
    write_rows(store, rows, "macro")

    scores = RatesCarry().compute(store.as_of(START + dt.timedelta(days=10)), [])
    assert scores.height == 1
    # FRED publishes percentages; the signal converts before computing.
    assert scores["score"].item() == pytest.approx(0.045 + 9.0 * 0.005)


def test_fx_carry_ranks_currencies_by_rate_differential(store) -> None:
    from quantlab.signals.fx.carry import FxCarry

    rows = []
    for series, level in (("DFF", 5.0), ("HIGH", 8.0), ("LOW", 1.0)):
        stamp = START
        rows.append(
            {
                "source": "fred",
                "dataset": "series_observations",
                "symbol": series,
                "as_of": stamp,
                "known_at": stamp,
                "ingested_at": stamp,
                "value": level,
                "units": "percent",
                "vintage": False,
            }
        )
    write_rows(store, rows, "macro")

    signal = FxCarry(rate_series={"AUD": "HIGH", "JPY": "LOW"})
    scores = signal.compute(store.as_of(START + dt.timedelta(days=1)), [])
    by_symbol = dict(zip(scores["symbol"], scores["score"], strict=True))
    assert by_symbol["AUD"] == pytest.approx(0.03)
    assert by_symbol["JPY"] == pytest.approx(-0.04)


def test_commodity_basis_computes_when_a_curve_is_configured(store) -> None:
    """The arithmetic is fine; it is the second contract that free data lacks."""
    from quantlab.signals.futures.basis import CommodityBasis

    rows = []
    for symbol, price in (("CL1", 102.0), ("CL2", 100.0)):
        stamp = dt.datetime(2020, 1, 3, 21, tzinfo=dt.UTC)
        rows.append(
            {
                "source": "yahoo",
                "dataset": "ohlcv_daily",
                "symbol": symbol,
                "as_of": stamp,
                "known_at": stamp,
                "ingested_at": stamp,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1e6,
                "adj_close": price,
                "currency": "USD",
                "venue": "CMES",
            }
        )
    write_rows(store, rows, "commodity")

    signal = CommodityBasis(curve_symbols={"WTI": ("CL1", "CL2", 3.0)})
    scores = signal.compute(store.as_of("2020-01-06"), [])
    assert scores["score"].item() == pytest.approx(annualised_basis(102.0, 100.0, 3.0))


def test_profitability_scores_a_real_cross_section(store) -> None:
    """Two filers with the same revenue but different asset bases rank correctly."""
    rows = []
    fundamentals = {
        "LEAN": {"revenue": 1000.0, "cost_of_goods_sold": 600.0, "total_assets": 1000.0},
        "HEAVY": {"revenue": 1000.0, "cost_of_goods_sold": 600.0, "total_assets": 4000.0},
    }
    filed = dt.datetime(2020, 3, 1, tzinfo=dt.UTC)
    for symbol, metrics in fundamentals.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "source": "sec_edgar",
                    "dataset": "fundamentals",
                    "symbol": symbol,
                    "as_of": dt.datetime(2019, 12, 31, tzinfo=dt.UTC),
                    "known_at": filed,
                    "ingested_at": filed,
                    "metric": metric,
                    "value": value,
                    "unit": "USD",
                    "fiscal_period": "FY2019",
                    "form": "10-K",
                }
            )
    write_rows(store, rows, "equity")

    from quantlab.signals.equity.profitability import EquityProfitability

    scores = EquityProfitability().compute(store.as_of("2020-06-01"), ["LEAN", "HEAVY"])
    by_symbol = dict(zip(scores["symbol"], scores["score"], strict=True))
    assert by_symbol["LEAN"] == pytest.approx(0.4)
    assert by_symbol["HEAVY"] == pytest.approx(0.1)


def test_profitability_respects_the_filed_date(store) -> None:
    """The whole value of EDGAR over every restated free source: a figure filed in
    March is not knowable in January."""
    filed = dt.datetime(2020, 3, 1, tzinfo=dt.UTC)
    rows = [
        {
            "source": "sec_edgar",
            "dataset": "fundamentals",
            "symbol": "ACME",
            "as_of": dt.datetime(2019, 12, 31, tzinfo=dt.UTC),
            "known_at": filed,
            "ingested_at": filed,
            "metric": metric,
            "value": value,
            "unit": "USD",
            "fiscal_period": "FY2019",
            "form": "10-K",
        }
        for metric, value in (
            ("revenue", 1000.0),
            ("cost_of_goods_sold", 600.0),
            ("total_assets", 2000.0),
        )
    ]
    write_rows(store, rows, "equity")

    from quantlab.signals.equity.profitability import EquityProfitability

    signal = EquityProfitability()
    with pytest.raises(SignalUnavailableError):
        signal.compute(store.as_of("2020-01-15"), ["ACME"])
    assert signal.compute(store.as_of("2020-06-01"), ["ACME"]).height == 1


def test_trend_scores_a_real_price_history(store) -> None:
    rows = []
    price = 100.0
    for day in range(400):
        stamp = dt.datetime(2020, 1, 1, 21, tzinfo=dt.UTC) + dt.timedelta(days=day)
        price *= float(np.exp(0.001))
        rows.append(
            {
                "source": "yahoo",
                "dataset": "ohlcv_daily",
                "symbol": "UP",
                "as_of": stamp,
                "known_at": stamp,
                "ingested_at": stamp,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1e6,
                "adj_close": price,
                "currency": "USD",
                "venue": "XNAS",
            }
        )
    write_rows(store, rows, "equity")

    scores = TimeSeriesMomentum().compute(store.as_of("2021-03-01"), ["UP"])
    assert scores.height == 1
    assert scores["score"].item() > 1.0
