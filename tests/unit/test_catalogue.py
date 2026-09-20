"""The catalogue is documentation that the code depends on, so it is tested.

Its invariants are not stylistic. A source with no recorded caveats produces a
tearsheet with an empty data-quality panel, which reads as "this data is clean" --
the exact impression the platform exists to prevent.
"""

from __future__ import annotations

import pytest

from quantlab.data.catalogue import (
    SOURCES,
    AssetClass,
    PitQuality,
    get_source,
    sources_for,
)


def test_every_source_declares_at_least_one_caveat() -> None:
    missing = [key for key, spec in SOURCES.items() if not spec.caveats]
    assert not missing, (
        f"sources with no recorded caveats: {missing}. Free data is never clean; "
        "an empty caveat list renders as a clean bill of health in the tearsheet."
    )


def test_caveats_are_substantive() -> None:
    for key, spec in SOURCES.items():
        for caveat in spec.caveats:
            assert len(caveat) > 40, f"{key}: caveat too terse to be useful: {caveat!r}"


def test_source_keys_match_their_registry_entry() -> None:
    for key, spec in SOURCES.items():
        assert key == spec.key


def test_every_source_declares_an_asset_class_and_dataset() -> None:
    for key, spec in SOURCES.items():
        assert spec.asset_classes, f"{key} declares no asset class"
        assert spec.datasets, f"{key} declares no dataset"


def test_restated_sources_are_blocked_from_the_signal_path() -> None:
    restated = [s for s in SOURCES.values() if s.pit_quality is PitQuality.RESTATED]
    assert restated, "expected some sources to serve restated data"
    for spec in restated:
        assert not spec.usable_in_signal_path


def test_fred_is_restated_and_alfred_is_not() -> None:
    """The single most important distinction in the macro data layer.

    FRED serves the latest vintage; using it for a revisable series is look-ahead
    bias. ALFRED serves what was actually known on a date. Confusing the two is how
    a macro backtest quietly becomes fiction.
    """
    assert get_source("fred").pit_quality is PitQuality.RESTATED
    assert get_source("alfred").pit_quality is PitQuality.VINTAGE


def test_sec_edgar_is_point_in_time() -> None:
    """EDGAR's `filed` dates are what make free point-in-time fundamentals possible."""
    assert get_source("sec_edgar").pit_quality is PitQuality.VINTAGE


def test_equity_price_sources_are_flagged_survivorship_biased() -> None:
    for key in ("stooq", "yfinance"):
        assert get_source(key).pit_quality is PitQuality.SURVIVORSHIP_BIASED


def test_rate_limits_are_positive_and_conservative() -> None:
    for key, spec in SOURCES.items():
        assert spec.max_requests_per_second > 0, f"{key} has a non-positive rate limit"
        assert spec.max_requests_per_second <= 10, (
            f"{key} allows {spec.max_requests_per_second} req/s; no free source "
            "should be hit that hard"
        )


def test_free_tier_sources_are_marked_cross_check_only() -> None:
    """Free tiers capped at tens of calls per day cannot be primary ingest (spec 3.2)."""
    for key in ("alpha_vantage", "tiingo", "fmp", "nasdaq_data_link"):
        assert get_source(key).cross_check_only


def test_sources_for_asset_class() -> None:
    crypto = sources_for(AssetClass.CRYPTO)
    assert {s.key for s in crypto} >= {"binance", "bybit", "okx", "coingecko"}
    for spec in crypto:
        assert AssetClass.CRYPTO in spec.asset_classes


def test_unknown_source_raises_with_a_helpful_message() -> None:
    with pytest.raises(KeyError, match="known sources"):
        get_source("bloomberg")


def test_key_settings_exist_on_settings() -> None:
    """A source naming a key setting that Settings does not define would report as
    permanently unavailable with no way to fix it."""
    from quantlab.config import Settings

    fields = set(Settings.model_fields)
    for key, spec in SOURCES.items():
        if spec.key_setting is not None:
            assert spec.key_setting in fields, f"{key} names unknown setting {spec.key_setting}"
