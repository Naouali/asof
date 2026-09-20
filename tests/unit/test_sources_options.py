"""Option chain snapshots.

The dataset that cannot be backfilled, so the tests are about the things that
would quietly corrupt a history there is no second chance to collect: the OCC
symbol parse, the snapshot's own timestamp, and what is dropped on the way in.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_sources import fixture_source, load_json_fixture

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.options import CboeOptionChain, parse_occ_symbol

WINDOW = (dt.datetime(2026, 9, 1, tzinfo=dt.UTC), dt.datetime(2026, 9, 30, tzinfo=dt.UTC))


def fetch(settings: Settings, **kwargs: object) -> pl.DataFrame:
    payload = load_json_fixture("cboe_options_spx.json")
    with fixture_source(CboeOptionChain, payload, settings, **kwargs) as source:
        return source.fetch(["SPX"], *WINDOW)


# -------------------------------------------------------------- OCC symbols --
def test_an_occ_symbol_decodes_to_a_contract() -> None:
    """`SPX261016C00200000` is the SPX 200 call expiring 2026-10-16. The strike
    is in thousandths, so a missed division is a factor of a thousand."""
    parsed = parse_occ_symbol("SPX261016C00200000")

    assert parsed == ("SPX", dt.date(2026, 10, 16), "C", 200.0)


@pytest.mark.parametrize(
    ("symbol", "strike"),
    [("SPX261016C00200000", 200.0), ("SPY260320P00450500", 450.5), ("A261016C99999000", 99999.0)],
)
def test_strikes_are_thousandths(symbol: str, strike: float) -> None:
    parsed = parse_occ_symbol(symbol)
    assert parsed is not None
    assert parsed[3] == strike


def test_a_malformed_symbol_returns_none_rather_than_raising() -> None:
    """CBOE lists the occasional non-standard series, and one of them must not
    abort a snapshot that cannot be taken again. The count is logged instead."""
    assert parse_occ_symbol("NOT-AN-OCC-SYMBOL") is None
    assert parse_occ_symbol("SPX261301C00200000") is None  # month 13
    assert parse_occ_symbol("") is None


# ------------------------------------------------------------------ the snapshot --
def test_a_snapshot_parses(settings: Settings) -> None:
    frame = fetch(settings)

    assert frame.height > 0
    assert frame["symbol"].unique().to_list() == ["SPX"]
    assert set(frame["right"].unique()) <= {"C", "P"}
    assert frame["strike"].min() > 0
    assert frame["underlying_price"].null_count() == 0


def test_the_quote_time_is_used_not_the_collection_time(settings: Settings) -> None:
    """The file is delayed. Dating a 16:15 close as though it were observed when
    the collector happened to run would misstate by hours what a snapshot is."""
    frame = fetch(settings)

    assert frame["as_of"].n_unique() == 1
    assert (frame["known_at"] > frame["as_of"]).all()


def test_strikes_nobody_holds_are_dropped_by_default(settings: Settings) -> None:
    """They are the bulk of a chain and its least trustworthy part: wide, stale,
    and worth storing for ever only if someone asked."""
    kept = fetch(settings)
    everything = fetch(settings, require_open_interest=False)

    assert everything.height > kept.height
    assert kept["open_interest"].min() > 0


def test_a_malformed_symbol_does_not_abort_the_snapshot(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        frame = fetch(settings)

    assert frame.height > 0
    assert "unparsed_symbols" in caplog.text


def test_greeks_are_carried_but_are_cboes(settings: Settings) -> None:
    """Stored because discarding a free field is worse than storing a documented
    one -- and documented as CBOE's, computed with a model, dividend assumption
    and rate curve that are not published alongside the numbers."""
    frame = fetch(settings)

    for greek in ("delta", "gamma", "vega", "theta", "implied_vol"):
        assert greek in frame.columns
    assert frame["delta"].drop_nulls().len() > 0


def test_a_payload_without_an_options_block_fails_loudly(settings: Settings) -> None:
    with (
        fixture_source(CboeOptionChain, {"data": {"symbol": "^SPX"}}, settings) as source,
        pytest.raises(SourceError, match="leading underscore"),
    ):
        source.fetch(["SPX"], *WINDOW)


def test_an_empty_chain_is_a_lost_day_not_a_retryable_query(settings: Settings) -> None:
    """The message has to say so: an empty options fetch is not like an empty
    price fetch, because there is no wider window that would recover it."""
    with (
        fixture_source(CboeOptionChain, {"data": {"options": []}}, settings) as source,
        pytest.raises(SourceError, match="cannot be backfilled"),
    ):
        source.fetch(["SPX"], *WINDOW)


def test_index_tickers_are_requested_with_an_underscore(settings: Settings) -> None:
    """CBOE serves index chains at _SPX and single names at their plain ticker.
    Getting it backwards returns nothing for either."""
    from quantlab.data.sources.options import INDEX_TICKERS

    assert "SPX" in INDEX_TICKERS
    assert "AAPL" not in INDEX_TICKERS


def test_the_dataset_records_that_it_cannot_be_backfilled() -> None:
    from quantlab.data.schemas import get_schema

    description = get_schema("chain_snapshot").description
    assert "CANNOT be backfilled" in description
    assert "never be recovered" in description
