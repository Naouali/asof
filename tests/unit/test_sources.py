"""Source parsers, tested against recorded payloads so the suite runs offline.

These assert the two things a parser can silently get wrong: the timestamp it
assigns to an observation, and whether that observation was knowable then.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json

import httpx
import polars as pl
import pytest
from tests.conftest import load_fixture, load_json_fixture

from quantlab.config import Settings
from quantlab.data.http import HttpClient, SourceError, SourceUnavailableError
from quantlab.data.sources.base import SOURCE_REGISTRY, Source, get_fetcher
from quantlab.data.sources.binance import BinanceFunding, BinanceInstruments, BinanceKlines
from quantlab.data.sources.fred import SERIES_POLICY, AlfredVintageSeries, FredSeries
from quantlab.data.sources.stooq import StooqDailyBars, parse_stooq_csv
from quantlab.data.sources.yahoo import YahooCorporateActions, YahooDailyBars

START = dt.datetime(2014, 5, 1, tzinfo=dt.UTC)
END = dt.datetime(2014, 6, 30, tzinfo=dt.UTC)


def fixture_source(
    cls: type[Source], payload: object, settings: Settings, **kwargs: object
) -> Source:
    """Build a fetcher whose transport always replies with ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(payload, str):
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json=payload)

    http = HttpClient(
        cls.spec, settings=settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return cls(client=http, settings=settings, **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------- registry --
def test_every_fetcher_is_registered_under_source_dot_dataset() -> None:
    """The registry name must match the catalogue key, because the lake partitions
    by the catalogue key: a mismatch files data under a source nobody can find."""
    for name, cls in SOURCE_REGISTRY.items():
        assert name == f"{cls.spec.key}.{cls.dataset}"


def test_unknown_fetcher_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known fetchers"):
        get_fetcher("bloomberg.everything")


# ------------------------------------------------------------------------- yahoo --
def test_yahoo_daily_bars_parse(settings: Settings) -> None:
    payload = load_json_fixture("yahoo_chart_aapl_2014.json")
    with fixture_source(YahooDailyBars, payload, settings) as source:
        frame = source.fetch(["AAPL"], START, END)

    assert frame.height > 30
    assert frame["venue"].unique().to_list() == ["XNAS"]
    assert frame["currency"].unique().to_list() == ["USD"]


def test_yahoo_as_of_is_the_session_close_not_the_session_open(settings: Settings) -> None:
    """Yahoo's daily timestamp is the session OPEN. Storing it unchanged would
    make every bar knowable six and a half hours before it existed."""
    payload = load_json_fixture("yahoo_chart_aapl_2014.json")
    with fixture_source(YahooDailyBars, payload, settings) as source:
        frame = source.fetch(["AAPL"], START, END)

    first = frame.sort("as_of")["as_of"][0]
    assert first.hour == 20, "May is EDT, so the NASDAQ close is 20:00 UTC"


def test_yahoo_known_at_equals_as_of(settings: Settings) -> None:
    payload = load_json_fixture("yahoo_chart_aapl_2014.json")
    with fixture_source(YahooDailyBars, payload, settings) as source:
        frame = source.fetch(["AAPL"], START, END)
    assert (frame["known_at"] == frame["as_of"]).all()


def test_yahoo_prices_are_retroactively_split_adjusted(settings: Settings) -> None:
    """The documented caveat, asserted against real recorded data.

    Apple closed near $600 in May 2014 and split 7:1 that June. Yahoo serves the
    pre-split closes already divided by seven. Returns are unaffected, but any
    price-LEVEL signal built on this is wrong and will not raise.
    """
    payload = load_json_fixture("yahoo_chart_aapl_2014.json")
    with fixture_source(YahooDailyBars, payload, settings) as source:
        frame = source.fetch(["AAPL"], START, END)

    may_close = frame.sort("as_of")["close"][0]
    assert may_close < 30, "a genuine May 2014 AAPL close was ~$591, not ~$21"
    # 7:1 in June 2014 and 4:1 in August 2020. Yahoo applies BOTH to a 2014 bar, so
    # the stored price has been restated by a corporate action six years after it.
    assert may_close * 28 == pytest.approx(591.48, abs=0.5)


def test_yahoo_corporate_actions_carry_their_own_ex_dates(settings: Settings) -> None:
    payload = load_json_fixture("yahoo_chart_aapl_2014.json")
    with fixture_source(YahooCorporateActions, payload, settings) as source:
        frame = source.fetch(["AAPL"], START, END)

    actions = dict(zip(frame["action"], frame["as_of"], strict=True))
    assert set(actions) == {"dividend", "split"}
    split = frame.filter(pl.col("action") == "split")
    assert split["split_numerator"].item() == 7.0
    assert split["split_denominator"].item() == 1.0
    assert frame.filter(pl.col("action") == "dividend")["amount"].item() == pytest.approx(0.1175)


def test_yahoo_unmapped_exchange_raises(settings: Settings) -> None:
    """Defaulting an unknown venue to a US calendar would misalign every bar of
    that instrument by hours, silently."""
    from quantlab.data.calendars import UnknownVenueError

    payload = json.loads(load_fixture("yahoo_chart_aapl_2014.json"))
    payload["chart"]["result"][0]["meta"]["fullExchangeName"] = "Narnia Main Board"
    payload["chart"]["result"][0]["meta"]["exchangeName"] = "NAR"
    with (
        fixture_source(YahooDailyBars, payload, settings) as source,
        pytest.raises(UnknownVenueError),
    ):
        source.fetch(["AAPL"], START, END)


def test_yahoo_error_payload_raises_rather_than_returning_empty(settings: Settings) -> None:
    payload = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    with fixture_source(YahooDailyBars, payload, settings) as source, pytest.raises(SourceError):
        source.fetch(["NOPE"], START, END)


def test_yahoo_shape_change_is_detected(settings: Settings) -> None:
    """Yahoo's API is unofficial and changes without notice; a response with
    neither a result nor an error must be loud, not an empty frame."""
    with (
        fixture_source(YahooDailyBars, {"chart": {}}, settings) as source,
        pytest.raises(SourceError, match="changed shape"),
    ):
        source.fetch(["AAPL"], START, END)


# ----------------------------------------------------------------------- binance --
def test_binance_klines_use_the_venue_bar_close(settings: Settings) -> None:
    payload = load_json_fixture("binance_klines_btcusdt_1d.json")
    with fixture_source(BinanceKlines, payload, settings, interval="1d") as source:
        frame = source.fetch(
            ["BTCUSDT"],
            dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 1, 5, tzinfo=dt.UTC),
        )

    first = frame.sort("as_of").row(0, named=True)
    assert first["as_of"] == dt.datetime(2024, 1, 1, 23, 59, 59, 999000, tzinfo=dt.UTC)
    assert first["interval"] == "1d"
    assert first["trades"] > 0
    assert (frame["known_at"] == frame["as_of"]).all()


def test_binance_funding_interval_is_measured_not_assumed(settings: Settings) -> None:
    """The funding interval has changed over time and differs per contract, so
    hardcoding eight hours makes a carry backtest wrong in the regimes that matter."""
    payload = load_json_fixture("binance_funding_btcusdt.json")
    with fixture_source(BinanceFunding, payload, settings) as source:
        frame = source.fetch(
            ["BTCUSDT"],
            dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 1, 5, tzinfo=dt.UTC),
        )

    assert frame["interval_hours"].unique().to_list() == [8.0]
    assert frame["funding_rate"].abs().max() < 0.01
    assert frame["mark_price"].null_count() == 0


def test_binance_instruments_snapshot_records_delisted_symbols(settings: Settings) -> None:
    """The survivorship defence: a symbol that has stopped trading is still
    recorded, so a past universe can be reconstructed rather than guessed."""
    payload = load_json_fixture("binance_exchangeinfo.json")
    with fixture_source(BinanceInstruments, payload, settings) as source:
        frame = source.fetch([], START, END)

    assert set(frame["symbol"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BCCUSDT"}
    assert frame.filter(pl.col("symbol") == "BCCUSDT")["status"].item() == "BREAK"
    # A universe snapshot is knowable at the moment we observed it, not before.
    assert (frame["known_at"] == frame["as_of"]).all()


def test_binance_rejects_an_unexpected_payload(settings: Settings) -> None:
    payload = {"code": -1121, "msg": "Invalid symbol."}
    with (
        fixture_source(BinanceKlines, payload, settings) as source,
        pytest.raises(SourceError, match="unexpected klines payload"),
    ):
        source.fetch(["NOPE"], START, END)


# -------------------------------------------------------------------------- fred --
def test_fred_refuses_a_revised_series_and_points_at_alfred(settings: Settings) -> None:
    payload = load_json_fixture("fred_observations_dgs10.json")
    with (
        fixture_source(FredSeries, payload, settings) as source,
        pytest.raises(SourceError, match="alfred"),
    ):
        source.fetch(["GDPC1"], START, END)


def test_fred_refuses_an_unclassified_series(settings: Settings) -> None:
    """Ingesting a series nobody has classified is how a macro backtest quietly
    starts using values that did not exist at the time."""
    payload = load_json_fixture("fred_observations_dgs10.json")
    with (
        fixture_source(FredSeries, payload, settings) as source,
        pytest.raises(SourceError, match="no revision policy"),
    ):
        source.fetch(["SOME_NEW_SERIES"], START, END)


def test_fred_requires_a_key(settings: Settings) -> None:
    payload = load_json_fixture("fred_observations_dgs10.json")
    with (
        fixture_source(FredSeries, payload, settings) as source,
        pytest.raises(SourceUnavailableError, match="QUANTLAB_FRED_API_KEY"),
    ):
        source.fetch(["DGS10"], START, END)


def test_fred_applies_the_publication_lag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Treasury yield dated Monday is published Tuesday afternoon. Setting
    known_at to the observation date would let a signal trade on it a day early."""
    monkeypatch.setenv("QUANTLAB_FRED_API_KEY", "x" * 32)
    from quantlab.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    payload = load_json_fixture("fred_observations_dgs10.json")
    with fixture_source(FredSeries, payload, settings) as source:
        frame = source.fetch(
            ["DGS10"],
            dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 1, 8, tzinfo=dt.UTC),
        )

    assert frame.height == 5, "the '.' missing-value row must be dropped, not parsed as zero"
    lag = (frame["known_at"] - frame["as_of"]).unique().to_list()
    assert lag == [dt.timedelta(hours=24)]
    assert not frame["vintage"].any()


def test_alfred_sets_known_at_to_the_vintage_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of ALFRED: the 2020 Q1 GDP figure published in April was
    revised twice. A snapshot in May must see the May vintage, not the June one."""
    monkeypatch.setenv("QUANTLAB_FRED_API_KEY", "x" * 32)
    from quantlab.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    payload = load_json_fixture("alfred_observations_gdpc1.json")
    with fixture_source(AlfredVintageSeries, payload, settings) as source:
        frame = source.fetch(
            ["GDPC1"],
            dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2020, 12, 31, tzinfo=dt.UTC),
        )

    assert frame["vintage"].all()
    q1 = frame.filter(pl.col("as_of") == dt.datetime(2020, 1, 1, tzinfo=dt.UTC)).sort("known_at")

    # Nine vintages of 2020 Q1 in the recorded payload -- the advance estimate
    # and eight revisions, the last of them five years later. The count is not
    # asserted: it grows every September when the annual revision lands, and a
    # test that has to be edited each year for a correct parser is noise.
    assert q1.height > 3, "a GDP quarter is revised many times"

    # known_at is the vintage's realtime_start: the day that value became the
    # published figure. Getting this wrong is what makes a macro backtest trade
    # on numbers that did not exist yet.
    assert q1["known_at"].to_list()[0] == dt.datetime(2020, 4, 29, tzinfo=dt.UTC)
    assert q1["value"].to_list()[0] == pytest.approx(18987.877)

    # Every vintage is distinct and ordered, and the revisions are material:
    # the advance estimate of 2020 Q1 is roughly 9% below the current figure.
    assert q1["known_at"].is_sorted()
    assert q1["known_at"].n_unique() == q1.height
    first, last = q1["value"].to_list()[0], q1["value"].to_list()[-1]
    assert abs(last / first - 1.0) > 0.05


def test_series_policy_covers_every_default_symbol() -> None:
    """A default symbol with no policy would fail on a user's first ingest."""
    for cls in (FredSeries, AlfredVintageSeries):
        for series_id in cls.default_symbols:
            assert series_id in SERIES_POLICY, f"{cls.name}: {series_id} is unclassified"


def test_fred_defaults_are_never_revised_and_alfred_defaults_are() -> None:
    assert all(not SERIES_POLICY[s].requires_vintages for s in FredSeries.default_symbols)
    assert all(SERIES_POLICY[s].requires_vintages for s in AlfredVintageSeries.default_symbols)


# ------------------------------------------------------------------------- stooq --
def test_stooq_detects_the_anti_bot_challenge_and_fails_loudly(settings: Settings) -> None:
    """Spec section 13: do not quietly substitute another source. The message must
    say what happened and what to do, not just that something went wrong."""
    challenge = load_fixture("stooq_challenge.html")
    with (
        fixture_source(StooqDailyBars, challenge, settings) as source,
        pytest.raises(SourceUnavailableError) as excinfo,
    ):
        source.fetch(["aapl.us"], START, END)

    message = str(excinfo.value)
    assert "proof-of-work" in message
    assert "does not solve anti-bot challenges" in message
    assert "LIMITATIONS" in message


def test_stooq_parser_still_works_for_the_day_access_returns() -> None:
    rows = parse_stooq_csv(load_fixture("stooq_aapl_sample.csv"), "aapl.us", "XNYS")
    assert len(rows) == 5
    assert rows[0]["as_of"] == dt.datetime(2024, 1, 2, 21, tzinfo=dt.UTC)
    assert rows[0]["close"] == 185.64
    # Stooq documents no adjustment methodology, so there is nothing honest to put
    # in adj_close.
    assert all(row["adj_close"] is None for row in rows)


def test_stooq_venue_inference_fails_loudly_on_an_unknown_suffix(settings: Settings) -> None:
    source = StooqDailyBars(settings=settings)
    assert source.venue_for("aapl.us") == "XNYS"
    with pytest.raises(SourceError, match="known suffixes"):
        source.venue_for("aapl.zz")


# ----------------------------------------------------------------------------------
# Regressions found by running the real APIs (Milestone 2)
# ----------------------------------------------------------------------------------
def test_binance_drops_the_current_unclosed_bar(settings: Settings) -> None:
    """Binance serves the in-progress daily bar with its close time in the future.

    Found live: at 12:04 UTC the lake held a bar stamped as closing at 23:59:59
    that day, whose "close" was simply the current price. Any snapshot taken later
    that day would have read a mid-session quote as a settled close.
    """
    from quantlab.data.store import utcnow

    now = utcnow()
    closed_ms = int((now - dt.timedelta(days=1)).timestamp() * 1000)
    open_ms = int((now + dt.timedelta(hours=6)).timestamp() * 1000)
    payload = [
        [closed_ms - 86_400_000, "1", "2", "0.5", "1.5", "10", closed_ms, "15", 5, "0", "0", "0"],
        [closed_ms, "1.5", "3", "1", "2.0", "4", open_ms, "8", 2, "0", "0", "0"],
    ]
    with fixture_source(BinanceKlines, payload, settings, interval="1d") as source:
        frame = source.fetch(["BTCUSDT"], now - dt.timedelta(days=3), now)

    assert frame.height == 1, "the unclosed bar must not be stored"
    assert frame["as_of"].max() < now


def test_yahoo_drops_the_current_unclosed_session(settings: Settings) -> None:
    """Same defect on the equity side: Yahoo reports a live price for today's
    still-open session."""
    from quantlab.data.calendars import session_close
    from quantlab.data.store import utcnow

    payload = json.loads(load_fixture("yahoo_chart_aapl_2014.json"))
    result = payload["chart"]["result"][0]
    now = utcnow()

    # A session whose close is still hours away.
    future_session = now.date() + dt.timedelta(days=365)
    while True:
        try:
            future_close = session_close("XNAS", future_session)
            break
        except ValueError:
            future_session += dt.timedelta(days=1)

    session_open = int((future_close - dt.timedelta(hours=7)).timestamp())
    result["timestamp"] = [session_open]
    # The symbol "listed" at that session, so the coverage check is satisfied and
    # this test isolates the unclosed-session rule.
    result["meta"]["firstTradeDate"] = session_open
    result["indicators"]["quote"][0] = {
        "open": [1.0],
        "high": [1.0],
        "low": [1.0],
        "close": [1.0],
        "volume": [1.0],
    }
    result["indicators"]["adjclose"][0] = {"adjclose": [1.0]}
    result.pop("events", None)

    with fixture_source(YahooDailyBars, payload, settings) as source:
        frame = source.fetch(["AAPL"], now - dt.timedelta(days=5), now)
    assert frame.height == 0


def test_yahoo_refuses_a_silently_truncated_history(settings: Settings) -> None:
    """The most dangerous defect found in Milestone 2.

    Yahoo truncates history non-deterministically: the identical SPY request
    returned 5030 bars from 2006-09-20 on one call and 5462 bars from 2005-01-03
    minutes later, with no error either time. A backtest run over 2006-2026
    instead of 2005-2026 is a different experiment, and nothing else in the stack
    would say so.
    """
    payload = json.loads(load_fixture("yahoo_chart_aapl_2014.json"))
    payload["chart"]["result"][0]["meta"]["firstTradeDate"] = 0  # listed long ago

    with (
        fixture_source(YahooDailyBars, payload, settings) as source,
        pytest.raises(SourceError, match="truncates history without reporting it"),
    ):
        # The fixture covers May-June 2014; asking from 2000 must not silently
        # return the short window as though it were complete.
        source.fetch(["AAPL"], dt.datetime(2000, 1, 1, tzinfo=dt.UTC), END)


def test_yahoo_accepts_history_that_starts_at_the_listing_date(settings: Settings) -> None:
    """A symbol that simply did not exist earlier is not a truncation."""
    payload = json.loads(load_fixture("yahoo_chart_aapl_2014.json"))
    first_bar = payload["chart"]["result"][0]["timestamp"][0]
    payload["chart"]["result"][0]["meta"]["firstTradeDate"] = first_bar

    with fixture_source(YahooDailyBars, payload, settings) as source:
        frame = source.fetch(["AAPL"], dt.datetime(2000, 1, 1, tzinfo=dt.UTC), END)
    assert frame.height > 30


def test_schema_rejects_timestamps_in_the_future(settings: Settings) -> None:
    """Defence in depth behind both source-level fixes: nothing can be knowable
    before it has happened, whatever a venue's bar-close convention says."""
    from tests.conftest import bar

    from quantlab.data.schemas import SchemaError, get_schema
    from quantlab.data.store import utcnow

    row = bar("AAPL", dt.date(2024, 1, 3), 100.0)
    ahead = utcnow() + dt.timedelta(hours=6)
    row["as_of"] = ahead
    row["known_at"] = ahead
    row["ingested_at"] = ahead
    with pytest.raises(SchemaError, match="knowable in the FUTURE"):
        get_schema("ohlcv_daily").validate(pl.DataFrame([row]))


def test_yahoo_requests_history_in_windows(settings: Settings) -> None:
    """The actual defence against the truncation: never ask for a span long enough
    to trigger the cap. A metadata check cannot help, because when Yahoo truncates
    it reports firstTradeDate as the start of the truncated range too.
    """
    from quantlab.data.sources.yahoo import CHUNK, _windows

    start = dt.datetime(2005, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2026, 9, 20, tzinfo=dt.UTC)
    windows = _windows(start, end)

    assert len(windows) >= 3
    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(b - a <= CHUNK for a, b in windows)
    # Contiguous and non-overlapping, so no session is requested twice or missed.
    for (_, first_end), (second_start, _) in itertools.pairwise(windows):
        assert second_start == first_end + dt.timedelta(days=1)


def test_yahoo_refuses_a_history_with_holes(settings: Settings) -> None:
    """A gap in the middle of a series is invisible to any metadata check, and it
    quietly changes every rolling statistic computed over it."""
    from quantlab.data.sources.yahoo import _check_session_coverage

    venue = "XNYS"
    days = [dt.date(2024, 1, 3) + dt.timedelta(days=n) for n in range(120)]
    # Keep only every third session: a series riddled with holes.
    rows = [
        {"as_of": dt.datetime(d.year, d.month, d.day, 21, tzinfo=dt.UTC)}
        for index, d in enumerate(days)
        if index % 3 == 0
    ]
    with pytest.raises(SourceError, match="silent holes"):
        _check_session_coverage("AAPL", venue, rows)


def test_yahoo_accepts_a_complete_history(settings: Settings) -> None:
    from quantlab.data.calendars import sessions
    from quantlab.data.sources.yahoo import _check_session_coverage

    venue = "XNYS"
    rows = [
        {"as_of": dt.datetime(d.year, d.month, d.day, 21, tzinfo=dt.UTC)}
        for d in sessions(venue, dt.date(2024, 1, 3), dt.date(2024, 3, 28))
    ]
    _check_session_coverage("AAPL", venue, rows)  # must not raise
