"""What the disclosures add up to.

An aggregate is where a small dishonesty compounds: a dollar total built from
ranges, a median of three trades, a contract counted before it was public. These
tests pin what each number is allowed to claim.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient
from tests.unit.test_api import at, congress_line, contract_line, frame, insider_line

from quantlab.api import analytics, create_app, track_record
from quantlab.api.events import insider_events
from quantlab.api.queries import Lens
from quantlab.config import Settings
from quantlab.data.store import Store

NOW = at(2026, 9, 18, 23)


def _prices(symbol: str, closes: dict[dt.datetime, float]) -> pl.DataFrame:
    rows = [
        {"symbol": symbol, "as_of": day, "known_at": day + dt.timedelta(hours=21), "close": close}
        for day, close in closes.items()
    ]
    return frame("ohlcv_daily", "yahoo", rows)


@pytest.fixture
def lake(store: Store) -> Store:
    insiders = [
        insider_line(),  # Oyelaran bought on 15 Sep, public 17 Sep
        insider_line(
            owner_cik=901,
            owner_name="FISCHER LENA",
            accession="0001-26-000002",
            as_of=at(2026, 9, 11),
            known_at=at(2026, 9, 15, 21),
        ),
        insider_line(
            owner_cik=902,
            owner_name="ABARA NKEM",
            accession="0001-26-000003",
            transaction_code="S",
            acquired_disposed="D",
            as_of=at(2026, 9, 9),
            known_at=at(2026, 9, 11, 21),
        ),
        # A grant: pay, not a decision. It must not count as a purchase.
        insider_line(
            owner_cik=903,
            owner_name="GRANT RECIPIENT",
            accession="0001-26-000004",
            transaction_code="A",
            as_of=at(2026, 9, 10),
            known_at=at(2026, 9, 12, 21),
        ),
    ]
    store.write(frame("insider_transactions", "sec_insider", insiders), asset_class="equity")
    store.write(
        frame(
            "congress_trades",
            "house_clerk",
            [
                congress_line(),  # bought 2 Jul, public 17 Sep
                congress_line(doc_id="20030002", member="Hon. Ada Reyes", state_district="NM01"),
            ],
        ),
        asset_class="equity",
    )
    store.write(
        _prices(
            "OSPR",
            {
                at(2026, 7, 2): 30.0,
                at(2026, 9, 9): 40.0,
                at(2026, 9, 11): 41.0,
                at(2026, 9, 15): 42.0,
                at(2026, 9, 16): 45.0,
            },
        ),
        asset_class="equity",
    )
    store.write(
        frame(
            "government_contracts",
            "usaspending",
            [
                contract_line(),  # Pentagon, signed 11 Jun, public 11 Sep
                contract_line(
                    transaction_key="civil",
                    awarding_agency="Department of Energy",
                    awarding_sub_agency="Department of Energy",
                    defense=False,
                    obligation_usd=-4_000_000.0,
                    as_of=at(2026, 8, 3),
                    known_at=at(2026, 8, 5),
                ),
            ],
        ),
        asset_class="equity",
    )
    return store


# ------------------------------------------------------------------------ traded --
def test_tickers_are_ranked_by_people_not_by_trades_or_dollars(lake: Store) -> None:
    found = analytics.traded(Lens(lake, NOW), days=90, kind=None)

    (ticker,) = found["tickers"]
    # Two insiders and two members bought; one insider sold. The grant is not a purchase.
    assert (ticker["ticker"], ticker["buyers"], ticker["sellers"]) == ("OSPR", 4, 1)
    assert ticker["name"] == "Osprey Therapeutics"
    assert found["trades"] == 5


def test_the_kind_filter_narrows_both_lists(lake: Store) -> None:
    found = analytics.traded(Lens(lake, NOW), days=90, kind="congress")

    assert (found["tickers"][0]["buyers"], found["tickers"][0]["sellers"]) == (2, 0)
    assert {person["kind"] for person in found["people"]} == {"congress"}
    assert [person["actor"] for person in found["people"]] == [
        "Rep. Tom Villareal",
        "Rep. Ada Reyes",
    ]


def test_nothing_disclosed_after_the_as_of_date_is_counted(lake: Store) -> None:
    """On 14 Sep only the sale (public 11 Sep) had surfaced."""
    found = analytics.traded(Lens(lake, at(2026, 9, 14, 23)), days=90, kind=None)
    assert (found["tickers"][0]["buyers"], found["tickers"][0]["sellers"]) == (0, 1)


# ------------------------------------------------------------------- price moves --
def test_a_move_is_measured_from_the_trade_to_the_day_it_became_public(lake: Store) -> None:
    found = analytics.price_moves(Lens(lake, NOW), days=90)

    assert found["trades"] == 5 and found["measured"] == 5 and found["priced_tickers"] == 1
    # The member bought at 30 on 2 Jul; it was 45 when the report surfaced on 16 Sep.
    assert found["examples"][0]["actor"] in {"Rep. Tom Villareal", "Rep. Ada Reyes"}
    assert found["examples"][0]["price_move_pct"] == 50.0


def test_a_median_of_a_few_trades_is_not_offered(lake: Store) -> None:
    groups = {
        group["key"]: group for group in analytics.price_moves(Lens(lake, NOW), days=90)["groups"]
    }

    assert groups["house"]["buy"]["n"] == 2 and groups["house"]["buy"]["median"] is None
    assert groups["senate"]["buy"]["n"] == 0


def test_with_enough_trades_the_spread_is_given() -> None:
    spread = analytics._spread([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    assert spread == {"n": 6, "median": 1.0, "p10": -3.0, "p25": -1.5, "p75": 3.5, "p90": 5.0}


def test_a_trade_in_a_ticker_with_no_prices_is_counted_as_unmeasured(lake: Store) -> None:
    lake.write(
        frame(
            "congress_trades",
            "house_clerk",
            [congress_line(symbol="ZZZZ", doc_id="20030009", asset="Unpriced Corp (ZZZZ)")],
        ),
        asset_class="equity",
    )
    found = analytics.price_moves(Lens(lake, NOW), days=90)
    assert found["trades"] == 6 and found["measured"] == 5


# --------------------------------------------------------------------- contracts --
def test_contract_totals_are_net_and_split_by_who_paid(lake: Store) -> None:
    found = analytics.contracts(Lens(lake, NOW), live=None)

    assert found["actions"] == 2
    assert found["net_usd"] == pytest.approx(510_412_527.67)
    assert found["defense_usd"] == pytest.approx(514_412_527.67)
    assert [agency["name"] for agency in found["agencies"]] == [
        "Department of Defense",
        "Department of Energy",
    ]
    assert found["companies"][0]["name"] == "OSPR" and found["companies"][0]["actions"] == 2
    assert found["hidden"] is None  # today cannot know what it cannot see yet


def test_months_are_the_months_things_became_public_not_the_months_they_were_signed(
    lake: Store,
) -> None:
    months = analytics.contracts(Lens(lake, NOW), live=None)["months"]

    assert [month["month"] for month in months] == [dt.date(2026, 8, 1), dt.date(2026, 9, 1)]
    assert months[0]["taken_back_usd"] == -4_000_000.0 and months[0]["committed_usd"] == 0.0
    assert months[1]["committed_usd"] == pytest.approx(514_412_527.67)  # signed in June


def test_looking_back_counts_what_was_signed_and_not_yet_public(lake: Store) -> None:
    """On 1 August the Pentagon's June award was six weeks from the public record."""
    past = Lens(lake, at(2026, 8, 1, 23))
    found = analytics.contracts(past, live=Lens(lake, NOW))

    assert found["actions"] == 0
    assert found["hidden"] == {
        "actions": 1,
        "net_usd": pytest.approx(514_412_527.67),
        "defense_actions": 1,
    }


# ------------------------------------------------- forms filed as somebody's owner --
def test_a_form_filed_as_an_owner_of_another_company_is_not_this_companys_insider() -> None:
    rows = [
        insider_line(),
        insider_line(accession="0001-26-000002", owner_cik=901, owner_name="FISCHER LENA"),
        insider_line(
            accession="0001-26-000777",
            issuer_cik=999,
            issuer_name="Some Start-up Inc.",
            owner_name="OSPREY THERAPEUTICS",
        ),
    ]
    events = insider_events(frame("insider_transactions", "sec_insider", rows))

    assert len(events) == 2
    assert {event.asset for event in events} == {"Osprey Therapeutics"}


# ----------------------------------------------------------------------- the API --
def test_the_three_analyses_are_served(lake: Store, settings: Settings, tmp_path: Path) -> None:
    client = TestClient(create_app(settings, ui_dir=tmp_path / "no-ui"))
    asked: dict[str, Any] = {"as_of": "2026-09-18"}

    traded = client.get("/api/analytics/traded", params={**asked, "kind": "insider"})
    moves = client.get("/api/analytics/price-moves", params=asked)
    money = client.get("/api/analytics/contracts", params={"as_of": "2026-08-01"})

    assert traded.status_code == moves.status_code == money.status_code == 200
    assert traded.json()["tickers"][0]["buyers"] == 2
    assert moves.json()["measured"] == 5
    assert money.json()["hidden"]["actions"] == 1
    assert client.get("/api/analytics/traded", params={"kind": "contract"}).status_code == 422


# ------------------------------------------------------------- track record --
# Everything here happens in the spring, because a bar is only storable once it is
# in the past: the lake refuses a price knowable in the future, which is the same
# discipline this module is about.
# A bar is stamped at the 21:00 close, which is the same day in Washington; at
# midnight UTC it would be the evening before, and land on the wrong session.
BOUGHT = at(2026, 3, 2, 21)
FILED = at(2026, 4, 17, 3, 59)  # public 16 April in Washington
MEMBER_EXIT = at(2026, 4, 1, 21)  # thirty days after the trade
PUBLIC = at(2026, 4, 16, 21)
AFTER_30 = at(2026, 5, 16, 21)  # thirty days after it became public
LOOKING = at(2026, 5, 20, 23)


def _spring(**overrides: Any) -> dict[str, Any]:
    return congress_line(as_of=at(2026, 3, 2), known_at=FILED, **overrides)


def _bench(store: Store, closes: dict[dt.datetime, float]) -> None:
    store.write(_prices("SPY", closes), asset_class="equity")


def test_a_record_is_measured_from_the_day_the_trade_became_public(store: Store) -> None:
    """The filer's own price was available to nobody else, so it is not the price
    a follower is scored at. Both are reported; the gap is what the delay cost."""
    store.write(frame("congress_trades", "house_clerk", [_spring()]), asset_class="equity")
    store.write(
        _prices(
            "OSPR",
            {BOUGHT: 100.0, MEMBER_EXIT: 132.0, PUBLIC: 120.0, AFTER_30: 132.0},
        ),
        asset_class="equity",
    )
    _bench(store, {BOUGHT: 100.0, MEMBER_EXIT: 102.0, PUBLIC: 100.0, AFTER_30: 102.0})

    found = track_record.records(Lens(store, LOOKING), horizon=30, min_trades=1)

    (person,) = found["people"]
    assert person["mean_excess_pct"] == 8.0  # a follower: in at 120, +10%, less 2%
    assert person["own_mean_excess_pct"] == 30.0  # the member: in at 100, +32%, less 2%
    assert person["trades"] == 1 and person["ranked"] is True


def test_a_window_that_has_not_finished_is_not_measured(store: Store) -> None:
    """Scoring a trade disclosed last week over thirty days means peeking."""
    store.write(frame("congress_trades", "house_clerk", [_spring()]), asset_class="equity")
    store.write(_prices("OSPR", {BOUGHT: 100.0, PUBLIC: 120.0}), asset_class="equity")
    _bench(store, {BOUGHT: 100.0, PUBLIC: 100.0})  # the window is still open

    found = track_record.records(Lens(store, at(2026, 4, 20, 23)), horizon=30, min_trades=1)

    assert found["measured"] == 0 and found["unfinished"] == 1
    assert found["people"] == []


def test_a_thin_record_is_measured_but_not_ranked(store: Store) -> None:
    """A mean of one trade is an anecdote with a decimal point."""
    store.write(frame("congress_trades", "house_clerk", [_spring()]), asset_class="equity")
    store.write(_prices("OSPR", {PUBLIC: 100.0, AFTER_30: 150.0}), asset_class="equity")
    _bench(store, {PUBLIC: 100.0, AFTER_30: 100.0})

    found = track_record.records(Lens(store, LOOKING), horizon=30, min_trades=10)

    (person,) = found["people"]
    assert person["ranked"] is False and found["ranked"] == 0
    # One trade can never be told from luck, however large it is.
    assert person["distinguishable"] is False and person["low_pct"] is None


def test_the_shuffle_says_how_good_chance_alone_would_look(store: Store) -> None:
    """With enough filers somebody always leads. The question the leader board has
    to answer is whether dealing the same trades out at random does as well."""
    flat = {PUBLIC: 100.0, AFTER_30: 100.0}
    rows = [
        _spring(
            doc_id=f"d{index}",
            line=index,
            symbol=f"T{index % 2}",
            member="Hon. Ada Reyes" if index % 2 else "Hon. Tom Villareal",
            state_district="NM01" if index % 2 else "TX21",
        )
        for index in range(24)
    ]
    store.write(frame("congress_trades", "house_clerk", rows), asset_class="equity")
    for ticker in ("T0", "T1"):  # both do exactly what the market does: no skill anywhere
        store.write(_prices(ticker, flat), asset_class="equity")
    _bench(store, flat)

    found = track_record.records(Lens(store, LOOKING), horizon=30, min_trades=5)

    assert found["luck"]["as_good_by_chance"] == 1.0
    assert found["standouts"] == 0


def test_a_trade_in_a_ticker_with_no_prices_is_counted_as_unpriced(store: Store) -> None:
    store.write(
        frame("congress_trades", "house_clerk", [_spring(symbol="ZZZZ")]), asset_class="equity"
    )
    _bench(store, {PUBLIC: 100.0, AFTER_30: 100.0})

    found = track_record.records(Lens(store, LOOKING), horizon=30, min_trades=1)

    assert found["measured"] == 0 and found["unpriced"] == 1


def test_without_a_benchmark_no_record_is_offered(store: Store) -> None:
    """An excess return needs something to be in excess of."""
    store.write(frame("congress_trades", "house_clerk", [_spring()]), asset_class="equity")
    store.write(_prices("OSPR", {PUBLIC: 100.0, AFTER_30: 150.0}), asset_class="equity")

    found = track_record.records(Lens(store, LOOKING), horizon=30, min_trades=1)

    assert found["measured"] == 0 and "SPY" in (found["why_empty"] or "")


def test_the_api_serves_the_track_record(store: Store, settings: Settings, tmp_path: Path) -> None:
    store.write(frame("congress_trades", "house_clerk", [_spring()]), asset_class="equity")
    store.write(
        _prices(
            "OSPR",
            {BOUGHT: 100.0, MEMBER_EXIT: 132.0, PUBLIC: 120.0, AFTER_30: 132.0},
        ),
        asset_class="equity",
    )
    _bench(store, {BOUGHT: 100.0, MEMBER_EXIT: 102.0, PUBLIC: 100.0, AFTER_30: 102.0})
    client = TestClient(create_app(settings, ui_dir=tmp_path / "no-ui"))

    found = client.get(
        "/api/analytics/track-record",
        params={"as_of": "2026-05-20", "horizon": 30, "min_trades": 1},
    )

    assert found.status_code == 200
    body = found.json()
    assert body["benchmark"] == "SPY" and body["horizon_days"] == 30
    assert body["people"][0]["mean_excess_pct"] == 8.0
