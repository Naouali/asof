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

from quantlab.api import analytics, create_app
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
