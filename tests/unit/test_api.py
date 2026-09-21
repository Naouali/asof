"""The web app.

Three things are worth testing here and the rest is plumbing. That the "as of"
date really does hide what was not yet public -- through the whole stack, not just
in the data layer. That each kind of filing becomes an honest event: a dozen lines
are one sale, a grant is not a purchase, a late report says so. And that an app
with no login tells the truth about that and stays on loopback.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from quantlab.api import create_app
from quantlab.api.app import _instant
from quantlab.api.events import (
    add_business_days,
    business_days_between,
    congress_events,
    contract_events,
    display_name,
    fund_events,
    insider_events,
    member_name,
    unread_report_events,
)
from quantlab.config import Settings
from quantlab.data.schemas import get_schema
from quantlab.data.store import Store

UTC = dt.UTC


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=UTC)


def frame(dataset: str, source: str, rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Rows as a frame of the dataset's exact schema, nulls where a row is silent."""
    schema = get_schema(dataset).polars_schema
    complete = [
        {"source": source, "dataset": dataset, "ingested_at": at(2026, 9, 20), **row}
        for row in rows
    ]
    built = pl.DataFrame(complete, infer_schema_length=None)
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in schema.items()
        if name not in built.columns
    ]
    return built.with_columns(missing).select(list(schema)).cast(schema)  # type: ignore[arg-type]


def insider_line(**overrides: Any) -> dict[str, Any]:
    return {
        "symbol": "OSPR",
        "as_of": at(2026, 9, 15),
        "known_at": at(2026, 9, 17, 22, 32),
        "issuer_cik": 111,
        "issuer_name": "Osprey Therapeutics",
        "owner_cik": 900,
        "owner_name": "OYELARAN MARCUS",
        "is_director": True,
        "is_officer": False,
        "is_ten_percent_owner": False,
        "form": "4",
        "accession": "0001-26-000001",
        "line": 1,
        "security_title": "Common Stock",
        "is_derivative": False,
        "transaction_code": "P",
        "acquired_disposed": "A",
        "shares": 12_000.0,
        "price": 41.30,
        "ownership": "D",
        "planned_10b5_1": False,
        **overrides,
    }


def congress_line(**overrides: Any) -> dict[str, Any]:
    return {
        "symbol": "OSPR",
        "as_of": at(2026, 7, 2),
        "known_at": at(2026, 9, 17, 3, 59),
        "chamber": "house",
        "doc_id": "20030001",
        "line": 1,
        "member": "Hon. Tom Tom Mr Villareal",
        "state_district": "TX21",
        "asset": "Osprey Therapeutics (OSPR)",
        "asset_type": "ST",
        "transaction_type": "purchase",
        "amount_min": 100_001.0,
        "amount_max": 250_000.0,
        "amount_text": "$100,001 - $250,000",
        "filing_status": "new",
        "url": "https://example.test/20030001.pdf",
        **overrides,
    }


def holding(period: dt.datetime, known: dt.datetime, cusip: str, shares: float) -> dict[str, Any]:
    return {
        "symbol": "5001",
        "as_of": period,
        "known_at": known,
        "manager_name": "Harrow Peak Capital",
        "cusip": cusip,
        "issuer_name": "OSPREY THERAPEUTICS" if cusip == "OSPR00001" else "OTHER CO",
        "put_call": "NONE",
        "shares_type": "SH",
        "shares": shares,
        "value_usd": shares * 40.0,
        "lines": 1,
        "form": "13F-HR",
        "accession": f"0002-{period:%y%m}",
    }


# ------------------------------------------------------------------------- names --
@pytest.mark.parametrize(
    ("raw", "person", "expected"),
    [
        ("COOK TIMOTHY D", True, "Timothy D Cook"),
        ("Newstead Jennifer", True, "Jennifer Newstead"),
        ("O'BRIEN DEIRDRE", True, "Deirdre O'Brien"),
        ("Vanguard Group Inc", True, "Vanguard Group Inc"),
        ("BERKSHIRE HATHAWAY INC", False, "BERKSHIRE HATHAWAY INC"),
        ("Madonna", True, "Madonna"),
        ("Kress Colette; Kress Family Trust", True, "Colette Kress"),
    ],
)
def test_a_filer_name_is_turned_round_only_when_it_is_safe_to(
    raw: str, person: bool, expected: str
) -> None:
    """EDGAR stores people LAST FIRST. A wrong guess renames a person, so anything
    that might be an entity is left exactly as filed."""
    assert display_name(raw, person=person) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hon. Scott Scott Franklin", "Scott Franklin"),
        ("Hon. John J Mr McGuire III", "John J McGuire III"),
        ("Cleo Fields", "Cleo Fields"),
        ("Hon. Suzan K. DelBene", "Suzan K. DelBene"),
    ],
)
def test_the_clerks_clutter_is_dropped_from_member_names(raw: str, expected: str) -> None:
    """These are real values from the Clerk's own index, not parsing artefacts."""
    assert member_name(raw) == expected


def test_business_days_skip_weekends_and_federal_holidays() -> None:
    friday = dt.date(2026, 7, 2)  # Thursday; Friday 3 July is the observed holiday
    assert add_business_days(friday, 2) == dt.date(2026, 7, 7)
    assert business_days_between(dt.date(2026, 9, 11), dt.date(2026, 9, 15)) == 2


# ---------------------------------------------------------------------- insiders --
def test_a_sale_reported_as_many_lines_is_one_event() -> None:
    lines = [
        insider_line(line=1, transaction_code="S", acquired_disposed="D", shares=100.0, price=10.0),
        insider_line(line=2, transaction_code="S", acquired_disposed="D", shares=300.0, price=20.0),
    ]
    (event,) = insider_events(frame("insider_transactions", "sec_insider", lines))

    assert event.verb == "Sold"
    assert event.direction == "sell"
    assert event.value_usd == 7_000.0
    assert event.size == "400 shares at $17.50"  # volume-weighted, not averaged
    assert event.actor == "Marcus Oyelaran"


def test_compensation_is_noise_and_an_open_market_purchase_is_not() -> None:
    lines = [
        insider_line(line=1),
        insider_line(line=2, transaction_code="A", price=None),
        insider_line(line=3, transaction_code="F", acquired_disposed="D"),
    ]
    events = {
        e.verb: e for e in insider_events(frame("insider_transactions", "sec_insider", lines))
    }

    assert not events["Bought"].noise
    assert events["Was granted"].noise
    assert events["Had withheld for tax"].noise
    assert events["Was granted"].value_usd is None  # no price, so no invented value


def test_a_pre_scheduled_sale_is_noise_even_though_it_is_a_sale() -> None:
    line = insider_line(transaction_code="S", acquired_disposed="D", planned_10b5_1=True)
    (event,) = insider_events(frame("insider_transactions", "sec_insider", [line]))

    assert event.noise
    assert "pre-scheduled" in (event.detail or "")


def test_an_insider_trade_is_dated_in_washington_not_in_utc() -> None:
    """Accepted 21:40 Eastern is 01:40 UTC the next day. It became public on the
    17th, and its lag is two days, not three."""
    line = insider_line(known_at=at(2026, 9, 18, 1, 40))
    (event,) = insider_events(frame("insider_transactions", "sec_insider", [line]))

    assert event.disclosed_on == dt.date(2026, 9, 17)
    assert event.lag_days == 2
    assert event.due_on == dt.date(2026, 9, 17)
    assert event.late_days == 0


def test_a_form_4_filed_after_two_business_days_is_late() -> None:
    line = insider_line(known_at=at(2026, 9, 22, 21, 0))
    (event,) = insider_events(frame("insider_transactions", "sec_insider", [line]))

    assert event.late_days == 3


# ---------------------------------------------------------------------- congress --
def test_a_house_trade_past_45_days_says_how_late_it_is() -> None:
    (event,) = congress_events(frame("congress_trades", "house_clerk", [congress_line()]))

    assert event.actor == "Rep. Tom Villareal"
    assert event.lag_days == 76
    assert event.late_days == 31
    assert event.due_on == dt.date(2026, 8, 16)
    assert event.size == "$100,001 to $250,000"
    assert event.value_usd is None  # a bracket is never summed as an amount


def test_a_trade_with_no_ticker_keeps_its_asset_name() -> None:
    line = congress_line(symbol="NO_TICKER", asset="Parkland, PA School District Bond", owner="SP")
    (event,) = congress_events(frame("congress_trades", "house_clerk", [line]))

    assert event.ticker is None
    assert event.asset == "Parkland, PA School District Bond"
    assert event.detail == "Held by their spouse"


def test_an_unread_report_is_bounded_to_the_period_the_trades_cover() -> None:
    """The index is ingested for years and the PDFs are not. A report nobody tried
    to open is not a scan."""
    trades = frame("congress_trades", "house_clerk", [congress_line()])

    def filing(doc_id: str, known: dt.datetime) -> dict[str, Any]:
        return {
            "symbol": "CA17",
            "as_of": known - dt.timedelta(hours=20),
            "known_at": known,
            "chamber": "house",
            "doc_id": doc_id,
            "filing_type": "P",
            "first_name": "Rohit",
            "last_name": "Khanna",
            "year": 2026,
            "url": f"https://example.test/{doc_id}.pdf",
        }

    filings = frame(
        "congress_filings",
        "house_clerk",
        [
            filing("20030001", at(2026, 9, 17, 3, 59)),  # read: it has trades
            filing("8220001", at(2026, 9, 18, 3, 59)),  # a scan
            filing("8110001", at(2024, 2, 1, 4, 59)),  # before anything was parsed
        ],
    )
    events = unread_report_events(filings, trades)

    assert [event.id for event in events] == ["unread:8220001"]
    assert events[0].actor == "Rep. Rohit Khanna"


def test_a_senator_is_addressed_as_one_and_has_no_district() -> None:
    line = congress_line(
        chamber="senate",
        doc_id="b999bc0e-3eb0-4ca9-ab07-8e8f2e04b41f",
        member="Alan Armstrong",
        state_district="SENATE",
        known_at=at(2026, 9, 17, 12, 55),
    )
    (event,) = congress_events(frame("congress_trades", "senate_efd", [line]))

    assert event.actor == "Sen. Alan Armstrong"
    assert event.role == "Senate"  # the Senate's index does not say which state
    assert event.disclosed_at == at(2026, 9, 17, 12, 55)


def test_a_paper_filer_indexed_in_capitals_is_not_shouted() -> None:
    trades = frame("congress_trades", "senate_efd", [congress_line(chamber="senate")])
    paper = {
        "symbol": "SENATE",
        "as_of": at(2026, 9, 18),
        "known_at": at(2026, 9, 19, 3, 59),
        "chamber": "senate",
        "doc_id": "929216d5",
        "filing_type": "P",
        "first_name": "RICHARD",
        "last_name": "BLUMENTHAL",
        "year": 2026,
        "url": "https://example.test/paper/929216d5/",
    }
    (event,) = unread_report_events(frame("congress_filings", "senate_efd", [paper]), trades)

    assert event.actor == "Sen. Richard Blumenthal"
    assert event.role == "Senate"


# ------------------------------------------------------------------------- funds --
def test_a_fund_event_is_the_change_from_the_quarter_before() -> None:
    q1, q2 = at(2026, 3, 31), at(2026, 6, 30)
    rows = [
        holding(q1, at(2026, 5, 15, 20), "OSPR00001", 2_000_000.0),
        holding(q1, at(2026, 5, 15, 20), "GONE00001", 500_000.0),
        holding(q2, at(2026, 8, 14, 20), "OSPR00001", 3_000_000.0),
        holding(q2, at(2026, 8, 14, 20), "NEW000001", 100_000.0),
    ]
    events = fund_events(
        frame("institutional_holdings", "sec_13f", rows), {"OSPR00001": "OSPR"}, per_filing=None
    )
    by_cusip = {event.id.rsplit(":", 1)[1]: event for event in events}

    assert by_cusip["OSPR00001"].verb == "Added"
    assert by_cusip["OSPR00001"].size == "50%, to 3.0M shares"
    assert by_cusip["OSPR00001"].ticker == "OSPR"
    assert by_cusip["GONE00001"].verb == "Sold out of"
    assert by_cusip["NEW000001"].verb == "Opened"
    assert by_cusip["NEW000001"].ticker is None  # no bridge entry, so no invented ticker
    # The form says what was held at quarter end, not when it was bought.
    assert by_cusip["OSPR00001"].traded_on == dt.date(2026, 6, 30)
    assert by_cusip["OSPR00001"].lag_days == 45


def test_a_ticker_page_sees_a_fund_leave_even_though_no_row_says_so() -> None:
    """Narrowed to one security, the quarter after an exit has no rows at all. The
    exit is that absence, so the filing's date has to come from the whole book."""
    q1, q2 = at(2026, 3, 31), at(2026, 6, 30)
    rows = [
        holding(q1, at(2026, 5, 15, 20), "OSPR00001", 2_000_000.0),
        holding(q1, at(2026, 5, 15, 20), "KEPT00001", 100.0),
        holding(q2, at(2026, 8, 14, 20), "KEPT00001", 100.0),
    ]
    (event,) = fund_events(
        frame("institutional_holdings", "sec_13f", rows),
        {"OSPR00001": "OSPR"},
        per_filing=None,
        only_cusips=["OSPR00001"],
    )

    assert event.verb == "Sold out of"
    assert event.disclosed_on == dt.date(2026, 8, 14)
    assert event.traded_on == dt.date(2026, 6, 30)


def test_an_amendment_is_not_lateness_and_does_not_redate_the_whole_book() -> None:
    """A fund files on time in August and amends in September, adding a position
    it had been allowed to keep confidential. Only that position became public in
    September, and nothing here was late."""
    q1, q2 = at(2026, 3, 31), at(2026, 6, 30)
    on_time, amended = at(2026, 8, 14, 20), at(2026, 9, 1, 20)
    rows = [
        holding(q1, at(2026, 5, 15, 20), "OSPR00001", 1_000_000.0),
        {**holding(q2, on_time, "OSPR00001", 2_000_000.0), "first_known_at": on_time},
        {
            **holding(q2, amended, "HELD00001", 900_000.0),
            "first_known_at": amended,
            "form": "13F-HR/A",
        },
    ]
    built = frame(
        "institutional_holdings",
        "sec_13f",
        [{k: v for k, v in r.items() if k != "first_known_at"} for r in rows],
    )
    built = built.with_columns(
        pl.Series("first_known_at", [r.get("first_known_at", r["known_at"]) for r in rows])
    )
    events = {e.id.rsplit(":", 1)[1]: e for e in fund_events(built, {}, per_filing=None)}

    assert events["OSPR00001"].disclosed_on == dt.date(2026, 8, 14)
    assert events["HELD00001"].disclosed_on == dt.date(2026, 9, 1)
    assert events["HELD00001"].lag_days == 63
    assert events["HELD00001"].late_days == 0
    assert events["OSPR00001"].late_days == 0


def test_the_first_report_held_opens_no_positions() -> None:
    """Calling everything in it "new" would be an artefact of where ingestion began."""
    rows = [holding(at(2026, 6, 30), at(2026, 8, 14, 20), "OSPR00001", 3_000_000.0)]
    assert fund_events(frame("institutional_holdings", "sec_13f", rows), {}) == []


def test_small_rebalancing_is_not_an_event() -> None:
    rows = [
        holding(at(2026, 3, 31), at(2026, 5, 15, 20), "OSPR00001", 1_000_000.0),
        holding(at(2026, 6, 30), at(2026, 8, 14, 20), "OSPR00001", 1_040_000.0),
    ]
    assert fund_events(frame("institutional_holdings", "sec_13f", rows), {}) == []


# --------------------------------------------------------------------- contracts --
def contract_line(**overrides: Any) -> dict[str, Any]:
    return {
        "symbol": "OSPR",
        "as_of": at(2026, 6, 11),
        "known_at": at(2026, 9, 11),
        "transaction_key": "9700_-NONE-_FA880718C0009_P00200_-NONE-_0",
        "award_id": "FA880718C0009",
        "modification_number": "P00200",
        "recipient_name": "OSPREY FEDERAL SYSTEMS LLC",
        "recipient_uei": "AAAAAAAAAAA1",
        "parent_name": "OSPREY THERAPEUTICS INC",
        "parent_uei": "BBBBBBBBBBB2",
        "awarding_agency": "Department of Defense",
        "awarding_sub_agency": "DEPT OF THE AIR FORCE",
        "defense": True,
        "action_type": "EXERCISE AN OPTION",
        "obligation_usd": 514_412_527.67,
        "potential_value_usd": 7_200_000_000.0,
        "description": "FIELD HOSPITAL ANTIVIRAL STOCKPILE LOT 23 AND 24 PURCHASE",
        "reported_at": at(2026, 6, 9, 11, 56),
        "url": "https://www.usaspending.gov/award/CONT_AWD_FA880718C0009_9700_-NONE-_-NONE-/",
        **overrides,
    }


def test_a_contract_is_dated_by_the_action_and_known_by_its_publication() -> None:
    (event,) = contract_events(frame("government_contracts", "usaspending", [contract_line()]))

    assert event.kind == "contract" and event.ticker == "OSPR"
    assert event.actor == "Dept of the Air Force"
    assert event.role == "Department of Defense"
    assert event.verb == "Exercised an option for" and event.size == "$514.41M"
    assert event.traded_on == dt.date(2026, 6, 11)
    assert event.disclosed_on == dt.date(2026, 9, 10)  # 00:00 UTC is the evening before
    assert event.lag_days == 91
    # A policy, not a breach: no deadline, so never "late".
    assert event.deadline_days is None and event.late_days == 0
    assert event.detail == (
        "Field hospital antiviral stockpile lot 23 and 24 purchase. "
        "Signed by Osprey Federal Systems LLC. "
        "The Pentagon publishes its contract actions 90 days late."
    )


def test_the_parent_under_another_legal_ending_did_not_sign_for_itself() -> None:
    line = contract_line(recipient_name="OSPREY THERAPEUTICS CORPORATION", defense=False)
    (event,) = contract_events(frame("government_contracts", "usaspending", [line]))

    assert event.detail == "Field hospital antiviral stockpile lot 23 and 24 purchase."


def test_what_a_contract_action_did_is_said_in_a_verb() -> None:
    rows = [
        contract_line(transaction_key="a", modification_number="0", action_type=None),
        contract_line(transaction_key="b", action_type="FUNDING ONLY ACTION"),
        contract_line(transaction_key="c", obligation_usd=-12_475_741.0),
    ]
    events = contract_events(frame("government_contracts", "usaspending", rows))

    assert [event.verb for event in events] == ["Awarded", "Added", "Took back"]
    assert [event.direction for event in events] == ["buy", "buy", "sell"]
    assert events[2].size == "$12.48M" and events[2].value_usd == -12_475_741.0


def test_small_contract_actions_stay_out_of_the_feed_by_a_floor() -> None:
    rows = [contract_line(), contract_line(transaction_key="small", obligation_usd=3_000_000.0)]
    built = frame("government_contracts", "usaspending", rows)

    assert len(contract_events(built)) == 2
    assert len(contract_events(built, min_usd=25_000_000.0)) == 1


# ----------------------------------------------------------------------- the app --
@pytest.fixture
def client(store: Store, settings: Settings, tmp_path: Path) -> TestClient:
    store.write(
        frame(
            "insider_transactions",
            "sec_insider",
            [
                insider_line(),
                insider_line(
                    owner_cik=901,
                    owner_name="FISCHER LENA",
                    accession="0001-26-000002",
                    as_of=at(2026, 9, 11),
                    known_at=at(2026, 9, 15, 21, 0),
                    shares=3_000.0,
                    price=40.85,
                ),
            ],
        ),
        asset_class="equity",
    )
    store.write(frame("congress_trades", "house_clerk", [congress_line()]), asset_class="equity")
    store.write(
        frame(
            "fails_to_deliver",
            "sec_ftd",
            [
                {
                    "symbol": "OSPR",
                    "as_of": at(2026, 8, 31),
                    "known_at": at(2026, 9, 15, 13),
                    "cusip": "OSPR00001",
                    "quantity": 48_210.0,
                    "description": "OSPREY THERAPEUTICS",
                    "price": 39.0,
                }
            ],
        ),
        asset_class="equity",
    )
    store.write(
        frame(
            "institutional_holdings",
            "sec_13f",
            [
                holding(at(2026, 3, 31), at(2026, 5, 15, 20), "OSPR00001", 2_000_000.0),
                holding(at(2026, 6, 30), at(2026, 8, 14, 20), "OSPR00001", 3_000_000.0),
            ],
        ),
        asset_class="equity",
    )
    return TestClient(create_app(settings, ui_dir=tmp_path / "no-ui"))


def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for group in payload["groups"] for event in group["events"]]


def test_the_feed_groups_by_the_day_things_became_public(client: TestClient) -> None:
    payload = client.get("/api/feed", params={"as_of": "2026-09-18"}).json()

    assert [group["date"] for group in payload["groups"]] == [
        "2026-09-17",
        "2026-09-16",
        "2026-09-15",
    ]
    assert {event["kind"] for event in _events(payload)} == {"insider", "congress"}
    assert payload["coverage"]["fund_period"] == "2026-06-30"
    assert payload["coverage"]["fund_period_age_days"] == 80


def test_two_insiders_buying_at_once_is_called_out(client: TestClient) -> None:
    (insight,) = client.get("/api/feed", params={"as_of": "2026-09-18"}).json()["insights"]

    assert insight["ticker"] == "OSPR"
    assert "2 Osprey Therapeutics insiders" in insight["text"]


def test_as_of_hides_what_was_not_public_yet_and_says_what_had_already_happened(
    client: TestClient,
) -> None:
    """On 16 September, Oyelaran had bought (the 15th) and nobody knew (the 17th).
    The feed must not show it; `beyond` must, because it had already happened."""
    payload = client.get("/api/feed", params={"as_of": "2026-09-16", "days": 30}).json()
    shown = {event["actor"] for event in _events(payload)}
    beyond = {event["actor"] for event in payload["beyond"]}

    assert payload["is_live"] is False
    assert "Marcus Oyelaran" not in shown
    assert "Lena Fischer" in shown
    assert "Marcus Oyelaran" in beyond
    assert payload["insights"] == []  # one buyer was public, not two


def test_a_date_means_the_end_of_that_day_in_washington() -> None:
    """A House report filed on the 16th is knowable at 23:59:59 Eastern that day,
    which is already the 17th in UTC. "As of the 16th" has to include it."""
    assert _instant(dt.date(2026, 9, 16)) > at(2026, 9, 17, 3, 59)
    assert _instant(dt.date(2026, 1, 16)) > at(2026, 1, 17, 4, 59)


def test_the_live_feed_has_no_beyond(client: TestClient) -> None:
    payload = client.get("/api/feed").json()

    assert payload["is_live"] is True
    assert payload["beyond"] == []


def test_the_feed_narrows_to_one_person(client: TestClient) -> None:
    payload = client.get(
        "/api/feed", params={"as_of": "2026-09-18", "days": 365, "actor": "insider:901"}
    ).json()

    assert payload["actor"]["name"] == "Lena Fischer"
    assert {event["actor"] for event in _events(payload)} == {"Lena Fischer"}


def test_a_ticker_page_joins_funds_to_it_through_the_cusip_bridge(client: TestClient) -> None:
    page = client.get("/api/tickers/ospr", params={"as_of": "2026-09-18"}).json()

    assert page["ticker"] == "OSPR"
    assert page["name"] == "Osprey Therapeutics"
    assert [holder["manager"] for holder in page["holders"]] == ["Harrow Peak Capital"]
    assert page["holders"][0]["change"] == 1_000_000.0
    assert page["holders"][0]["age_days"] == 80
    assert page["fails_to_deliver"]["quantity"] == 48_210.0
    assert {event["kind"] for event in page["events"]} == {"insider", "congress", "fund"}
    assert any("2 insiders bought" in sentence for sentence in page["brief"])
    assert page["prices"] == []  # none ingested: the page says so rather than failing


def test_an_unknown_ticker_is_an_empty_page_not_an_error(client: TestClient) -> None:
    page = client.get("/api/tickers/ZZZZ").json()

    assert page["known"] is False
    assert page["brief"] == ["No disclosure in the lake names ZZZZ yet."]


def test_search_finds_tickers_people_and_funds(client: TestClient) -> None:
    def kinds(query: str) -> set[str]:
        return {hit["kind"] for hit in client.get("/api/search", params={"q": query}).json()}

    assert kinds("ospr") == {"ticker"}
    assert kinds("oyel") == {"person"}
    assert kinds("harrow") == {"fund"}
    assert client.get("/api/search", params={"q": "o"}).json() == []


@pytest.fixture
def contractor(client: TestClient, store: Store) -> TestClient:
    store.write(
        frame(
            "government_contracts",
            "usaspending",
            [
                contract_line(),
                contract_line(
                    transaction_key="civil",
                    modification_number="0",
                    action_type=None,
                    awarding_agency="Department of Health and Human Services",
                    awarding_sub_agency="Centers for Disease Control and Prevention",
                    defense=False,
                    obligation_usd=4_000_000.0,
                    as_of=at(2026, 9, 10),
                    known_at=at(2026, 9, 12, 15),
                ),
            ],
        ),
        asset_class="equity",
    )
    return client


def test_the_feed_carries_large_contracts_and_the_ticker_page_all_of_them(
    contractor: TestClient,
) -> None:
    feed = contractor.get("/api/feed", params={"as_of": "2026-09-18", "days": 30}).json()
    in_feed = [event for event in _events(feed) if event["kind"] == "contract"]
    assert [event["size"] for event in in_feed] == ["$514.41M"]  # the $4M award is under the floor

    page = contractor.get("/api/tickers/OSPR", params={"as_of": "2026-09-18"}).json()
    contracts = page["contracts"]
    assert contracts["actions"] == 2 and contracts["net_usd"] == pytest.approx(518_412_527.67)
    assert [event["size"] for event in contracts["events"]] == ["$514.41M", "$4.00M"]
    assert contracts["defense_share"] == pytest.approx(0.992)
    assert contracts["agencies"][0]["name"] == "Department of Defense"
    # Contracts sit in their own section, not among the people who traded.
    assert not [event for event in page["events"] if event["kind"] == "contract"]
    assert "99% of it from the Pentagon, which publishes 90 days late." in page["brief"][-1]


def test_an_embargoed_award_is_behind_the_curtain_and_counted_apart(
    contractor: TestClient,
) -> None:
    """On 1 August the June award had happened and was six weeks from public."""
    past = contractor.get("/api/feed", params={"as_of": "2026-08-01", "days": 30}).json()

    assert not [event for event in _events(past) if event["kind"] == "contract"]
    assert [event["kind"] for event in past["beyond"]].count("contract") == 1
    assert past["beyond_contracts_total"] == 1
    assert (
        contractor.get("/api/tickers/OSPR", params={"as_of": "2026-08-01"}).json()["contracts"]
        is None
    )


def test_an_empty_lake_is_an_empty_feed_with_a_way_forward(
    settings: Settings, tmp_path: Path
) -> None:
    settings.layout.ensure()
    payload = TestClient(create_app(settings, ui_dir=tmp_path / "no-ui")).get("/api/feed").json()

    assert payload["empty_lake"] is True
    assert payload["groups"] == []


def test_data_health_lists_what_the_lake_holds(client: TestClient) -> None:
    payload = client.get("/api/data").json()

    assert {row["dataset"] for row in payload["datasets"]} >= {
        "insider_transactions",
        "congress_trades",
    }


# ---------------------------------------------------------- no login, said plainly --
def test_health_says_there_is_no_authentication(client: TestClient) -> None:
    payload = client.get("/api/health").json()

    assert payload["authentication"] == "none"
    assert payload["status"] == "ok"


def test_responses_refuse_framing_sniffing_and_caching(client: TestClient) -> None:
    headers = client.get("/api/feed").headers

    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["cache-control"] == "no-store"


def test_nothing_in_the_api_writes(client: TestClient) -> None:
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)("/api/feed").status_code == 405


def test_a_bad_date_is_refused_not_guessed(client: TestClient) -> None:
    assert client.get("/api/feed", params={"as_of": "last tuesday"}).status_code == 422
    assert client.get("/api/feed", params={"days": 0}).status_code == 422
    assert client.get("/api/nope").status_code == 404


def test_serving_beyond_loopback_is_allowed_and_said_out_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from quantlab.cli import app

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda _app, **kwargs: served.update(kwargs))
    runner = CliRunner()

    quiet = runner.invoke(app, ["serve"])
    assert served["host"] == "127.0.0.1"
    assert "no login" not in quiet.output

    loud = runner.invoke(app, ["serve", "--host", "0.0.0.0"])  # noqa: S104
    assert loud.exit_code == 0
    assert "no login" in loud.output


# ------------------------------------------------------------------------ the ui --
def test_without_a_built_interface_the_app_says_how_to_build_it(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 503
    assert "npm run build" in response.json()["detail"]


def test_any_page_path_serves_the_app_and_nothing_outside_it(
    settings: Settings, tmp_path: Path
) -> None:
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>asof</title>", encoding="utf-8")
    (ui / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not yours", encoding="utf-8")
    settings.layout.ensure()
    client = TestClient(create_app(settings, ui_dir=ui))

    assert "asof" in client.get("/t/NVDA").text  # a reload on a deep link lands on the app
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert "not yours" not in client.get("/%2e%2e/secret.txt").text
    assert "not yours" not in client.get("/..%2fsecret.txt").text
