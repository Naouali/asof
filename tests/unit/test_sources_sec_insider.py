"""SEC insider transactions.

The parser is simple; what these tests guard is the meaning. A trade is dated by
when it happened and knowable from when EDGAR accepted the report of it, a
restated position is not a trade, and a line that is compensation is recorded as
what it is rather than dressed up as a decision.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.conftest import load_fixture, load_json_fixture
from tests.unit.test_sources import routed_source

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.schemas import get_schema
from quantlab.data.sources.sec_filings import Filing, parse_accepted
from quantlab.data.sources.sec_insider import InsiderTransactions, parse_ownership_document

START = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 19, tzinfo=dt.UTC)
TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def _routes() -> dict[str, object]:
    """Each filing in the index gets the recorded Form 4, dated to its own report
    period -- one real document stands in for three, but never claims a trade
    later than the filing that reports it."""
    index = load_json_fixture("sec_submissions_aapl.json")
    assert isinstance(index, dict)
    recent = index["filings"]["recent"]
    document = load_fixture("sec_form4_aapl.xml")
    routes: dict[str, object] = {"company_tickers.json": TICKERS, "submissions/": index}
    for form, accession, period in zip(
        recent["form"], recent["accessionNumber"], recent["reportDate"], strict=True
    ):
        if form == "4":
            routes[accession.replace("-", "")] = document.replace("2026-09-15", period)
    return routes


def _fetch(
    settings: Settings, start: dt.datetime = START, end: dt.datetime = END
) -> tuple[pl.DataFrame, list[str]]:
    source, requested = routed_source(InsiderTransactions, _routes(), settings)
    return source.fetch(["AAPL"], start, end), requested


# ------------------------------------------------------------------- known_at --
def test_the_acceptance_instant_is_used_as_given() -> None:
    """It is genuine UTC, so 22:30Z is 18:30 in Washington -- after the close."""
    accepted = parse_accepted("2026-09-17T22:30:24.000Z", "2026-09-17")
    assert accepted == dt.datetime(2026, 9, 17, 22, 30, 24, tzinfo=dt.UTC)


@pytest.mark.parametrize("raw", [None, "", "not a timestamp", "2026-09-17T22:30:24"])
def test_an_unusable_acceptance_instant_falls_back_to_the_end_of_the_filing_day(
    raw: str | None,
) -> None:
    """Including a NAIVE timestamp: guessing its zone could move a filing earlier,
    and the fallback can only ever move it later."""
    accepted = parse_accepted(raw, "2026-09-17")
    assert accepted == dt.datetime(2026, 9, 17, 21, 30, tzinfo=dt.UTC)  # 17:30 Eastern


def test_a_trade_is_dated_by_the_trade_and_knowable_from_the_filing(settings: Settings) -> None:
    frame, _ = _fetch(settings)
    first = frame.sort("known_at", "line").row(0, named=True)

    assert first["as_of"] < first["known_at"]
    assert (frame["known_at"] - frame["as_of"]).min() >= dt.timedelta(days=2)
    assert frame["symbol"].unique().to_list() == ["AAPL"]


def test_the_window_filters_on_acceptance_not_on_the_trade(settings: Settings) -> None:
    """A trade on the 15th reported on the 17th is not in a window ending the 16th
    -- on the 16th, nobody outside Apple could have known about it."""
    cutoff = dt.datetime(2026, 9, 16, tzinfo=dt.UTC)
    everything, _ = _fetch(settings)
    frame, requested = _fetch(settings, end=cutoff)

    assert everything["as_of"].max() == dt.datetime(2026, 9, 15, tzinfo=dt.UTC)
    assert frame["known_at"].max() < cutoff
    assert frame["as_of"].max() < dt.datetime(2026, 9, 15, tzinfo=dt.UTC)
    assert not [url for url in requested if "000114036126037020" in url]


# -------------------------------------------------------------------- content --
def test_only_ownership_forms_are_opened(settings: Settings) -> None:
    """The index also lists a Form 3, a 144, a 10-Q and an 8-K. None reports an
    insider transaction, and each would cost a request."""
    _, requested = _fetch(settings)
    documents = [url for url in requested if "/Archives/" in url]

    assert len(documents) == 3
    assert all(url.endswith("/form4.xml") for url in documents)


def test_the_raw_xml_is_requested_not_the_rendered_view() -> None:
    filing = Filing(320193, "4", "0001140361-26-037020", START, None, "xslF345X06/form4.xml")

    assert filing.raw_primary_document == "form4.xml"
    assert filing.url(filing.raw_primary_document) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000114036126037020/form4.xml"
    )


def test_compensation_lines_are_recorded_as_what_they_are() -> None:
    """One real Form 4: a sale, an option exercise on both sides of the table, and
    shares withheld for tax. Only the sale is a decision about the stock."""
    rows = parse_ownership_document("test", load_fixture("sec_form4_aapl.xml"), "fixture")
    codes = [row["transaction_code"] for row in rows]

    assert codes.count("S") == 1
    assert {"M", "F"} <= set(codes)
    assert [row["line"] for row in rows] == list(range(1, len(rows) + 1))

    sale = next(row for row in rows if row["transaction_code"] == "S")
    assert sale["shares"] == 1438
    assert sale["price"] == 330.19
    assert sale["acquired_disposed"] == "D"
    assert sale["shares_owned_after"] == 32914
    assert sale["is_officer"] and not sale["is_director"]
    assert sale["officer_title"] == "SVP, GC and Government Affairs"
    assert sale["planned_10b5_1"] is True
    assert sale["is_derivative"] is False


def test_a_price_given_only_in_a_footnote_is_null_not_zero() -> None:
    rows = parse_ownership_document("test", load_fixture("sec_form4_aapl.xml"), "fixture")
    exercise = next(row for row in rows if row["transaction_code"] == "M")

    assert exercise["price"] is None


def test_a_restated_holding_is_not_a_transaction() -> None:
    document = """<ownershipDocument>
      <issuer><issuerCik>1</issuerCik><issuerName>X</issuerName></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>2</rptOwnerCik>
        <rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId></reportingOwner>
      <nonDerivativeTable><nonDerivativeHolding>
        <securityTitle><value>Common Stock</value></securityTitle>
        <postTransactionAmounts><sharesOwnedFollowingTransaction><value>500</value>
        </sharesOwnedFollowingTransaction></postTransactionAmounts>
      </nonDerivativeHolding></nonDerivativeTable>
    </ownershipDocument>"""

    assert parse_ownership_document("test", document, "holding only") == []


def test_the_10b5_1_flag_is_null_when_the_form_did_not_ask() -> None:
    """The checkbox dates from April 2023. Before it, absence is not a "no"."""
    document = load_fixture("sec_form4_aapl.xml").replace("<aff10b5One>true</aff10b5One>", "")
    rows = parse_ownership_document("test", document, "old form")

    assert rows and all(row["planned_10b5_1"] is None for row in rows)


def test_joint_filers_are_all_named() -> None:
    document = load_fixture("sec_form4_aapl.xml")
    owner = document[document.index("<reportingOwner>") : document.index("</reportingOwner>")]
    second = owner.replace("Newstead Jennifer", "Fund GP LLC").replace(
        "<isOfficer>true</isOfficer>", "<isTenPercentOwner>1</isTenPercentOwner>"
    )
    document = document.replace(
        "</reportingOwner>", "</reportingOwner>" + second + "</reportingOwner>", 1
    )
    rows = parse_ownership_document("test", document, "joint")

    assert rows[0]["owner_name"] == "Newstead Jennifer; Fund GP LLC"
    assert rows[0]["is_officer"] and rows[0]["is_ten_percent_owner"]


# -------------------------------------------------------------------- failure --
def test_a_document_that_is_not_an_ownership_form_raises() -> None:
    with pytest.raises(SourceError, match="ownershipDocument"):
        parse_ownership_document("test", "<html><body>Rate limited</body></html>", "x")


def test_malformed_xml_raises_rather_than_yielding_nothing() -> None:
    with pytest.raises(SourceError, match="well-formed"):
        parse_ownership_document("test", "<ownershipDocument><issuer>", "truncated")


def test_an_entity_expansion_attack_is_refused() -> None:
    """These documents come off the network; the stdlib parser would expand this."""
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
        "<ownershipDocument>&b;</ownershipDocument>"
    )
    with pytest.raises(SourceError, match="well-formed"):
        parse_ownership_document("test", bomb, "bomb")


def test_a_trade_dated_after_its_own_report_is_dropped_not_fatal(settings: Settings) -> None:
    """A typo in one filing -- "2206" for "2026" -- must not cost the issuer."""
    routes = _routes()
    latest = "000114036126037020"
    assert isinstance(routes[latest], str)
    routes[latest] = routes[latest].replace(
        "<value>2026-09-15</value>", "<value>2206-09-15</value>", 1
    )
    source, _ = routed_source(InsiderTransactions, routes, settings)
    clean, _ = _fetch(settings)
    frame = source.fetch(["AAPL"], START, END)

    assert frame.height == clean.height - 1
    assert frame["as_of"].max() < END


def test_an_unknown_ticker_is_refused_not_guessed(settings: Settings) -> None:
    source, _ = routed_source(InsiderTransactions, {"company_tickers.json": TICKERS}, settings)
    with pytest.raises(SourceError, match="current registrants"):
        source.fetch(["LEHMQ"], START, END)


def test_a_window_with_no_filings_is_empty_not_an_error(settings: Settings) -> None:
    frame, requested = _fetch(settings, start=dt.datetime(2026, 9, 18, tzinfo=dt.UTC))

    assert frame.height == 0
    assert dict(frame.schema) == get_schema("insider_transactions").polars_schema
    assert not [url for url in requested if "/Archives/" in url]


# ----------------------------------------------------------------- pagination --
def test_older_index_pages_are_read_and_pages_before_the_window_are_not(
    settings: Settings,
) -> None:
    """`recent` holds about a thousand filings and the rest sit in numbered pages.
    Apple's insiders pass a thousand in under two years, so a reader of `recent`
    alone silently truncates history -- and one that reads every page wastes a
    request on each page that cannot hold anything in the window."""
    routes = _routes()
    index = routes["submissions/"]
    assert isinstance(index, dict)
    recent = index["filings"]["recent"]
    older = {column: [values[2]] for column, values in recent.items()}  # the 1 Sept Form 4
    older["accessionNumber"] = ["0001140361-26-030000"]
    older["acceptanceDateTime"] = ["2026-08-04T22:30:00.000Z"]
    older["filingDate"], older["reportDate"] = ["2026-08-04"], ["2026-08-03"]
    index["filings"]["files"] = [
        {
            "name": "CIK0000320193-submissions-001.json",
            "filingFrom": "2025-01-01",
            "filingTo": "2026-08-05",
        },
        {
            "name": "CIK0000320193-submissions-002.json",
            "filingFrom": "2020-01-01",
            "filingTo": "2024-12-31",
        },
    ]
    document = load_fixture("sec_form4_aapl.xml").replace("2026-09-15", "2026-08-03")
    paged = {
        "company_tickers.json": TICKERS,
        "submissions-001.json": older,
        "000114036126030000": document,
        **{key: value for key, value in routes.items() if key != "company_tickers.json"},
    }
    source, requested = routed_source(InsiderTransactions, paged, settings)
    frame = source.fetch(["AAPL"], START, END)

    assert "0001140361-26-030000" in frame["accession"].to_list()
    assert any("submissions-001.json" in url for url in requested)
    assert not any("submissions-002.json" in url for url in requested)
