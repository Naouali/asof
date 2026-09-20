"""US House financial disclosures.

The transactions live in PDFs, so most of what can go wrong is layout: cells that
wrap, an asset-type code the member left off, a comment long enough to spill into
the next row. The rest is honesty about what could not be read -- a scanned report
must show up as a gap that can be measured, never as a member who did not trade.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import polars as pl
import pytest
from pypdf import PdfWriter
from tests.conftest import FIXTURES, load_fixture
from tests.unit.test_sources import routed_source

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.house_clerk import (
    EASTERN,
    NO_TICKER,
    HouseDisclosureFilings,
    HouseTrades,
    filed_known_at,
    parse_index,
    parse_ptr_text,
    pdf_text,
)

START = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
PETERS, SMUCKER, KHANNA_SCAN = "20034342", "20019182", "8221322"


def _index_zip(xml: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # The Clerk's file opens with a byte-order mark, which breaks a naive parse.
        body = xml if xml is not None else load_fixture("house_fd_index_2026.xml")
        archive.writestr("2026FD.xml", "\ufeff" + body)
        archive.writestr("2026FD.txt", "not used")
    return buffer.getvalue()


def _scan() -> bytes:
    """A PDF with pages and no text layer, which is what a scanned report is."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _report() -> bytes:
    return Path(FIXTURES / f"house_ptr_{PETERS}.pdf").read_bytes()


def _routes(**overrides: object) -> dict[str, object]:
    return {
        "2026FD.zip": _index_zip(),
        f"{PETERS}.pdf": _report(),
        f"{SMUCKER}.pdf": _report(),
        f"{KHANNA_SCAN}.pdf": _scan(),
        **overrides,
    }


# ---------------------------------------------------------------------- dates --
def test_a_report_is_knowable_at_the_end_of_the_day_it_was_filed() -> None:
    """The Clerk gives a date and no time. The end of the day can only make a
    report knowable late; midnight would make every report knowable a day early."""
    known = filed_known_at(dt.date(2026, 4, 29))

    assert known == dt.datetime(2026, 4, 30, 3, 59, 59, tzinfo=dt.UTC)  # 23:59:59 EDT
    assert filed_known_at(dt.date(2026, 1, 15)).hour == 4  # 23:59:59 EST


def test_a_trade_is_dated_by_the_trade_and_knowable_from_the_report(settings: Settings) -> None:
    source, _ = routed_source(HouseTrades, _routes(), settings)
    frame = source.fetch([], START, END)
    peters = frame.filter(pl.col("doc_id") == PETERS)

    assert peters["known_at"].unique().to_list() == [filed_known_at(dt.date(2026, 4, 29))]
    assert peters["as_of"].max() == dt.datetime(2026, 3, 30, tzinfo=dt.UTC)
    assert (peters["known_at"] - peters["as_of"]).min() > dt.timedelta(days=29)


def test_a_report_filed_today_is_not_knowable_until_the_day_is_over(settings: Settings) -> None:
    """Its `known_at` is tonight, which is in the future, so it waits for the next
    run rather than being stored as though the day had already ended."""
    now = dt.datetime.now(dt.UTC)
    today = now.astimezone(EASTERN).date()
    xml = load_fixture("house_fd_index_2026.xml").replace(
        "<FilingDate>4/29/2026</FilingDate>",
        f"<FilingDate>{today.month}/{today.day}/{today.year}</FilingDate>",
    )
    assert filed_known_at(today) > now

    source, _ = routed_source(HouseDisclosureFilings, {"FD.zip": _index_zip(xml)}, settings)
    frame = source.fetch([], dt.datetime(2026, 1, 1, tzinfo=dt.UTC), now)

    assert SMUCKER in frame["doc_id"].to_list()
    assert PETERS not in frame["doc_id"].to_list()


# ---------------------------------------------------------------------- index --
def test_the_index_lists_every_kind_of_document(settings: Settings) -> None:
    source, _ = routed_source(HouseDisclosureFilings, _routes(), settings)
    frame = source.fetch([], START, END)
    peters = frame.filter(pl.col("doc_id") == PETERS).row(0, named=True)

    assert sorted(frame["filing_type"].unique().to_list()) == ["A", "C", "P"]
    assert peters["symbol"] == "CA50"
    assert peters["last_name"] == "Peters"
    assert peters["chamber"] == "house"
    assert peters["filing_type_name"] == "periodic transaction report"
    assert peters["as_of"] == dt.datetime(2026, 4, 29, tzinfo=dt.UTC)
    assert peters["url"] == (
        f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/{PETERS}.pdf"
    )


def test_an_undated_entry_is_skipped_because_it_cannot_be_placed_in_time() -> None:
    entries = parse_index("test", _index_zip(), 2026)

    assert len(entries) == 5  # the fixture holds six; one withdrawal has no date
    assert all(entry.filing_type != "W" for entry in entries)


def test_a_filing_type_with_no_certain_meaning_gets_no_name(settings: Settings) -> None:
    xml = load_fixture("house_fd_index_2026.xml").replace(
        "<FilingType>C</FilingType>", "<FilingType>D</FilingType>"
    )
    source, _ = routed_source(HouseDisclosureFilings, {"2026FD.zip": _index_zip(xml)}, settings)
    frame = source.fetch([], START, END)

    assert frame.filter(pl.col("filing_type") == "D")["filing_type_name"].to_list() == [None]


def test_an_index_entry_without_a_document_id_is_a_layout_change() -> None:
    xml = load_fixture("house_fd_index_2026.xml").replace(f"<DocID>{PETERS}</DocID>", "<DocID />")
    with pytest.raises(SourceError, match="layout"):
        parse_index("test", _index_zip(xml), 2026)


# -------------------------------------------------------------------- the gap --
def test_a_scanned_report_is_a_measurable_gap_not_a_member_who_did_not_trade(
    settings: Settings,
) -> None:
    """The index lists three transaction reports. One is a scan. It must be
    findable as listed-but-unread, which is the whole reason the index is stored."""
    filings, _ = routed_source(HouseDisclosureFilings, _routes(), settings)
    trades, _ = routed_source(HouseTrades, _routes(), settings)
    listed = filings.fetch([], START, END).filter(pl.col("filing_type") == "P")
    read = trades.fetch([], START, END)

    unread = set(listed["doc_id"]) - set(read["doc_id"])
    assert set(listed["doc_id"]) == {PETERS, SMUCKER, KHANNA_SCAN}
    assert unread == {KHANNA_SCAN}


def test_only_transaction_reports_are_downloaded(settings: Settings) -> None:
    source, requested = routed_source(HouseTrades, _routes(), settings)
    source.fetch([], START, END)
    documents = sorted(url.rsplit("/", 1)[-1] for url in requested if url.endswith(".pdf"))

    assert documents == sorted(f"{doc}.pdf" for doc in (PETERS, SMUCKER, KHANNA_SCAN))


def test_every_report_unreadable_means_the_layout_changed(settings: Settings) -> None:
    routes = _routes(**{f"{PETERS}.pdf": _scan(), f"{SMUCKER}.pdf": _scan()})
    source, _ = routed_source(HouseTrades, routes, settings)

    with pytest.raises(SourceError, match="layout has changed"):
        source.fetch([], START, END)


def test_a_payload_that_is_not_a_pdf_reads_as_unreadable_not_as_a_crash() -> None:
    assert pdf_text(b"<html>Access Denied</html>") is None
    assert parse_ptr_text(pdf_text(_scan()) or "") == ([], 0)


# --------------------------------------------------------------------- layout --
def test_a_real_report_parses_completely() -> None:
    """Spouse-owned funds and municipal bonds: no tickers, an open-ended amount
    bracket, a partial sale, and a table that runs onto a second page."""
    rows, date_pairs = parse_ptr_text(pdf_text(_report()) or "")

    assert len(rows) == date_pairs == 10
    assert [row["line"] for row in rows] == list(range(1, 11))
    assert {row["owner"] for row in rows} == {"SP"}
    assert {row["symbol"] for row in rows} == {NO_TICKER}
    assert {row["filing_status"] for row in rows} == {"new"}
    assert {row["transaction_type"] for row in rows} == {"purchase", "sale", "sale_partial"}
    assert rows[0]["asset"] == "Allocate Alpha Fund II LP"
    assert rows[0]["asset_type"] == "OT"
    assert (rows[0]["amount_min"], rows[0]["amount_max"]) == (1001.0, 15000.0)


def test_field_labels_never_leak_into_an_asset_name() -> None:
    """The labels are small caps, which extract as a capital and a run of NUL
    bytes. A terminal draws NUL as a space, so a leak is invisible on screen."""
    text = pdf_text(_report()) or ""
    rows, _ = parse_ptr_text(text)

    assert "\x00" not in text
    assert all(":" not in (row["asset"] or "") for row in rows)
    assert all(len(row["asset"] or "") < 60 for row in rows)


def test_an_open_ended_bracket_has_no_upper_bound() -> None:
    rows, _ = parse_ptr_text(pdf_text(_report()) or "")
    over = next(row for row in rows if "Over" in row["amount_text"])

    assert over["amount_text"] == "Spouse/DC Over $1,000,000"
    assert over["amount_min"] == 1_000_000
    assert over["amount_max"] is None


def test_a_transaction_without_an_asset_type_code_is_still_a_transaction() -> None:
    """Members do leave the code off. Requiring it lost one row in twenty-six."""
    rows, date_pairs = parse_ptr_text(load_fixture("house_ptr_20034585.txt"))
    uncoded = [row for row in rows if row["asset_type"] is None]

    assert len(rows) == date_pairs == 26
    assert [row["symbol"] for row in uncoded] == ["IFNNY"]
    assert uncoded[0]["owner"] == "JT"
    assert uncoded[0]["transaction_type"] == "purchase"


def test_tickers_come_from_the_end_of_the_asset_name() -> None:
    rows, _ = parse_ptr_text(load_fixture("house_ptr_20034585.txt"))
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["IBM"]["asset"].endswith("(IBM)")
    assert by_symbol["NTES"]["asset"].startswith("NetEase, Inc.")
    assert NO_TICKER not in by_symbol


def test_a_wrapped_comment_does_not_become_part_of_the_next_asset() -> None:
    """A comment long enough to wrap leaves continuation lines that look, to a
    line-based reader, exactly like the first line of the next asset's name."""
    rows, date_pairs = parse_ptr_text(load_fixture("house_ptr_20034202.txt"))

    assert len(rows) == date_pairs == 2
    assert [row["asset"] for row in rows] == [
        "Parkland, PA School District Municipal Bond",
        "Pennsylvania State Turnpike Commission Bond",
    ]


def test_a_line_holding_a_transaction_is_never_discarded_as_a_continuation() -> None:
    """Even straight after a full-width comment: losing a name is tolerable,
    losing a trade is not."""
    comment = "C : " + "x" * 120
    table = (
        "ID Owner Asset Transaction Type Date Notification Date Amount Cap. Gains > $200?\n"
        "First Co (AAA) [ST] P 01/05/2026 01/06/2026 $1,001 - $15,000\n"
        "F S : New\n"
        f"{comment}\n"
        "Second Co (BBB) [ST] S 01/07/2026 01/08/2026 $15,001 - $50,000\n"
        "F S : Amended\n"
    )
    rows, date_pairs = parse_ptr_text(table)

    assert len(rows) == date_pairs == 2
    assert [row["symbol"] for row in rows] == ["AAA", "BBB"]
    assert [row["filing_status"] for row in rows] == ["new", "amended"]


def test_a_shortfall_against_the_dates_on_the_page_is_detectable() -> None:
    """The second row has a transaction type this parser has never seen."""
    table = (
        "ID Owner Asset Transaction Type Date Notification Date Amount Cap. Gains > $200?\n"
        "First Co (AAA) [ST] P 01/05/2026 01/06/2026 $1,001 - $15,000\n"
        "Second Co (BBB) [ST] Z 01/07/2026 01/08/2026 $15,001 - $50,000\n"
    )
    rows, date_pairs = parse_ptr_text(table)

    assert (len(rows), date_pairs) == (1, 2)


# -------------------------------------------------------------------- filters --
def test_a_symbol_filter_keeps_only_those_tickers(settings: Settings) -> None:
    source, _ = routed_source(HouseTrades, _routes(), settings)

    assert source.fetch(["AAPL"], START, END).height == 0
    assert source.fetch([NO_TICKER], START, END).height == 20  # two copies of one report


def test_the_member_is_named_as_the_report_names_them(settings: Settings) -> None:
    source, _ = routed_source(HouseTrades, _routes(), settings)
    frame = source.fetch([], START, END).filter(pl.col("doc_id") == PETERS)

    assert frame["member"].unique().to_list() == ["Hon. Scott H. Peters"]
    assert frame["state_district"].unique().to_list() == ["CA50"]
