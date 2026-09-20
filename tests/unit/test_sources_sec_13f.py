"""SEC Form 13F institutional holdings.

Four ways this dataset lies if read carelessly, each with a test: the unit of
`value` changed in 2023, one position is split over several lines, a holding is
knowable six weeks after the date it describes, and the information table's
filename is the filer's choice.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest
from tests.conftest import load_fixture, load_json_fixture
from tests.unit.test_sources import routed_source

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.sec_13f import (
    DOLLARS_FROM,
    InstitutionalHoldings,
    parse_information_table,
)

BERKSHIRE = "1067983"
START = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 19, tzinfo=dt.UTC)


def _index(*filings: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """A submissions payload. The default row is Berkshire's real June 2026 filing."""
    rows = filings or (
        ("13F-HR", "0001193125-26-352200", "2026-08-14", "2026-06-30", "2026-08-14T20:05:04.000Z"),
    )
    columns = ("form", "accessionNumber", "filingDate", "reportDate", "acceptanceDateTime")
    recent: dict[str, list[str]] = {
        column: [row[i] for row in rows] for i, column in enumerate(columns)
    }
    recent["primaryDocument"] = ["xslForm13F_X02/primary_doc.xml"] * len(rows)
    return {"name": "BERKSHIRE HATHAWAY INC", "filings": {"recent": recent, "files": []}}


def _routes(index: dict[str, Any] | None = None) -> dict[str, object]:
    return {
        "submissions/": index or _index(),
        "index.json": load_json_fixture("sec_13f_index_brk.json"),
        "primary_doc.xml": load_fixture("sec_13f_primary_brk.xml"),
        "56757.xml": load_fixture("sec_13f_infotable_brk.xml"),
    }


def _fetch(settings: Settings, routes: dict[str, object] | None = None) -> pl.DataFrame:
    source, _ = routed_source(InstitutionalHoldings, routes or _routes(), settings)
    return source.fetch([BERKSHIRE], START, END)


# ---------------------------------------------------------------- aggregation --
def test_a_position_split_across_lines_is_summed() -> None:
    """Berkshire reports for co-managers, so Ally Financial is on six lines. The
    first line alone is under half of the position."""
    positions = parse_information_table("test", load_fixture("sec_13f_infotable_brk.xml"), "brk")
    ally = next(p for p in positions if p["cusip"] == "02005N100")

    assert ally["lines"] == 6
    assert ally["shares"] > 12_561_737  # the first line
    assert len([p for p in positions if p["cusip"] == "02005N100"]) == 1


def test_a_put_is_not_merged_with_the_shares_it_is_written_on() -> None:
    table = """<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>X</nameOfIssuer><titleOfClass>COM</titleOfClass>
        <cusip>000000001</cusip><value>100</value>
        <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
      </infoTable>
      <infoTable><nameOfIssuer>X</nameOfIssuer><titleOfClass>COM</titleOfClass>
        <cusip>000000001</cusip><value>900</value>
        <shrsOrPrnAmt><sshPrnamt>90</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
        <putCall>Put</putCall>
      </infoTable></informationTable>"""
    positions = parse_information_table("test", table, "x")

    assert sorted((p["put_call"], p["shares"]) for p in positions) == [("NONE", 10), ("PUT", 90)]


def test_a_prefixed_namespace_parses_like_a_default_one() -> None:
    """Filers' software declares the namespace either way."""
    default = load_fixture("sec_13f_infotable_brk.xml")
    prefixed = (
        default.replace('xmlns="http://www.sec.gov', 'xmlns:ns1="http://www.sec.gov')
        .replace("<", "<ns1:")
        .replace("<ns1:/", "</ns1:")
    )
    assert parse_information_table("test", prefixed, "p") == parse_information_table(
        "test", default, "d"
    )


# ---------------------------------------------------------------------- dates --
def test_a_holding_is_knowable_at_acceptance_not_at_quarter_end(settings: Settings) -> None:
    frame = _fetch(settings)

    assert frame["as_of"].unique().to_list() == [dt.datetime(2026, 6, 30, tzinfo=dt.UTC)]
    assert frame["known_at"].unique().to_list() == [
        dt.datetime(2026, 8, 14, 20, 5, 4, tzinfo=dt.UTC)
    ]
    assert frame["symbol"].unique().to_list() == [BERKSHIRE]
    assert frame["manager_name"][0] == "BERKSHIRE HATHAWAY INC"


def test_a_filing_outside_the_window_is_not_opened(settings: Settings) -> None:
    source, requested = routed_source(InstitutionalHoldings, _routes(), settings)
    frame = source.fetch([BERKSHIRE], dt.datetime(2026, 8, 15, tzinfo=dt.UTC), END)

    assert frame.height == 0
    assert all("submissions/" in url for url in requested)


# ----------------------------------------------------------------------- unit --
def test_value_is_dollars_for_a_filing_made_under_the_new_form(settings: Settings) -> None:
    frame = _fetch(settings)
    ally = frame.filter(pl.col("cusip") == "02005N100").row(0, named=True)

    # About $46 a share: dollars. Read as thousands it would be $46,000 a share.
    assert 40 < ally["value_usd"] / ally["shares"] < 55


def test_value_is_scaled_for_a_filing_made_under_the_old_form(settings: Settings) -> None:
    """Same payload, filed the day before the unit changed: `value` is thousands."""
    day_before = DOLLARS_FROM - dt.timedelta(days=1)
    index = _index(
        (
            "13F-HR",
            "0001193125-26-352200",
            day_before.isoformat(),
            "2022-09-30",
            f"{day_before.isoformat()}T21:05:04.000Z",
        )
    )
    source, _ = routed_source(InstitutionalHoldings, _routes(index), settings)
    old = source.fetch([BERKSHIRE], dt.datetime(2022, 1, 1, tzinfo=dt.UTC), END)
    new = _fetch(settings)

    assert old["value_usd"].sum() == pytest.approx(new["value_usd"].sum() * 1000)


# ------------------------------------------------------------ table discovery --
def test_the_table_is_found_by_listing_the_filing_not_by_guessing_a_name(
    settings: Settings,
) -> None:
    source, requested = routed_source(InstitutionalHoldings, _routes(), settings)
    source.fetch([BERKSHIRE], START, END)
    archive = [url.rsplit("/", 1)[-1] for url in requested if "/Archives/" in url]

    assert archive == ["index.json", "56757.xml", "primary_doc.xml"]


def test_two_candidate_tables_is_an_error_not_a_choice(settings: Settings) -> None:
    routes = _routes()
    listing = load_json_fixture("sec_13f_index_brk.json")
    assert isinstance(listing, dict)
    listing["directory"]["item"].append({"name": "second.xml", "size": "1"})
    routes["index.json"] = listing

    with pytest.raises(SourceError, match="candidate information tables"):
        _fetch(settings, routes)


def test_a_filing_with_no_xml_table_is_skipped_not_guessed(settings: Settings) -> None:
    """Before mid-2013 the holdings were free text inside the submission."""
    routes = _routes()
    routes["index.json"] = {
        "directory": {"item": [{"name": "0001.txt"}, {"name": "primary_doc.xml"}]}
    }

    assert _fetch(settings, routes).height == 0


def test_a_position_missing_its_value_refuses_the_whole_table() -> None:
    table = load_fixture("sec_13f_infotable_brk.xml").replace("<value>577211815</value>", "", 1)
    with pytest.raises(SourceError, match="partial portfolio"):
        parse_information_table("test", table, "brk")


# ------------------------------------------------------------------- managers --
@pytest.mark.parametrize("symbol", ["0001067983", "Berkshire Hathaway", "1067983.0", ""])
def test_a_manager_must_be_a_cik_without_leading_zeros(settings: Settings, symbol: str) -> None:
    """Zero-padded, the symbol would not match what the lake already holds, and
    unquoted in YAML a padded number of octal digits becomes a different manager."""
    source, _ = routed_source(InstitutionalHoldings, _routes(), settings)
    with pytest.raises(ValueError, match="leading zeros"):
        source.fetch([symbol], START, END)


def test_an_amendment_records_whether_it_restates_or_adds(settings: Settings) -> None:
    index = _index(
        ("13F-HR/A", "0001193125-26-352200", "2026-08-20", "2026-06-30", "2026-08-20T20:05:04.000Z")
    )
    routes = _routes(index)
    primary = load_fixture("sec_13f_primary_brk.xml")
    routes["primary_doc.xml"] = primary.replace(
        "<isAmendment>false</isAmendment>",
        "<isAmendment>true</isAmendment><amendmentType>NEW HOLDINGS</amendmentType>",
    )
    frame = _fetch(settings, routes)

    assert frame["form"].unique().to_list() == ["13F-HR/A"]
    assert frame["amendment_type"].unique().to_list() == ["NEW HOLDINGS"]
