"""US Senate financial disclosures.

The Senate publishes real tables, so parsing is the easy part. What can go wrong
is everything around it: a site that answers only from inside the United States,
an agreement that has to be accepted before anything can be searched, a mirror
standing in for the site everywhere else, and paper reports that must show up as
a gap that can be measured, never as a senator who did not trade.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from tests.conftest import load_fixture, load_json_fixture
from tests.unit.test_sources import routed_source

from quantlab.config import Settings
from quantlab.data.http import HttpClient, RateLimiter, SourceError
from quantlab.data.sources.house_clerk import NO_TICKER
from quantlab.data.sources.senate_efd import (
    DATA_URL,
    HOME_URL,
    SenateDisclosureFilings,
    SenateSession,
    SenateTrades,
    parse_index_page,
    parse_report,
    update_mirror,
)

START = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 20, tzinfo=dt.UTC)
ARMSTRONG = "b999bc0e-3eb0-4ca9-ab07-8e8f2e04b41f"
BOOZMAN_AMENDMENT = "51455bcd-4966-4e77-b481-09897ada81ae"
BLUMENTHAL_PAPER = "929216d5-5dbd-429c-858c-1e9332924627"

HOME = '<form><input type="hidden" name="csrfmiddlewaretoken" value="home-token"></form>'
SEARCH = '<form><input type="hidden" name="csrfmiddlewaretoken" value="search-token"></form>'


class Senate:
    """The Senate's site, as far as the reader can tell. Records what it was sent."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        form = dict(httpx.QueryParams(request.content.decode()))
        self.calls.append((request.method, url, form))
        if self.refuse:
            return httpx.Response(403, text="Access Denied")
        if url == HOME_URL:
            return httpx.Response(200, text=SEARCH if request.method == "POST" else HOME)
        if url == DATA_URL:
            return httpx.Response(200, json=load_json_fixture("senate_ptr_index.json"))
        for report_id in (ARMSTRONG, BOOZMAN_AMENDMENT):
            if report_id in url:
                return httpx.Response(200, text=load_fixture(f"senate_ptr_{report_id[:8]}.html"))
        return httpx.Response(404, text="not routed")

    def client(self, settings: Settings) -> HttpClient:
        return HttpClient(
            SenateTrades.spec,
            settings=settings,
            client=httpx.Client(transport=httpx.MockTransport(self)),
            rate_limiter=RateLimiter(1000.0),
        )


def _mirror(tmp_path: Path, settings: Settings) -> dict[str, Any]:
    """A mirror file built the way the daily job builds it."""
    update_mirror(SenateSession(Senate().client(settings), "test"), tmp_path, dt.date(2026, 1, 1))
    return json.loads((tmp_path / "ptr-2026.json").read_text("utf-8"))


# ---------------------------------------------------------------------- parsing --
def test_the_index_tells_paper_reports_and_amendments_apart() -> None:
    refs, total = parse_index_page("test", load_json_fixture("senate_ptr_index.json"))

    assert total == 3
    assert [ref.report_id for ref in refs] == [ARMSTRONG, BLUMENTHAL_PAPER, BOOZMAN_AMENDMENT]
    assert [ref.paper for ref in refs] == [False, True, False]
    assert [ref.amended for ref in refs] == [False, False, True]
    assert refs[0].filed == dt.date(2026, 9, 17)
    assert refs[0].url == f"https://efdsearch.senate.gov/search/view/ptr/{ARMSTRONG}/"
    assert refs[0].title == "Periodic Transaction Report for 09/17/2026"


def test_an_index_in_an_unknown_shape_is_refused() -> None:
    with pytest.raises(SourceError, match="not the expected JSON"):
        parse_index_page("test", {"rows": []})
    with pytest.raises(SourceError, match=r"not\s+five"):
        parse_index_page("test", {"data": [["only", "three", "cells"]]})


def test_a_report_gives_its_filing_minute_and_its_own_count() -> None:
    report = parse_report("test", load_fixture("senate_ptr_b999bc0e.html"), "fixture")

    # "Filed 09/17/2026 @ 8:55 AM", in Washington, which is on daylight time.
    assert report.filed_at == dt.datetime(2026, 9, 17, 12, 55, tzinfo=dt.UTC)
    assert report.declared == 4
    assert [row["line"] for row in report.rows] == ["4", "3", "2", "1"]


def test_cells_are_kept_as_the_page_showed_them() -> None:
    report = parse_report("test", load_fixture("senate_ptr_b999bc0e.html"), "fixture")
    option, no_ticker = report.rows[0], report.rows[1]

    assert option["ticker"] == "WMB"
    assert option["asset"] == (
        "Williams Companies, Inc. (The) Common Stock Option Type: Call "
        "Strike price: $75.00 Expires: 2026-08-21"
    )
    assert no_ticker["ticker"] == "--"
    assert no_ticker["comment"] == "Sale due to corporate transaction"


def test_a_row_with_the_wrong_number_of_cells_is_refused() -> None:
    page = "<table><tbody><tr><td>1</td><td>01/02/2026</td></tr></tbody></table>"
    with pytest.raises(SourceError, match="2 cells, not 9"):
        parse_report("test", page, "fixture")


# ------------------------------------------------------------ the Senate itself --
def test_the_agreement_is_accepted_before_anything_is_searched(settings: Settings) -> None:
    senate = Senate()
    refs = list(SenateSession(senate.client(settings), "test").index(dt.date(2026, 1, 1)))

    assert len(refs) == 3
    (_, _, _), (method, url, agreed), (_, search_url, search) = senate.calls
    assert (method, url) == ("POST", HOME_URL)
    assert agreed == {"csrfmiddlewaretoken": "home-token", "prohibition_agreement": "1"}
    assert search_url == DATA_URL
    assert search["csrfmiddlewaretoken"] == "search-token"
    assert search["report_types"] == "[11]"
    assert search["submitted_start_date"] == "01/01/2026 00:00:00"


def test_a_refusal_is_explained_as_the_geographic_block_it_is(settings: Settings) -> None:
    fetcher = SenateTrades(
        client=Senate(refuse=True).client(settings), settings=settings, direct=True
    )
    with pytest.raises(SourceError, match="inside the United States"):
        fetcher.fetch([], START, END)


def test_reading_directly_yields_the_same_rows_as_the_mirror(
    settings: Settings, tmp_path: Path
) -> None:
    direct = SenateTrades(client=Senate().client(settings), settings=settings, direct=True)
    mirrored, _ = routed_source(
        SenateTrades, {"ptr-2026.json": _mirror(tmp_path, settings)}, settings
    )

    columns = ["symbol", "doc_id", "line", "as_of", "known_at", "transaction_type", "amount_min"]
    by_line = ["doc_id", "line"]  # the Senate lists newest first, the mirror oldest first
    assert (
        direct.fetch([], START, END)
        .select(columns)
        .sort(by_line)
        .equals(mirrored.fetch([], START, END).select(columns).sort(by_line))
    )


# ------------------------------------------------------------------- the mirror --
def test_the_mirror_never_asks_for_a_report_twice(settings: Settings, tmp_path: Path) -> None:
    first = _mirror(tmp_path, settings)
    assert [r["report_id"] for r in first["reports"]] == [
        BOOZMAN_AMENDMENT,
        BLUMENTHAL_PAPER,
        ARMSTRONG,
    ]
    paper = first["reports"][1]
    assert paper["paper"] is True and paper["rows"] == [] and paper["filed_at"] is None

    senate = Senate()
    added = update_mirror(
        SenateSession(senate.client(settings), "test"), tmp_path, dt.date(2026, 1, 1)
    )

    assert added == 0
    assert not [url for _, url, _ in senate.calls if "/view/" in url]
    # Unchanged, first-seen stamps included: a quiet day must not make a commit.
    assert json.loads((tmp_path / "ptr-2026.json").read_text("utf-8")) == first


def test_a_year_the_mirror_does_not_hold_is_an_error_not_an_empty_year(settings: Settings) -> None:
    fetcher, _ = routed_source(SenateTrades, {}, settings)
    with pytest.raises(SourceError, match="the mirror has no file for 2026"):
        fetcher.fetch([], START, END)


def test_a_file_that_is_not_a_mirror_is_refused(settings: Settings) -> None:
    fetcher, _ = routed_source(SenateTrades, {"ptr-2026.json": {"filings": []}}, settings)
    with pytest.raises(SourceError, match="not a mirror file"):
        fetcher.fetch([], START, END)


# ---------------------------------------------------------------------- fetchers --
@pytest.fixture
def trades(settings: Settings, tmp_path: Path) -> pl.DataFrame:
    fetcher, _ = routed_source(
        SenateTrades, {"ptr-2026.json": _mirror(tmp_path, settings)}, settings
    )
    return fetcher.fetch([], START, END)


def test_trades_are_dated_by_the_trade_and_known_by_the_filing_minute(trades: pl.DataFrame) -> None:
    armstrong = trades.filter(pl.col("doc_id") == ARMSTRONG).sort("line")

    assert armstrong.height == 4
    assert armstrong["as_of"].to_list()[0] == dt.datetime(2026, 8, 19, tzinfo=dt.UTC)
    assert set(armstrong["known_at"].to_list()) == {dt.datetime(2026, 9, 17, 12, 55, tzinfo=dt.UTC)}
    assert set(armstrong["chamber"].to_list()) == {"senate"}
    assert set(armstrong["member"].to_list()) == {"Alan Armstrong"}
    assert set(armstrong["source"].to_list()) == {"senate_efd"}


def test_the_senates_words_become_the_houses_codes(trades: pl.DataFrame) -> None:
    armstrong = trades.filter(pl.col("doc_id") == ARMSTRONG).sort("line")

    assert armstrong["transaction_type"].to_list() == ["purchase", "exchange", "sale", "purchase"]
    assert set(armstrong["owner"].to_list()) == {"JT"}
    assert armstrong["amount_min"].to_list() == [1001.0, 1001.0, 1001.0, 15001.0]
    assert armstrong["amount_max"].to_list() == [15000.0, 15000.0, 15000.0, 50000.0]


def test_a_ticker_left_blank_is_read_from_the_asset_except_on_an_exchange(
    trades: pl.DataFrame,
) -> None:
    armstrong = trades.filter(pl.col("doc_id") == ARMSTRONG).sort("line")
    # Line 3 names "(EA)" in the asset. Line 2 is an exchange naming two securities.
    assert armstrong["symbol"].to_list() == ["WMB", NO_TICKER, "EA", "WMB"]


def test_an_amendment_is_marked_as_one(trades: pl.DataFrame) -> None:
    status = dict(trades.group_by("doc_id").agg(pl.col("filing_status").first()).iter_rows())
    assert status == {ARMSTRONG: "new", BOOZMAN_AMENDMENT: "amended"}


def test_a_paper_report_is_listed_and_yields_no_trades(settings: Settings, tmp_path: Path) -> None:
    mirror = {"ptr-2026.json": _mirror(tmp_path, settings)}
    trades = routed_source(SenateTrades, mirror, settings)[0].fetch([], START, END)
    filings = routed_source(SenateDisclosureFilings, mirror, settings)[0].fetch([], START, END)

    assert BLUMENTHAL_PAPER not in trades["doc_id"].to_list()
    assert sorted(filings["doc_id"].to_list()) == sorted(
        [ARMSTRONG, BOOZMAN_AMENDMENT, BLUMENTHAL_PAPER]
    )
    paper = filings.filter(pl.col("doc_id") == BLUMENTHAL_PAPER).row(0, named=True)
    assert paper["filing_type"] == "P" and paper["chamber"] == "senate"
    # No filing minute on paper: knowable at the end of that day in Washington.
    assert paper["known_at"] == dt.datetime(2026, 9, 1, 3, 59, 59, tzinfo=dt.UTC)


def test_nothing_filed_after_the_window_is_returned(settings: Settings, tmp_path: Path) -> None:
    fetcher, _ = routed_source(
        SenateTrades, {"ptr-2026.json": _mirror(tmp_path, settings)}, settings
    )
    early = fetcher.fetch([], START, dt.datetime(2026, 9, 1, tzinfo=dt.UTC))
    assert set(early["doc_id"].to_list()) == {BOOZMAN_AMENDMENT}


def test_an_unknown_transaction_type_is_refused_not_guessed(
    settings: Settings, tmp_path: Path
) -> None:
    mirror = _mirror(tmp_path, settings)
    mirror["reports"][-1]["rows"][0]["type"] = "Gift"
    fetcher, _ = routed_source(SenateTrades, {"ptr-2026.json": mirror}, settings)
    with pytest.raises(SourceError, match="unknown transaction type 'Gift'"):
        fetcher.fetch([], START, END)
