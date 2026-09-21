"""Federal contract actions.

The arithmetic is trivial and the parsing is a CSV. What can go wrong is the
claim each row makes: when anyone could have known about it -- three months after
the fact, for the Pentagon -- and which listed company it belongs to, in a record
that knows legal entities and has never heard of a ticker.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from tests.conftest import load_fixture

from quantlab.config import Settings
from quantlab.data.http import HttpClient, RateLimiter, SourceError
from quantlab.data.sources.usaspending import (
    DOWNLOAD_URL,
    ContractActions,
    known_at,
    load_contractors,
    parse_transactions,
)

START = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 21, tzinfo=dt.UTC)
STATUS_URL = "https://api.usaspending.gov/api/v2/download/status?file_name=t.zip"
FILE_URL = "https://files.usaspending.gov/generated_downloads/t.zip"


def _zip(name: str = "Contracts_PrimeTransactions_2026-09-21_1.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # The real file opens with a byte-order mark, which breaks a naive header read.
        archive.writestr(name, "﻿" + load_fixture("usaspending_transactions_lmt.csv"))
        archive.writestr("Contracts_Subawards_2026-09-21_1.csv", "not,used\n")
    return buffer.getvalue()


class Treasury:
    """The download service: a ticket, a status that is not ready at first, a file."""

    def __init__(self, *, states: tuple[str, ...] = ("running", "finished")) -> None:
        self.states = list(states)
        self.queries: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        import json

        url = str(request.url)
        if url == DOWNLOAD_URL:
            self.queries.append(json.loads(request.content))
            return httpx.Response(200, json={"status_url": STATUS_URL, "file_url": FILE_URL})
        if url == STATUS_URL:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return httpx.Response(200, json={"status": state, "message": "it broke"})
        if url == FILE_URL:
            return httpx.Response(200, content=_zip())
        return httpx.Response(404, text="not routed")

    def fetcher(self, settings: Settings, **options: Any) -> ContractActions:
        http = HttpClient(
            ContractActions.spec,
            settings=settings,
            client=httpx.Client(transport=httpx.MockTransport(self)),
            rate_limiter=RateLimiter(1000.0),
        )
        return ContractActions(client=http, settings=settings, poll_seconds=0, **options)


@pytest.fixture
def actions(settings: Settings) -> pl.DataFrame:
    return Treasury().fetcher(settings).fetch(["LMT"], START, END).sort("as_of")


# ------------------------------------------------------------------- known_at --
def test_a_civilian_action_is_knowable_two_days_after_it_is_reported() -> None:
    reported = dt.datetime(2026, 6, 10, 16, 47, tzinfo=dt.UTC)
    assert known_at(dt.date(2026, 6, 10), reported, defense=False) == dt.datetime(
        2026, 6, 12, 16, 47, tzinfo=dt.UTC
    )


def test_a_pentagon_action_waits_out_the_embargo() -> None:
    reported = dt.datetime(2026, 6, 9, 11, 56, tzinfo=dt.UTC)
    # Reported two days BEFORE the action date: the clock starts at the action.
    assert known_at(dt.date(2026, 6, 11), reported, defense=True) == dt.datetime(
        2026, 9, 11, tzinfo=dt.UTC
    )


def test_an_action_reported_late_is_knowable_late() -> None:
    reported = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)
    assert known_at(dt.date(2026, 6, 1), reported, defense=False) == dt.datetime(
        2026, 8, 3, tzinfo=dt.UTC
    )


# -------------------------------------------------------------------- parsing --
def test_the_download_is_read_past_its_byte_order_mark() -> None:
    records = list(parse_transactions("test", _zip()))
    assert len(records) == 6
    assert records[0]["contract_transaction_unique_key"]  # the first header survived


def test_a_download_with_no_transactions_file_is_refused() -> None:
    with pytest.raises(SourceError, match="no transactions file"):
        list(parse_transactions("test", _zip(name="Something_Else.csv")))
    with pytest.raises(SourceError, match="not a zip"):
        list(parse_transactions("test", b"<html>busy</html>"))


# -------------------------------------------------------------------- fetcher --
def test_rows_are_filed_under_the_listed_parent_not_the_entity_that_signed(
    actions: pl.DataFrame,
) -> None:
    assert set(actions["symbol"].to_list()) == {"LMT"}
    assert "SIKORSKY AIRCRAFT CORPORATION" in actions["recipient_name"].to_list()
    assert set(actions["source"].to_list()) == {"usaspending"}


def test_a_joint_venture_belongs_to_no_ticker(actions: pl.DataFrame) -> None:
    assert not [name for name in actions["recipient_name"].to_list() if "JOINT VENTURE" in name]


def test_small_actions_are_dropped_and_the_minimum_is_a_choice(settings: Settings) -> None:
    default = Treasury().fetcher(settings).fetch(["LMT"], START, END)
    everything = Treasury().fetcher(settings, min_obligation=0).fetch(["LMT"], START, END)

    assert default.height == 4
    assert everything.height == 5  # the $402,126 Homeland Security modification


def test_money_taken_back_is_negative_and_is_kept(actions: pl.DataFrame) -> None:
    assert actions["obligation_usd"].min() == pytest.approx(-12_475_741, abs=1)


def test_the_pentagon_is_dated_by_its_embargo(actions: pl.DataFrame) -> None:
    gps = actions.filter(pl.col("modification_number") == "P00200").row(0, named=True)

    assert gps["defense"] is True
    assert gps["as_of"] == dt.datetime(2026, 6, 11, tzinfo=dt.UTC)
    assert gps["known_at"] == dt.datetime(2026, 9, 11, tzinfo=dt.UTC)
    assert gps["obligation_usd"] == pytest.approx(514_412_527.67)
    assert gps["url"].startswith("https://www.usaspending.gov/award/")


def test_an_action_still_under_embargo_is_not_returned(settings: Settings) -> None:
    """Read on 1 September, the June defence actions are ten days from public."""
    early = (
        Treasury().fetcher(settings).fetch(["LMT"], START, dt.datetime(2026, 9, 1, tzinfo=dt.UTC))
    )

    assert early.height == 2
    assert early["defense"].to_list() == [False, False]


def test_the_query_names_parents_by_identifier_and_looks_back(settings: Settings) -> None:
    treasury = Treasury()
    treasury.fetcher(settings).fetch(["LMT"], START, END)

    (query,) = treasury.queries
    filters = query["filters"]
    assert filters["recipient_search_text"] == ["CWM4UN76ZQW8", "ZFN2JJXBLZT3"]
    assert filters["award_type_codes"] == ["A", "B", "C", "D"]
    assert filters["award_amounts"] == [{"lower_bound": 1_000_000.0}]
    assert "initial_report_date" in query["columns"] and len(query["columns"]) == 20
    # 200 days before the window: an embargoed action surfaces long after its date.
    assert filters["time_period"] == [{"start_date": "2025-06-15", "end_date": "2026-09-21"}]


def test_with_no_minimum_the_service_is_not_asked_to_filter_by_award(settings: Settings) -> None:
    treasury = Treasury()
    treasury.fetcher(settings, min_obligation=0).fetch(["LMT"], START, END)
    assert "award_amounts" not in treasury.queries[0]["filters"]


def test_an_action_returned_by_two_downloads_is_kept_once(settings: Settings) -> None:
    """Thirty-seven companies need four downloads, and the text search lets one
    company's records answer another's query. The real first ingest held 533
    actions twice."""
    treasury = Treasury()
    everyone = treasury.fetcher(settings).fetch([], START, END)

    assert len(treasury.queries) > 1  # every download serves the same fixture file
    assert everyone.height == 4
    assert everyone["transaction_key"].n_unique() == 4


def test_a_ticker_with_no_recorded_parent_is_refused(settings: Settings) -> None:
    with pytest.raises(SourceError, match="no parent-company identifiers"):
        Treasury().fetcher(settings).fetch(["AAPL"], START, END)


def test_a_failed_download_is_an_error(settings: Settings) -> None:
    with pytest.raises(SourceError, match="the download failed: 'it broke'"):
        Treasury(states=("failed",)).fetcher(settings).fetch(["LMT"], START, END)


def test_a_download_that_never_finishes_gives_up(settings: Settings) -> None:
    stuck = Treasury(states=("running",)).fetcher(settings, max_wait_seconds=0)
    with pytest.raises(SourceError, match="still 'running'"):
        stuck.fetch(["LMT"], START, END)


# ------------------------------------------------------------------------ map --
def test_the_shipped_list_loads_and_no_identifier_is_claimed_twice(repo_root: Path) -> None:
    contractors = load_contractors(repo_root / "configs" / "contractors.yaml")

    claimed = [uei for contractor in contractors for uei in contractor.parent_ueis]
    assert len(contractors) >= 30
    assert len(claimed) == len(set(claimed))
    assert {"LMT", "GD", "PLTR"} <= {contractor.ticker for contractor in contractors}


def _list(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "contractors.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_an_identifier_claimed_by_two_companies_is_refused(tmp_path: Path) -> None:
    """It would file one company's contracts under another's ticker."""
    body = (
        "contractors:\n"
        "  - {ticker: AAA, name: A, parent_ueis: [ZFN2JJXBLZT3]}\n"
        "  - {ticker: BBB, name: B, parent_ueis: [ZFN2JJXBLZT3]}\n"
    )
    with pytest.raises(SourceError, match="claimed by both AAA and BBB"):
        load_contractors(_list(tmp_path, body))


@pytest.mark.parametrize(
    ("body", "complaint"),
    [
        ("contractors: []\n", "lists no contractors"),
        (
            "contractors:\n  - {ticker: AAA, name: A, parent_ueis: [SHORT]}\n",
            "not a 12-character UEI",
        ),
        ("contractors:\n  - {ticker: AAA, name: A}\n", "lacks ticker, name or parent_ueis"),
        (
            "contractors:\n  - {ticker: AAA, name: A, parent_ueis: []}\n",
            "lists no parent identifier",
        ),
        ("contractors: [\n", "not valid YAML"),
    ],
)
def test_a_malformed_list_is_refused_with_the_reason(
    tmp_path: Path, body: str, complaint: str
) -> None:
    with pytest.raises(SourceError, match=complaint):
        load_contractors(_list(tmp_path, body))


def test_without_the_list_nothing_can_be_fetched(settings: Settings, tmp_path: Path) -> None:
    fetcher = Treasury().fetcher(settings, contractors=tmp_path / "missing.yaml")
    with pytest.raises(SourceError, match="the only link between the two"):
        fetcher.fetch(["LMT"], START, END)


def test_a_job_can_point_at_a_list_of_its_own(settings: Settings, tmp_path: Path) -> None:
    body = "contractors:\n  - {ticker: LOCK, name: Renamed, parent_ueis: [ZFN2JJXBLZT3]}\n"
    found = Treasury().fetcher(settings, contractors=_list(tmp_path, body)).fetch([], START, END)
    assert set(found["symbol"].to_list()) == {"LOCK"}
