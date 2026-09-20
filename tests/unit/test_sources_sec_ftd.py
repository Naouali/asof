"""SEC fails-to-deliver data.

The rows are trivial to parse. The tests are about the two things that are not:
when a row became knowable, which only the server's Last-Modified header records,
and whether the file that arrived is the whole file.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import httpx
import pytest

from quantlab.config import Settings
from quantlab.data.http import HttpClient, SourceError
from quantlab.data.sources.sec_ftd import (
    INDEX_URL,
    FailsToDeliver,
    half_months,
    parse_fails_file,
)

# The first rows of the real cnsfails202608a file, with the trailer recomputed for
# the rows kept.
ROWS = (
    "20260803|B5950S113|MDXH|8960|MDXHEALTH SA SHS NEW(BELGIUM) |0.42",
    "20260803|D18190898|DB|31|DEUTSCHE BANK AG NAMEN AKT (DE|36.70",
    "20260803|F92124100|TTE|14|TOTALENERGIES SE ORDINARY SHAR|87.86",
    "20260814|98985Y108|ZYME|282|ZYMEWORKS INC COM(DE)|25.76",
    "20260814|000000001|NOPX|5|NO PRICE CORP|.",
)
HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"
POSTED = "Mon, 31 Aug 2026 11:54:19 GMT"


def _file(rows: tuple[str, ...] = ROWS, *, count: int | None = None) -> bytes:
    quantity = sum(int(row.split("|")[3]) for row in rows)
    body = [
        HEADER,
        *rows,
        f"Trailer record count {len(rows) if count is None else count}",
        f"Trailer total quantity of shares {quantity}",
    ]
    return "\r\n".join(body).encode("latin-1")


def _zip(payload: bytes, member: str = "cnsfails202608a.txt") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


START = dt.datetime(2026, 8, 20, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 5, tzinfo=dt.UTC)
LONG_AGO = "Tue, 30 Jan 2024 17:24:34 GMT"


def _sec(
    settings: Settings,
    files: dict[str, tuple[bytes, str | None]],
    *,
    missing: tuple[str, ...] = (),
    moved: tuple[str, ...] = (),
) -> tuple[FailsToDeliver, list[str]]:
    """A fetcher talking to a fake SEC on which every half-month file exists.

    ``files`` gives the ones a test cares about, as (zip, Last-Modified). Every
    other period answers with a valid file posted long ago, which is what the
    real server does for the periods a fetch has to look back over. ``missing``
    names periods the index page does not list; ``moved`` names ones it lists in
    another directory, as the SEC really does.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == INDEX_URL:
            return httpx.Response(200, text=_index_page(missing, moved=moved))
        name = str(request.url).rsplit("/", 1)[-1].removesuffix(".zip")
        requested.append(str(request.url) if moved else name)
        payload, posted = files.get(name, (_zip(_file()), LONG_AGO))
        return httpx.Response(
            200, content=payload, headers={"last-modified": posted} if posted else None
        )

    http = HttpClient(
        FailsToDeliver.spec,
        settings=settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return FailsToDeliver(client=http, settings=settings), requested


AUGUST_A = "cnsfails202608a"


def _index_page(missing: tuple[str, ...] = (), *, moved: tuple[str, ...] = ()) -> str:
    """The SEC's index: a link for every half-month since 2018, relative, as served."""
    links = []
    for year, month, half, _ in half_months(dt.date(2018, 1, 1), dt.date(2030, 12, 31)):
        name = f"cnsfails{year}{month:02d}{half}"
        if name in missing:
            continue
        folder = (
            "/files/data/other/fails-deliver-data"
            if name in moved
            else "/files/data/fails-deliver-data"
        )
        links.append(f'<a href="{folder}/{name}.zip">{name}</a>')
    return "<html><body>" + "\n".join(links) + "</body></html>"


# ---------------------------------------------------------------------- dates --
def test_rows_are_knowable_when_the_file_was_posted_not_when_the_trade_settled(
    settings: Settings,
) -> None:
    source, _ = _sec(settings, {AUGUST_A: (_zip(_file()), POSTED)})
    frame = source.fetch([], START, END)

    assert frame.height == len(ROWS)
    assert frame["known_at"].unique().to_list() == [
        dt.datetime(2026, 8, 31, 11, 54, 19, tzinfo=dt.UTC)
    ]
    assert frame["as_of"].max() == dt.datetime(2026, 8, 14, tzinfo=dt.UTC)
    assert (frame["known_at"] - frame["as_of"]).min() > dt.timedelta(days=16)


def test_a_file_posted_outside_the_window_contributes_nothing(settings: Settings) -> None:
    source, _ = _sec(settings, {})
    assert source.fetch([], START, END).height == 0


def test_a_file_with_no_posting_time_is_refused(settings: Settings) -> None:
    """Dating it by its contents would make every row knowable weeks early."""
    source, _ = _sec(settings, {"cnsfails202606b": (_zip(_file()), None)})
    with pytest.raises(SourceError, match="Last-Modified"):
        source.fetch([], START, END)


def test_files_are_requested_back_far_enough_to_cover_the_publication_lag(
    settings: Settings,
) -> None:
    """A file POSTED inside the window covers settlement dates well before it."""
    source, requested = _sec(settings, {})
    source.fetch([], START, END)

    assert requested[0] == "cnsfails202606b"
    assert AUGUST_A in requested


def test_half_months_end_on_the_15th_and_on_the_last_day() -> None:
    periods = list(half_months(dt.date(2024, 2, 1), dt.date(2024, 3, 31)))

    assert [(y, m, h) for y, m, h, _ in periods] == [
        (2024, 2, "a"),
        (2024, 2, "b"),
        (2024, 3, "a"),
        (2024, 3, "b"),
    ]
    assert periods[1][3] == dt.date(2024, 2, 29)  # a leap year
    assert [p[:3] for p in half_months(dt.date(2025, 12, 20), dt.date(2026, 1, 20))] == [
        (2025, 12, "b"),
        (2026, 1, "a"),
    ]


# ------------------------------------------------------------------- not yet --
def _period(day: dt.date) -> str:
    return f"cnsfails{day.year}{day.month:02d}{'a' if day.day <= 15 else 'b'}"


def test_a_recent_period_that_is_not_posted_yet_is_normal(settings: Settings) -> None:
    now = dt.datetime.now(dt.UTC)
    recent = tuple({_period((now - dt.timedelta(days=back)).date()) for back in range(0, 30)})
    source, requested = _sec(settings, {}, missing=recent)

    assert source.fetch([], now - dt.timedelta(days=5), now).height == 0
    assert not set(recent) & set(requested)  # not listed yet, so not asked for


def test_a_period_the_sec_never_published_is_skipped_not_guessed(settings: Settings) -> None:
    """The SEC's page lists 23 files for 2019, not 24. Its page is the authority:
    an address is never constructed for a file it does not link to."""
    source, requested = _sec(settings, {}, missing=("cnsfails202002a",))
    frame = source.fetch(
        [], dt.datetime(2020, 3, 1, tzinfo=dt.UTC), dt.datetime(2020, 3, 20, tzinfo=dt.UTC)
    )

    assert frame.height == 0
    assert "cnsfails202002a" not in requested
    assert "cnsfails202002b" in requested


def test_a_file_is_fetched_from_wherever_the_index_says_it_is(settings: Settings) -> None:
    """The two May 2026 files sit in a directory of their own, between April and
    June. A constructed address 404s, and once cost a whole multi-year ingest."""
    source, requested = _sec(settings, {AUGUST_A: (_zip(_file()), POSTED)}, moved=(AUGUST_A,))
    frame = source.fetch([], START, END)

    assert frame.height == len(ROWS)
    assert (
        "https://www.sec.gov/files/data/other/fails-deliver-data/cnsfails202608a.zip" in requested
    )


def test_an_index_page_with_no_links_is_a_changed_page_not_an_empty_history(
    settings: Settings,
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>Moved</html>"))
    http = HttpClient(
        FailsToDeliver.spec, settings=settings, client=httpx.Client(transport=transport)
    )
    with pytest.raises(SourceError, match="page has changed"):
        FailsToDeliver(client=http, settings=settings).fetch([], START, END)


# -------------------------------------------------------------------- content --
def test_a_missing_price_is_null_and_the_bridge_columns_are_kept() -> None:
    rows = parse_fails_file("test", _file(), "fixture")
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["NOPX"]["price"] is None
    assert by_symbol["DB"]["price"] == 36.70
    assert by_symbol["DB"]["cusip"] == "D18190898"
    assert by_symbol["MDXH"]["quantity"] == 8960
    assert by_symbol["MDXH"]["description"] == "MDXHEALTH SA SHS NEW(BELGIUM)"


def test_a_symbol_filter_keeps_only_those_tickers(settings: Settings) -> None:
    source, _ = _sec(settings, {AUGUST_A: (_zip(_file()), POSTED)})
    frame = source.fetch(["db", "TTE"], START, END)

    assert sorted(frame["symbol"].to_list()) == ["DB", "TTE"]


def test_older_files_whose_member_has_no_extension_still_open(settings: Settings) -> None:
    source, _ = _sec(settings, {AUGUST_A: (_zip(_file(), member=AUGUST_A), POSTED)})
    assert source.fetch([], START, END).height == len(ROWS)


# ----------------------------------------------------------------- truncation --
def test_a_truncated_file_is_caught_by_its_own_trailer() -> None:
    """A download cut short is otherwise a perfectly valid shorter file."""
    with pytest.raises(SourceError, match="Truncated"):
        parse_fails_file("test", _file(ROWS[:-2], count=len(ROWS)), "short")


def test_a_file_with_no_trailer_is_refused() -> None:
    body = "\r\n".join([HEADER, *ROWS]).encode("latin-1")
    with pytest.raises(SourceError, match="no trailer"):
        parse_fails_file("test", body, "headless")


def test_a_changed_header_is_refused() -> None:
    body = _file().replace(b"QUANTITY (FAILS)", b"QUANTITY")
    with pytest.raises(SourceError, match="unexpected header"):
        parse_fails_file("test", body, "renamed")


def test_a_payload_that_is_not_a_zip_is_refused(settings: Settings) -> None:
    source, _ = _sec(
        settings, {AUGUST_A: (b"<html>Request Rate Threshold Exceeded</html>", POSTED)}
    )
    with pytest.raises(SourceError, match="not a zip"):
        source.fetch([], START, END)
