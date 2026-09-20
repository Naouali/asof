"""SEC fails-to-deliver data.

Twice a month the SEC posts, for every security with a balance of undelivered
shares at NSCC, that balance for each settlement day. It is here for two reasons:
the data itself, and because each row pairs a CUSIP with a ticker -- the only
free, official bridge from ``institutional_holdings``, which knows securities by
CUSIP alone, to anything with a price.

**Three things here are easy to get silently wrong.**

*The date.* The first half of a month is posted at the end of that month and the
second half around the 15th of the next, so a row is two to six weeks old when it
becomes knowable. The file says nothing about when it was posted, but the server
does: ``Last-Modified`` on the zip matches the SEC's stated schedule to the day
(the first half of August 2026 was modified on 31 August, the second half on 15
September). That is used as ``known_at``.

*The quantity.* It is the balance outstanding on that day, not that day's new
fails. A position that stays unsettled for a week appears seven times.

*Truncation.* Each file ends with a trailer giving its record count and total
share quantity. Both are checked, because a download cut short is otherwise a
perfectly valid shorter file.

*Where the files are.* Their addresses cannot be constructed. The SEC has kept
them in four different directories over the years, and not in date order: the two
May 2026 files sit in a directory of their own between April and June. The
fetcher therefore reads the SEC's index page and follows the links it publishes,
which also makes that page the authority on whether a period exists at all -- it
lists 23 files, not 24, for 2019 and for 2023.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
from collections.abc import Iterator, Sequence
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["FailsToDeliver", "half_months", "parse_fails_file"]

log = get_logger("quantlab.data.sources.sec_ftd")

INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
SITE = "https://www.sec.gov"
_FILE_LINK = re.compile(r'href="([^"]*?cnsfails(\d{4})(\d{2})([ab])\.zip)"')
#: The page has listed some four hundred files since 2009. Far fewer means it is
#: no longer the page this was written against.
MIN_INDEXED_FILES = 100

HEADER = ("SETTLEMENT DATE", "CUSIP", "SYMBOL", "QUANTITY (FAILS)", "DESCRIPTION", "PRICE")

#: A file is posted two to six weeks after the last settlement date it covers, so
#: a half-month this recent may simply not be out yet, and its absence is normal.
NOT_YET_DUE = dt.timedelta(days=45)

#: How far before a window's start a file's period can end and still have been
#: POSTED inside the window.
PUBLICATION_LAG = dt.timedelta(days=60)


def half_months(first: dt.date, last: dt.date) -> Iterator[tuple[int, int, str, dt.date]]:
    """Every (year, month, half, period end) whose period end is in ``[first, last]``."""
    year, month = first.year, first.month
    while dt.date(year, month, 1) <= last:
        next_month = dt.date(year + month // 12, month % 12 + 1, 1)
        for half, period_end in (
            ("a", dt.date(year, month, 15)),
            ("b", next_month - dt.timedelta(days=1)),
        ):
            if first <= period_end <= last:
                yield year, month, half, period_end
        year, month = next_month.year, next_month.month


def parse_fails_file(source: str, raw: bytes, what: str) -> list[dict[str, Any]]:
    """Rows of one fails file, checked against its own trailer."""
    # The SEC declares no encoding and issuer names carry the odd non-ASCII byte;
    # latin-1 cannot fail and keeps every field boundary intact.
    lines = raw.decode("latin-1").splitlines()
    if not lines or tuple(part.strip() for part in lines[0].split("|")) != HEADER:
        raise SourceError(
            source,
            f"{what}: unexpected header {lines[0][:120] if lines else '(empty file)'!r}. "
            "The SEC changed the layout; the parser must be re-checked.",
        )

    rows: list[dict[str, Any]] = []
    declared_count: int | None = None
    declared_quantity: int | None = None
    for line in lines[1:]:
        if not line.strip():
            continue
        if line.startswith("Trailer"):
            number = line.rsplit(" ", 1)[-1]
            if "record count" in line:
                declared_count = int(number)
            elif "total quantity" in line:
                declared_quantity = int(number)
            continue
        parts = line.split("|")
        if len(parts) != len(HEADER):
            raise SourceError(source, f"{what}: row with {len(parts)} fields: {line[:120]!r}")
        settled, cusip, symbol, quantity, description, price = (part.strip() for part in parts)
        rows.append(
            {
                "settlement_date": dt.datetime.strptime(settled, "%Y%m%d").replace(tzinfo=dt.UTC),
                "cusip": cusip.upper(),
                # Should a row carry no ticker, the CUSIP still identifies the
                # security, so it stands in rather than the row being lost.
                "symbol": symbol.upper() or cusip.upper(),
                "quantity": float(quantity),
                "description": description or None,
                "price": None if price in {"", "."} else float(price),
            }
        )

    if declared_count is None or declared_quantity is None:
        raise SourceError(
            source, f"{what}: no trailer. The file is truncated, or the layout has changed."
        )
    total = int(sum(row["quantity"] for row in rows))
    if declared_count != len(rows) or declared_quantity != total:
        raise SourceError(
            source,
            f"{what}: trailer declares {declared_count} records and {declared_quantity} "
            f"shares, but {len(rows)} records and {total} shares were read. Truncated.",
        )
    return rows


@register
class FailsToDeliver(Source):
    """Fails-to-deliver balances, for every security or a chosen list of tickers."""

    name: ClassVar[str] = "sec_ftd.fails_to_deliver"
    dataset: ClassVar[str] = "fails_to_deliver"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY
    spec: ClassVar = get_source("sec_ftd")

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        wanted = {symbol.upper() for symbol in symbols}
        today = utcnow().date()

        # One frame per file rather than one list of rows: a file is some sixty
        # thousand rows and a backfill is dozens of files, which as Python dicts
        # is gigabytes and as columns is not.
        ingested_at = utcnow()
        frames: list[pl.DataFrame] = []
        index = self._index()
        for year, month, half, period_end in half_months(
            (start - PUBLICATION_LAG).date(), min(end.date(), today)
        ):
            url = index.get((year, month, half))
            if url is None:
                # The SEC's page lists no file for this half-month. Recently, that
                # is a file not posted yet. Long ago, it is a period the SEC never
                # published, and its own page is the authority on that.
                event = (
                    "ftd.not_yet_posted"
                    if period_end + NOT_YET_DUE > today
                    else "ftd.not_published"
                )
                log.info(event, period=f"{year}-{month:02d}{half}")
                continue
            response = self.client.request(url)

            known_at = self._posted_at(response.headers.get("last-modified"), url)
            if not (start <= known_at <= end):
                continue
            what = f"cnsfails{year}{month:02d}{half}"
            rows: list[dict[str, Any]] = []
            for row in parse_fails_file(self.name, _unzip(self.name, response.content, what), what):
                if wanted and row["symbol"] not in wanted:
                    continue
                row["as_of"] = row.pop("settlement_date")
                row["known_at"] = known_at
                rows.append(row)
            if rows:
                frames.append(self.finalise(rows, ingested_at=ingested_at))
            log.info("ftd.file", file=what, posted=known_at.isoformat(), rows=len(rows))
        return pl.concat(frames) if frames else self.empty()

    def _index(self) -> dict[tuple[int, int, str], str]:
        """Every published file's address, keyed by (year, month, half)."""
        page = self.client.get_text(INDEX_URL)
        found = {
            (int(year), int(month), half): link if link.startswith("https://") else SITE + link
            for link, year, month, half in _FILE_LINK.findall(page)
        }
        if len(found) < MIN_INDEXED_FILES:
            raise SourceError(
                self.name,
                f"{INDEX_URL} lists {len(found)} files, where it has listed hundreds. The "
                "page has changed; refusing to read an absent link as an absent file.",
            )
        return found

    def _posted_at(self, header: str | None, url: str) -> dt.datetime:
        if not header:
            raise SourceError(
                self.name,
                f"{url} carries no Last-Modified header, which is the only record of "
                "when the file was posted. Refusing to date it by its contents: the "
                "settlement dates are weeks earlier than anyone could have known them.",
            )
        return parsedate_to_datetime(header).astimezone(dt.UTC)


def _unzip(source: str, payload: bytes, what: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.namelist()
            if len(members) != 1:
                raise SourceError(source, f"{what}: expected one member, got {members}")
            return archive.read(members[0])
    except zipfile.BadZipFile as exc:
        raise SourceError(source, f"{what}: not a zip archive") from exc
