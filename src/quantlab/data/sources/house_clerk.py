"""US House of Representatives financial disclosures.

The STOCK Act requires members of Congress to report a securities trade within
45 days. The Clerk of the House publishes a yearly index of every disclosure
document as XML, and the documents themselves only as PDF. Two fetchers read
them: one stores the index, the other opens each periodic transaction report and
pulls the trades out of its text.

**Four things here are easy to get silently wrong.**

*The date.* ``as_of`` is the day of the trade and ``known_at`` is the day the
report was filed. The law allows 45 days between them and late reports run far
longer. Dating a trade by when it happened is how "Congress beats the market"
results are manufactured: nobody outside could have followed the trade until the
report existed.

*The time of day.* The Clerk gives a filing date and nothing finer, and no record
of when a document appeared on the site. ``known_at`` is therefore the END of the
filing day in Washington. A report filed at 9 a.m. becomes knowable fifteen hours
late; one filed at 11 p.m. does not become knowable fifteen hours early.

*Scans.* A report filed on paper is published as a scanned image with no text
layer. It cannot be read here and yields no rows -- and some of the most active
traders in the House file on paper. That absence must never read as "did not
trade", which is why the index is stored as a dataset of its own: a report listed
in ``congress_filings`` with no rows in ``congress_trades`` was not read.

*The layout is recovered, not given.* Text comes out of the PDF with line breaks
wherever the table cell wrapped: a ticker on its own line, an amount split in
two. Each transaction is found by its invariant core -- ``P|S date date amount``,
usually behind an ``[asset type]`` -- and the count is checked against the date pairs on the page, so a
report that does not parse cleanly is reported rather than quietly shortened.
The dates, type, amount and ticker are anchored that way. The ``asset`` name is
whatever text precedes them, and is best effort: a member's long comment can, in
a narrow case, lend a line to the name of the asset below it.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import polars as pl
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.sources.sec_filings import child_text, local_name, parse_xml
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = [
    "NO_TICKER",
    "HouseDisclosureFilings",
    "HouseTrades",
    "IndexEntry",
    "filed_known_at",
    "parse_index",
    "parse_ptr_text",
    "pdf_text",
]

log = get_logger("quantlab.data.sources.house_clerk")

# pypdf reports every recoverable oddity in a PDF at WARNING. Government PDFs are
# full of them and none is actionable here.
logging.getLogger("pypdf").setLevel(logging.ERROR)

INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
DOCUMENT_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}/{doc_id}.pdf"

CHAMBER = "house"
EASTERN = ZoneInfo("America/New_York")
PTR = "P"
NO_TICKER = "NO_TICKER"

#: Only the codes whose meaning is certain. The index uses others (D, H, O, W and
#: more) that the Clerk does not define anywhere machine-readable; those keep
#: their code and get no name rather than a guessed one.
FILING_TYPE_NAMES = {
    "A": "annual report",
    "C": "candidate report",
    "P": "periodic transaction report",
    "T": "termination report",
    "X": "extension request",
}

TRANSACTION_TYPES = {
    "P": "purchase",
    "S": "sale",
    "S (partial)": "sale_partial",
    "E": "exchange",
}

_TABLE_HEADER = re.compile(
    r"ID\s+Owner\s+Asset\s+Transaction\s+Type\s+Date\s+Notification\s+Date\s+"
    r"Amount\s+Cap\.\s+Gains\s+>\s+\$200\?"
)
_TABLE_END = "* For the complete list of asset type abbreviations"
_PAGE_FOOTER = re.compile(r"Filing ID #\d+")
#: Small-caps field labels lose their lowercase letters in extraction, so
#: "Filing Status: New" arrives as "F      S     : New".
_FILING_STATUS = re.compile(r"^F +S *: *(\w+)")
_FIELD_LINE = re.compile(r"^[A-Z](?: +[A-Z])* *:")
#: Free-text fields run the full width of the page, about 120 characters, and the
#: asset column is about 45. A line can only be the continuation of a wrapped
#: comment if the line before it was long enough to have wrapped.
_WRAP_WIDTH = 95
_DATE_PAIR = re.compile(r"\d{2}/\d{2}/\d{4}\s+\d{2}/\d{2}/\d{4}")
#: The asset-type code is optional: members do leave it off, and a transaction is
#: still a transaction without it.
_CORE = re.compile(
    r"(?:\[(?P<asset_type>[A-Z0-9]{2})\]\s+)?"
    r"(?<!\S)(?P<type>P|S \(partial\)|S|E)\s+"
    r"(?P<traded>\d{2}/\d{2}/\d{4})\s+(?P<notified>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<amount>(?:Spouse/DC\s+)?Over\s+\$[\d,]+|\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+)"
)
_STATUS_MARK = re.compile(r"<<status:(\w+)>>")
_OWNER = re.compile(r"^(SP|JT|DC)\s+")
_TICKER = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)\s*$")
_DOLLARS = re.compile(r"\$([\d,]+)")


@dataclass(frozen=True, slots=True)
class IndexEntry:
    year: int
    doc_id: str
    filing_type: str
    filed: dt.date
    state_district: str
    prefix: str | None
    first_name: str | None
    last_name: str
    suffix: str | None

    @property
    def url(self) -> str:
        template = PTR_URL if self.filing_type == PTR else DOCUMENT_URL
        return template.format(year=self.year, doc_id=self.doc_id)

    @property
    def member(self) -> str:
        return " ".join(part for part in (self.first_name, self.last_name, self.suffix) if part)


def filed_known_at(filed: dt.date) -> dt.datetime:
    """The end of the filing day in Washington, as UTC."""
    return dt.datetime(filed.year, filed.month, filed.day, 23, 59, 59, tzinfo=EASTERN).astimezone(
        dt.UTC
    )


def parse_index(source: str, payload: bytes, year: int) -> list[IndexEntry]:
    """Every document in one year's disclosure index."""
    what = f"{year}FD.zip"
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next((n for n in archive.namelist() if n.lower().endswith(".xml")), None)
            if member is None:
                raise SourceError(source, f"{what}: no XML index among {archive.namelist()}")
            text = archive.read(member).decode("utf-8-sig")
    except zipfile.BadZipFile as exc:
        raise SourceError(source, f"{what}: not a zip archive") from exc

    entries: list[IndexEntry] = []
    undated: list[str] = []
    for node in parse_xml(source, text, what):
        if local_name(node) != "Member":
            continue
        doc_id, filing_type = child_text(node, "DocID"), child_text(node, "FilingType")
        filed, last_name = child_text(node, "FilingDate"), child_text(node, "Last")
        if not doc_id or not filing_type or not last_name:
            raise SourceError(
                source,
                f"{what}: an entry lacks DocID, FilingType or Last. The index layout "
                "may have changed; the reader must be re-checked.",
            )
        if not filed:
            # The index carries a few dozen of these a year, all withdrawals. With
            # no date there is no honest `known_at`, so they cannot be placed.
            undated.append(filing_type.upper())
            continue
        entries.append(
            IndexEntry(
                year=year,
                doc_id=doc_id,
                filing_type=filing_type.upper(),
                filed=dt.datetime.strptime(filed, "%m/%d/%Y").date(),
                state_district=child_text(node, "StateDst") or "UNKNOWN",
                prefix=child_text(node, "Prefix"),
                first_name=child_text(node, "First"),
                last_name=last_name,
                suffix=child_text(node, "Suffix"),
            )
        )
    if not entries:
        raise SourceError(source, f"{what}: the index is empty")
    if undated:
        log.info("house.index.undated", year=year, skipped=len(undated), types=sorted(set(undated)))
        if PTR in undated:
            log.warning("house.index.undated_transaction_report", year=year)
    return entries


def pdf_text(payload: bytes) -> str | None:
    """The text layer of a PDF, or ``None`` when it cannot be read at all."""
    try:
        reader = PdfReader(io.BytesIO(payload))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PyPdfError, ValueError, KeyError, OSError):
        return None
    # The small-caps field labels come out with a NUL where each lowercase letter
    # was. A terminal draws NUL as a space, so it looks harmless and matches nothing.
    return text.replace("\x00", " ")


def parse_ptr_text(text: str) -> tuple[list[dict[str, Any]], int]:
    """Transactions in a periodic transaction report's text.

    Returns the rows and the number of date pairs seen in the table. The two
    agree when every transaction was recognised; the caller reports it when they
    do not. Rows carry no provenance -- the caller knows which document this is.
    """
    header = _TABLE_HEADER.search(text)
    if header is None:
        return [], 0
    end = text.find(_TABLE_END, header.end())
    table = text[header.end() : end if end != -1 else len(text)]

    expected = len(_DATE_PAIR.findall(table))
    table = _TABLE_HEADER.sub("\n", table)
    table = _PAGE_FOOTER.sub("\n", table)
    flat = " ".join(" ".join(_without_fields(table.splitlines())).split())

    matches = list(_CORE.finditer(flat))
    rows: list[dict[str, Any]] = []
    previous_end = 0
    for line, match in enumerate(matches, start=1):
        following = matches[line].start() if line < len(matches) else len(flat)
        status = _STATUS_MARK.search(flat, match.end(), following)

        asset = _STATUS_MARK.sub(" ", flat[previous_end : match.start()])
        asset = " ".join(asset.split())
        previous_end = match.end()

        owner = _OWNER.match(asset)
        if owner is not None:
            asset = asset[owner.end() :]
        ticker = _TICKER.search(asset)

        amount_text = " ".join(match["amount"].split())
        bounds = [float(number.replace(",", "")) for number in _DOLLARS.findall(amount_text)]
        open_ended = "Over" in amount_text or len(bounds) == 1
        rows.append(
            {
                "line": line,
                "symbol": ticker.group(1) if ticker else NO_TICKER,
                # None means the member's own account; the form leaves it blank.
                "owner": owner.group(1) if owner else None,
                "asset": asset or None,
                "asset_type": match["asset_type"],
                "transaction_type": TRANSACTION_TYPES[match["type"]],
                "transaction_date": dt.datetime.strptime(match["traded"], "%m/%d/%Y").date(),
                "notification_date": dt.datetime.strptime(match["notified"], "%m/%d/%Y").date(),
                "amount_min": bounds[0],
                "amount_max": None if open_ended else bounds[1],
                "amount_text": amount_text,
                "filing_status": status.group(1).lower() if status else None,
            }
        )
    return rows, expected


def _without_fields(lines: Sequence[str]) -> Iterator[str]:
    """Table lines with the per-transaction fields removed.

    Under each transaction sit labelled fields -- filing status, subholding,
    location, description, comments -- which are not part of the next asset's
    name. The filing status is kept as a marker; the rest is dropped, including
    the continuation lines of a field long enough to wrap. A line holding a
    transaction's dates is never dropped, whatever precedes it.
    """
    in_field, previous_width = False, 0
    for line in lines:
        status = _FILING_STATUS.match(line)
        if status is not None:
            yield f"<<status:{status.group(1)}>>"
        if status is not None or _FIELD_LINE.match(line):
            in_field, previous_width = True, len(line)
            continue
        if in_field and previous_width >= _WRAP_WIDTH and not _DATE_PAIR.search(line):
            previous_width = len(line)
            continue
        in_field = False
        yield line


def _member_name(text: str) -> str | None:
    found = re.search(r"^Name:\s*(.+)$", text, re.M)
    return found.group(1).strip() if found else None


class _HouseSource(Source):
    """Shared index handling for the two House fetchers."""

    spec: ClassVar = get_source("house_clerk")

    def _entries(self, start: dt.datetime, end: dt.datetime) -> Iterator[IndexEntry]:
        """Index entries that became knowable within ``[start, end]`` -- and that
        are knowable yet: a report filed today is not, until the day is over."""
        horizon = min(end, utcnow())
        for year in range(start.year, horizon.year + 1):
            payload = self.client.get_bytes(INDEX_URL.format(year=year))
            for entry in parse_index(self.name, payload, year):
                if start <= filed_known_at(entry.filed) <= horizon:
                    yield entry


@register
class HouseDisclosureFilings(_HouseSource):
    """The index of every House disclosure document."""

    name: ClassVar[str] = "house_clerk.congress_filings"
    dataset: ClassVar[str] = "congress_filings"
    asset_class: ClassVar[AssetClass] = AssetClass.REFERENCE

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        wanted = {symbol.upper() for symbol in symbols}
        rows = [
            {
                "symbol": entry.state_district,
                "as_of": dt.datetime(
                    entry.filed.year, entry.filed.month, entry.filed.day, tzinfo=dt.UTC
                ),
                "known_at": filed_known_at(entry.filed),
                "chamber": CHAMBER,
                "doc_id": entry.doc_id,
                "filing_type": entry.filing_type,
                "filing_type_name": FILING_TYPE_NAMES.get(entry.filing_type),
                "prefix": entry.prefix,
                "first_name": entry.first_name,
                "last_name": entry.last_name,
                "suffix": entry.suffix,
                "year": entry.year,
                "url": entry.url,
            }
            for entry in self._entries(start, end)
            if not wanted or entry.state_district.upper() in wanted
        ]
        return self.finalise(rows)


@register
class HouseTrades(_HouseSource):
    """Transactions parsed out of House periodic transaction reports."""

    name: ClassVar[str] = "house_clerk.congress_trades"
    dataset: ClassVar[str] = "congress_trades"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        wanted = {symbol.upper() for symbol in symbols}

        rows: list[dict[str, Any]] = []
        reports = unreadable = partial = impossible = 0
        for entry in self._entries(start, end):
            if entry.filing_type != PTR:
                continue
            reports += 1
            text = pdf_text(self.client.get_bytes(entry.url))
            parsed, expected = parse_ptr_text(text) if text else ([], 0)
            if not parsed:
                # A scan, or a PDF that would not open. Every real report holds at
                # least one transaction, so nothing parsed means nothing read.
                unreadable += 1
                log.warning("house.ptr.unreadable", doc_id=entry.doc_id, member=entry.member)
                continue
            if len(parsed) != expected:
                partial += 1
                log.warning(
                    "house.ptr.partial",
                    doc_id=entry.doc_id,
                    member=entry.member,
                    parsed=len(parsed),
                    date_pairs_on_page=expected,
                    url=entry.url,
                )

            known_at = filed_known_at(entry.filed)
            member = _member_name(text or "") or entry.member
            for row in parsed:
                traded: dt.date = row.pop("transaction_date")
                as_of = dt.datetime(traded.year, traded.month, traded.day, tzinfo=dt.UTC)
                if as_of > known_at:
                    impossible += 1  # a trade dated after its own report: a typo
                    continue
                if wanted and row["symbol"] not in wanted:
                    continue
                rows.append(
                    {
                        **row,
                        "as_of": as_of,
                        "known_at": known_at,
                        "chamber": CHAMBER,
                        "doc_id": entry.doc_id,
                        "member": member,
                        "state_district": entry.state_district,
                        "url": entry.url,
                    }
                )

        log.info(
            "house.trades",
            reports=reports,
            unreadable=unreadable,
            partially_parsed=partial,
            dropped_dated_after_filing=impossible,
            rows=len(rows),
        )
        if reports and unreadable == reports:
            raise SourceError(
                self.name,
                f"none of {reports} periodic transaction reports could be read. A few "
                "scans are normal; all of them means the PDF layout has changed.",
            )
        return self.finalise(rows)
