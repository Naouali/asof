"""US Senate financial disclosures (eFD).

The Senate half of the STOCK Act record. Senators report a securities trade within
45 days, as the House does, but the Senate publishes each electronic report as an
HTML page with a real table, so nothing here is recovered from a PDF.

**Four things here are easy to get silently wrong.**

*Where it can be reached from.* ``efdsearch.senate.gov`` answers only connections
from inside the United States; everywhere else gets HTTP 403 whatever the client.
So there are two ways in. ``direct`` talks to the Senate and works from a US
address. The default reads a *mirror*: a daily job on a US-hosted runner
(``.github/workflows/senate-mirror.yml``) runs the direct reader and commits what
it saw, as JSON, to a branch of this project's own repository. The mirror holds
the cells of each report exactly as the page showed them; every interpretation
below happens at ingest, so a parsing fix never needs the Senate asked again.

*The agreement.* The search sits behind a form whose one checkbox restates
5 U.S.C. 13107(c): the reports may not be used for an unlawful purpose, for a
commercial purpose other than news dissemination, to establish a credit rating,
or to solicit money. The direct reader ticks it, because a person decided it
should. The same statute covers the House reports, which simply have no checkbox.

*The date.* ``as_of`` is the day of the trade. ``known_at`` is the minute the
report was filed, which the page prints ("Filed 10/10/2025 @ 11:27 AM") and which
is read as Washington time. Electronic reports are published as they are filed.
Where a page gives no time, ``known_at`` falls back to the end of the filing day.

*Paper.* A report filed on paper is a set of scanned images. It is listed in
``congress_filings`` and yields nothing in ``congress_trades``; that absence is a
report nobody could read, never a senator who did not trade.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import HttpClient, SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.sources.house_clerk import EASTERN, NO_TICKER, filed_known_at
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = [
    "MIRROR_URL",
    "Report",
    "ReportRef",
    "SenateDisclosureFilings",
    "SenateSession",
    "SenateTrades",
    "parse_index_page",
    "parse_report",
    "update_mirror",
]

log = get_logger("quantlab.data.sources.senate_efd")

BASE = "https://efdsearch.senate.gov"
HOME_URL = f"{BASE}/search/home/"
SEARCH_URL = f"{BASE}/search/"
DATA_URL = f"{BASE}/search/report/data/"

#: Where the daily job publishes what it read. One file per filing year.
MIRROR_URL = "https://raw.githubusercontent.com/Naouali/asof/senate-mirror/senate"
MIRROR_FILE = "ptr-{year}.json"

CHAMBER = "senate"
#: The Senate's index does not say which state a senator sits for.
STATE_UNKNOWN = "SENATE"
PTR = "P"
PTR_NAME = "periodic transaction report"
#: The site's own code for periodic transaction reports in its search form.
PTR_REPORT_TYPE = 11
PAGE_SIZE = 100

TRANSACTION_TYPES = {
    "purchase": "purchase",
    "sale (full)": "sale",
    "sale (partial)": "sale_partial",
    "exchange": "exchange",
}
#: The House's codes for the same thing, so one column means one thing.
OWNERS = {"self": None, "joint": "JT", "spouse": "SP", "child": "DC", "dependent child": "DC"}

_GEO_BLOCKED = (
    "the Senate's site answers only connections from inside the United States, and "
    "this one was refused (HTTP 403). Read the mirror instead (the default), or run "
    "the direct reader from a US address."
)
_TOKEN = re.compile(r"name=[\"']csrfmiddlewaretoken[\"'][^>]*value=[\"']([^\"']+)")
_HREF = re.compile(r"href=[\"']([^\"']+)[\"']")
_TAG = re.compile(r"<[^>]+>")
_FILED_AT = re.compile(r"Filed\s+(\d{2}/\d{2}/\d{4})\s*@\s*(\d{1,2}:\d{2}\s*[AP]M)", re.I)
_DECLARED = re.compile(r"\((\d+)\s+transactions?\s+total\)", re.I)
_TICKER = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)\s*$")
_PLAIN_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_DOLLARS = re.compile(r"\$([\d,]+)")


@dataclass(frozen=True, slots=True)
class ReportRef:
    """One line of the Senate's index: a report exists, and where."""

    report_id: str
    first_name: str
    last_name: str
    office: str
    title: str
    filed: dt.date
    url: str
    paper: bool

    @property
    def member(self) -> str:
        return " ".join(part for part in (self.first_name, self.last_name) if part)

    @property
    def amended(self) -> bool:
        return "amendment" in self.title.lower()


@dataclass(slots=True)
class Report:
    """What one report page said. Cells are kept as the page showed them."""

    filed_at: dt.datetime | None = None
    declared: int | None = None
    rows: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------- parsing --
def parse_index_page(source: str, payload: Any) -> tuple[list[ReportRef], int]:
    """One page of the search results, and how many results there are in all."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise SourceError(source, f"the report index is not the expected JSON: {payload!r:.200}")

    refs: list[ReportRef] = []
    for row in payload["data"]:
        if not isinstance(row, list) or len(row) < 5:
            raise SourceError(
                source,
                f"an index row has {len(row) if isinstance(row, list) else 'no'} cells, not "
                "five. The index layout may have changed; the reader must be re-checked.",
            )
        first, last, office, link, filed = (str(cell) for cell in row[:5])
        href = _HREF.search(link)
        if href is None:
            raise SourceError(source, f"an index row links to no report: {link!r:.200}")
        path = href.group(1)
        refs.append(
            ReportRef(
                report_id=path.strip("/").split("/")[-1],
                first_name=_tidy(first).strip(" ,"),
                last_name=_tidy(last).strip(" ,"),
                office=_tidy(office),
                title=_tidy(_TAG.sub(" ", link)),
                filed=dt.datetime.strptime(filed.strip(), "%m/%d/%Y").date(),
                url=path if path.startswith("https://") else f"{BASE}{path}",
                paper="/view/paper/" in path,
            )
        )
    return refs, int(payload.get("recordsFiltered") or 0)


class _Table(HTMLParser):
    """The cells of the first table body on a page, row by row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._in_body = self._done = False
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        if tag == "tbody":
            self._in_body = True
        elif self._in_body and tag == "tr":
            self.rows.append([])
        elif self._in_body and tag == "td":
            self._cell = []
        elif self._cell is not None and tag in {"br", "div"}:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell is not None and self.rows:
            self.rows[-1].append(_tidy("".join(self._cell)))
            self._cell = None
        elif tag == "tbody" and self._in_body:
            self._in_body, self._done = False, True

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


_COLUMNS = ("line", "date", "owner", "ticker", "asset", "asset_type", "type", "amount", "comment")


def parse_report(source: str, html: str, what: str) -> Report:
    """The transactions on one electronic report page.

    Rows come back as the page's own text. The page states how many transactions
    it holds; the caller compares that with what was found.
    """
    text = _TAG.sub(" ", html)
    report = Report()
    filed = _FILED_AT.search(text)
    if filed is not None:
        stamp = dt.datetime.strptime(
            f"{filed.group(1)} {filed.group(2).upper().replace(' ', '')}", "%m/%d/%Y %I:%M%p"
        )
        report.filed_at = stamp.replace(tzinfo=EASTERN).astimezone(dt.UTC)
    declared = _DECLARED.search(text)
    report.declared = int(declared.group(1)) if declared else None

    table = _Table()
    table.feed(html)
    for cells in table.rows:
        if not cells:
            continue
        if len(cells) != len(_COLUMNS):
            raise SourceError(
                source,
                f"{what}: a transaction row has {len(cells)} cells, not {len(_COLUMNS)}. "
                "The report layout may have changed; the reader must be re-checked.",
            )
        report.rows.append(dict(zip(_COLUMNS, cells, strict=True)))
    return report


def _tidy(text: str) -> str:
    return " ".join(text.split())


def _transactions(source: str, ref: ReportRef, report: Report) -> Iterator[dict[str, Any]]:
    """Schema rows from a report's raw cells. Provenance is the caller's."""
    for cells in report.rows:
        kind = TRANSACTION_TYPES.get(cells["type"].lower())
        owner_key = cells["owner"].lower()
        if kind is None or owner_key not in OWNERS:
            raise SourceError(
                source,
                f"{ref.url}: unknown transaction type {cells['type']!r} or owner "
                f"{cells['owner']!r}. Guessing would file it under the wrong meaning.",
            )
        bounds = [float(number.replace(",", "")) for number in _DOLLARS.findall(cells["amount"])]
        if not bounds:
            raise SourceError(source, f"{ref.url}: no dollar amount in {cells['amount']!r}")

        ticker = cells["ticker"].upper()
        if not _PLAIN_TICKER.match(ticker):
            # "--": the senator named no ticker. An exchange names two securities,
            # so there the asset's own text is not a safe place to look for one.
            named = _TICKER.search(cells["asset"]) if kind != "exchange" else None
            ticker = named.group(1) if named else NO_TICKER

        yield {
            "line": int(cells["line"]),
            "symbol": ticker,
            "owner": OWNERS[owner_key],
            "asset": cells["asset"] or None,
            "asset_type": cells["asset_type"] or None,
            "transaction_type": kind,
            "transaction_date": dt.datetime.strptime(cells["date"], "%m/%d/%Y").date(),
            "amount_min": bounds[0],
            "amount_max": bounds[1]
            if len(bounds) > 1 and "over" not in cells["amount"].lower()
            else None,
            "amount_text": cells["amount"],
            "filing_status": "amended" if ref.amended else "new",
        }


# --------------------------------------------------------------- the Senate itself --
class SenateSession:
    """A search session on the Senate's site, opened by accepting its agreement."""

    def __init__(self, client: HttpClient, source: str) -> None:
        self.client, self.source = client, source
        self._token: str | None = None

    def _open(self) -> str:
        if self._token is not None:
            return self._token
        try:
            home = self.client.request(HOME_URL).text
        except SourceError as exc:
            if "HTTP 403" in str(exc):
                raise SourceError(self.source, _GEO_BLOCKED) from exc
            raise
        token = _TOKEN.search(home)
        if token is None:
            raise SourceError(self.source, "no form token on the Senate's home page")
        agreed = self.client.request(
            HOME_URL,
            data={"csrfmiddlewaretoken": token.group(1), "prohibition_agreement": "1"},
            headers={"Referer": HOME_URL},
        ).text
        # The search page that follows carries a token of its own; either is valid
        # for the session, and the newer one is what a browser would send.
        self._token = (_TOKEN.search(agreed) or token).group(1)
        return self._token

    def index(self, start: dt.date) -> Iterator[ReportRef]:
        """Every transaction report filed since ``start``, newest first."""
        token, offset = self._open(), 0
        while True:
            response = self.client.request(
                DATA_URL,
                data={
                    "start": offset,
                    "length": PAGE_SIZE,
                    "report_types": f"[{PTR_REPORT_TYPE}]",
                    "filer_types": "[]",
                    "submitted_start_date": f"{start:%m/%d/%Y} 00:00:00",
                    "submitted_end_date": "",
                    "candidate_state": "",
                    "senator_state": "",
                    "office_id": "",
                    "first_name": "",
                    "last_name": "",
                    "csrfmiddlewaretoken": token,
                },
                headers={"Referer": SEARCH_URL},
            )
            try:
                payload = response.json()
            except ValueError as exc:
                raise SourceError(
                    self.source, f"the report index is not JSON: {response.text[:200]!r}"
                ) from exc
            refs, total = parse_index_page(self.source, payload)
            yield from refs
            offset += PAGE_SIZE
            if not refs or offset >= total:
                return

    def report(self, ref: ReportRef) -> Report:
        self._open()
        return parse_report(self.source, self.client.request(ref.url).text, ref.url)


# ------------------------------------------------------------------- the mirror --
def _record(ref: ReportRef, report: Report | None, first_seen_at: dt.datetime) -> dict[str, Any]:
    record = asdict(ref) | {"filed": ref.filed.isoformat()}
    record["first_seen_at"] = first_seen_at.isoformat(timespec="seconds")
    record["filed_at"] = (
        report.filed_at.isoformat(timespec="seconds") if report and report.filed_at else None
    )
    record["declared"] = report.declared if report else None
    record["rows"] = report.rows if report else []
    return record


def _from_record(source: str, record: dict[str, Any]) -> tuple[ReportRef, Report]:
    try:
        ref = ReportRef(
            report_id=record["report_id"],
            first_name=record["first_name"],
            last_name=record["last_name"],
            office=record["office"],
            title=record["title"],
            filed=dt.date.fromisoformat(record["filed"]),
            url=record["url"],
            paper=bool(record["paper"]),
        )
        filed_at = record["filed_at"]
        report = Report(
            filed_at=dt.datetime.fromisoformat(filed_at) if filed_at else None,
            declared=record["declared"],
            rows=list(record["rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceError(source, f"a mirror record is malformed: {exc!r}") from exc
    return ref, report


def update_mirror(session: SenateSession, directory: Path, start: dt.date) -> int:
    """Bring the mirror files in ``directory`` up to date. Returns reports added.

    A report already in the mirror is never asked for again, and keeps the moment
    it was first seen: that is the one hard bound this project has on when a
    Senate report was really public.
    """
    directory.mkdir(parents=True, exist_ok=True)
    years: dict[int, dict[str, dict[str, Any]]] = {}

    def year_of(year: int) -> dict[str, dict[str, Any]]:
        if year not in years:
            path = directory / MIRROR_FILE.format(year=year)
            held = json.loads(path.read_text("utf-8"))["reports"] if path.exists() else []
            years[year] = {record["report_id"]: record for record in held}
        return years[year]

    now, added = utcnow(), 0
    for ref in session.index(start):
        held = year_of(ref.filed.year)
        if ref.report_id in held:
            continue
        held[ref.report_id] = _record(ref, None if ref.paper else session.report(ref), now)
        added += 1

    for year, held in years.items():
        body = {
            "source": BASE,
            "reports": sorted(held.values(), key=lambda r: (r["filed"], r["report_id"])),
        }
        path = directory / MIRROR_FILE.format(year=year)
        path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", "utf-8")
    log.info("senate.mirror.updated", added=added, years=sorted(years))
    return added


# -------------------------------------------------------------------- fetchers --
class _SenateSource(Source):
    """Shared report handling for the two Senate fetchers."""

    spec: ClassVar = get_source("senate_efd")

    def __init__(
        self, *, direct: bool = False, mirror_url: str = MIRROR_URL, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.direct = direct
        self.mirror_url = mirror_url.rstrip("/")

    def _reports(
        self, start: dt.datetime, end: dt.datetime, *, read: bool
    ) -> Iterator[tuple[ReportRef, Report, dt.datetime]]:
        """Reports that became knowable within ``[start, end]``, with when.

        ``read`` is whether the transactions are wanted; the index alone is one
        request a hundred reports, and each report read directly is one more.
        """
        horizon = min(end, utcnow())
        for ref, report in (
            self._direct(start, read) if self.direct else self._mirrored(start, horizon)
        ):
            known_at = report.filed_at or filed_known_at(ref.filed)
            if start <= known_at <= horizon:
                yield ref, report, known_at

    def _direct(self, start: dt.datetime, read: bool) -> Iterator[tuple[ReportRef, Report]]:
        session = SenateSession(self.client, self.name)
        # A day of margin: the index is searched by date, the window is an instant.
        for ref in session.index(start.date() - dt.timedelta(days=1)):
            yield ref, session.report(ref) if read and not ref.paper else Report()

    def _mirrored(
        self, start: dt.datetime, horizon: dt.datetime
    ) -> Iterator[tuple[ReportRef, Report]]:
        for year in range(start.year, horizon.year + 1):
            url = f"{self.mirror_url}/{MIRROR_FILE.format(year=year)}"
            try:
                payload = self.client.get_json(url)
            except SourceError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                raise SourceError(
                    self.name,
                    f"the mirror has no file for {year} ({url}). It is written by the "
                    "senate-mirror workflow; check that the job has run, or start the "
                    "ingest no earlier than the mirror does.",
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("reports"), list):
                raise SourceError(self.name, f"{url} is not a mirror file")
            for record in payload["reports"]:
                yield _from_record(self.name, record)


@register
class SenateDisclosureFilings(_SenateSource):
    """The index of Senate periodic transaction reports, paper ones included."""

    name: ClassVar[str] = "senate_efd.congress_filings"
    dataset: ClassVar[str] = "congress_filings"
    asset_class: ClassVar[AssetClass] = AssetClass.REFERENCE

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        rows = [
            {
                "symbol": STATE_UNKNOWN,
                "as_of": dt.datetime(ref.filed.year, ref.filed.month, ref.filed.day, tzinfo=dt.UTC),
                "known_at": known_at,
                "chamber": CHAMBER,
                "doc_id": ref.report_id,
                "filing_type": PTR,
                "filing_type_name": PTR_NAME,
                "first_name": ref.first_name,
                "last_name": ref.last_name,
                "year": ref.filed.year,
                "url": ref.url,
            }
            # The mirror knows each report's filing minute; a bare index does not,
            # so reading directly has to open the reports to date them the same way.
            for ref, _report, known_at in self._reports(start, end, read=True)
        ]
        return self.finalise(rows)


@register
class SenateTrades(_SenateSource):
    """Transactions from Senate periodic transaction reports."""

    name: ClassVar[str] = "senate_efd.congress_trades"
    dataset: ClassVar[str] = "congress_trades"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        wanted = {symbol.upper() for symbol in symbols}

        rows: list[dict[str, Any]] = []
        reports = paper = partial = impossible = 0
        for ref, report, known_at in self._reports(start, end, read=True):
            reports += 1
            if ref.paper:
                paper += 1
                log.warning("senate.ptr.paper", report_id=ref.report_id, member=ref.member)
                continue
            if report.declared is not None and report.declared != len(report.rows):
                partial += 1
                log.warning(
                    "senate.ptr.partial",
                    report_id=ref.report_id,
                    member=ref.member,
                    parsed=len(report.rows),
                    declared_on_page=report.declared,
                    url=ref.url,
                )
            for row in _transactions(self.name, ref, report):
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
                        "doc_id": ref.report_id,
                        "member": ref.member,
                        "state_district": STATE_UNKNOWN,
                        "url": ref.url,
                    }
                )

        log.info(
            "senate.trades",
            reports=reports,
            paper=paper,
            partially_parsed=partial,
            dropped_dated_after_filing=impossible,
            rows=len(rows),
        )
        return self.finalise(rows)
