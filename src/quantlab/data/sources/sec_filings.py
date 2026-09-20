"""EDGAR's filing index, shared by the fetchers that read individual filings.

Registers no fetcher of its own. The ownership and 13F fetchers both start from
the same place -- the list of what an entity has filed -- and both need the same
three things from it, each of which is easy to get subtly wrong.

*When a filing became knowable.* The submissions API carries
``acceptanceDateTime``, the instant EDGAR accepted the filing, to the second. It
is genuine UTC, not Eastern time mislabelled: Berkshire's 16:05 Washington
filings read ``20:05Z`` in August and ``21:05Z`` in February, and a ``22:30Z``
Form 4 shows "Accepted 18:30" on its own index page. That is used directly,
which is strictly better than deriving a time from the filing date. Only when it
is absent does this fall back to the 17:30 Eastern cutoff the fundamentals
fetcher uses.

*Where the older filings are.* ``filings.recent`` holds roughly the last thousand
and the rest sit in numbered pages, each advertising the date range it covers. A
fetcher that reads only ``recent`` silently truncates the history of any prolific
filer -- Apple's insiders pass a thousand filings in under two years.

*Where the machine-readable document is.* ``primaryDocument`` names the
stylesheet-rendered HTML view, ``xslF345X06/form4.xml``. The XML itself is the
same filename without the ``xsl...`` directory.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import Any
from xml.etree.ElementTree import Element  # the type only; parsing is defusedxml's

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from quantlab.data.http import HttpClient, SourceError
from quantlab.data.sources.edgar import TICKER_URL, filing_known_at
from quantlab.logging import get_logger

__all__ = [
    "Filing",
    "Submissions",
    "child_text",
    "load_ticker_map",
    "local_name",
    "parse_accepted",
    "parse_xml",
]

log = get_logger("quantlab.data.sources.sec_filings")

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{document}"

_COLUMNS = ("form", "accessionNumber", "filingDate", "primaryDocument")


@dataclass(frozen=True, slots=True)
class Filing:
    """One row of an entity's filing index."""

    cik: int
    form: str
    accession: str
    accepted_at: dt.datetime
    report_date: dt.date | None
    primary_document: str

    @property
    def folder(self) -> str:
        return self.accession.replace("-", "")

    @property
    def raw_primary_document(self) -> str:
        """The XML itself, rather than the stylesheet-rendered view of it."""
        directory, _, filename = self.primary_document.rpartition("/")
        return filename if directory.startswith("xsl") else self.primary_document

    def url(self, document: str) -> str:
        return ARCHIVE_URL.format(cik=self.cik, folder=self.folder, document=document)


def parse_accepted(raw: str | None, filing_date: str) -> dt.datetime:
    """The instant a filing became knowable, as UTC."""
    if raw:
        try:
            accepted = dt.datetime.fromisoformat(raw)
        except ValueError:
            accepted = None
        if accepted is not None and accepted.tzinfo is not None:
            return accepted.astimezone(dt.UTC)
    # No usable acceptance instant: fall back to the end of EDGAR's filing day,
    # which can only make the filing knowable later than it was, never earlier.
    return filing_known_at(dt.date.fromisoformat(filing_date))


class Submissions:
    """An entity's filing index, across every page EDGAR splits it into."""

    def __init__(self, client: HttpClient, source: str, cik: int) -> None:
        self._client = client
        self._source = source
        self.cik = cik
        payload = client.get_json(SUBMISSIONS_URL.format(cik=cik))
        if not isinstance(payload, dict) or "filings" not in payload:
            raise SourceError(source, f"CIK {cik}: submissions payload has no `filings`")
        self.name = str(payload.get("name") or "")
        self._recent: dict[str, Any] = payload["filings"].get("recent") or {}
        self._pages: list[dict[str, Any]] = list(payload["filings"].get("files") or [])

    def filings(
        self, forms: Collection[str], start: dt.datetime, end: dt.datetime
    ) -> Iterator[Filing]:
        """Filings of the given forms ACCEPTED within ``[start, end]``."""
        yield from self._from_table(self._recent, forms, start, end)
        for page in self._pages:
            covers_to = str(page.get("filingTo") or "")
            if covers_to and covers_to < start.date().isoformat():
                continue  # the whole page predates the window
            table = self._client.get_json(SUBMISSIONS_PAGE_URL.format(name=page["name"]))
            if not isinstance(table, dict):
                raise SourceError(self._source, f"CIK {self.cik}: {page['name']} is not an object")
            yield from self._from_table(table, forms, start, end)

    def _from_table(
        self, table: dict[str, Any], forms: Collection[str], start: dt.datetime, end: dt.datetime
    ) -> Iterator[Filing]:
        if not table:
            return
        missing = [column for column in _COLUMNS if column not in table]
        if missing:
            raise SourceError(
                self._source,
                f"CIK {self.cik}: the filing index has no {missing} column. EDGAR "
                "changed the submissions layout; the reader must be re-checked.",
            )
        accepted_column = table.get("acceptanceDateTime") or [None] * len(table["form"])
        report_column = table.get("reportDate") or [""] * len(table["form"])

        for index, form in enumerate(table["form"]):
            if form not in forms:
                continue
            accepted_at = parse_accepted(accepted_column[index], table["filingDate"][index])
            if not (start <= accepted_at <= end):
                continue
            report_raw = report_column[index]
            yield Filing(
                cik=self.cik,
                form=str(form),
                accession=str(table["accessionNumber"][index]),
                accepted_at=accepted_at,
                report_date=dt.date.fromisoformat(report_raw) if report_raw else None,
                primary_document=str(table["primaryDocument"][index] or ""),
            )


def load_ticker_map(client: HttpClient, source: str) -> dict[str, int]:
    """Ticker to CIK, from the SEC's own mapping of current registrants."""
    payload = client.get_json(TICKER_URL)
    if not isinstance(payload, dict):
        raise SourceError(source, "company_tickers.json was not an object")
    return {
        str(row["ticker"]).upper(): int(row["cik_str"])
        for row in payload.values()
        if isinstance(row, dict) and row.get("ticker")
    }


# ------------------------------------------------------------------------- xml --
def parse_xml(source: str, text: str, what: str) -> Element:
    try:
        root: Element = SafeET.fromstring(text)
    except (SafeET.ParseError, DefusedXmlException) as exc:
        raise SourceError(source, f"{what} is not well-formed XML: {exc}") from exc
    return root


def local_name(element: Element) -> str:
    """Tag without its namespace. 13F documents declare theirs inconsistently --
    default on one filer's table, prefixed on the next -- and the names are what
    carry the meaning."""
    return element.tag.rpartition("}")[2]


def child_text(element: Element, *path: str) -> str | None:
    """Stripped text at a path of local names, or ``None`` if any step is absent
    or the text is empty."""
    node: Element | None = element
    for step in path:
        if node is None:
            return None
        node = next((child for child in node if local_name(child) == step), None)
    if node is None or node.text is None:
        return None
    return node.text.strip() or None
