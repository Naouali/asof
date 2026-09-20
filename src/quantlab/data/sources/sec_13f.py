"""Institutional holdings, from SEC Form 13F.

Every manager with over $100 million in 13(f) securities reports its long
positions each quarter. This fetcher follows a list of managers -- the "whales" --
rather than the whole population, which is several thousand filers a quarter.

**Four things here are easy to get silently wrong.**

*The date.* ``as_of`` is the quarter end the positions are counted on and
``known_at`` is when EDGAR accepted the filing, up to 45 days later. Dating a
holding by its quarter end is the standard way to build a "follow the smart
money" result that could not have been traded.

*The unit.* ``value`` was reported in thousands of dollars until the SEC changed
the form, and in dollars by every filing made on or after 3 January 2023. Same
field, same schema, nothing in the payload to say which. It is normalised to
dollars here by filing date. Left alone, every position appears to grow a
thousandfold overnight.

*One position, several lines.* A manager reporting for co-managers splits a
holding across lines -- Berkshire's June 2026 filing lists Apple on twelve of them.
Lines are summed per (CUSIP, put/call, share type), because a consumer reading
the first line as the position understates it by whatever the others held.

*No ticker.* A 13F identifies securities by CUSIP only, and there is no free
CUSIP master. ``symbol`` is therefore the manager, which is what a job asks for,
and the security stays a CUSIP. ``fails_to_deliver`` is the free bridge to a
ticker, for the securities it happens to cover.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.sources.sec_filings import (
    Filing,
    Submissions,
    child_text,
    local_name,
    parse_xml,
)
from quantlab.logging import get_logger

__all__ = [
    "DOLLARS_FROM",
    "HOLDINGS_FORMS",
    "InstitutionalHoldings",
    "parse_information_table",
]

log = get_logger("quantlab.data.sources.sec_13f")

#: 13F-NT is a notice that another manager reports the holdings; it has no table.
HOLDINGS_FORMS = frozenset({"13F-HR", "13F-HR/A"})

#: Filings made on or after this date report `value` in dollars; earlier ones
#: report thousands of dollars.
DOLLARS_FROM = dt.date(2023, 1, 3)

PRIMARY_DOCUMENT = "primary_doc.xml"


def parse_information_table(source: str, text: str, what: str) -> list[dict[str, Any]]:
    """Positions from one information table, summed over split lines.

    ``value`` is returned exactly as reported; the caller knows the filing date
    and therefore the unit.
    """
    root = parse_xml(source, text, what)
    if local_name(root) != "informationTable":
        raise SourceError(source, f"{what}: expected an informationTable, got <{local_name(root)}>")

    positions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in root:
        if local_name(entry) != "infoTable":
            continue
        cusip = (child_text(entry, "cusip") or "").upper()
        value = child_text(entry, "value")
        shares = child_text(entry, "shrsOrPrnAmt", "sshPrnamt")
        shares_type = child_text(entry, "shrsOrPrnAmt", "sshPrnamtType")
        if not cusip or value is None or shares is None or shares_type is None:
            raise SourceError(
                source,
                f"{what}: a position is missing its cusip, value or amount. The table "
                "layout may have changed; refusing to load a partial portfolio.",
            )
        put_call = (child_text(entry, "putCall") or "NONE").upper()

        key = (cusip, put_call, shares_type.upper())
        position = positions.setdefault(
            key,
            {
                "cusip": cusip,
                "put_call": put_call,
                "shares_type": shares_type.upper(),
                "issuer_name": child_text(entry, "nameOfIssuer"),
                "title_of_class": child_text(entry, "titleOfClass"),
                "shares": 0.0,
                "value": 0.0,
                "lines": 0,
            },
        )
        position["shares"] += float(shares)
        position["value"] += float(value)
        position["lines"] += 1
    return list(positions.values())


@register
class InstitutionalHoldings(Source):
    """Quarterly 13F holdings for a list of managers, identified by CIK."""

    name: ClassVar[str] = "sec_13f.institutional_holdings"
    dataset: ClassVar[str] = "institutional_holdings"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY
    spec: ClassVar = get_source("sec_13f")

    #: Berkshire Hathaway, Bridgewater, Renaissance Technologies, Pershing Square,
    #: Scion. CIKs, because manager names are not unique and change.
    default_symbols: ClassVar[tuple[str, ...]] = (
        "1067983",
        "1350694",
        "1037389",
        "1336528",
        "1649339",
    )

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        managers = list(dict.fromkeys(symbols)) or list(self.default_symbols)

        rows: list[dict[str, Any]] = []
        for manager in managers:
            index = Submissions(self.client, self.name, _cik(self.name, manager))
            filings = no_table = 0
            for filing in index.filings(HOLDINGS_FORMS, start, end):
                filings += 1
                positions = self._positions(manager, index.name, filing)
                if positions is None:
                    no_table += 1
                    continue
                rows.extend(positions)
            log.info(
                "13f.manager",
                manager=manager,
                name=index.name,
                filings=filings,
                skipped_no_xml_table=no_table,
            )
        # No rows is a legitimate answer for a window with no filing in it.
        return self.finalise(rows)

    def _positions(
        self, manager: str, manager_name: str, filing: Filing
    ) -> list[dict[str, Any]] | None:
        what = f"{manager} {filing.form} {filing.accession}"
        if filing.report_date is None:
            raise SourceError(self.name, f"{what}: the filing index gives no report period")

        table_document = self._table_document(filing, what)
        if table_document is None:
            return None  # before mid-2013 the table was free text

        positions = parse_information_table(
            self.name, self.client.get_text(filing.url(table_document)), what
        )
        amendment_type = self._check_cover_page(filing, what, positions)
        scale = 1.0 if filing.accepted_at.date() >= DOLLARS_FROM else 1000.0
        period = filing.report_date
        as_of = dt.datetime(period.year, period.month, period.day, tzinfo=dt.UTC)

        return [
            {
                "symbol": manager,
                "as_of": as_of,
                "known_at": filing.accepted_at,
                "manager_name": manager_name,
                "cusip": position["cusip"],
                "issuer_name": position["issuer_name"],
                "title_of_class": position["title_of_class"],
                "put_call": position["put_call"],
                "shares_type": position["shares_type"],
                "shares": position["shares"],
                "value_usd": position["value"] * scale,
                "lines": position["lines"],
                "form": filing.form,
                "amendment_type": amendment_type,
                "accession": filing.accession,
            }
            for position in positions
        ]

    def _table_document(self, filing: Filing, what: str) -> str | None:
        """The information table's filename, which the filer chooses.

        Nothing in the filing index names it, so the filing's directory is listed
        and the table is whichever XML document is not the cover page.
        """
        listing = self.client.get_json(filing.url("index.json"))
        try:
            items = listing["directory"]["item"]
        except (KeyError, TypeError) as exc:
            raise SourceError(self.name, f"{what}: unreadable directory listing") from exc

        candidates = [
            str(item["name"])
            for item in items
            if str(item.get("name", "")).lower().endswith(".xml")
            and str(item["name"]).lower() != PRIMARY_DOCUMENT
        ]
        if len(candidates) > 1:
            raise SourceError(
                self.name,
                f"{what}: {len(candidates)} candidate information tables {candidates}. "
                "Choosing one would load part of a portfolio as though it were all of it.",
            )
        return candidates[0] if candidates else None

    def _check_cover_page(
        self, filing: Filing, what: str, positions: Sequence[dict[str, Any]]
    ) -> str | None:
        """Compare the table with the totals its own cover page declares, and
        return the amendment type: RESTATEMENT replaces the original table, NEW
        HOLDINGS adds to it.

        A disagreement is logged, not raised. The totals are typed by the filer
        and are sometimes simply wrong, and refusing a manager's whole quarter
        over a filer's arithmetic would lose real data to a clerical error.
        """
        root = parse_xml(self.name, self.client.get_text(filing.url(PRIMARY_DOCUMENT)), what)
        found = {local_name(element): (element.text or "").strip() for element in root.iter()}

        lines = sum(position["lines"] for position in positions)
        value = sum(position["value"] for position in positions)
        declared_lines, declared_value = found.get("tableEntryTotal"), found.get("tableValueTotal")
        # Every line is rounded to a whole dollar (once, to a whole thousand), so a
        # table of four thousand lines can honestly miss its total by a few thousand.
        tolerance = max(1.0, lines / 2)
        if (declared_lines and declared_lines.isdigit() and int(declared_lines) != lines) or (
            declared_value
            and declared_value.isdigit()
            and abs(int(declared_value) - value) > tolerance
        ):
            log.warning(
                "13f.cover_page_mismatch",
                filing=what,
                declared_lines=declared_lines,
                read_lines=lines,
                declared_value=declared_value,
                read_value=value,
            )
        return (found.get("amendmentType") or "").upper() or None


def _cik(source: str, manager: str) -> int:
    """A manager symbol as a CIK.

    Written without leading zeros, and refused otherwise. The symbol is stored as
    given and is what incremental ingest looks up, so ``0001067983`` and
    ``1067983`` would be two managers to the lake. Unquoted in YAML, a zero-padded
    number made only of the digits 0-7 is also read as OCTAL, which turns one
    manager into a different one without any error.
    """
    if not manager.isdigit() or manager != str(int(manager)):
        raise ValueError(
            f"{source}: manager {manager!r} is not a CIK written without leading zeros. "
            "Names are not accepted: they are not unique, and they change."
        )
    return int(manager)
