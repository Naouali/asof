"""Insider transactions, from SEC Forms 4 and 5.

Officers, directors and 10% owners must report a trade in their own company's
securities within two business days. Each report is a small XML document, and
this fetcher reads them one at a time: the filing index says which exist, and the
document says what was traded.

**Three things here are easy to get silently wrong.**

*The date.* ``as_of`` is the day of the trade and ``known_at`` is the instant
EDGAR accepted the report. The gap is normally two business days, but late
reports run to months and a Form 5 sweeps up a whole year. Dating a trade by when
it happened makes it visible before anyone outside the company could have known.

*What counts as a trade.* Most lines on most Form 4s are compensation: a grant
(A), an option exercise (M), shares withheld to pay the tax on a vesting (F). They
are recorded, because discarding them is a judgement that belongs to whoever
reads the data, but ``transaction_code`` must be filtered before any of it is
read as a view on the stock. Only P and S are open-market decisions.

*Holdings are not transactions.* The same tables carry ``...Holding`` rows that
restate a position without any trade having happened. They are skipped: filing
them beside transactions would invent activity on the filing date.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar
from xml.etree.ElementTree import Element  # the type only; parsing is defusedxml's

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.sources.sec_filings import (
    Filing,
    Submissions,
    child_text,
    load_ticker_map,
    local_name,
    parse_xml,
)
from quantlab.logging import get_logger

__all__ = ["OWNERSHIP_FORMS", "InsiderTransactions", "parse_ownership_document"]

log = get_logger("quantlab.data.sources.sec_insider")

#: Form 3 is an initial statement of holdings and reports no transaction, so it
#: is not read. Form 5 is the annual catch-up for trades exempt from Form 4.
OWNERSHIP_FORMS = frozenset({"4", "4/A", "5", "5/A"})

_TRUE = frozenset({"1", "true"})


def _flag(element: Element, *path: str) -> bool:
    return (child_text(element, *path) or "").lower() in _TRUE


def _number(element: Element, *path: str) -> float | None:
    raw = child_text(element, *path)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_ownership_document(source: str, text: str, what: str) -> list[dict[str, Any]]:
    """Transaction rows from one ownership document, without provenance columns.

    Several people can report on one form -- a fund and its general partner, say.
    The first is recorded as the owner by CIK, every name is kept, and the
    relationship flags are true if they are true of any of them.
    """
    root = parse_xml(source, text, what)
    if local_name(root) != "ownershipDocument":
        raise SourceError(
            source, f"{what}: expected an ownershipDocument, got <{local_name(root)}>"
        )

    owners = [child for child in root if local_name(child) == "reportingOwner"]
    if not owners:
        raise SourceError(source, f"{what}: no reportingOwner")
    names = [child_text(owner, "reportingOwnerId", "rptOwnerName") or "" for owner in owners]
    owner_cik = child_text(owners[0], "reportingOwnerId", "rptOwnerCik")
    relationship = "reportingOwnerRelationship"
    titles = [child_text(owner, relationship, "officerTitle") for owner in owners]

    # Absent on filings from before April 2023, when the checkbox was added, so
    # null means "not asked", which is different from "no".
    planned = child_text(root, "aff10b5One")
    issuer_cik = child_text(root, "issuer", "issuerCik")

    shared = {
        "issuer_cik": int(issuer_cik) if issuer_cik and issuer_cik.isdigit() else None,
        "issuer_name": child_text(root, "issuer", "issuerName"),
        "owner_cik": int(owner_cik) if owner_cik and owner_cik.isdigit() else None,
        "owner_name": "; ".join(name for name in names if name),
        "is_director": any(_flag(owner, relationship, "isDirector") for owner in owners),
        "is_officer": any(_flag(owner, relationship, "isOfficer") for owner in owners),
        "is_ten_percent_owner": any(
            _flag(owner, relationship, "isTenPercentOwner") for owner in owners
        ),
        "officer_title": next((title for title in titles if title), None),
        "planned_10b5_1": None if planned is None else planned.lower() in _TRUE,
    }

    rows: list[dict[str, Any]] = []
    line = 0
    for table_name, is_derivative in (("nonDerivativeTable", False), ("derivativeTable", True)):
        table = next((child for child in root if local_name(child) == table_name), None)
        if table is None:
            continue
        for entry in table:
            if not local_name(entry).endswith("Transaction"):
                continue  # a ...Holding row: a position restated, not a trade
            line += 1
            traded = child_text(entry, "transactionDate", "value")
            if traded is None:
                raise SourceError(source, f"{what}: transaction {line} has no date")
            amounts = "transactionAmounts"
            rows.append(
                {
                    **shared,
                    "line": line,
                    "transaction_date": dt.date.fromisoformat(traded[:10]),
                    "security_title": child_text(entry, "securityTitle", "value"),
                    "is_derivative": is_derivative,
                    "transaction_code": child_text(entry, "transactionCoding", "transactionCode"),
                    "acquired_disposed": child_text(
                        entry, amounts, "transactionAcquiredDisposedCode", "value"
                    ),
                    "shares": _number(entry, amounts, "transactionShares", "value"),
                    # Null when the price exists only as a footnote. Not guessed.
                    "price": _number(entry, amounts, "transactionPricePerShare", "value"),
                    "shares_owned_after": _number(
                        entry, "postTransactionAmounts", "sharesOwnedFollowingTransaction", "value"
                    ),
                    "ownership": child_text(
                        entry, "ownershipNature", "directOrIndirectOwnership", "value"
                    ),
                }
            )
    return rows


@register
class InsiderTransactions(Source):
    """Form 4 and 5 transactions for a list of issuers."""

    name: ClassVar[str] = "sec_insider.insider_transactions"
    dataset: ClassVar[str] = "insider_transactions"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY
    spec: ClassVar = get_source("sec_insider")

    #: Kept short on purpose: every filing is one request, and a large issuer
    #: files several hundred a year.
    default_symbols: ClassVar[tuple[str, ...]] = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ticker_map: dict[str, int] | None = None

    def cik_for(self, ticker: str) -> int:
        if self._ticker_map is None:
            self._ticker_map = load_ticker_map(self.client, self.name)
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise SourceError(
                self.name,
                f"{ticker!r} is not in the SEC's ticker-to-CIK mapping. It lists "
                "current registrants only, so a delisted or acquired issuer cannot be "
                "requested by ticker. It is not substituted with a guess.",
            )
        return cik

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        tickers = list(dict.fromkeys(symbols)) or list(self.default_symbols)

        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            index = Submissions(self.client, self.name, self.cik_for(ticker))
            filings = not_xml = impossible = 0
            for filing in index.filings(OWNERSHIP_FORMS, start, end):
                filings += 1
                document = filing.raw_primary_document
                if not document.lower().endswith(".xml"):
                    not_xml += 1  # pre-2003 paper-era filing, carried as text
                    continue
                kept, dropped = self._rows_for(ticker, filing, document)
                rows.extend(kept)
                impossible += dropped
            log.info(
                "insider.issuer",
                ticker=ticker,
                filings=filings,
                skipped_not_xml=not_xml,
                dropped_dated_after_filing=impossible,
            )
        # No rows is a legitimate answer: a narrow window in which no insider of
        # these issuers filed. Failure would have raised above.
        return self.finalise(rows)

    def _rows_for(
        self, ticker: str, filing: Filing, document: str
    ) -> tuple[list[dict[str, Any]], int]:
        what = f"{ticker} {filing.form} {filing.accession}"
        parsed = parse_ownership_document(
            self.name, self.client.get_text(filing.url(document)), what
        )
        kept: list[dict[str, Any]] = []
        dropped = 0
        for row in parsed:
            traded: dt.date = row.pop("transaction_date")
            as_of = dt.datetime(traded.year, traded.month, traded.day, tzinfo=dt.UTC)
            if as_of > filing.accepted_at:
                # A trade dated after the report of it is a typo in the filing --
                # "2206" for "2026". It cannot be placed in time, so it is counted
                # and dropped rather than allowed to fail the whole issuer.
                dropped += 1
                continue
            kept.append(
                {
                    **row,
                    "symbol": ticker,
                    "as_of": as_of,
                    "known_at": filing.accepted_at,
                    "form": filing.form,
                    "accession": filing.accession,
                }
            )
        return kept, dropped
