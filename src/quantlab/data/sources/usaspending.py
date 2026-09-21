"""Federal contract actions, from USAspending.gov.

The Treasury's record of what the US government has committed to pay whom. Every
award and every change to one -- funding added, an option exercised, money taken
back -- is a row, reported by the agency that signed it.

**Three things here are easy to get silently wrong.**

*When anyone could know.* The record gives an action date. It does not give the
date the action became public, and for the Department of Defense those differ by
policy: defence actions are withheld for 90 days. The record itself carries no
trace of the embargo. ``known_at`` applies it -- see :func:`known_at` -- and a
chart that dates defence contracts by action date is looking three months ahead.

*Who got the money.* A contract names a legal entity: "Sikorsky Aircraft
Corporation", "Electric Boat Corporation", "CSRA LLC". The government groups
entities under a parent, identified by a UEI, but one listed company is spread
over several parents -- General Dynamics over five -- and no record anywhere links
a parent to a ticker. ``configs/contractors.yaml`` is that link, curated by hand
from the government's own list of its largest contractors, and every ticker in it
was checked against the SEC's list of registrants. A company not in it is absent from
the dataset, not idle. Joint ventures belong to no single ticker and are left out;
so are parents listed abroad, whose US tickers are depositary receipts.

*What the number means.* ``obligation_usd`` is what one action committed, and is
negative when money is de-obligated. ``potential_value_usd`` is the ceiling of the
whole award with every option exercised, which is the figure a press release
quotes and the one that must never be summed across actions.

The search API returns a dozen fields and no reporting date, so this reads the
download service instead: a query is posted, a zip of CSVs is generated, and the
columns asked for come back. Generation is slow -- half a minute for one company
and one quarter, eight minutes for the four largest contractors over three years
-- so a first ingest takes the better part of an hour and a daily one minutes.

To keep those files small the service is asked only for awards worth at least the
minimum action size. An action that large inside an award whose NET total is
smaller -- money committed and then mostly taken back -- is therefore missed.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import time
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import yaml

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = [
    "DEFAULT_MIN_OBLIGATION",
    "DEFENSE_EMBARGO",
    "PUBLICATION_LAG",
    "ContractActions",
    "Contractor",
    "known_at",
    "load_contractors",
    "parse_transactions",
]

log = get_logger("quantlab.data.sources.usaspending")

DOWNLOAD_URL = "https://api.usaspending.gov/api/v2/download/transactions/"
CONTRACT_TYPES = ("A", "B", "C", "D")  # purchase orders, delivery orders, definitive contracts
DEFENSE = "department of defense"
#: The nightly load from the procurement system into the public record.
PUBLICATION_LAG = dt.timedelta(days=2)
#: The Pentagon's embargo on its own contract actions.
DEFENSE_EMBARGO = dt.timedelta(days=90)
#: How far before the window to look by ACTION date, so that an action reported
#: late, or released from the embargo, inside the window is still found.
LOOKBACK = dt.timedelta(days=200)
#: Actions smaller than this, in absolute value, are dropped unless the job says
#: otherwise. Most actions are administrative modifications of a few thousand dollars.
DEFAULT_MIN_OBLIGATION = 1_000_000.0
CONTRACTORS_FILE = "contractors.yaml"
#: Parent identifiers per download. The filter matches any of them.
BATCH = 12
#: The record has some 300 columns. Asking for these alone makes the file a
#: fourteenth of the size; generating it still takes minutes for a long window.
COLUMNS = (
    "contract_transaction_unique_key",
    "award_id_piid",
    "modification_number",
    "parent_award_id_piid",
    "federal_action_obligation",
    "total_dollars_obligated",
    "potential_total_value_of_award",
    "action_date",
    "awarding_agency_name",
    "awarding_sub_agency_name",
    "recipient_uei",
    "recipient_name",
    "recipient_parent_uei",
    "recipient_parent_name",
    "action_type",
    "transaction_description",
    "naics_code",
    "naics_description",
    "usaspending_permalink",
    "initial_report_date",
)


@dataclass(frozen=True, slots=True)
class Contractor:
    ticker: str
    name: str
    parent_ueis: tuple[str, ...]


def load_contractors(path: Path) -> tuple[Contractor, ...]:
    """The curated link from the government's parent identifiers to tickers.

    It lives in ``configs/contractors.yaml`` because it is a judgement somebody has
    to keep current, not a fact of the code. It is checked as it is read: an
    identifier claimed by two companies would file one company's contracts under
    another's ticker, which is the one mistake this list exists to prevent.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise SourceError(
            "usaspending",
            f"{path} does not exist. Contracts name legal entities, not tickers, and that "
            "file is the only link between the two; nothing can be fetched without it.",
        ) from None
    except yaml.YAMLError as exc:
        raise SourceError("usaspending", f"{path} is not valid YAML: {exc}") from exc

    entries = raw.get("contractors") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SourceError("usaspending", f"{path} lists no contractors")

    found: list[Contractor] = []
    owner_of: dict[str, str] = {}
    for entry in entries:
        try:
            ticker = str(entry["ticker"]).upper()
            name = str(entry["name"])
            ueis = tuple(str(uei).upper() for uei in entry["parent_ueis"])
        except (KeyError, TypeError) as exc:
            raise SourceError(
                "usaspending", f"{path}: an entry lacks ticker, name or parent_ueis: {entry!r}"
            ) from exc
        if not ueis:
            raise SourceError("usaspending", f"{path}: {ticker} lists no parent identifier")
        if any(contractor.ticker == ticker for contractor in found):
            raise SourceError("usaspending", f"{path}: {ticker} is listed twice")
        for uei in ueis:
            if len(uei) != 12 or not uei.isalnum():
                raise SourceError(
                    "usaspending", f"{path}: {uei!r} under {ticker} is not a 12-character UEI"
                )
            if uei in owner_of:
                raise SourceError(
                    "usaspending",
                    f"{path}: {uei} is claimed by both {owner_of[uei]} and {ticker}",
                )
            owner_of[uei] = ticker
        found.append(Contractor(ticker, name, ueis))
    return tuple(found)


def known_at(action: dt.date, reported_at: dt.datetime | None, *, defense: bool) -> dt.datetime:
    """When an action could first have been read in the public record.

    It cannot be public before it happened or before the agency reported it, so
    the clock starts at the later of the two. Then comes the nightly load, and for
    the Pentagon its 90-day embargo. This is a rule standing in for a publication
    timestamp the record does not have; it is set to err late, not early.
    """
    happened = dt.datetime(action.year, action.month, action.day, tzinfo=dt.UTC)
    clock = max(happened, reported_at) if reported_at else happened
    return clock + PUBLICATION_LAG + (DEFENSE_EMBARGO if defense else dt.timedelta())


def parse_transactions(source: str, payload: bytes) -> Iterator[dict[str, str]]:
    """Every prime contract transaction in a generated download."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if "PrimeTransactions" in n]
            if not names:
                raise SourceError(
                    source, f"no transactions file among {archive.namelist()} in the download"
                )
            for name in names:
                text = archive.read(name).decode("utf-8-sig")
                yield from csv.DictReader(io.StringIO(text))
    except zipfile.BadZipFile as exc:
        raise SourceError(source, "the download is not a zip archive") from exc


def _money(raw: str | None) -> float | None:
    return float(raw) if raw else None


def _stamp(raw: str | None) -> dt.datetime | None:
    """ "2026-06-09 11:56:25+00" -- the record's timestamps, which are UTC."""
    if not raw:
        return None
    return dt.datetime.fromisoformat(
        raw.replace("+00", "+00:00", 1) if raw.endswith("+00") else raw
    )


@register
class ContractActions(Source):
    """Contract actions of listed federal contractors, by parent company."""

    name: ClassVar[str] = "usaspending.government_contracts"
    spec: ClassVar = get_source("usaspending")
    dataset: ClassVar[str] = "government_contracts"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY

    def __init__(
        self,
        *,
        min_obligation: float = DEFAULT_MIN_OBLIGATION,
        contractors: str | Path | None = None,
        poll_seconds: float = 10.0,
        max_wait_seconds: float = 2400.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_obligation = float(min_obligation)
        self.contractors_path = (
            Path(contractors) if contractors else self.settings.layout.configs / CONTRACTORS_FILE
        )
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        by_ticker = {c.ticker: c for c in load_contractors(self.contractors_path)}
        # No symbols named means every company in the list.
        wanted = [symbol.upper() for symbol in symbols] or list(by_ticker)
        unknown = sorted(set(wanted) - set(by_ticker))
        if unknown:
            raise SourceError(
                self.name,
                f"no parent-company identifiers are recorded for {unknown}. Contracts name "
                f"legal entities, not tickers; add the company to {self.contractors_path} first.",
            )
        ticker_of = {uei: t for t in wanted for uei in by_ticker[t].parent_ueis}

        horizon = min(end, utcnow())
        ueis = sorted(ticker_of)
        rows: list[dict[str, Any]] = []
        keys: set[str] = set()
        seen = small = embargoed = 0
        for offset in range(0, len(ueis), BATCH):
            payload = self._download(ueis[offset : offset + BATCH], start - LOOKBACK, horizon)
            for record in parse_transactions(self.name, payload):
                ticker = ticker_of.get(record.get("recipient_parent_uei", ""))
                if ticker is None:
                    continue  # the text search matched something that is not this parent
                # The search is by text, and one company's identifiers turn up in
                # another's records -- a subsidiary that is a parent elsewhere -- so
                # two downloads can return the same action.
                key = record.get("contract_transaction_unique_key", "")
                if key in keys:
                    continue
                keys.add(key)
                seen += 1
                row = self._row(ticker, record)
                if abs(row["obligation_usd"]) < self.min_obligation:
                    small += 1
                elif row["known_at"] > horizon:
                    embargoed += 1
                elif row["known_at"] >= start:
                    rows.append(row)

        log.info(
            "usaspending.contracts",
            companies=len(wanted),
            actions=seen,
            under_minimum=small,
            not_yet_public=embargoed,
            rows=len(rows),
        )
        return self.finalise(rows)

    def _row(self, ticker: str, record: dict[str, str]) -> dict[str, Any]:
        try:
            action = dt.date.fromisoformat(record["action_date"])
            agency = record["awarding_agency_name"]
            reported_at = _stamp(record.get("initial_report_date"))
            defense = agency.strip().lower() == DEFENSE
            return {
                "symbol": ticker,
                "as_of": dt.datetime(action.year, action.month, action.day, tzinfo=dt.UTC),
                "known_at": known_at(action, reported_at, defense=defense),
                "transaction_key": record["contract_transaction_unique_key"],
                "award_id": record["award_id_piid"],
                "modification_number": record.get("modification_number") or None,
                "parent_award_id": record.get("parent_award_id_piid") or None,
                "recipient_name": record["recipient_name"],
                "recipient_uei": record.get("recipient_uei") or None,
                "parent_name": record.get("recipient_parent_name") or None,
                "parent_uei": record["recipient_parent_uei"],
                "awarding_agency": agency,
                "awarding_sub_agency": record.get("awarding_sub_agency_name") or None,
                "defense": defense,
                "action_type": record.get("action_type") or None,
                "obligation_usd": _money(record["federal_action_obligation"]) or 0.0,
                "total_obligated_usd": _money(record.get("total_dollars_obligated")),
                "potential_value_usd": _money(record.get("potential_total_value_of_award")),
                "description": record.get("transaction_description") or None,
                "naics_code": record.get("naics_code") or None,
                "naics_description": record.get("naics_description") or None,
                "reported_at": reported_at,
                "url": record.get("usaspending_permalink") or None,
            }
        except (KeyError, ValueError) as exc:
            raise SourceError(
                self.name,
                f"a transaction could not be read ({exc!r}). The download's columns may "
                "have changed; the reader must be re-checked.",
            ) from exc

    def _download(self, ueis: Sequence[str], start: dt.datetime, end: dt.datetime) -> bytes:
        """Ask for a download, wait for it to be generated, and fetch it."""
        ticket = self.client.request(
            DOWNLOAD_URL,
            json={
                "filters": {
                    "award_type_codes": list(CONTRACT_TYPES),
                    "time_period": [
                        {"start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}"}
                    ],
                    "recipient_search_text": list(ueis),
                    **(
                        {"award_amounts": [{"lower_bound": self.min_obligation}]}
                        if self.min_obligation > 0
                        else {}
                    ),
                },
                "columns": list(COLUMNS),
            },
        ).json()
        status_url, file_url = ticket.get("status_url"), ticket.get("file_url")
        if not status_url or not file_url:
            raise SourceError(self.name, f"the download request was not accepted: {ticket!r:.300}")

        deadline = time.monotonic() + self.max_wait_seconds
        while True:
            status = self.client.get_json(status_url)
            state = status.get("status")
            if state == "finished":
                return self.client.get_bytes(file_url)
            if state == "failed":
                raise SourceError(self.name, f"the download failed: {status.get('message')!r}")
            if time.monotonic() > deadline:
                raise SourceError(
                    self.name,
                    f"the download was still {state!r} after {self.max_wait_seconds:.0f}s",
                )
            time.sleep(self.poll_seconds)
