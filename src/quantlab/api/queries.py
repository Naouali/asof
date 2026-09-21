"""What the app asks of the lake.

Every read goes through a :class:`Lens`: one point-in-time snapshot, and the
frames and events built from it. A page is a pure function of a lens, so "as of 14
August" is decided once, where the lens is made, and cannot be forgotten by a
query written later.

There is one deliberate exception, and it is labelled. When the reader has
travelled to a past date, the feed also reports what had ALREADY HAPPENED by then
but was not public yet. That needs today's knowledge, so it is read through a
second, live lens, and it is returned under its own key (``beyond``) so the
interface can only ever draw it behind the curtain. Things that had not happened
yet are not returned at all: they are the future, and nobody's business here.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import replace
from functools import cached_property
from typing import Any

import polars as pl
from polars.exceptions import ComputeError

from quantlab.api.events import (
    Event,
    congress_events,
    contract_events,
    eastern_date,
    fund_events,
    insider_events,
    money,
    unread_report_events,
)
from quantlab.data.ingest import recent_runs, symbols_in
from quantlab.data.schemas import empty_frame
from quantlab.data.sources.usaspending import DEFENSE_EMBARGO
from quantlab.data.store import Store, utcnow
from quantlab.data.unpriceable import Unpriceable

__all__ = ["Lens", "data_health", "feed", "search", "ticker_page"]

#: How far past the as-of date the feed looks for things that had already
#: happened but were not public yet.
BEYOND_DAYS = 200
#: How many of them are sent. The interface draws forty; the count is sent whole.
BEYOND_LIMIT = 120
PRICE_HISTORY_DAYS = 370
TICKER_EVENT_DAYS = 370
#: The two windows a ticker page's opening paragraph talks about.
BRIEF_RECENT_DAYS = 30
BRIEF_QUARTER_DAYS = 90
#: Datasets whose tickers the app expects to have prices for. Mirrors the
#: `symbols_from` of the price job in configs/ingest.yaml.
PRICED_FROM = ("congress_trades", "insider_transactions", "government_contracts")
UNPRICEABLE_SHOWN = 60
#: Fund reports older than this are not compared at all. The longest window any
#: page shows is a year, and a report surfaces up to 45 days after its quarter.
FUND_HISTORY_DAYS = 550
#: Contract actions this size and over reach the feed: about thirty a week across
#: the mapped contractors. Below it there are fifty a day, nearly all of them
#: funding increments, and they would bury every trade on the page.
FEED_CONTRACT_MIN_USD = 25_000_000.0
#: How many of a company's contract actions its page lists, largest first.
TICKER_CONTRACTS = 30
#: How many of those are also drawn on the price chart.
CHART_CONTRACTS = 8


class Lens:
    """The lake as it was knowable at one instant, and what the app derives from it."""

    def __init__(self, store: Store, as_of: dt.datetime) -> None:
        self.store = store
        self.as_of = as_of
        self.snapshot = store.as_of(as_of)
        self._frames: dict[str, pl.DataFrame] = {}

    @property
    def today(self) -> dt.date:
        return eastern_date(self.as_of)

    def frame(self, dataset: str) -> pl.DataFrame:
        if dataset not in self._frames:
            try:
                found = self.snapshot.frame(dataset)
            except (FileNotFoundError, ComputeError):
                # Nothing ingested yet. An empty page with a way forward is the
                # right answer to that; a 500 is not.
                found = empty_frame(dataset)
            if dataset == "institutional_holdings":
                found = self._with_first_known(found)
            self._frames[dataset] = found
        return self._frames[dataset]

    def _with_first_known(self, holdings: pl.DataFrame) -> pl.DataFrame:
        """Add ``first_known_at``: when each position FIRST became public.

        The snapshot keeps the latest version of a row, which is right for its
        value and wrong for its date. A fund that files on time in August and
        amends in September has every restated position stamped September, and
        the whole book then reads as disclosed late. The date a position became
        public is the earliest filing that showed it.
        """
        if holdings.height == 0:
            return holdings.with_columns(pl.col("known_at").alias("first_known_at"))
        key = ["symbol", "cusip", "put_call", "shares_type", "as_of"]
        first = (
            self.store.scan("institutional_holdings", known_before=self.as_of, dedup=False)
            .group_by(key)
            .agg(pl.col("known_at").min().alias("first_known_at"))
            .collect()
        )
        return holdings.join(first, on=key, how="left").with_columns(
            pl.col("first_known_at").fill_null(pl.col("known_at"))
        )

    @cached_property
    def bridge(self) -> dict[str, str]:
        """CUSIP to ticker, from the fails-to-deliver files.

        The most frequent pairing wins: a handful of CUSIPs carry more than one
        ticker over time, and a handful of tickers more than one CUSIP.
        """
        try:
            pairs = (
                self.store.scan("fails_to_deliver", known_before=self.as_of, dedup=False)
                .group_by("cusip")
                .agg(pl.col("symbol").mode().first().alias("ticker"))
                .collect()
            )
        except (FileNotFoundError, ComputeError):
            return {}
        return dict(zip(pairs["cusip"].to_list(), pairs["ticker"].to_list(), strict=True))

    @property
    def fund_since(self) -> dt.date:
        return self.today - dt.timedelta(days=FUND_HISTORY_DAYS)

    @cached_property
    def events(self) -> list[Event]:
        trades = self.frame("congress_trades")
        found = [
            *insider_events(self.frame("insider_transactions")),
            *congress_events(trades),
            *unread_report_events(self.frame("congress_filings"), trades),
            *fund_events(self.frame("institutional_holdings"), self.bridge, since=self.fund_since),
            *contract_events(self.frame("government_contracts"), min_usd=FEED_CONTRACT_MIN_USD),
        ]
        found.sort(key=lambda event: (event.disclosed_at, event.id), reverse=True)
        return found

    @cached_property
    def fund_changes(self) -> list[Event]:
        """Every change in every tracked fund's book, unabridged. The feed keeps each
        filing's largest few; a count of how many funds bought something needs all."""
        return fund_events(
            self.frame("institutional_holdings"),
            self.bridge,
            per_filing=None,
            since=self.fund_since,
        )

    def cusips_for(self, ticker: str) -> list[str]:
        return [cusip for cusip, symbol in self.bridge.items() if symbol == ticker]


# -------------------------------------------------------------------------- feed --
def _department_id(event: Event) -> str | None:
    """A contract is signed by an office ("Dept of the Navy") inside a department
    ("Department of Defense"). Both have a page; this is the department's id."""
    return f"agency:{event.role.lower()}" if event.kind == "contract" and event.role else None


def _by(event: Event, actor: str | None) -> bool:
    return actor is None or event.actor_id == actor or _department_id(event) == actor


def feed(lens: Lens, *, days: int, actor: str | None, live: Lens | None) -> dict[str, Any]:
    since = lens.today - dt.timedelta(days=days)
    chosen = [event for event in lens.events if event.disclosed_on > since and _by(event, actor)]

    groups: dict[dt.date, list[Event]] = defaultdict(list)
    for event in chosen:
        groups[event.disclosed_on].append(event)

    beyond: list[Event] = []
    if live is not None:
        horizon = lens.today + dt.timedelta(days=BEYOND_DAYS)
        beyond = [
            event
            for event in live.events
            if event.kind != "unread"
            and event.traded_on <= lens.today < event.disclosed_on <= horizon
            and _by(event, actor)
        ]
        # Soonest to surface first: the secret that was about to stop being one.
        beyond.sort(key=lambda event: (event.disclosed_at, event.id))
    # Capped apart. The Pentagon's embargo keeps hundreds of awards behind the
    # curtain at any moment, and one cap would leave no room there for a trade.
    hidden_trades = [event for event in beyond if event.kind != "contract"]
    hidden_contracts = [event for event in beyond if event.kind == "contract"]

    holdings = lens.frame("institutional_holdings")
    latest_period = holdings["as_of"].max() if holdings.height else None
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "is_live": live is None,
        "days": days,
        "actor": _actor_label(lens, actor),
        "groups": [
            {"date": day, "events": [event.as_dict() for event in events]}
            for day, events in sorted(groups.items(), reverse=True)
        ],
        "insights": _cluster_buys(chosen),
        "beyond": [
            event.as_dict()
            for event in [*hidden_trades[:BEYOND_LIMIT], *hidden_contracts[:BEYOND_LIMIT]]
        ],
        "beyond_total": len(hidden_trades),
        "beyond_contracts_total": len(hidden_contracts),
        "coverage": {
            "unread_reports": sum(1 for event in chosen if event.kind == "unread"),
            "fund_period": latest_period.date() if isinstance(latest_period, dt.datetime) else None,
            "fund_period_age_days": (lens.today - latest_period.date()).days
            if isinstance(latest_period, dt.datetime)
            else None,
        },
        "empty_lake": not lens.events,
    }


def _actor_label(lens: Lens, actor: str | None) -> dict[str, Any] | None:
    if actor is None:
        return None
    match = next((event for event in lens.events if event.actor_id == actor), None)
    if match is None:
        # A department's page: named after the department, which is its offices' role.
        office = next((event for event in lens.events if _department_id(event) == actor), None)
        if office is not None:
            return {"id": actor, "name": office.role, "role": "Federal department"}
    return {
        "id": actor,
        "name": match.actor if match else actor,
        "role": match.role if match else None,
    }


def _cluster_buys(events: list[Event]) -> list[dict[str, Any]]:
    """Several insiders of one company buying on the open market at once -- the
    one pattern in this data with a long record of meaning something."""
    buyers: dict[str, dict[str, float]] = defaultdict(dict)
    names: dict[str, str] = {}
    for event in events:
        if (
            event.kind == "insider"
            and event.direction == "buy"
            and not event.noise
            and event.ticker
        ):
            previous = buyers[event.ticker].get(event.actor_id, 0.0)
            buyers[event.ticker][event.actor_id] = previous + (event.value_usd or 0.0)
            names[event.ticker] = event.asset or event.ticker
    return [
        {
            "ticker": ticker,
            "text": f"{len(who)} {names[ticker]} insiders bought on the open market in this "
            f"window, {money(sum(who.values()))} in total. None of it was pre-scheduled.",
        }
        for ticker, who in buyers.items()
        if len(who) >= 2
    ]


# ------------------------------------------------------------------------ ticker --
def ticker_page(lens: Lens, ticker: str) -> dict[str, Any]:
    ticker = ticker.upper()
    since = lens.today - dt.timedelta(days=TICKER_EVENT_DAYS)
    cusips = lens.cusips_for(ticker)

    prices = _prices(lens, ticker)
    fund_changes = fund_events(
        lens.frame("institutional_holdings"),
        lens.bridge,
        per_filing=None,
        only_cusips=cusips,
        since=lens.fund_since,
    )
    # The feed keeps only each filing's largest fund changes. This page wants every
    # change in THIS security, so funds come from the unabridged list instead.
    # Contracts have a section of their own: a large contractor has a thousand a year.
    named = [
        event
        for event in lens.events
        if event.ticker == ticker and event.kind not in {"fund", "contract"}
    ]
    events = [event for event in [*named, *fund_changes] if event.disclosed_on > since]
    events.sort(key=lambda event: (event.disclosed_at, event.id), reverse=True)
    events = [replace(event, price_move_pct=_move(prices, event)) for event in events]

    holders = _holders(lens, cusips)
    contracts = _contracts(lens, ticker, since, prices)
    name = _company_name(events) or _ftd_name(lens, ticker)
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "ticker": ticker,
        "name": name,
        "known": bool(events or holders or prices or contracts),
        "prices": [{"date": day, "close": close} for day, close in prices],
        "events": [event.as_dict() for event in events],
        "holders": holders,
        "contracts": contracts,
        "fails_to_deliver": _latest_fails(lens, ticker),
        "brief": [
            *_brief(lens, ticker, events, holders, bool(contracts)),
            *_contract_brief(contracts),
        ],
    }


def _contract_brief(contracts: dict[str, Any] | None) -> list[str]:
    if contracts is None:
        return []
    actions = contracts["actions"]
    sentence = (
        f"The federal government committed a net {money(contracts['net_usd'])} to it across "
        f"{actions} contract action{'' if actions == 1 else 's'} made public in the last year"
    )
    share = contracts["defense_share"]
    if share is not None and share >= 0.5:
        sentence += f", {share:.0%} of it from the Pentagon, which publishes {DEFENSE_EMBARGO.days} days late"
    return [sentence + "."]


def _contracts(
    lens: Lens, ticker: str, since: dt.date, prices: list[tuple[dt.date, float]]
) -> dict[str, Any] | None:
    """A company's federal contract actions that became public in the last year.

    Every action of $1 million and over is counted; the largest are listed. The
    net figure is money committed less money taken back, and is NOT revenue: it is
    spent over the life of each contract, which runs to years.
    """
    frame = lens.frame("government_contracts").filter(pl.col("symbol") == ticker)
    found = [event for event in contract_events(frame) if event.disclosed_on > since]
    if not found:
        return None
    found.sort(key=lambda event: abs(event.value_usd or 0.0), reverse=True)
    listed = [
        replace(event, price_move_pct=_move(prices, event)) for event in found[:TICKER_CONTRACTS]
    ]

    by_agency: dict[str, float] = defaultdict(float)
    for event in found:
        by_agency[event.role or event.actor] += event.value_usd or 0.0
    top = sorted(by_agency.items(), key=lambda pair: pair[1], reverse=True)[:3]
    return {
        "actions": len(found),
        "net_usd": sum(event.value_usd or 0.0 for event in found),
        "taken_back": sum(1 for event in found if event.direction == "sell"),
        "defense_share": _defense_share(frame, since),
        "agencies": [{"name": name, "net_usd": net} for name, net in top],
        "events": [event.as_dict() for event in listed],
        "on_chart": [event.id for event in listed[:CHART_CONTRACTS]],
    }


def _defense_share(frame: pl.DataFrame, since: dt.date) -> float | None:
    """The share of money committed that came from the Pentagon, which is the share
    of this page that is three months old before anyone can read it."""
    # Washington dates, as everywhere else in this app: a contract published at
    # 20:00 in Washington is stamped the next day in UTC, and 13,438 of the lake's
    # rows fall on a different day under the two readings.
    public_on = pl.col("known_at").dt.convert_time_zone("America/New_York").dt.date()
    recent = frame.filter((public_on > since) & (pl.col("obligation_usd") > 0))
    total = float(recent["obligation_usd"].sum()) if recent.height else 0.0
    if total <= 0:
        return None
    defense = float(recent.filter(pl.col("defense"))["obligation_usd"].sum())
    return round(defense / total, 3)


def _company_name(events: list[Event]) -> str | None:
    """The issuer's name as the SEC has it, before anybody else's spelling of it.

    A House member writes "Apple Inc. - Common Stock (AAPL)" and a 13F writes
    "APPLE INC". The insider filings carry the registrant's own name.
    """
    for kind in ("insider", "fund", "congress"):
        found = next((event.asset for event in events if event.kind == kind and event.asset), None)
        if found:
            return found.title() if found.isupper() else found
    return None


def _prices(lens: Lens, ticker: str) -> list[tuple[dt.date, float]]:
    start = lens.as_of - dt.timedelta(days=PRICE_HISTORY_DAYS)
    try:
        bars = lens.snapshot.ohlcv_daily(symbols=[ticker], start=start)
    except (FileNotFoundError, ComputeError):
        return []
    bars = bars.filter(pl.col("close").is_not_null()).sort("as_of")
    # One source per chart. Two sources for one ticker disagree on adjustments,
    # and a line that alternates between them draws a sawtooth that is not there.
    if bars.height and bars["source"].n_unique() > 1:
        bars = bars.filter(pl.col("source") == bars["source"].mode().first())
    return [
        (eastern_date(moment), float(close))
        for moment, close in zip(bars["as_of"], bars["close"], strict=True)
    ]


def _move(prices: list[tuple[dt.date, float]], event: Event) -> float | None:
    """How far the price moved between the trade and the day it became public."""
    if not prices or event.kind == "unread":
        return None

    def close_on(day: dt.date) -> float | None:
        found = None
        for date, close in prices:
            if date > day:
                break
            found = close
        return found

    before, after = close_on(event.traded_on), close_on(event.disclosed_on)
    if not before or after is None:
        return None
    return round((after - before) / before * 100, 1)


def _holders(lens: Lens, cusips: list[str]) -> list[dict[str, Any]]:
    holdings = lens.frame("institutional_holdings")
    if not cusips or holdings.height == 0:
        return []
    shares_only = holdings.filter((pl.col("put_call") == "NONE") & (pl.col("shares_type") == "SH"))

    out: list[dict[str, Any]] = []
    for (manager,), book in shares_only.group_by("symbol", maintain_order=True):
        periods = sorted(book["as_of"].unique().to_list())
        now = book.filter((pl.col("as_of") == periods[-1]) & pl.col("cusip").is_in(cusips))
        if now.height == 0:
            continue
        before = (
            book.filter((pl.col("as_of") == periods[-2]) & pl.col("cusip").is_in(cusips))
            if len(periods) > 1
            else None
        )
        held = float(now["shares"].sum())
        held_before = float(before["shares"].sum()) if before is not None else None
        period: dt.datetime = periods[-1]
        out.append(
            {
                "manager_id": f"fund:{manager}",
                "manager": str(now["manager_name"][0]),
                "shares": held,
                "change": None if held_before is None else held - held_before,
                "value_usd": float(now["value_usd"].sum()),
                "period": period.date(),
                "disclosed_on": eastern_date(now["first_known_at"].min()),  # type: ignore[arg-type]
                "age_days": (lens.today - period.date()).days,
            }
        )
    out.sort(key=lambda row: row["value_usd"], reverse=True)
    return out


def _ftd_name(lens: Lens, ticker: str) -> str | None:
    rows = _fails(lens, ticker)
    return str(rows["description"][-1]).title() if rows.height and rows["description"][-1] else None


def _fails(lens: Lens, ticker: str) -> pl.DataFrame:
    try:
        return (
            lens.store.scan("fails_to_deliver", symbols=[ticker], known_before=lens.as_of)
            .sort("as_of")
            .collect()
        )
    except (FileNotFoundError, ComputeError):
        return empty_frame("fails_to_deliver")


def _latest_fails(lens: Lens, ticker: str) -> dict[str, Any] | None:
    rows = _fails(lens, ticker)
    if rows.height == 0:
        return None
    last = rows.row(-1, named=True)
    return {
        "quantity": last["quantity"],
        "settled_on": last["as_of"].date(),
        "posted_on": eastern_date(last["known_at"]),
    }


def _brief(
    lens: Lens,
    ticker: str,
    events: list[Event],
    holders: list[dict[str, Any]],
    has_contracts: bool = False,
) -> list[str]:
    """The page's opening paragraph: what the disclosures add up to, in sentences.

    Every sentence is a count of rows on this page, so it can be checked against
    them by eye. Nothing here is a judgement.
    """
    recent = lens.today - dt.timedelta(days=BRIEF_RECENT_DAYS)
    quarter = lens.today - dt.timedelta(days=BRIEF_QUARTER_DAYS)
    insiders = [e for e in events if e.kind == "insider" and not e.noise]
    buys = [e for e in insiders if e.direction == "buy" and e.disclosed_on > recent]
    sells = [e for e in insiders if e.direction == "sell" and e.disclosed_on > recent]
    planned = [
        e
        for e in events
        if e.kind == "insider" and e.noise and e.verb == "Sold" and e.disclosed_on > recent
    ]

    sentences: list[str] = []
    if buys:
        people = len({e.actor_id for e in buys})
        total = money(sum(e.value_usd or 0.0 for e in buys))
        who = "One insider" if people == 1 else f"{people} insiders"
        sentences.append(
            f"{who} bought {total} on the open market in the last {BRIEF_RECENT_DAYS} days, none of it pre-scheduled."
        )
    if sells:
        total = money(sum(e.value_usd or 0.0 for e in sells))
        sentences.append(f"Insiders sold {total} by choice in the last {BRIEF_RECENT_DAYS} days.")
    if planned and not sells:
        total = money(sum(e.value_usd or 0.0 for e in planned))
        sentences.append(
            f"Insiders sold {total} in the last {BRIEF_RECENT_DAYS} days, all of it under plans scheduled in advance."
        )
    if not buys and not sells and not planned:
        last = next((e for e in insiders), None)
        if last is not None:
            sentences.append(
                f"No insider has traded by choice in the last {BRIEF_RECENT_DAYS} days. The last was "
                f"{last.actor}, who {last.verb.lower()} on {last.traded_on:%-d %B}."
            )

    members = [e for e in events if e.kind == "congress" and e.disclosed_on > quarter]
    if members:
        bought = sum(1 for e in members if e.direction == "buy")
        people = len({e.actor_id for e in members})
        who = "One member of Congress" if people == 1 else f"{people} members of Congress"
        sentences.append(
            f"{who} disclosed trades in the last {BRIEF_QUARTER_DAYS} days: {bought} bought, {len(members) - bought} sold."
        )

    if holders:
        oldest = max(row["age_days"] for row in holders)
        funds = "One tracked fund" if len(holders) == 1 else f"{len(holders)} tracked funds"
        sentences.append(f"{funds} reported holding it, as of up to {oldest} days ago.")

    if not sentences and not has_contracts:
        sentences.append(f"No disclosure in the lake names {ticker} yet.")
    return sentences


# ------------------------------------------------------------------------ search --
def search(lens: Lens, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    needle = query.strip().lower()
    if len(needle) < 2:
        return []
    seen: set[tuple[str, str]] = set()
    starts: list[dict[str, Any]] = []
    contains: list[dict[str, Any]] = []

    def offer(kind: str, key: str, label: str, note: str | None) -> None:
        if (kind, key) in seen:
            return
        haystack = f"{key} {label}".lower()
        if needle not in haystack:
            return
        seen.add((kind, key))
        hit = {"kind": kind, "key": key, "label": label, "note": note}
        (
            starts
            if key.lower().startswith(needle) or label.lower().startswith(needle)
            else contains
        ).append(hit)

    for event in lens.events:
        # A contract names the entity that signed, often a subsidiary: a poor label
        # for the ticker, so other disclosures name it first.
        if event.ticker and event.kind != "contract":
            offer("ticker", event.ticker, event.ticker, event.asset)
        if event.kind in {"insider", "congress"}:
            offer("person", event.actor_id, event.actor, event.role)
        elif event.kind == "fund":
            offer("fund", event.actor_id, event.actor, "Fund")
        # A member of Congress has two pages: their trades, and what those add up to.
        if event.kind == "congress":
            offer("portfolio", event.actor_id, event.actor, "Compiled portfolio")
    for event in lens.events:
        if event.kind != "contract":
            continue
        if event.ticker:
            offer("ticker", event.ticker, event.ticker, "Federal contractor")
        offer("agency", event.actor_id, event.actor, event.role or "Federal agency")
        department = _department_id(event)
        if department and event.role:
            offer("agency", department, event.role, "Federal department")
    return [*starts, *contains][:limit]


# ------------------------------------------------------------------- data health --
def data_health(store: Store, lens: Lens) -> dict[str, Any]:
    now = utcnow()
    datasets = [
        {
            "source": stat.source,
            "dataset": stat.dataset,
            "rows": stat.rows,
            "symbols": stat.symbols,
            "first": stat.first_as_of.date() if stat.first_as_of else None,
            "last": stat.last_as_of.date() if stat.last_as_of else None,
            "newest_known": stat.last_known_at,
            "age_days": round((now - stat.last_known_at).total_seconds() / 86400, 1)
            if stat.last_known_at
            else None,
            "bytes": stat.bytes,
        }
        for stat in store.stats()
    ]
    unread = [event.as_dict() for event in lens.events if event.kind == "unread"]
    runs = recent_runs(store.layout.state, limit=10)
    return {
        "datasets": datasets,
        "unread_reports": unread,
        "runs": list(reversed(runs)),
        "unpriceable": _unpriceable(store, lens),
    }


def _unpriceable(store: Store, lens: Lens) -> dict[str, Any]:
    """Tickers somebody disclosed that the lake cannot price.

    Every return figure in the app is measured on the tickers that have prices.
    This says how many do not, and names the ones a source refused, so the gap is
    a number on a page rather than an absence nobody sees.
    """
    wanted = set(symbols_in(store, PRICED_FROM))
    try:
        priced = set(
            store.scan("ohlcv_daily", dedup=False)
            .select("symbol")
            .unique()
            .collect()["symbol"]
            .to_list()
        )
    except (FileNotFoundError, ComputeError):
        priced = set()
    record = Unpriceable(store.layout.state)
    refused = [row for row in record.all() if row.symbol in wanted]
    return {
        "wanted": len(wanted),
        "priced": len(wanted & priced),
        "refused": [
            {
                "symbol": row.symbol,
                "reason": row.reason,
                "attempts": row.attempts,
                "retry_after": eastern_date(row.retry_after),
            }
            for row in refused[:UNPRICEABLE_SHOWN]
        ],
        "refused_total": len(refused),
    }
