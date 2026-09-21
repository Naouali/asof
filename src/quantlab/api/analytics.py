"""Aggregates over the disclosures: what the rows add up to.

Three questions, each a pure function of a :class:`~quantlab.api.queries.Lens`, so
each is answered as of a date like every other page.

*What is being traded* counts PEOPLE, not dollars. Congress discloses a range and
a fund discloses a position, so a dollar total across them would be invented; how
many distinct filers bought and how many sold is a number every source supports.

*Price moved before you knew* measures the one thing this data is about: between
the day somebody traded and the day anyone could find out, what did the price do?
It is measured only where the lake holds prices, which is a short list of large
companies, and the answer says how short.

*Government money* adds up contract actions. Its figures are money committed less
money taken back, spent over the life of contracts that run for years. It is not
revenue, and nothing here calls it that.
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_right
from collections import defaultdict
from statistics import median, quantiles
from typing import Any

import polars as pl
from polars.exceptions import ComputeError

from quantlab.api.events import Event, eastern_date
from quantlab.api.queries import Lens

__all__ = ["contracts", "price_moves", "traded"]

TOP_TICKERS = 14
TOP_PEOPLE = 9
TOP_COMPANIES = 12
TOP_AGENCIES = 8
EXAMPLES = 8
#: Fewer trades than this and a median is an anecdote; the group is listed without one.
MIN_SAMPLE = 5
#: A trade to disclosure gap longer than this is a typo in a date, not a delay.
MAX_PLAUSIBLE_LAG_DAYS = 400


def _trades(lens: Lens, since: dt.date, kind: str | None) -> list[Event]:
    """Decisions that became public after ``since``: no routine filings, no
    contracts, no unread reports. Funds are read unabridged, because the feed's
    list keeps only each filing's four largest changes."""
    people = [event for event in lens.events if event.kind in {"insider", "congress"}]
    funds = lens.fund_changes
    return [
        event
        for event in [*people, *funds]
        if event.disclosed_on > since
        and not event.noise
        and event.direction != "none"
        and (kind is None or event.kind == kind)
    ]


# ------------------------------------------------------------------------- traded --
def traded(lens: Lens, *, days: int, kind: str | None) -> dict[str, Any]:
    since = lens.today - dt.timedelta(days=days)
    events = _trades(lens, since, kind)

    sides: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"buy": set(), "sell": set()})
    names: dict[str, str] = {}
    people: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.ticker:
            sides[event.ticker][event.direction].add(event.actor_id)
            if event.asset and event.kind == "insider":
                names[event.ticker] = event.asset.title() if event.asset.isupper() else event.asset
        if event.kind == "fund":
            continue  # a fund changes thousands of positions; it would top every list
        person = people.setdefault(
            event.actor_id,
            {
                "actor_id": event.actor_id,
                "actor": event.actor,
                "role": event.role,
                "kind": event.kind,
                "buys": 0,
                "sells": 0,
                "tickers": set(),
            },
        )
        person["buys" if event.direction == "buy" else "sells"] += 1
        if event.ticker:
            person["tickers"].add(event.ticker)

    counted = [(ticker, len(side["buy"]), len(side["sell"])) for ticker, side in sides.items()]
    counted.sort(key=lambda row: (row[1] + row[2], row[1], row[0]), reverse=True)
    tickers = [
        {"ticker": ticker, "name": names.get(ticker), "buyers": buyers, "sellers": sellers}
        for ticker, buyers, sellers in counted
    ]
    ranked = sorted(
        people.values(),
        key=lambda person: (person["buys"] + person["sells"], person["actor"]),
        reverse=True,
    )
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "days": days,
        "kind": kind,
        "trades": len(events),
        "tickers_total": len(tickers),
        "tickers": tickers[:TOP_TICKERS],
        "people": [{**person, "tickers": len(person["tickers"])} for person in ranked[:TOP_PEOPLE]],
    }


# -------------------------------------------------------------------- price moves --
def _closes(
    lens: Lens, tickers: set[str], start: dt.date
) -> dict[str, tuple[list[dt.date], list[float]]]:
    if not tickers:
        return {}
    begin = dt.datetime(start.year, start.month, start.day, tzinfo=dt.UTC)
    try:
        bars = lens.snapshot.ohlcv_daily(symbols=sorted(tickers), start=begin)
    except (FileNotFoundError, ComputeError):
        return {}
    bars = bars.filter(pl.col("close").is_not_null()).sort("as_of")
    series: dict[str, tuple[list[dt.date], list[float]]] = {}
    for (ticker,), rows in bars.group_by("symbol", maintain_order=True):
        if rows["source"].n_unique() > 1:  # one source per series, as on the ticker page
            rows = rows.filter(pl.col("source") == rows["source"].mode().first())
        days = [eastern_date(moment) for moment in rows["as_of"]]
        series[str(ticker)] = (days, [float(close) for close in rows["close"]])
    return series


def _close_on(series: tuple[list[dt.date], list[float]], day: dt.date) -> float | None:
    """The last close on or before ``day``."""
    days, closes = series
    index = bisect_right(days, day)
    return closes[index - 1] if index else None


def _spread(values: list[float]) -> dict[str, Any]:
    if len(values) < MIN_SAMPLE:
        return {
            "n": len(values),
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    tenths = quantiles(values, n=10, method="inclusive")
    quarters = quantiles(values, n=4, method="inclusive")
    return {
        "n": len(values),
        "median": round(median(values), 2),
        "p10": round(tenths[0], 2),
        "p25": round(quarters[0], 2),
        "p75": round(quarters[2], 2),
        "p90": round(tenths[8], 2),
    }


_GROUPS = (
    ("insider", "Company insiders"),
    ("house", "Members of the House"),
    ("senate", "Senators"),
    ("fund", "Tracked funds"),
)


def _group_of(event: Event) -> str:
    if event.kind == "congress":
        return "senate" if event.role == "Senate" else "house"
    return event.kind


def price_moves(lens: Lens, *, days: int) -> dict[str, Any]:
    since = lens.today - dt.timedelta(days=days)
    events = [event for event in _trades(lens, since, None) if event.ticker]
    start = since - dt.timedelta(days=MAX_PLAUSIBLE_LAG_DAYS + 10)
    series = _closes(lens, {event.ticker for event in events if event.ticker}, start)

    moves: dict[tuple[str, str], list[float]] = defaultdict(list)
    measured: list[tuple[float, Event]] = []
    for event in events:
        if event.lag_days > MAX_PLAUSIBLE_LAG_DAYS or event.ticker not in series:
            continue
        before = _close_on(series[event.ticker], event.traded_on)
        after = _close_on(series[event.ticker], event.disclosed_on)
        if not before or after is None:
            continue
        move = (after - before) / before * 100
        moves[(_group_of(event), event.direction)].append(move)
        measured.append((move, event))

    # "Against the follower": the price rose before a purchase was public, or fell
    # before a sale was. That is the move a reader of the disclosure had missed.
    missed = sorted(
        measured,
        key=lambda pair: pair[0] if pair[1].direction == "buy" else -pair[0],
        reverse=True,
    )
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "days": days,
        "trades": len(events),
        "measured": len(measured),
        "priced_tickers": len(series),
        "groups": [
            {
                "key": key,
                "label": label,
                "buy": _spread(moves[(key, "buy")]),
                "sell": _spread(moves[(key, "sell")]),
            }
            for key, label in _GROUPS
        ],
        # People only: a fund's "trade date" is a quarter end, not a decision.
        "examples": [
            {**event.as_dict(), "price_move_pct": round(move, 1)}
            for move, event in [pair for pair in missed if pair[1].kind != "fund"][:EXAMPLES]
        ],
    }


# ---------------------------------------------------------------------- contracts --
def contracts(lens: Lens, *, live: Lens | None) -> dict[str, Any]:
    year_ago = lens.today - dt.timedelta(days=365)
    frame = lens.frame("government_contracts")
    if frame.height:
        frame = frame.with_columns(
            pl.col("known_at").dt.convert_time_zone("America/New_York").dt.date().alias("public_on")
        ).filter(pl.col("public_on") > year_ago)

    def ranked(column: str, limit: int) -> list[dict[str, Any]]:
        if frame.height == 0:
            return []
        totals = (
            frame.group_by(column)
            .agg(
                pl.col("obligation_usd").sum().alias("net_usd"),
                pl.col("obligation_usd").filter(pl.col("defense")).sum().alias("defense_usd"),
                pl.len().alias("actions"),
            )
            .sort("net_usd", descending=True)
            .head(limit)
        )
        return [
            {
                "name": row[column],
                "net_usd": float(row["net_usd"]),
                "defense_usd": float(row["defense_usd"] or 0.0),
                "actions": int(row["actions"]),
            }
            for row in totals.iter_rows(named=True)
        ]

    months: list[dict[str, Any]] = []
    if frame.height:
        monthly = (
            frame.with_columns(pl.col("public_on").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(
                pl.col("obligation_usd")
                .filter(pl.col("obligation_usd") > 0)
                .sum()
                .alias("committed"),
                pl.col("obligation_usd")
                .filter(pl.col("obligation_usd") < 0)
                .sum()
                .alias("taken_back"),
            )
            .sort("month")
        )
        months = [
            {
                "month": row["month"],
                "committed_usd": float(row["committed"] or 0.0),
                "taken_back_usd": float(row["taken_back"] or 0.0),
            }
            for row in monthly.iter_rows(named=True)
        ]

    # What had been signed by the as-of date and was not public yet. Only a later
    # date can know that, so it exists only when the reader has travelled back.
    hidden = None
    if live is not None:
        later = live.frame("government_contracts")
        moment = lens.as_of
        if later.height:
            unseen = later.filter((pl.col("as_of") <= moment) & (pl.col("known_at") > moment))
            hidden = {
                "actions": unseen.height,
                "net_usd": float(unseen["obligation_usd"].sum()) if unseen.height else 0.0,
                "defense_actions": int(unseen["defense"].sum()) if unseen.height else 0,
            }

    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "is_live": live is None,
        "actions": frame.height,
        "net_usd": float(frame["obligation_usd"].sum()) if frame.height else 0.0,
        "defense_usd": float(frame.filter(pl.col("defense"))["obligation_usd"].sum())
        if frame.height
        else 0.0,
        "companies": ranked("symbol", TOP_COMPANIES),
        "agencies": ranked("awarding_agency", TOP_AGENCIES),
        "months": months,
        "hidden": hidden,
    }
