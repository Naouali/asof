"""A member's portfolio, compiled from the trades they disclosed.

This is an ESTIMATE of a thing the law does not ask anyone to disclose, and the
page has to be read knowing four things it cannot get round.

*There is no starting point.* A transaction report says what was bought and sold,
never what was already held. So a portfolio here is what a member has bought AND
KEPT since the first report the lake holds for their chamber -- not what they own. A sale
of something never seen bought is evidence of a holding from before, and is listed
as that; it is never netted into a negative position.

*Sizes are ranges.* A trade is disclosed as "$15,001 - $50,000". A position's
size is the MIDPOINT of each range, netted, and its honest low and high are kept
beside it: the least it can be is every purchase at its floor and every sale at
its ceiling, the most is the reverse. An open-ended range ("over $50,000,000")
counts at its floor, because nothing else is known about it.

*An amendment repeats itself.* An amended report re-lists the trades of the one it
corrects, under a new document. Where the same line appears in two documents and
one of them is an amendment, only the latest document's copy is kept. Identical
lines inside ONE document are left alone: people do buy the same bracket of the
same stock twice in a day.

*Options are not shares.* An option trade's range is a premium, not a position.
Options are counted and left out.

Returns are given only where the lake holds prices, and in two versions: since the
member bought, and since the day anyone could have known -- which is the only one
a follower could have had.

*The portfolio's return over time* replays every priced trade: a CLOSED trade earns
its exit price over its entry price and stays there; an OPEN one is marked to each
day's close. The figure on any day is everything gained so far over everything put
in so far. Trades in tickers without prices take no part, and the page says how
many that leaves.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from quantlab.api.analytics import _close_from, _close_on, _closes
from quantlab.api.events import _OWNER_NOTES, _seat, eastern_date, member_id, member_name
from quantlab.api.queries import Lens

__all__ = ["MIN_PRICED_SHARE", "members", "portfolio"]

NO_TICKER = "NO_TICKER"
#: Asset types that are options, in the House's codes and the Senate's words.
OPTION_TYPES = frozenset({"OP", "Stock Option"})
#: A portfolio-wide return is only offered when at least this share of the
#: portfolio, by estimated size, is in tickers the lake holds prices for.
MIN_PRICED_SHARE = 0.5
_LINE = ["member", "symbol", "transaction_type", "as_of", "amount_text", "owner"]


def _trades(lens: Lens) -> pl.DataFrame:
    """Every disclosed transaction that counts, with amendments' repeats removed."""
    frame = lens.frame("congress_trades")
    if frame.height == 0:
        return frame
    frame = frame.filter(pl.col("filing_status").is_null() | (pl.col("filing_status") != "deleted"))
    # The same line in two documents, one of them an amendment: keep the latest document's.
    latest = (
        frame.group_by(_LINE)
        .agg(
            pl.col("doc_id").n_unique().alias("_docs"),
            (pl.col("filing_status") == "amended").any().alias("_amended"),
            pl.col("doc_id").sort_by("known_at").last().alias("_latest"),
        )
        .filter((pl.col("_docs") > 1) & pl.col("_amended"))
        .select(*_LINE, "_latest")
    )
    if latest.height == 0:
        return frame
    return (
        frame.join(latest, on=_LINE, how="left", nulls_equal=True)
        .filter(pl.col("_latest").is_null() | (pl.col("doc_id") == pl.col("_latest")))
        .drop("_latest")
    )


def _identity(row: dict[str, Any]) -> tuple[str, str, str | None]:
    name = member_name(str(row["member"]))
    title, seat = _seat(row["chamber"], row["state_district"])
    return member_id(name, row["state_district"]), f"{title} {name}", seat


@dataclass(frozen=True, slots=True)
class _Trade:
    traded: dt.date
    public: dt.date
    kind: str  # purchase, sale (in full) or sale_partial
    mid: float
    line: int


@dataclass(slots=True)
class _Position:
    ticker: str
    asset: str | None = None
    bought: list[tuple[dt.date, dt.date, float, float]] = field(default_factory=list)
    sold: list[tuple[float, float]] = field(default_factory=list)
    #: Every purchase and sale in order, for replaying at entry and exit prices.
    trades: list[_Trade] = field(default_factory=list)
    accounts: set[str] = field(default_factory=set)
    last_trade: dt.date | None = None

    def add(self, row: dict[str, Any]) -> None:
        low = float(row["amount_min"])
        high = float(row["amount_max"]) if row["amount_max"] is not None else low
        traded = row["as_of"].date()
        public = eastern_date(row["known_at"])
        kind = str(row["transaction_type"])
        if kind == "purchase":
            self.bought.append((traded, public, low, high))
        else:
            self.sold.append((low, high))
        self.trades.append(_Trade(traded, public, kind, (low + high) / 2, int(row["line"] or 0)))
        self.asset = self.asset or row["asset"]
        self.accounts.add(_OWNER_NOTES.get(row["owner"] or "", "their own"))
        self.last_trade = max(self.last_trade or traded, traded)

    @property
    def mid(self) -> float:
        bought = sum((low + high) / 2 for _, _, low, high in self.bought)
        return bought - sum((low + high) / 2 for low, high in self.sold)

    @property
    def low(self) -> float:
        return max(
            0.0, sum(low for _, _, low, _ in self.bought) - sum(high for _, high in self.sold)
        )

    @property
    def high(self) -> float:
        return max(
            0.0, sum(high for _, _, _, high in self.bought) - sum(low for low, _ in self.sold)
        )


# ------------------------------------------------------------------------ members --
def members(lens: Lens) -> dict[str, Any]:
    trades = _trades(lens)
    found: dict[str, dict[str, Any]] = {}
    for row in trades.iter_rows(named=True):
        actor_id, actor, seat = _identity(row)
        entry = found.setdefault(
            actor_id,
            {
                "actor_id": actor_id,
                "actor": actor,
                "role": seat,
                "chamber": row["chamber"],
                "trades": 0,
                "tickers": set(),
                "last_disclosed": None,
            },
        )
        entry["trades"] += 1
        if row["symbol"] != NO_TICKER:
            entry["tickers"].add(row["symbol"])
        disclosed = eastern_date(row["known_at"])
        entry["last_disclosed"] = max(entry["last_disclosed"] or disclosed, disclosed)

    ranked = sorted(found.values(), key=lambda m: (len(m["tickers"]), m["trades"]), reverse=True)
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "members": [{**entry, "tickers": len(entry["tickers"])} for entry in ranked],
    }


# ---------------------------------------------------------------------- portfolio --
def portfolio(lens: Lens, actor_id: str) -> dict[str, Any] | None:
    trades = _trades(lens)
    rows = [row for row in trades.iter_rows(named=True) if _identity(row)[0] == actor_id]
    if not rows:
        return None
    _, actor, seat = _identity(rows[0])
    chamber = rows[0]["chamber"]
    # Where this chamber's record begins: the first REPORT the lake holds. A report
    # can disclose a trade from years earlier, so the first trade date says nothing.
    since = trades.filter(pl.col("chamber") == chamber)["known_at"].min()

    positions: dict[str, _Position] = {}
    unnamed = options = exchanges = 0
    for row in rows:
        if row["asset_type"] in OPTION_TYPES:
            options += 1
        elif row["transaction_type"] == "exchange":
            exchanges += 1
        elif row["symbol"] == NO_TICKER:
            unnamed += 1
        else:
            positions.setdefault(row["symbol"], _Position(row["symbol"])).add(row)

    held = [p for p in positions.values() if p.bought and p.mid > 0]
    closed = [p for p in positions.values() if p.bought and p.mid <= 0]
    prior = [p for p in positions.values() if not p.bought]

    bought = [p for p in positions.values() if p.bought]
    start = min((traded for p in bought for traded, _, _, _ in p.bought), default=lens.today)
    series = _closes(lens, {p.ticker for p in bought}, start - dt.timedelta(days=10))
    performance, exits = _performance(bought, series, lens.today)
    total = sum(p.mid for p in held)

    holdings: list[dict[str, Any]] = []
    for position in sorted(held, key=lambda p: p.mid, reverse=True):
        bought_return = public_return = None
        if position.ticker in series:
            now = _close_on(series[position.ticker], lens.today)
            bought_return = _weighted_return(series[position.ticker], position, now, by="traded")
            public_return = _weighted_return(series[position.ticker], position, now, by="public")
        holdings.append(
            {
                "ticker": position.ticker,
                "asset": position.asset,
                # Not rounded for display here: a member with six hundred holdings
                # has hundreds below 0.05%, and rounding each to a tenth of a
                # percent threw away five percent of the portfolio. The interface
                # rounds what it prints; the ring needs the real shares to add up.
                "weight_pct": round(position.mid / total * 100, 6) if total else 0.0,
                "mid_usd": position.mid,
                "low_usd": position.low,
                "high_usd": position.high,
                "purchases": len(position.bought),
                "sales": len(position.sold),
                "first_bought": min(traded for traded, _, _, _ in position.bought),
                "last_trade": position.last_trade,
                "accounts": sorted(position.accounts),
                "return_since_bought_pct": bought_return,
                "return_since_public_pct": public_return,
            }
        )

    priced = [h for h in holdings if h["return_since_public_pct"] is not None]
    priced_share = sum(h["mid_usd"] for h in priced) / total if total else 0.0
    overall = None
    if priced and priced_share >= MIN_PRICED_SHARE:
        weight = sum(h["mid_usd"] for h in priced)
        overall = {
            "since_bought_pct": round(
                sum(h["return_since_bought_pct"] * h["mid_usd"] for h in priced) / weight, 1
            ),
            "since_public_pct": round(
                sum(h["return_since_public_pct"] * h["mid_usd"] for h in priced) / weight, 1
            ),
        }

    def brief(found: list[_Position]) -> list[dict[str, Any]]:
        ranked = sorted(found, key=lambda p: sum(high for _, high in p.sold), reverse=True)
        return [
            {
                "ticker": p.ticker,
                "asset": p.asset,
                # Exit price over entry price, where the lake has both.
                "return_pct": exits.get(p.ticker),
                "sold_low_usd": sum(low for low, _ in p.sold),
                "sold_high_usd": sum(high for _, high in p.sold),
                "last_trade": p.last_trade,
            }
            for p in ranked
        ]

    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "actor_id": actor_id,
        "actor": actor,
        "role": seat,
        "chamber": chamber,
        "since": eastern_date(since) if isinstance(since, dt.datetime) else None,
        "trades": len(rows),
        "mid_usd": total,
        "low_usd": sum(p.low for p in held),
        "high_usd": sum(p.high for p in held),
        "holdings": holdings,
        "closed": brief(closed),
        "held_before": brief(prior),
        "left_out": {"options": options, "exchanges": exchanges, "no_ticker": unnamed},
        "performance": performance,
        "priced_share": round(priced_share, 3),
        "returns": overall,
    }


# -------------------------------------------------------------------- performance --
def _replay(
    positions: list[_Position],
    series: dict[str, tuple[list[dt.date], list[float]]],
    today: dt.date,
    *,
    by: str,
) -> tuple[list[tuple[dt.date, float]], dict[str, float | None], int]:
    """Replay every priced trade at its entry and exit prices, day by day.

    A purchase buys its midpoint's worth of shares at that day's close. A sale in
    full sells every share held, at that day's close; a partial sale sells its
    midpoint's worth, and never more than is held. From then on a sold share is
    worth what it was sold for, and a share still held is worth the day's close.
    The return on any day is everything gained so far over everything put in so
    far: closed trades at their exit prices, open ones marked to that day.

    ``by="traded"`` dates each trade when it happened. ``by="public"`` dates it
    when it was disclosed, which is the first day anybody else could have made it.
    A sale of shares never seen bought is skipped: it has no entry price.
    """
    moves: list[tuple[dt.date, int, str, _Trade]] = []
    for position in positions:
        if position.ticker not in series:
            continue
        for trade in position.trades:
            day = trade.traded if by == "traded" else trade.public
            moves.append((day, 0 if trade.kind == "purchase" else 1, position.ticker, trade))
    moves.sort(key=lambda move: (move[0], move[1], move[2], move[3].line))
    if not moves:
        return [], {}, 0

    days = sorted({day for ticker in series for day in series[ticker][0]} | {today})
    days = [day for day in days if moves[0][0] <= day <= today]
    shares: dict[str, float] = {}
    cost: dict[str, float] = {}
    realised: dict[str, float] = {}
    invested = proceeds = 0.0
    lots = cursor = 0
    points: list[tuple[dt.date, float]] = []
    for day in days:
        while cursor < len(moves) and moves[cursor][0] <= day:
            when, _, ticker, trade = moves[cursor]
            cursor += 1
            # Filled at the first price available on or after the day, never the
            # last one before it: on a Saturday disclosure that would be Friday's
            # close, a price that existed before the news did.
            price = _close_from(series[ticker], when)
            if not price:
                continue
            if trade.kind == "purchase":
                shares[ticker] = shares.get(ticker, 0.0) + trade.mid / price
                cost[ticker] = cost.get(ticker, 0.0) + trade.mid
                invested += trade.mid
                lots += 1
                continue
            held = shares.get(ticker, 0.0)
            if held <= 0:
                continue  # held from before the record: no entry price to measure from
            selling = held if trade.kind == "sale" else min(held, trade.mid / price)
            shares[ticker] = held - selling
            proceeds += selling * price
            realised[ticker] = realised.get(ticker, 0.0) + selling * price
        if invested <= 0:
            continue
        value = sum(
            held * (_close_on(series[ticker], day) or 0.0)
            for ticker, held in shares.items()
            if held
        )
        points.append((day, round((value + proceeds - invested) / invested * 100, 2)))

    # What each position that is now fully sold earned, exit over entry.
    closed = {
        ticker: round((realised.get(ticker, 0.0) / cost[ticker] - 1) * 100, 1)
        for ticker in cost
        if cost[ticker] > 0 and shares.get(ticker, 0.0) <= 1e-9
    }
    return points, dict(closed), lots


def _performance(
    positions: list[_Position],
    series: dict[str, tuple[list[dt.date], list[float]]],
    today: dt.date,
) -> tuple[dict[str, Any] | None, dict[str, float | None]]:
    member, closed, lots = _replay(positions, series, today, by="traded")
    follower, _, _ = _replay(positions, series, today, by="public")
    if len(member) < 2:
        return None, closed
    copied = dict(follower)
    priced = [
        p for p in positions if p.ticker in series and any(t.kind == "purchase" for t in p.trades)
    ]
    return (
        {
            "points": [
                {"date": day, "member_pct": value, "follower_pct": copied.get(day)}
                for day, value in member
            ],
            "member_pct": member[-1][1],
            "follower_pct": follower[-1][1] if follower else None,
            "purchases": lots,
            "tickers": len(priced),
        },
        closed,
    )


def _weighted_return(
    series: tuple[list[dt.date], list[float]], position: _Position, now: float | None, *, by: str
) -> float | None:
    """The position's return to now, each purchase weighted by its midpoint.

    ``by="traded"`` starts each purchase on the day it was made; ``by="public"``
    on the day it was disclosed, which is the first day anyone could have copied it.
    """
    if not now:
        return None
    gained = weight = 0.0
    for traded, public, low, high in position.bought:
        entry = _close_from(series, traded if by == "traded" else public)
        if not entry:
            continue
        size = (low + high) / 2
        gained += (now / entry - 1) * size
        weight += size
    return round(gained / weight * 100, 1) if weight else None
