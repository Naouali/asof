"""The API's response shapes.

Declared rather than implied, so that ``/api/docs`` is an accurate contract and the
interface's TypeScript types have something fixed to be written against.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel

__all__ = [
    "DataHealth",
    "EventModel",
    "FeedResponse",
    "Health",
    "SearchHit",
    "TickerResponse",
]


class EventModel(BaseModel):
    id: str
    kind: Literal["insider", "congress", "fund", "unread", "contract"]
    ticker: str | None
    asset: str | None
    actor: str
    actor_id: str
    role: str | None
    direction: Literal["buy", "sell", "none"]
    verb: str
    size: str
    detail: str | None
    value_usd: float | None
    traded_on: dt.date
    disclosed_on: dt.date
    disclosed_at: dt.datetime
    lag_days: int
    deadline_days: int | None
    due_on: dt.date | None
    late_days: int
    noise: bool
    source_url: str | None
    price_move_pct: float | None = None


class FeedGroup(BaseModel):
    date: dt.date
    events: list[EventModel]


class Insight(BaseModel):
    ticker: str
    text: str


class Coverage(BaseModel):
    unread_reports: int
    fund_period: dt.date | None
    fund_period_age_days: int | None


class ActorLabel(BaseModel):
    id: str
    name: str
    role: str | None


class FeedResponse(BaseModel):
    as_of: dt.datetime
    today: dt.date
    is_live: bool
    days: int
    actor: ActorLabel | None
    groups: list[FeedGroup]
    insights: list[Insight]
    #: Already happened on the as-of date, not public until after it. Read with
    #: today's knowledge, so the interface draws it only behind the curtain.
    beyond: list[EventModel]
    #: How many there were in all; `beyond` holds the ones that surfaced soonest.
    beyond_total: int
    #: Contract actions behind the curtain, counted apart from the trades.
    beyond_contracts_total: int
    coverage: Coverage
    empty_lake: bool


class PricePoint(BaseModel):
    date: dt.date
    close: float


class Holder(BaseModel):
    manager_id: str
    manager: str
    shares: float
    change: float | None
    value_usd: float
    period: dt.date
    disclosed_on: dt.date
    age_days: int


class Fails(BaseModel):
    quantity: float
    settled_on: dt.date
    posted_on: dt.date


class AgencyTotal(BaseModel):
    name: str
    net_usd: float


class Contracts(BaseModel):
    """A company's federal contract actions made public in the last year."""

    actions: int
    #: Money committed less money taken back. Not revenue: it is spent over years.
    net_usd: float
    taken_back: int
    #: Share of money committed that came from the Pentagon, published 90 days late.
    defense_share: float | None
    agencies: list[AgencyTotal]
    #: The largest actions, largest first.
    events: list[EventModel]
    #: Ids of the ones drawn on the price chart.
    on_chart: list[str]


class TickerResponse(BaseModel):
    as_of: dt.datetime
    today: dt.date
    ticker: str
    name: str | None
    known: bool
    prices: list[PricePoint]
    events: list[EventModel]
    holders: list[Holder]
    contracts: Contracts | None
    fails_to_deliver: Fails | None
    brief: list[str]


class SearchHit(BaseModel):
    kind: Literal["ticker", "person", "fund", "portfolio", "agency"]
    key: str
    label: str
    note: str | None


class DatasetHealth(BaseModel):
    source: str
    dataset: str
    rows: int
    symbols: int
    first: dt.date | None
    last: dt.date | None
    newest_known: dt.datetime | None
    age_days: float | None
    bytes: int


class RefusedSymbol(BaseModel):
    symbol: str
    reason: str
    attempts: int
    retry_after: dt.date


class Unpriced(BaseModel):
    """How much of what was disclosed the lake can actually price. Every return
    figure in the app is measured on `priced` of `wanted` tickers."""

    wanted: int
    priced: int
    refused: list[RefusedSymbol]
    refused_total: int


class DataHealth(BaseModel):
    datasets: list[DatasetHealth]
    unread_reports: list[EventModel]
    runs: list[dict[str, object]]
    unpriceable: Unpriced


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    #: Today in Washington, as the server sees it. The interface uses this rather
    #: than the browser's clock, because the server is the one that read the lake.
    today: dt.date
    authentication: Literal["none"]


# -------------------------------------------------------------------- portfolios --
class PortfolioMember(BaseModel):
    actor_id: str
    actor: str
    role: str | None
    chamber: str
    trades: int
    tickers: int
    last_disclosed: dt.date | None


class PortfolioMembers(BaseModel):
    as_of: dt.datetime
    today: dt.date
    members: list[PortfolioMember]


class Holding(BaseModel):
    """What a member has bought and kept in one ticker. `mid_usd` nets the
    midpoints of the disclosed ranges; `low_usd` and `high_usd` are the least and
    the most the same trades allow."""

    ticker: str
    asset: str | None
    weight_pct: float
    mid_usd: float
    low_usd: float
    high_usd: float
    purchases: int
    sales: int
    first_bought: dt.date
    last_trade: dt.date | None
    accounts: list[str]
    #: Null where the lake holds no prices for the ticker.
    return_since_bought_pct: float | None
    #: From the day the purchase became public: the return a follower could have had.
    return_since_public_pct: float | None


class SoldPosition(BaseModel):
    ticker: str
    asset: str | None
    #: Exit price over entry price. Null without prices, or if shares are still held.
    return_pct: float | None = None
    sold_low_usd: float
    sold_high_usd: float
    last_trade: dt.date | None


class PerformancePoint(BaseModel):
    date: dt.date
    member_pct: float
    #: The same trades made on the days they became public. Null before the first one.
    follower_pct: float | None


class Performance(BaseModel):
    """The portfolio's return over time: closed trades at their exit over their entry
    price, open ones marked to each day's close, over everything put in so far."""

    points: list[PerformancePoint]
    member_pct: float
    follower_pct: float | None
    #: Purchases that took part, and the tickers they were in: only priced ones can.
    purchases: int
    tickers: int


class LeftOut(BaseModel):
    options: int
    exchanges: int
    no_ticker: int


class PortfolioReturns(BaseModel):
    since_bought_pct: float
    since_public_pct: float


class PortfolioResponse(BaseModel):
    as_of: dt.datetime
    today: dt.date
    actor_id: str
    actor: str
    role: str | None
    chamber: str
    #: The day the first report the lake holds for this chamber became public.
    since: dt.date | None
    trades: int
    mid_usd: float
    low_usd: float
    high_usd: float
    holdings: list[Holding]
    #: Bought and then sold again, in full or more.
    closed: list[SoldPosition]
    #: Sold without ever being seen bought: held from before the record begins.
    held_before: list[SoldPosition]
    left_out: LeftOut
    #: Null when fewer than two days of priced trades exist.
    performance: Performance | None
    priced_share: float
    #: Null unless enough of the portfolio has prices for a figure to mean anything.
    returns: PortfolioReturns | None


# ---------------------------------------------------------------- track record --
class Record(BaseModel):
    """A filer's measured record: the mean and what the sample can support."""

    trades: int
    mean_excess_pct: float
    #: The 95% interval of the mean. Null with a single trade.
    low_pct: float | None
    high_pct: float | None
    median_excess_pct: float
    beat_rate: float
    #: True only when that interval is clear of zero.
    distinguishable: bool


class FilerRecord(Record):
    actor_id: str
    actor: str
    role: str | None
    #: The same trades entered on the day the filer made them, not the day they
    #: became public. The gap between the two is what the delay cost a follower.
    own_mean_excess_pct: float | None
    first: dt.date
    last: dt.date
    ranked: bool


class Luck(BaseModel):
    """How good the leader would look if nobody had any skill at all."""

    best_mean_excess_pct: float
    shuffles: int
    #: The share of shuffles producing a leader at least this good. Near 1 is noise.
    as_good_by_chance: float


class TrackRecords(BaseModel):
    as_of: dt.datetime
    today: dt.date
    horizon_days: int
    benchmark: str
    min_trades: int
    measured: int
    filers: int
    ranked: int
    #: Trades left out: no prices, or the holding window has not closed yet.
    unpriced: int
    unfinished: int
    everyone: Record | None
    standouts: int
    luck: Luck | None
    expected_by_luck: float
    people: list[FilerRecord]
    why_empty: str | None


# ------------------------------------------------------------------------- rules --
class DeadlineRules(BaseModel):
    congress_days: int
    fund_days: int
    insider_business_days: int


class ContractRules(BaseModel):
    #: The smallest action the ingest keeps, from the ingest plan.
    min_action_usd: float
    #: The smallest action the feed shows.
    feed_min_usd: float
    publication_lag_days: int
    defense_embargo_days: int
    #: Companies in configs/contractors.yaml. Null if that file cannot be read.
    companies: int | None
    listed_per_ticker: int
    on_chart: int


class FeedRules(BaseModel):
    fund_changes_per_filing: int
    fund_min_change_pct: int


class AnalyticsRules(BaseModel):
    min_sample: int
    max_lag_days: int


class PortfolioRules(BaseModel):
    #: Share of a portfolio that must have prices before an overall return is given.
    min_priced_share_pct: int


class Rules(BaseModel):
    """Every threshold the app applies, so the interface prints them, never repeats them."""

    deadlines: DeadlineRules
    contracts: ContractRules
    feed: FeedRules
    analytics: AnalyticsRules
    portfolios: PortfolioRules


# --------------------------------------------------------------------- analytics --
class TradedTicker(BaseModel):
    ticker: str
    name: str | None
    #: Distinct filers, not trades and not dollars: the one count every source supports.
    buyers: int
    sellers: int


class TradedPerson(BaseModel):
    actor_id: str
    actor: str
    role: str | None
    kind: Literal["insider", "congress"]
    buys: int
    sells: int
    tickers: int


class TradedResponse(BaseModel):
    as_of: dt.datetime
    today: dt.date
    days: int
    kind: str | None
    trades: int
    tickers_total: int
    tickers: list[TradedTicker]
    people: list[TradedPerson]


class Spread(BaseModel):
    """Percent price moves between the trade and its disclosure. The percentiles
    are null under five trades: a median of three is an anecdote."""

    n: int
    median: float | None
    p10: float | None
    p25: float | None
    p75: float | None
    p90: float | None


class MoveGroup(BaseModel):
    key: Literal["insider", "house", "senate", "fund"]
    label: str
    buy: Spread
    sell: Spread


class PriceMovesResponse(BaseModel):
    as_of: dt.datetime
    today: dt.date
    days: int
    trades: int
    #: How many of them name a ticker the lake holds prices for.
    measured: int
    priced_tickers: int
    groups: list[MoveGroup]
    #: The moves a reader of the disclosure had missed by the most.
    examples: list[EventModel]


class ContractTotal(BaseModel):
    name: str
    net_usd: float
    defense_usd: float
    actions: int


class ContractMonth(BaseModel):
    month: dt.date
    committed_usd: float
    taken_back_usd: float


class HiddenContracts(BaseModel):
    actions: int
    net_usd: float
    defense_actions: int


class ContractsResponse(BaseModel):
    """Contract actions made public in the year to the as-of date. Net figures are
    money committed less money taken back, spent over years: not revenue."""

    as_of: dt.datetime
    today: dt.date
    is_live: bool
    actions: int
    net_usd: float
    defense_usd: float
    companies: list[ContractTotal]
    agencies: list[ContractTotal]
    months: list[ContractMonth]
    #: Signed by the as-of date and not public yet. Null when looking at today,
    #: because only a later date can know it.
    hidden: HiddenContracts | None
