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
    kind: Literal["insider", "congress", "fund", "unread"]
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


class TickerResponse(BaseModel):
    as_of: dt.datetime
    today: dt.date
    ticker: str
    name: str | None
    known: bool
    prices: list[PricePoint]
    events: list[EventModel]
    holders: list[Holder]
    fails_to_deliver: Fails | None
    brief: list[str]


class SearchHit(BaseModel):
    kind: Literal["ticker", "person", "fund"]
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


class DataHealth(BaseModel):
    datasets: list[DatasetHealth]
    unread_reports: list[EventModel]
    runs: list[dict[str, object]]


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    #: Today in Washington, as the server sees it. The interface uses this rather
    #: than the browser's clock, because the server is the one that read the lake.
    today: dt.date
    authentication: Literal["none"]
