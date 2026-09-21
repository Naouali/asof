"""One shape for four kinds of disclosure.

An insider's Form 4, a member of Congress's transaction report, a fund's 13F and a
federal contract action have nothing in common as documents. To a reader they are
the same thing: somebody did something on one day, and the world found out on
another. This module turns each
dataset into that one shape -- an :class:`Event` -- so the feed, the ticker page
and the search index never need to know which filing a row came from.

Everything here is a pure function of frames that were already read through a
point-in-time snapshot. Nothing in this module touches the lake, which is what
keeps the "as of" guarantee in one place.

**Dates are Washington dates.** A Form 4 accepted at 21:40 Eastern is stamped
01:40 UTC the next day. Grouping the feed by the UTC date would file it under a
day on which nobody in the market saw it, and count its lag one day long.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, Literal
from zoneinfo import ZoneInfo

import polars as pl

from quantlab.data.calendars import is_federal_workday, next_federal_workday
from quantlab.data.sources.usaspending import DEFENSE_EMBARGO

__all__ = [
    "CONGRESS_DEADLINE_DAYS",
    "FUND_DEADLINE_DAYS",
    "Event",
    "congress_events",
    "contract_events",
    "display_name",
    "due_after",
    "fund_events",
    "insider_events",
    "member_name",
    "unread_report_events",
]

EASTERN = ZoneInfo("America/New_York")

Kind = Literal["insider", "congress", "fund", "unread", "contract"]
Direction = Literal["buy", "sell", "none"]

#: The STOCK Act allows 45 days from the trade to the report.
CONGRESS_DEADLINE_DAYS = 45
#: A 13F is due 45 days after the quarter it describes.
FUND_DEADLINE_DAYS = 45
#: A Form 4 is due before the end of the second business day after the trade.
INSIDER_DEADLINE_BUSINESS_DAYS = 2

#: How many of a fund's position changes reach the feed, largest first. A large
#: quant manager changes thousands of positions a quarter, and a feed that lists
#: them all is a feed nobody reads.
FUND_CHANGES_PER_FILING = 4
#: A change in share count smaller than this is rebalancing, not a decision.
FUND_MIN_CHANGE = 0.10

#: The SEC's transaction codes, as a reader would say them. Only P and S are a
#: decision about the share price; the rest are pay, gifts and paperwork.
_INSIDER_VERBS: dict[str, str] = {
    "P": "Bought",
    "S": "Sold",
    "A": "Was granted",
    "M": "Exercised options for",
    "X": "Exercised options for",
    "F": "Had withheld for tax",
    "G": "Gave away",
    "C": "Converted",
    "D": "Returned to the company",
    "W": "Inherited or bequeathed",
    "J": "Reported another change of",
}
_INSIDER_WHY_NOISE: dict[str, str] = {
    "A": "Compensation, not a decision about the price",
    "M": "An option exercise, not a decision about the price",
    "X": "An option exercise, not a decision about the price",
    "F": "Automatic when stock vests, not a sale by choice",
    "G": "A gift, not a sale",
}
_CORPORATE_WORDS = frozenset(
    {"inc", "llc", "lp", "ltd", "corp", "co", "trust", "fund", "partners", "capital", "group",
     "holdings", "management", "advisors", "associates", "foundation", "plc", "sa", "ag", "nv"}
)  # fmt: skip
_OWNER_NOTES = {
    "SP": "held by their spouse",
    "JT": "in a joint account",
    "DC": "held by a dependent child",
}


@dataclass(frozen=True, slots=True)
class Event:
    """Somebody did something on one day; the world found out on another."""

    id: str
    kind: Kind
    ticker: str | None
    asset: str | None
    actor: str
    actor_id: str
    role: str | None
    direction: Direction
    verb: str
    size: str
    detail: str | None
    value_usd: float | None
    traded_on: dt.date
    disclosed_on: dt.date
    disclosed_at: dt.datetime
    lag_days: int
    deadline_days: int | None
    #: The last day the law allowed this to become public, where there is one.
    due_on: dt.date | None
    late_days: int
    noise: bool
    source_url: str | None
    price_move_pct: float | None = field(default=None)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------- helpers --
def eastern_date(moment: dt.datetime) -> dt.date:
    return moment.astimezone(EASTERN).date()


def display_name(raw: str, *, person: bool) -> str:
    """An SEC filer name, as a reader would write it.

    EDGAR stores people as ``LAST FIRST MIDDLE``, often in capitals: ``COOK
    TIMOTHY D``. Entities are stored as they are. Only a name that belongs to an
    officer or director, has two to four words and no corporate word in it is
    turned round; anything else is left exactly as filed, because a wrong guess
    here renames a person.
    """
    first_owner = raw.split(";")[0].strip()
    words = first_owner.split()
    looks_corporate = any(word.strip(".,").lower() in _CORPORATE_WORDS for word in words)
    if not person or looks_corporate or not 2 <= len(words) <= 4:
        return first_owner
    if first_owner.isupper():
        words = [word.title() for word in words]
    return " ".join([*words[1:], words[0]])


_HONORIFICS = frozenset({"hon", "mr", "mrs", "ms", "miss", "dr", "rep"})


def member_name(raw: str) -> str:
    """A member's name, without the clutter of the record it came from.

    The Clerk's own records carry honorifics in the middle of names and the odd
    doubled word -- ``Hon. John J Mr McGuire III``, ``Hon. Scott Scott Franklin``
    -- in the index and in the reports alike. Those are dropped for display. The
    words that identify the person are left alone. The Senate indexes its paper
    filers in capitals, which are lowered: a name is not a shout.
    """
    kept: list[str] = []
    for word in (raw.title() if raw.isupper() else raw).split():
        if word.strip(".,").lower() in _HONORIFICS:
            continue
        if kept and kept[-1].lower() == word.lower():
            continue
        kept.append(word)
    return " ".join(kept)


def _seat(chamber: str | None, district: str | None) -> tuple[str, str | None]:
    """How a member is addressed, and the seat shown beside the name."""
    if chamber == "senate":
        return "Sen.", "Senate"  # the Senate's index does not say which state
    return "Rep.", district


def member_id(name: str, district: str | None) -> str:
    """One key for a member whether the row came from a report or from the index."""
    return f"congress:{(district or '').lower()}:{name.lower()}"


#: Scale, suffix and decimals, largest first.
_UNITS = ((1e9, "B", 2), (1e6, "M", 2), (1e3, "K", 0))


def _scaled(magnitude: float, units: tuple[tuple[float, str, int], ...]) -> str:
    """A magnitude in the largest unit it fills, never as a thousand of a smaller
    one: $999,999,999 is a billion dollars, not $1000.00M."""
    for index, (scale, suffix, digits) in enumerate(units):
        if magnitude < scale:
            continue
        if round(magnitude / scale, digits) >= 1000 and index > 0:
            bigger, big_suffix, big_digits = units[index - 1]
            return f"{magnitude / bigger:,.{big_digits}f}{big_suffix}"
        return f"{magnitude / scale:,.{digits}f}{suffix}"
    return f"{magnitude:,.0f}"


def money(value: float) -> str:
    # The sign belongs in front of the currency, not between it and the digits.
    sign = "-" if value < 0 else ""
    return f"{sign}${_scaled(abs(value), _UNITS)}"


#: Share counts read better with one decimal: "2.9M shares", not "2.90M shares".
_SHARE_UNITS = ((1e9, "B", 1), (1e6, "M", 1))


def count(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{_scaled(abs(value), _SHARE_UNITS)}"


def due_after(start: dt.date, days: int) -> dt.date:
    """The day a report filed ``days`` calendar days after ``start`` is due.

    A deadline that lands on a weekend or a federal holiday moves to the next
    working day. Both the date shown to a reader and the judgement of lateness
    come from here, so the page cannot say "the deadline was Sunday" and then
    call a Monday filing late.
    """
    return next_federal_workday(start + dt.timedelta(days=days))


def add_business_days(start: dt.date, days: int) -> dt.date:
    cursor = start
    while days > 0:
        cursor += dt.timedelta(days=1)
        if is_federal_workday(cursor):
            days -= 1
    return cursor


def business_days_between(start: dt.date, end: dt.date) -> int:
    """Federal working days after ``start``, up to and including ``end``."""
    days, cursor = 0, start
    while cursor < end:
        cursor += dt.timedelta(days=1)
        if is_federal_workday(cursor):
            days += 1
    return days


# ---------------------------------------------------------------------- insiders --
def _own_issuer_only(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop forms a company filed as an OWNER of somebody else's stock.

    A company's filing index lists those beside the forms its own insiders file,
    and the fetcher used to keep both under the company's ticker: Alphabet's trades
    in a start-up it backs, shown as insider trades in Alphabet. The fetcher now
    drops them, but the lake is append-only and still holds the old rows. A
    ticker's own issuer is the one most of its rows name; the rest are dropped.
    Rows with no issuer recorded are kept, since nothing says they are wrong.
    """
    if "issuer_cik" not in frame.columns:
        return frame
    own = frame.group_by("symbol").agg(
        pl.col("issuer_cik").drop_nulls().mode().first().alias("_own")
    )
    return (
        frame.join(own, on="symbol", how="left")
        .filter(pl.col("issuer_cik").is_null() | (pl.col("issuer_cik") == pl.col("_own")))
        .drop("_own")
    )


def insider_events(frame: pl.DataFrame) -> list[Event]:
    """One event per filing, per owner, per kind of transaction.

    A single sale is often reported as a dozen lines at a dozen prices. Those are
    one decision, so they are summed, and the price shown is the volume-weighted
    one.
    """
    if frame.height == 0:
        return []
    frame = _own_issuer_only(frame)

    grouped = (
        frame.with_columns((pl.col("shares") * pl.col("price")).alias("_value"))
        .group_by(
            "symbol",
            "accession",
            "owner_name",
            "transaction_code",
            "acquired_disposed",
            "is_derivative",
            maintain_order=True,
        )
        .agg(
            pl.col("shares").sum().alias("shares"),
            pl.col("_value").sum().alias("value"),
            pl.col("shares").filter(pl.col("price").is_not_null()).sum().alias("priced_shares"),
            pl.col("as_of").min().alias("first_trade"),
            pl.col("known_at").max().alias("known_at"),
            pl.col("issuer_name").first(),
            pl.col("issuer_cik").first(),
            pl.col("owner_cik").first(),
            pl.col("officer_title").first(),
            pl.col("is_director").any(),
            pl.col("is_officer").any(),
            pl.col("is_ten_percent_owner").any(),
            pl.col("planned_10b5_1").any().alias("planned"),
            pl.col("form").first(),
            pl.col("security_title").first(),
        )
    )

    events: list[Event] = []
    for row in grouped.iter_rows(named=True):
        code = str(row["transaction_code"])
        open_market = code in {"P", "S"} and not row["is_derivative"]
        planned = bool(row["planned"])
        person = bool(row["is_officer"] or row["is_director"])
        traded = row["first_trade"].date()
        disclosed_at: dt.datetime = row["known_at"]
        disclosed = eastern_date(disclosed_at)

        shares = float(row["shares"] or 0.0)
        priced = float(row["priced_shares"] or 0.0)
        value = float(row["value"]) if priced > 0 else None
        if value is not None and priced > 0:
            size = f"{count(shares)} shares at ${value / priced:,.2f}"
        else:
            size = f"{count(shares)} shares"

        if not open_market:
            detail = _INSIDER_WHY_NOISE.get(code, row["security_title"])
        elif planned:
            detail = f"{money(value)} under a pre-scheduled plan" if value else "Pre-scheduled"
        else:
            detail = f"{money(value)} on the open market, not pre-scheduled" if value else None

        late = 0
        if open_market and row["form"] == "4":
            over = business_days_between(traded, disclosed) - INSIDER_DEADLINE_BUSINESS_DAYS
            late = max(0, over)

        role = row["officer_title"] or (
            "Director"
            if row["is_director"]
            else "10% owner"
            if row["is_ten_percent_owner"]
            else None
        )
        folder = str(row["accession"]).replace("-", "")
        events.append(
            Event(
                id=f"insider:{row['accession']}:{row['owner_cik']}:{code}:{row['acquired_disposed']}"
                f":{int(bool(row['is_derivative']))}",
                kind="insider",
                ticker=row["symbol"],
                asset=row["issuer_name"],
                actor=display_name(str(row["owner_name"]), person=person),
                actor_id=f"insider:{row['owner_cik']}",
                role=role,
                direction="buy" if row["acquired_disposed"] == "A" else "sell",
                verb=_INSIDER_VERBS.get(code, f"Reported a transaction (code {code}) of"),
                size=size,
                detail=detail,
                value_usd=value,
                traded_on=traded,
                disclosed_on=disclosed,
                disclosed_at=disclosed_at,
                lag_days=(disclosed - traded).days,
                deadline_days=None,
                due_on=add_business_days(traded, INSIDER_DEADLINE_BUSINESS_DAYS)
                if open_market and row["form"] == "4"
                else None,
                late_days=late,
                noise=not open_market or planned,
                source_url=f"https://www.sec.gov/Archives/edgar/data/{row['issuer_cik']}/{folder}/"
                if row["issuer_cik"]
                else None,
            )
        )
    return events


# ---------------------------------------------------------------------- congress --
def congress_events(frame: pl.DataFrame) -> list[Event]:
    events: list[Event] = []
    for row in frame.iter_rows(named=True):
        kind = str(row["transaction_type"])
        traded = row["as_of"].date()
        disclosed_at: dt.datetime = row["known_at"]
        disclosed = eastern_date(disclosed_at)
        lag = (disclosed - traded).days
        member = member_name(str(row["member"]))
        title, seat = _seat(row["chamber"], row["state_district"])
        ticker = None if row["symbol"] == "NO_TICKER" else row["symbol"]

        notes = [note for note in (_OWNER_NOTES.get(row["owner"] or ""),) if note]
        if kind == "sale_partial":
            notes.append("part of the holding")
        if row["filing_status"] == "amended":
            notes.append("an amended report")

        events.append(
            Event(
                id=f"congress:{row['doc_id']}:{row['line']}",
                kind="congress",
                ticker=ticker,
                asset=row["asset"],
                actor=f"{title} {member}",
                actor_id=member_id(member, row["state_district"]),
                role=seat,
                direction="buy" if kind == "purchase" else "none" if kind == "exchange" else "sell",
                verb={"purchase": "Bought", "exchange": "Exchanged"}.get(kind, "Sold"),
                size=str(row["amount_text"]).replace(" - ", " to "),
                detail=", ".join(notes).capitalize() if notes else None,
                value_usd=None,  # a bracket is not an amount, and is never summed as one
                traded_on=traded,
                disclosed_on=disclosed,
                disclosed_at=disclosed_at,
                lag_days=lag,
                deadline_days=CONGRESS_DEADLINE_DAYS,
                due_on=(due := due_after(traded, CONGRESS_DEADLINE_DAYS)),
                late_days=max(0, (disclosed - due).days),
                noise=False,
                source_url=row["url"],
            )
        )
    return events


def unread_report_events(filings: pl.DataFrame, trades: pl.DataFrame) -> list[Event]:
    """Transaction reports that were filed and could not be read.

    Bounded to the period the trades cover. The index is cheap and is ingested
    for years; the PDFs are not. A report nobody tried to open is not a scan.
    """
    if filings.height == 0 or trades.height == 0:
        return []
    covered_from = trades["known_at"].min()
    unread = filings.filter(
        (pl.col("filing_type") == "P")
        & (pl.col("known_at") >= covered_from)
        & ~pl.col("doc_id").is_in(trades["doc_id"].unique().to_list())
    )
    events: list[Event] = []
    for row in unread.iter_rows(named=True):
        member = member_name(
            " ".join(part for part in (row["first_name"], row["last_name"], row["suffix"]) if part)
        )
        title, seat = _seat(row["chamber"], row["symbol"])
        filed = eastern_date(row["known_at"])
        events.append(
            Event(
                id=f"unread:{row['doc_id']}",
                kind="unread",
                ticker=None,
                asset=None,
                actor=f"{title} {member}",
                actor_id=member_id(member, row["symbol"]),
                role=seat,
                direction="none",
                verb="Filed",
                size="a transaction report on paper",
                detail="A scan, so it can't be read automatically. These trades are missing, not zero.",
                value_usd=None,
                traded_on=filed,
                disclosed_on=filed,
                disclosed_at=row["known_at"],
                lag_days=0,
                deadline_days=None,
                due_on=None,
                late_days=0,
                noise=False,
                source_url=row["url"],
            )
        )
    return events


# ------------------------------------------------------------------------- funds --
def fund_events(
    holdings: pl.DataFrame,
    tickers: dict[str, str],
    *,
    per_filing: int | None = FUND_CHANGES_PER_FILING,
    only_cusips: Iterable[str] | None = None,
    since: dt.date | None = None,
) -> list[Event]:
    """What changed between each fund's report and the one before it.

    A 13F is a list of positions, not of trades, so the event is the difference
    from the previous quarter -- and it carries the dishonesty of that in its
    dates: ``traded_on`` is the quarter END, because the form does not say when in
    the quarter anything happened, only what was held when it closed.

    ``since`` leaves out quarters that closed before it. The lake holds a decade of
    reports and the app shows a year; comparing fifty quarters to draw four is
    most of what a first visit to a date used to wait for.
    """
    if holdings.height == 0:
        return []
    if "first_known_at" not in holdings.columns:
        holdings = holdings.with_columns(pl.col("known_at").alias("first_known_at"))
    shares_only = holdings.filter((pl.col("put_call") == "NONE") & (pl.col("shares_type") == "SH"))
    wanted = set(only_cusips) if only_cusips is not None else None

    events: list[Event] = []
    for (manager,), book in shares_only.group_by("symbol", maintain_order=True):
        periods = sorted(book["as_of"].unique().to_list())
        # What each report WAS is read from the whole book, before it is narrowed:
        # a quarter in which the fund held none of the wanted securities has no
        # rows left to read a filing date from, and that quarter is the exit.
        filings = {
            row["as_of"]: row
            for row in book.group_by("as_of")
            .agg(
                # The ORIGINAL report: the earliest moment anything about this
                # quarter was public. Amendments come later and are not lateness.
                pl.col("first_known_at").min().alias("known_at"),
                pl.col("manager_name").first(),
                pl.col("accession").sort_by("first_known_at").first(),
            )
            .iter_rows(named=True)
        }
        if wanted is not None:
            # The periods come from the whole book -- a quarter in which the fund
            # held none of these is how an exit is seen -- but only these rows are
            # worth comparing. It is the difference between a page in a second and
            # a page in a blink for a manager with four thousand positions.
            book = book.filter(pl.col("cusip").is_in(list(wanted)))
        # The first report held has nothing to be compared with, so it opens no
        # events: calling every position in it "new" would be an artefact of where
        # ingestion happened to start.
        for earlier, period in pairwise(periods):
            if since is not None and period.date() < since:
                continue
            now = book.filter(pl.col("as_of") == period)
            before = book.filter(pl.col("as_of") == earlier)
            if now.height == 0 and before.height == 0:
                continue
            events.extend(
                _changes(str(manager), filings[period], now, before, tickers, per_filing=per_filing)
            )
    return events


def _changes(
    manager: str,
    filing: dict[str, Any],
    now: pl.DataFrame,
    before: pl.DataFrame,
    tickers: dict[str, str],
    *,
    per_filing: int | None,
) -> list[Event]:
    joined = now.select(
        "cusip", "issuer_name", "shares", "value_usd", "first_known_at", "accession"
    ).join(
        before.select("cusip", pl.col("shares").alias("shares_before"), pl.col("issuer_name").alias("name_before")),
        on="cusip",
        how="full",
        coalesce=True,
    )  # fmt: skip
    original: dt.datetime = filing["known_at"]
    period: dt.datetime = filing["as_of"]
    manager_name = str(filing["manager_name"])
    traded = period.date()
    # Lateness belongs to the fund's original report, not to a position that a
    # later amendment added: holding one back under confidential treatment is legal.
    # Lateness is judged against the day the report was actually due, weekends and
    # federal holidays included, not against a flat count of calendar days.
    due = due_after(traded, FUND_DEADLINE_DAYS)
    filed_late = max(0, (eastern_date(original) - due).days)

    rows: list[tuple[float, Event]] = []
    for row in joined.iter_rows(named=True):
        cusip = str(row["cusip"])
        held, held_before = float(row["shares"] or 0.0), float(row["shares_before"] or 0.0)
        if held == held_before:
            continue
        if held_before and held and abs(held - held_before) / held_before < FUND_MIN_CHANGE:
            continue

        price = (float(row["value_usd"]) / held) if held and row["value_usd"] else None
        if held_before == 0:
            verb, size = "Opened", f"a position of {count(held)} shares"
        elif held == 0:
            verb, size = "Sold out of", f"{count(held_before)} shares"
        else:
            change = (held - held_before) / held_before
            verb = "Added" if change > 0 else "Cut"
            size = f"{abs(change):.0%}, to {count(held)} shares"
        value = held * price if price else None
        weight = abs(held - held_before) * (price or 0.0)

        # A position still held became public when it was first filed. An exit has
        # no row of its own: it became public with the original report.
        disclosed_at: dt.datetime = row["first_known_at"] or original
        accession = str(row["accession"] or filing["accession"])
        disclosed = eastern_date(disclosed_at)
        lag = (disclosed - traded).days
        rows.append(
            (
                weight,
                Event(
                    id=f"fund:{accession}:{cusip}",
                    kind="fund",
                    ticker=tickers.get(cusip),
                    asset=row["issuer_name"] or row["name_before"],
                    actor=manager_name,
                    actor_id=f"fund:{manager}",
                    role="Quarterly holdings report",
                    direction="buy" if held > held_before else "sell",
                    verb=verb,
                    size=size,
                    detail=f"Worth {money(value)} when the quarter closed" if value else None,
                    value_usd=value,
                    traded_on=traded,
                    disclosed_on=disclosed,
                    disclosed_at=disclosed_at,
                    lag_days=lag,
                    deadline_days=FUND_DEADLINE_DAYS,
                    due_on=due,
                    late_days=filed_late,
                    noise=False,
                    source_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={manager}&type=13F",
                ),
            )
        )

    rows.sort(key=lambda pair: pair[0], reverse=True)
    chosen = rows if per_filing is None else rows[:per_filing]
    return [event for _, event in chosen]


# --------------------------------------------------------------------- contracts --
#: Words a government clerk capitalises that a reader would not.
_SMALL_WORDS = frozenset({"of", "the", "and", "for", "in", "to", "on", "a", "an", "at", "by"})


#: Words that are initials, and stay that way.
_INITIALS = frozenset({"LLC", "LLP", "LP", "USA", "US", "U.S.", "II", "III", "IV", "IT", "NASA"})


def _named(raw: str) -> str:
    """ "DEPT OF THE AIR FORCE" is shouting; "Dept of the Air Force" is a name."""
    if not raw.isupper():
        return raw
    return " ".join(
        word
        if word.strip(".,") in _INITIALS or any(char.isdigit() for char in word)
        else word.lower()
        if index and word.lower() in _SMALL_WORDS
        else word.capitalize()
        for index, word in enumerate(raw.split())
    )


def _sentence(raw: str | None, limit: int = 180) -> str | None:
    """A description typed in capitals, lowered to a sentence and cut at a word."""
    if not raw:
        return None
    text = " ".join(raw.split())
    text = text.capitalize() if text.isupper() else text
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return text


#: Legal endings that do not make "Lockheed Martin Corp" a different company from
#: "Lockheed Martin Corporation".
_LEGAL_ENDINGS = frozenset(
    {"corp", "corporation", "inc", "incorporated", "co", "company", "llc", "lp", "ltd", "the"}
)


def _same_company(one: str, other: str) -> bool:
    def core(name: str) -> list[str]:
        words = [word.strip(".,").lower() for word in name.split()]
        return [word for word in words if word and word not in _LEGAL_ENDINGS]

    return core(one) == core(other)


def contract_events(frame: pl.DataFrame, *, min_usd: float = 0.0) -> list[Event]:
    """Federal contract actions, as events.

    ``direction`` says which way the money moved: ``buy`` is money committed to
    the company and ``sell`` is money taken back. No deadline applies: the gap
    between an action and its publication is policy, not lateness -- two days for
    a civilian agency, three months for the Pentagon.
    """
    if frame.height and min_usd > 0:
        frame = frame.filter(pl.col("obligation_usd").abs() >= min_usd)

    events: list[Event] = []
    for row in frame.iter_rows(named=True):
        amount = float(row["obligation_usd"])
        acted = row["as_of"].date()
        disclosed_at: dt.datetime = row["known_at"]
        disclosed = eastern_date(disclosed_at)

        if amount < 0:
            verb = "Took back"
        elif row["modification_number"] in (None, "0"):
            verb = "Awarded"
        elif (row["action_type"] or "").upper() == "EXERCISE AN OPTION":
            verb = "Exercised an option for"
        else:
            verb = "Added"

        office = _named(row["awarding_sub_agency"] or row["awarding_agency"])
        department = _named(row["awarding_agency"])
        signer = _named(str(row["recipient_name"]))
        described = _sentence(row["description"])
        notes = [
            described if described is None or described.endswith((".", "...")) else f"{described}."
        ]
        if row["parent_name"] and not _same_company(row["recipient_name"], row["parent_name"]):
            notes.append(f"Signed by {signer}.")
        if row["defense"]:
            notes.append(
                f"The Pentagon publishes its contract actions {DEFENSE_EMBARGO.days} days late."
            )

        events.append(
            Event(
                id=f"contract:{row['transaction_key']}",
                kind="contract",
                ticker=row["symbol"],
                asset=signer,
                actor=office,
                actor_id=f"agency:{office.lower()}",
                role=department if department != office else None,
                direction="sell" if amount < 0 else "buy",
                verb=verb,
                size=money(abs(amount)),
                detail=" ".join(note for note in notes if note) or None,
                value_usd=amount,
                traded_on=acted,
                disclosed_on=disclosed,
                disclosed_at=disclosed_at,
                lag_days=(disclosed - acted).days,
                deadline_days=None,
                due_on=None,
                late_days=0,
                noise=False,
                source_url=row["url"],
            )
        )
    return events
