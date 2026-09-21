"""Trading calendars per venue.

Calendars exist here for one reason that matters more than scheduling: they supply
the **session close instant in UTC** for every daily bar. A daily bar's information
does not exist until the session closes, and the close of a US session is 21:00 UTC
in winter and 20:00 UTC in summer. Storing a bare date instead throws that away and
makes cross-venue alignment silently wrong -- a Tokyo close and a New York close on
the "same date" are fourteen hours apart, and anything that treats them as
simultaneous is reading tomorrow's Tokyo close on today's New York open.

Venue codes are ISO MICs where one exists (`XNYS`, `XLON`), plus `24/7` for crypto.
"""

from __future__ import annotations

import calendar as stdlib_calendar
import datetime as dt
import functools
from collections.abc import Sequence
from functools import lru_cache

import exchange_calendars as xcals

from quantlab.logging import get_logger

__all__ = [
    "CRYPTO_VENUE",
    "VENUE_ALIASES",
    "UnknownVenueError",
    "get_calendar",
    "is_session",
    "previous_session",
    "resolve_venue",
    "session_close",
    "session_closes",
    "sessions",
]

log = get_logger("quantlab.data.calendars")

CRYPTO_VENUE = "24/7"


class UnknownVenueError(KeyError):
    """A venue has no calendar mapping.

    Raised rather than defaulting to a US calendar. Guessing a venue's hours is a
    silent, systematic time-misalignment error across every bar of that instrument,
    and it would never surface as an exception -- only as numbers that are
    slightly, persistently wrong.
    """


#: Exchange names as the sources report them, mapped to calendar codes. Kept
#: explicit: a source that starts returning an unmapped venue must fail loudly.
VENUE_ALIASES: dict[str, str] = {
    # Yahoo `fullExchangeName` / `exchangeName` values
    "nyse": "XNYS",
    "nyq": "XNYS",
    "nysearca": "XNYS",
    "nyseamerican": "XNYS",
    "pcx": "XNYS",
    "ase": "XNYS",
    "nasdaqgs": "XNAS",
    "nasdaqgm": "XNAS",
    "nasdaqcm": "XNAS",
    "ngm": "XNAS",
    "nms": "XNAS",
    "ncm": "XNAS",
    "bats": "XNYS",
    "cboe bzx": "XNYS",
    # OTC Markets: the ADRs and foreign shares members disclose constantly. It is
    # a quotation system, not an exchange, and it keeps the NYSE's hours and
    # holidays -- so the NYSE calendar is the right one, not a guess at one.
    "otc markets otcpk": "XNYS",
    "otc markets otcqb": "XNYS",
    "otc markets otcqx": "XNYS",
    "otc markets otcid": "XNYS",
    "otc markets": "XNYS",
    # Cboe's US equity venues, where a great many ETFs are listed. Same hours and
    # holidays as the NYSE.
    "cboe us": "XNYS",
    "cboe": "XNYS",
    "pnk": "XNYS",
    "otc": "XNYS",
    "lse": "XLON",
    "lon": "XLON",
    "xetra": "XETR",
    "ger": "XETR",
    "gettex": "XETR",
    "tokyo": "XTKS",
    "jpx": "XTKS",
    "toronto": "XTSE",
    "tor": "XTSE",
    "asx": "XASX",
    "hkse": "XHKG",
    "hkg": "XHKG",
    "swx": "XSWX",
    "paris": "XPAR",
    "amsterdam": "XAMS",
    "milan": "XMIL",
    "madrid": "XMAD",
    "stockholm": "XSTO",
    "oslo": "XOSL",
    "copenhagen": "XCSE",
    # Futures venues
    "cme": "CMES",
    "cmes": "CMES",
    "nymex": "CMES",
    "comex": "CMES",
    "cbot": "CMES",
    "ice": "IEPA",
    "iepa": "IEPA",
    # Crypto trades continuously.
    "binance": CRYPTO_VENUE,
    "bybit": CRYPTO_VENUE,
    "okx": CRYPTO_VENUE,
    "kraken": CRYPTO_VENUE,
    "coinbase": CRYPTO_VENUE,
    "crypto": CRYPTO_VENUE,
    "24/7": CRYPTO_VENUE,
}


def resolve_venue(name: str) -> str:
    """Map a source's exchange label to a calendar code.

    Raises :class:`UnknownVenueError` rather than guessing.
    """
    key = name.strip().lower()
    if key in VENUE_ALIASES:
        return VENUE_ALIASES[key]
    upper = name.strip().upper()
    if upper in xcals.get_calendar_names():
        return upper
    raise UnknownVenueError(
        f"no calendar mapping for venue {name!r}. Add it to VENUE_ALIASES rather "
        "than defaulting to a US calendar: guessing a venue's hours misaligns every "
        "bar of that instrument by hours, which never raises and always flatters."
    )


@functools.lru_cache(maxsize=64)
def get_calendar(venue: str) -> xcals.ExchangeCalendar:
    """Return the calendar for a venue code or alias.

    Calendars are expensive to build and immutable once built, so they are cached
    for the life of the process.
    """
    code = resolve_venue(venue)
    # `side="right"` is irrelevant for session boundaries but keeps minute
    # semantics consistent if the event-driven engine later asks for them.
    return xcals.get_calendar(code)


def sessions(venue: str, start: dt.date, end: dt.date) -> list[dt.date]:
    """Trading sessions in ``[start, end]`` inclusive."""
    calendar = get_calendar(venue)
    lo = max(start, calendar.first_session.date())
    hi = min(end, calendar.last_session.date())
    if lo > hi:
        return []
    return [ts.date() for ts in calendar.sessions_in_range(lo, hi)]


def is_session(venue: str, day: dt.date) -> bool:
    return bool(get_calendar(venue).is_session(day))


def previous_session(venue: str, day: dt.date) -> dt.date:
    """The last session strictly before ``day``."""
    calendar = get_calendar(venue)
    previous: dt.date = calendar.previous_session(day).date()
    return previous


def session_close(venue: str, day: dt.date) -> dt.datetime:
    """The UTC instant at which ``day``'s session closes on ``venue``.

    This is the ``as_of`` of a daily bar. Raises if ``day`` is not a session:
    a bar dated on a non-session day means the source and the calendar disagree,
    and resolving that by guessing would corrupt every subsequent alignment.
    """
    calendar = get_calendar(venue)
    if not calendar.is_session(day):
        raise ValueError(
            f"{day} is not a trading session on {resolve_venue(venue)}. The source "
            "returned a bar the calendar does not recognise; investigate rather than "
            "coercing it to a neighbouring session."
        )
    close: dt.datetime = calendar.session_close(day).to_pydatetime()
    return close.astimezone(dt.UTC)


def session_closes(venue: str, days: Sequence[dt.date]) -> list[dt.datetime | None]:
    """Vectorised :func:`session_close`, with ``None`` for non-sessions.

    Used by the ingest path, which must report how many bars a source returned on
    days the calendar does not recognise instead of failing the whole pull.
    """
    calendar = get_calendar(venue)
    out: list[dt.datetime | None] = []
    for day in days:
        if calendar.is_session(day):
            out.append(calendar.session_close(day).to_pydatetime().astimezone(dt.UTC))
        else:
            out.append(None)
    return out


# ======================================================================================
# US federal holidays
# ======================================================================================
# Federal agencies do not publish on federal holidays, and the federal calendar is
# not the exchange calendar: the NYSE trades on Columbus Day and Veterans Day and
# closes on Good Friday, which is not a federal holiday at all. Using an exchange
# calendar to predict when a government report is released is wrong in both
# directions, so the rules are written out here.
#
# `pandas.tseries.holiday` has these, but pandas is banned in this repository
# (see tests/unit/test_repo_structure.py) and the rules are eleven lines of
# arithmetic.

#: Juneteenth became a federal holiday when signed into law on 2021-06-17.
JUNETEENTH_FIRST_YEAR = 2021


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The ``n``-th ``weekday`` of a month; ``n = -1`` means the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))
    last_day = stdlib_calendar.monthrange(year, month)[1]
    last = dt.date(year, month, last_day)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: dt.date) -> dt.date:
    """A fixed-date holiday falling at a weekend is observed on the nearest weekday."""
    if day.weekday() == 5:  # Saturday -> the Friday before
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> the Monday after
        return day + dt.timedelta(days=1)
    return day


@lru_cache(maxsize=256)
def federal_holidays(year: int) -> frozenset[dt.date]:
    """Observed US federal holidays in ``year``.

    Inauguration Day is deliberately excluded: it is a holiday only for federal
    employees in the DC area, and CFTC publication is not suspended for it.
    """
    days = [
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(dt.date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 10, 0, 2),  # Columbus Day
        _observed(dt.date(year, 11, 11)),  # Veterans Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(dt.date(year, 12, 25)),  # Christmas
    ]
    if year >= JUNETEENTH_FIRST_YEAR:
        days.append(_observed(dt.date(year, 6, 19)))

    # New Year's Day is the one holiday whose observance can cross a year
    # boundary, so it is handled by asking which new year lands *in* this year.
    # A Saturday 1 January is observed on 31 December of the year before -- as in
    # 1993, 2021 and 2027 -- and that Friday is exactly when a government report
    # would otherwise be released. Filing it under the wrong year hides it from
    # every lookup, since a date is only ever sought in its own year's set.
    for new_year in (dt.date(year, 1, 1), dt.date(year + 1, 1, 1)):
        observed = _observed(new_year)
        if observed.year == year:
            days.append(observed)
    return frozenset(days)


def is_federal_holiday(day: dt.date) -> bool:
    return day in federal_holidays(day.year)


def is_federal_workday(day: dt.date) -> bool:
    return day.weekday() < 5 and not is_federal_holiday(day)


def next_federal_workday(day: dt.date, *, inclusive: bool = True) -> dt.date:
    """The first federal working day on or after ``day``."""
    candidate = day if inclusive else day + dt.timedelta(days=1)
    for _ in range(14):
        if is_federal_workday(candidate):
            return candidate
        candidate += dt.timedelta(days=1)
    raise ValueError(f"no federal working day within a fortnight of {day}")
