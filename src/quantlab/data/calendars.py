"""Trading calendars per venue.

Calendars exist here for one reason that matters more than scheduling: they supply
the **session close instant in UTC** for every daily bar. A daily bar's information
does not exist until the session closes, and the close of a US session is 21:00 UTC
in winter and 20:00 UTC in summer. Storing a bare date instead throws that away and
makes cross-venue alignment silently wrong -- a Tokyo close and a New York close on
the "same date" are fourteen hours apart, and a signal that treats them as
simultaneous is using tomorrow's Tokyo data to trade today's New York open.

Venue codes are ISO MICs where one exists (`XNYS`, `XLON`), plus `24/7` for crypto.
"""

from __future__ import annotations

import datetime as dt
import functools
from collections.abc import Sequence

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
    and it would never surface as an exception -- only as a slightly-too-good
    backtest.
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
