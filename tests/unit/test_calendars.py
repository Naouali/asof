"""Calendars supply the session close that every daily bar's `as_of` depends on."""

from __future__ import annotations

import datetime as dt

import pytest

from quantlab.data.calendars import (
    CRYPTO_VENUE,
    UnknownVenueError,
    is_session,
    previous_session,
    resolve_venue,
    session_close,
    session_closes,
    sessions,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("NasdaqGS", "XNAS"),
        ("NYSE", "XNYS"),
        ("nyse", "XNYS"),
        ("LSE", "XLON"),
        ("binance", CRYPTO_VENUE),
        ("XNYS", "XNYS"),
    ],
)
def test_venue_aliases(label: str, expected: str) -> None:
    assert resolve_venue(label) == expected


def test_unknown_venue_raises_instead_of_defaulting_to_the_us() -> None:
    """Guessing a venue's hours misaligns every bar of that instrument by hours.
    It never raises and it always flatters, so it must be impossible."""
    with pytest.raises(UnknownVenueError, match=r"do not let it default|rather than"):
        resolve_venue("Bourse de Narnia")


def test_session_close_is_utc_and_dst_aware() -> None:
    """January closes at 21:00 UTC, July at 20:00. A backtest that hardcodes
    either is wrong for half the year."""
    assert session_close("XNYS", dt.date(2024, 1, 3)).hour == 21
    assert session_close("XNYS", dt.date(2024, 7, 2)).hour == 20
    assert session_close("XNYS", dt.date(2024, 1, 3)).tzinfo is dt.UTC


def test_half_days_close_early() -> None:
    """3 July 2024 was a US half day. Treating it as a full session misprices any
    intraday alignment and misstates that day's volume profile."""
    assert session_close("XNYS", dt.date(2024, 7, 3)).hour == 17


def test_non_session_raises() -> None:
    with pytest.raises(ValueError, match="not a trading session"):
        session_close("XNYS", dt.date(2024, 1, 6))  # a Saturday


def test_crypto_trades_every_day() -> None:
    days = sessions(CRYPTO_VENUE, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert len(days) == 31


def test_equity_sessions_exclude_holidays() -> None:
    january = sessions("XNYS", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert dt.date(2024, 1, 1) not in january  # New Year's Day
    assert dt.date(2024, 1, 15) not in january  # MLK Day
    assert len(january) == 21


def test_is_session_and_previous_session() -> None:
    assert is_session("XNYS", dt.date(2024, 1, 3))
    assert not is_session("XNYS", dt.date(2024, 1, 1))
    assert previous_session("XNYS", dt.date(2024, 1, 2)) == dt.date(2023, 12, 29)


def test_session_closes_reports_gaps_rather_than_snapping() -> None:
    """A bar on a non-session day means the source and the calendar disagree.
    Snapping it to a neighbour would corrupt every later alignment."""
    days = [dt.date(2024, 1, 3), dt.date(2024, 1, 6), dt.date(2024, 1, 4)]
    closes = session_closes("XNYS", days)
    assert closes[1] is None
    assert closes[0] is not None and closes[2] is not None


def test_range_outside_calendar_coverage_is_empty_not_an_error() -> None:
    assert sessions("XNYS", dt.date(1700, 1, 1), dt.date(1700, 2, 1)) == []
