"""Market hours scheduler for NYSE/NASDAQ.

NYSE regular session: 09:30–16:00 Eastern Time (America/New_York).
This module uses only stdlib so it works without pytz.

Market holidays are NOT tracked — use a trading calendar library (e.g.,
exchange_calendars) if holiday-precise scheduling is needed.
TODO: integrate exchange_calendars when adding holiday awareness.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone

log = logging.getLogger(__name__)

# UTC offsets for Eastern Time:
# EST = UTC-5  (standard time Nov–Mar)
# EDT = UTC-4  (daylight saving Mar–Nov)
# We approximate: if month in [4..10] assume EDT, else EST.
_EDT = timezone(timedelta(hours=-4), name="EDT")
_EST = timezone(timedelta(hours=-5), name="EST")

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_MARKET_DAYS = {0, 1, 2, 3, 4}    # Mon–Fri


def _eastern_tz(dt: datetime) -> timezone:
    """Approximate EDT/EST offset for a given UTC datetime."""
    month = dt.month
    return _EDT if 4 <= month <= 10 else _EST


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_eastern(dt: datetime) -> datetime:
    """Convert a UTC datetime to Eastern Time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    tz = _eastern_tz(dt)
    return dt.astimezone(tz)


def is_market_open(as_of: datetime | None = None) -> bool:
    """
    Return True if the NYSE regular session is currently open.

    Parameters
    ----------
    as_of : UTC datetime to check (defaults to now).
    """
    as_of = as_of or utc_now()
    et = to_eastern(as_of)
    if et.weekday() not in _MARKET_DAYS:
        return False
    current = et.time().replace(second=0, microsecond=0)
    return _MARKET_OPEN <= current < _MARKET_CLOSE


def is_market_closed_for_day(as_of: datetime | None = None) -> bool:
    """True after 16:00 ET on a weekday, or on a weekend."""
    as_of = as_of or utc_now()
    et = to_eastern(as_of)
    if et.weekday() not in _MARKET_DAYS:
        return True
    return et.time() >= _MARKET_CLOSE


def seconds_until_market_open(as_of: datetime | None = None) -> float:
    """Seconds until the next NYSE market open (9:30 ET)."""
    as_of = as_of or utc_now()
    et = to_eastern(as_of)

    # Build next open datetime in ET
    candidate = et.replace(hour=9, minute=30, second=0, microsecond=0)
    if candidate <= et:
        candidate += timedelta(days=1)
    # Advance to a weekday
    while candidate.weekday() not in _MARKET_DAYS:
        candidate += timedelta(days=1)

    delta = candidate - et
    return max(0.0, delta.total_seconds())


def seconds_until_market_close(as_of: datetime | None = None) -> float:
    """Seconds until 16:00 ET today. Returns 0 if market already closed."""
    as_of = as_of or utc_now()
    et = to_eastern(as_of)
    close_today = et.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = close_today - et
    return max(0.0, delta.total_seconds())


def market_session_label(as_of: datetime | None = None) -> str:
    """Return a human-readable session label."""
    as_of = as_of or utc_now()
    if is_market_open(as_of):
        return "OPEN"
    et = to_eastern(as_of)
    if et.weekday() not in _MARKET_DAYS:
        return "WEEKEND"
    if et.time() < _MARKET_OPEN:
        return "PRE_MARKET"
    return "AFTER_HOURS"
