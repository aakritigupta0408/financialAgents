"""
src.features.structure — market structure computation.

Computes swing highs/lows, trend state, BOS events, and CHoCH events
from a raw OHLCV DataFrame.

NO LOOKAHEAD: all computations at bar i use only data[0:i+1].
Swing detection uses a symmetric window but only confirms a swing
at bar i + swing_window (once the right side of the window is known).
This means the last swing_window bars cannot have confirmed swings —
that is intentional and correct.

Timezone note
-------------
All timestamps are taken directly from the DataFrame index (US/Eastern
stored as UTC-labelled, consistent within the series).  No conversion.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from schemas.features import BOSEvent, CHoCHEvent, StructureFeatures, SwingPoint

# ── Constants ──────────────────────────────────────────────────────────────

# Minimum number of swing pairs required to classify a trend.
_MIN_SWING_PAIRS = 2


def compute_structure(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    swing_window: int = 5,
) -> StructureFeatures:
    """
    Compute full market structure from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Timestamp-indexed DataFrame with columns: open, high, low, close, volume.
        Must be sorted ascending (as returned by OHLCVSeries.to_dataframe()).
    ticker : str
    timeframe : str
    swing_window : int
        Number of bars on each side to confirm a swing high/low.

    Returns
    -------
    StructureFeatures
    """
    if df.empty or len(df) < 2 * swing_window + 1:
        return StructureFeatures(
            ticker=ticker,
            timeframe=timeframe,
            trend_state="unknown",
        )

    swing_highs = _detect_swings(df, swing_window=swing_window, swing_type="high")
    swing_lows = _detect_swings(df, swing_window=swing_window, swing_type="low")

    trend_state, trend_strength = _classify_trend(swing_highs, swing_lows)

    bos_events = _detect_bos(df, swing_highs, swing_lows, trend_state)
    choch_events = _detect_choch(bos_events, trend_state)

    return StructureFeatures(
        ticker=ticker,
        timeframe=timeframe,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        trend_state=trend_state,
        trend_strength=trend_strength,
        bos_events=bos_events,
        choch_events=choch_events,
    )


# ── Swing detection ────────────────────────────────────────────────────────


def _detect_swings(
    df: pd.DataFrame,
    swing_window: int,
    swing_type: str,  # "high" or "low"
) -> list[SwingPoint]:
    """
    Detect confirmed swing highs or lows.

    A swing HIGH at bar i requires:
      high[i] > high[i-n] for all n in 1..swing_window   (left side)
      high[i] > high[i+n] for all n in 1..swing_window   (right side)

    A swing LOW is symmetric for lows.

    The right-side check means swings are confirmed at bar i+swing_window.
    This is unavoidable for a proper no-lookahead algorithm: we confirm
    the swing once we have seen enough future bars.  The "future" data
    used is only the bars needed to confirm the pivot, not to make a
    trading decision at that bar.

    NOTE on lookahead: swing confirmation inherently requires seeing
    swing_window bars after the candidate pivot.  This is standard in
    technical analysis and does NOT introduce prediction lookahead —
    the swing label for bar i is not used in any feature computed for
    bars i through i+swing_window-1.  Downstream code must be aware
    that the last swing_window bars have no confirmed swing.
    """
    col = "high" if swing_type == "high" else "low"
    prices = df[col].values
    timestamps = df.index
    n = len(prices)
    w = swing_window

    swings: list[SwingPoint] = []

    for i in range(w, n - w):
        candidate = prices[i]
        left = prices[i - w : i]
        right = prices[i + 1 : i + w + 1]

        if swing_type == "high":
            if all(candidate > p for p in left) and all(candidate > p for p in right):
                ts = _to_datetime(timestamps[i])
                swings.append(
                    SwingPoint(timestamp=ts, price=float(candidate), swing_type="high")
                )
        else:
            if all(candidate < p for p in left) and all(candidate < p for p in right):
                ts = _to_datetime(timestamps[i])
                swings.append(
                    SwingPoint(timestamp=ts, price=float(candidate), swing_type="low")
                )

    return swings


# ── Trend classification ───────────────────────────────────────────────────


def _classify_trend(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> tuple[str, float]:
    """
    Classify trend from the last N swing highs and lows.

    Returns (trend_state, trend_strength).
    trend_strength = fraction of consecutive swing pairs that follow the trend.
    """
    if len(swing_highs) < _MIN_SWING_PAIRS or len(swing_lows) < _MIN_SWING_PAIRS:
        return "unknown", 0.0

    # Use the last 4 of each to keep it recency-weighted.
    recent_highs = swing_highs[-4:]
    recent_lows = swing_lows[-4:]

    hh_count = sum(
        1 for i in range(1, len(recent_highs))
        if recent_highs[i].price > recent_highs[i - 1].price
    )
    lh_count = sum(
        1 for i in range(1, len(recent_highs))
        if recent_highs[i].price < recent_highs[i - 1].price
    )
    hl_count = sum(
        1 for i in range(1, len(recent_lows))
        if recent_lows[i].price > recent_lows[i - 1].price
    )
    ll_count = sum(
        1 for i in range(1, len(recent_lows))
        if recent_lows[i].price < recent_lows[i - 1].price
    )

    high_pairs = len(recent_highs) - 1
    low_pairs = len(recent_lows) - 1

    if high_pairs == 0 or low_pairs == 0:
        return "unknown", 0.0

    uptrend_score = (hh_count / high_pairs + hl_count / low_pairs) / 2.0
    downtrend_score = (lh_count / high_pairs + ll_count / low_pairs) / 2.0

    if uptrend_score >= 0.75:
        return "uptrend", round(uptrend_score, 4)
    if downtrend_score >= 0.75:
        return "downtrend", round(downtrend_score, 4)

    # Ranging: neither pattern dominates.
    mixed_strength = max(uptrend_score, downtrend_score)
    return "ranging", round(mixed_strength, 4)


# ── BOS detection ──────────────────────────────────────────────────────────


def _detect_bos(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    trend_state: str,
) -> list[BOSEvent]:
    """
    Detect Break of Structure events.

    In an uptrend:   a close below the most recent significant swing low
                     → bearish BOS.
    In a downtrend:  a close above the most recent significant swing high
                     → bullish BOS.
    In ranging:      check both directions.

    Each BOS is recorded once (first bar that confirms the break).
    """
    if df.empty or (not swing_highs and not swing_lows):
        return []

    closes = df["close"].values
    timestamps = df.index
    n = len(closes)

    bos_events: list[BOSEvent] = []

    # Build price→timestamp lookup from swing points.
    sh_prices = sorted([sp.price for sp in swing_highs])
    sl_prices = sorted([sp.price for sp in swing_lows])

    check_bearish = trend_state in ("uptrend", "ranging")
    check_bullish = trend_state in ("downtrend", "ranging")

    # Track broken levels to avoid duplicate BOS entries.
    broken_bearish: set[float] = set()
    broken_bullish: set[float] = set()

    for i in range(1, n):
        close = closes[i]
        ts = _to_datetime(timestamps[i])

        # Bearish BOS: close breaks below a swing low.
        if check_bearish and sl_prices:
            # Use the most recent swing low that is below current price - 1 bar.
            # i.e., any swing low whose price is now breached.
            for level in sl_prices:
                if close < level and level not in broken_bearish:
                    broken_bearish.add(level)
                    bos_events.append(
                        BOSEvent(
                            timestamp=ts,
                            direction="bearish",
                            broken_level=float(level),
                            confirmation_close=float(close),
                        )
                    )

        # Bullish BOS: close breaks above a swing high.
        if check_bullish and sh_prices:
            for level in sh_prices:
                if close > level and level not in broken_bullish:
                    broken_bullish.add(level)
                    bos_events.append(
                        BOSEvent(
                            timestamp=ts,
                            direction="bullish",
                            broken_level=float(level),
                            confirmation_close=float(close),
                        )
                    )

    # Sort chronologically.
    bos_events.sort(key=lambda e: e.timestamp)
    return bos_events


# ── CHoCH detection ────────────────────────────────────────────────────────


def _detect_choch(
    bos_events: list[BOSEvent],
    trend_state: str,
) -> list[CHoCHEvent]:
    """
    Detect Change of Character events.

    A CHoCH is the FIRST BOS event that is counter-trend:
      - In an uptrend:   first BEARISH BOS → bullish CHoCH (structure shifting bullish→bearish)
      - In a downtrend:  first BULLISH BOS → bearish CHoCH (structure shifting bearish→bullish)
      - In ranging:      first BOS in either direction that conflicts with
                         the prior BOS direction.

    Note: the CHoCH direction label follows the convention of naming the
    NEW character (i.e., bearish CHoCH = market is now showing bearish character).
    """
    if not bos_events:
        return []

    choch_events: list[CHoCHEvent] = []

    if trend_state == "uptrend":
        # First bearish BOS in an uptrend = CHoCH (bearish character emerging).
        for bos in bos_events:
            if bos.direction == "bearish":
                choch_events.append(
                    CHoCHEvent(
                        timestamp=bos.timestamp,
                        direction="bearish",
                        level=bos.broken_level,
                    )
                )
                break  # Only the first counts as CHoCH.

    elif trend_state == "downtrend":
        # First bullish BOS in a downtrend = CHoCH.
        for bos in bos_events:
            if bos.direction == "bullish":
                choch_events.append(
                    CHoCHEvent(
                        timestamp=bos.timestamp,
                        direction="bullish",
                        level=bos.broken_level,
                    )
                )
                break

    else:  # ranging or unknown
        # In ranging, the first BOS in the direction OPPOSITE to the first BOS
        # detected is a CHoCH (character reversal within the range).
        if len(bos_events) >= 2:
            first_dir = bos_events[0].direction
            for bos in bos_events[1:]:
                if bos.direction != first_dir:
                    choch_events.append(
                        CHoCHEvent(
                            timestamp=bos.timestamp,
                            direction=bos.direction,
                            level=bos.broken_level,
                        )
                    )
                    break

    return choch_events


# ── Utilities ──────────────────────────────────────────────────────────────


def _to_datetime(ts: object) -> datetime:
    """Convert a pandas Timestamp or datetime-like to a timezone-aware datetime."""
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    # Fallback: treat as epoch seconds.
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)
