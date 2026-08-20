"""Feature engineering: turn raw 1m bars + sentiment into a state vector.

Feature families (from the factor research — see README):
  momentum   ret_1m / ret_5m / ret_15m / ret_60m
  volatility std of 1m returns over the last 30m
  technical  RSI(14), distance from EMA(30)
  liquidity  volume ratio (last 5m vs last 60m average)
  sentiment  daily Fear & Greed index
"""
from __future__ import annotations

import math


def _ret(closes: list[float], minutes: int) -> float:
    if len(closes) <= minutes or closes[-1 - minutes] == 0:
        return 0.0
    return closes[-1] / closes[-1 - minutes] - 1.0


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = losses = 0.0
    for prev, cur in zip(closes[-period - 1:-1], closes[-period:]):
        change = cur - prev
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)


def ema(closes: list[float], period: int = 30) -> float:
    k = 2.0 / (period + 1)
    value = closes[0]
    for c in closes[1:]:
        value = c * k + value * (1 - k)
    return value


def compute_features(bars: list[dict], fng: int | None) -> dict:
    """Continuous features from the lookback window ending at the decision bar."""
    closes = [b["close"] for b in bars]
    vols = [b["volume"] for b in bars]
    rets_1m = [c2 / c1 - 1.0 for c1, c2 in zip(closes[-31:-1], closes[-30:])]
    mean = sum(rets_1m) / len(rets_1m)
    vol_30m = math.sqrt(sum((r - mean) ** 2 for r in rets_1m) / len(rets_1m))
    avg_vol_60 = sum(vols[-60:]) / min(60, len(vols))
    avg_vol_5 = sum(vols[-5:]) / min(5, len(vols))
    return {
        "price": closes[-1],
        "ret_1m": _ret(closes, 1),
        "ret_5m": _ret(closes, 5),
        "ret_15m": _ret(closes, 15),
        "ret_60m": _ret(closes, 60),
        "vol_30m": vol_30m,
        "rsi_14": rsi(closes),
        "ema_dist": closes[-1] / ema(closes[-60:]) - 1.0,
        "vol_ratio": avg_vol_5 / avg_vol_60 if avg_vol_60 else 1.0,
        "fng": fng if fng is not None else 50,
    }


def _bucket3(x: float, lo: float, hi: float) -> int:
    """3-way bucket: 0 below lo, 1 between, 2 above hi."""
    if x < lo:
        return 0
    if x > hi:
        return 2
    return 1


def discretize(feat: dict) -> tuple[int, ...]:
    """Map continuous features to a small discrete state for tabular Q.

    Thresholds are round numbers chosen from typical 1m BTC behavior, not
    fitted to data (keeps the POC honest — no leakage from the test set).
    """
    return (
        _bucket3(feat["ret_5m"], -0.0005, 0.0005),
        _bucket3(feat["ret_15m"], -0.001, 0.001),
        _bucket3(feat["vol_30m"], 0.0004, 0.0009),
        _bucket3(feat["rsi_14"], 35, 65),
    )
