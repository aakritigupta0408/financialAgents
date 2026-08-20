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
    sign = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)
    lo60, hi60 = min(closes[-60:]), max(closes[-60:])
    r5, r15, r60 = _ret(closes, 5), _ret(closes, 15), _ret(closes, 60)
    return {
        "ret_240m": _ret(closes, 240),
        "trend_align": (sign(r5) + sign(r15) + sign(r60)) / 3.0,
        "range_pos": ((closes[-1] - lo60) / (hi60 - lo60) - 0.5
                      if hi60 > lo60 else 0.0),
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


def feature_vector(feat: dict) -> list[float]:
    """Normalized continuous context for linear/bandit models (LinUCB).

    Leading 1.0 is the intercept. Returns are in ~basis-point scale, RSI and
    Fear&Greed centered at 0 — everything roughly O(1) so one ridge prior
    (lambda*I) suits all dimensions.
    """
    import math as _m
    return [
        1.0,
        feat["ret_1m"] * 1e4 / 10,
        feat["ret_5m"] * 1e4 / 20,
        feat["ret_15m"] * 1e4 / 30,
        feat["ret_60m"] * 1e4 / 60,
        feat["vol_30m"] * 1e4 / 10,
        (feat["rsi_14"] - 50) / 50,
        feat["ema_dist"] * 1e4 / 30,
        _m.log(max(feat["vol_ratio"], 1e-6)) / 2,
        (feat["fng"] - 50) / 50,
    ]


LIVE_DIM = 5
TREND_DIM = 3
BOOK_DIM = 2
LLM_DIM = 3


def llm_feature_vector(snap: dict | None) -> list[float]:
    """CryptoBERT-derived context (t5): sentiment, its 1h momentum, intensity."""
    if not snap:
        return [0.0] * LLM_DIM
    return [
        snap.get("sent") or 0.0,               # mean P(bull)-P(bear), [-1, 1]
        snap.get("sent_mom") or 0.0,           # sentiment change vs ~1h ago
        min((snap.get("news_n") or 0) / 20, 1.5),  # headline intensity
    ]


def trend_feature_vector(feat: dict) -> list[float]:
    """Trend context (t4): 4h momentum, multi-scale alignment, range position."""
    return [
        feat.get("ret_240m", 0.0) * 1e4 / 120,
        feat.get("trend_align", 0.0),
        feat.get("range_pos", 0.0) * 2,
    ]


def book_feature_vector(snap: dict | None) -> list[float]:
    """Order-book context (t4): bid/ask imbalance and spread, neutral-at-0."""
    if not snap:
        return [0.0] * BOOK_DIM
    return [
        snap.get("imb") or 0.0,
        (snap.get("spread_bp") or 0.0) / 5,
    ]


def live_feature_vector(snap: dict | None) -> list[float]:
    """Streamed market-microstructure features, normalized to ~O(1).

    All centered so a missing snapshot (None) legitimately reads as neutral:
      funding rate (x1e4), perp basis bp (/10), cross-exchange dispersion
      bp (/5), BRTI-composite-vs-Coinbase gap bp (/5), mempool fee (/10).
    """
    if not snap:
        return [0.0] * LIVE_DIM
    return [
        (snap.get("funding") or 0.0) * 1e4,
        (snap.get("basis_bp") or 0.0) / 10,
        (snap.get("disp_bp") or 0.0) / 5,
        (snap.get("gap_bp") or 0.0) / 5,
        (snap.get("fee") or 0.0) / 10,
    ]


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
