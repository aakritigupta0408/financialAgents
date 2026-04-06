"""Feature schemas — output contract from feature engineering."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SwingPoint(BaseModel):
    timestamp: datetime
    price: float
    swing_type: Literal["high", "low"]


class BOSEvent(BaseModel):
    """Break of Structure event."""

    timestamp: datetime
    direction: Literal["bullish", "bearish"]
    broken_level: float
    confirmation_close: float


class CHoCHEvent(BaseModel):
    """Change of Character event."""

    timestamp: datetime
    direction: Literal["bullish", "bearish"]
    level: float


class StructureFeatures(BaseModel):
    ticker: str
    timeframe: str
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    swing_highs: list[SwingPoint] = Field(default_factory=list)
    swing_lows: list[SwingPoint] = Field(default_factory=list)
    trend_state: Literal["uptrend", "downtrend", "ranging", "unknown"] = "unknown"
    trend_strength: float = 0.0  # 0–1
    bos_events: list[BOSEvent] = Field(default_factory=list)
    choch_events: list[CHoCHEvent] = Field(default_factory=list)


class PriceZone(BaseModel):
    low: float
    high: float
    strength: float  # 0–1; how many times tested
    zone_type: Literal["support", "resistance"]


class LevelFeatures(BaseModel):
    ticker: str
    timeframe: str
    support_zones: list[PriceZone] = Field(default_factory=list)
    resistance_zones: list[PriceZone] = Field(default_factory=list)


class VolatilityFeatures(BaseModel):
    ticker: str
    timeframe: str
    atr: float  # Average True Range in price units
    atr_pct: float  # ATR as % of close
    volatility_regime: Literal["low", "normal", "high", "extreme"] = "normal"
    is_expanding: bool = False


class LiquidityFeatures(BaseModel):
    ticker: str
    timeframe: str
    avg_volume: float
    relative_volume: float  # current / avg
    spread_estimate: float  # in price units; 0.0 if unknown
