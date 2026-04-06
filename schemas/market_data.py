"""Market data schemas — output contract from the data layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class OHLCVBar(BaseModel):
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: str
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_ohlcv(self) -> "OHLCVBar":
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Open {self.open} outside [low={self.low}, high={self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Close {self.close} outside [low={self.low}, high={self.high}]")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")
        return self


class OHLCVSeries(BaseModel):
    """Validated series of OHLCV bars for one ticker and timeframe."""

    ticker: str
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    bars: list[OHLCVBar] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame indexed by timestamp, sorted ascending."""
        if not self.bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame([b.model_dump() for b in self.bars])
        df = df.set_index("timestamp").sort_index()
        df = df.drop(columns=["ticker", "timeframe"])
        return df

    @property
    def latest_close(self) -> float | None:
        if not self.bars:
            return None
        return max(self.bars, key=lambda b: b.timestamp).close


class MarketSnapshot(BaseModel):
    """
    All timeframes for a single ticker at a point in time.
    Used as the primary input to feature engineering and FTA.
    """

    ticker: str
    snapshot_time: datetime
    tf_1m: OHLCVSeries | None = None
    tf_5m: OHLCVSeries | None = None
    tf_15m: OHLCVSeries | None = None
    tf_1h: OHLCVSeries | None = None
    tf_4h: OHLCVSeries | None = None
    tf_1d: OHLCVSeries | None = None

    model_config = {"arbitrary_types_allowed": True}

    def get(self, timeframe: str) -> OHLCVSeries | None:
        mapping = {
            "1m": self.tf_1m,
            "5m": self.tf_5m,
            "15m": self.tf_15m,
            "1h": self.tf_1h,
            "4h": self.tf_4h,
            "1d": self.tf_1d,
        }
        return mapping.get(timeframe)
