"""TimesFM forecast output contract."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ForecastOutput(BaseModel):
    """
    Standardised forecast from TimesFM.
    This is a feature input to FTA and the meta-model — not a trade signal.
    """

    ticker: str
    timeframe: str
    forecast_at: datetime = Field(default_factory=datetime.utcnow)

    direction: Literal["up", "down"]
    expected_return: float  # fractional, e.g. 0.012 = +1.2%
    confidence: float  # 0–1
    horizon: int  # bars ahead

    # Optional: full quantile path if available
    quantile_50: list[float] = Field(default_factory=list)
    quantile_10: list[float] = Field(default_factory=list)
    quantile_90: list[float] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))
