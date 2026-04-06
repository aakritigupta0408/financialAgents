"""Meta-model input/output contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class MetaModelInput(BaseModel):
    """
    Flat feature vector fed to the meta-model classifier.
    All values must be numeric or None (filled with sentinel before training).
    """

    ticker: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    # Forecast features
    forecast_direction_up: float  # 1.0 = up, 0.0 = down
    forecast_expected_return: float
    forecast_confidence: float

    # FTA features
    fta_reward_risk: float
    fta_distance_to_fta_pct: float
    fta_structure_score: float
    fta_liquidity_score: float
    fta_volatility_ok: float  # 1.0 / 0.0

    # Volatility features
    atr_pct: float
    volatility_regime_encoded: float  # low=0, normal=1, high=2, extreme=3

    # Liquidity features
    relative_volume: float

    # Structure features
    trend_strength: float
    trend_state_encoded: float  # uptrend=1, downtrend=-1, ranging=0, unknown=0


class MetaModelOutput(BaseModel):
    """Output from the meta-model."""

    ticker: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    probability_of_success: float  # 0–1
    confidence: float  # model confidence; may differ from raw probability
    should_trade: bool  # threshold applied

    @field_validator("probability_of_success", "confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))
