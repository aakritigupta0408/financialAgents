"""FTA input/output contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from schemas.features import StructureFeatures, LevelFeatures, VolatilityFeatures, LiquidityFeatures
from schemas.forecast import ForecastOutput


class FTACandidate(BaseModel):
    """The proposed trade entry that FTA must validate."""

    ticker: str
    side: Literal["long", "short"]
    entry_price: float
    stop_price: float

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)


class FTAInput(BaseModel):
    """Full structured input required by the FTA engine."""

    candidate: FTACandidate
    structure: StructureFeatures
    levels: LevelFeatures
    volatility: VolatilityFeatures
    liquidity: LiquidityFeatures
    forecast: ForecastOutput

    model_config = {"arbitrary_types_allowed": True}


class FTARejectionReason(BaseModel):
    code: str
    detail: str


class FTAOutput(BaseModel):
    """Deterministic output from the FTA engine."""

    ticker: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    verdict: "FTAVerdict"

    # Computed metrics — always populated regardless of verdict
    nearest_trouble_price: float | None = None  # First Trouble Area
    distance_to_fta_pct: float | None = None  # as % of entry
    reward_risk: float | None = None
    structure_score: float | None = None  # 0–1
    liquidity_score: float | None = None  # 0–1
    volatility_ok: bool | None = None

    rejection_reasons: list[FTARejectionReason] = Field(default_factory=list)


class FTAVerdict(BaseModel):
    accepted: bool
    score: float  # 0–1 composite quality score
    summary: str
