"""Signal schemas — candidate and ranked trades flowing through the pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from schemas.fta import FTAOutput
from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelOutput


class CandidateTrade(BaseModel):
    """
    A trade candidate produced after FTA acceptance.
    Has not yet been scored by the meta-model.
    """

    candidate_id: str
    ticker: str
    side: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float | None = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    fta_output: FTAOutput
    forecast: ForecastOutput


class RankedTrade(BaseModel):
    """
    A CandidateTrade scored and ranked by the meta-model.
    Ready for portfolio/risk review.
    """

    candidate: CandidateTrade
    meta_output: MetaModelOutput
    rank: int | None = None  # lower = better; set by the ranker
    approved_for_execution: bool = False
