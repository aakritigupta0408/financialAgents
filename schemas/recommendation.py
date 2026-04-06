"""TradeRecommendation — user-facing output from the recommendation engine."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class TradeRecommendation(BaseModel):
    """
    The final user-facing recommendation produced by the recommendation engine.

    action / position_action semantics
    ────────────────────────────────────
    action=BUY,  position_action=OPEN    → enter a long position
    action=SELL, position_action=OPEN    → enter a short position (if supported)
    action=SELL, position_action=CLOSE   → exit an existing long position
    action=BUY,  position_action=CLOSE   → cover an existing short
    action=SELL, position_action=REDUCE  → trim an existing long
    action=HOLD, position_action=HOLD_POSITION → no change to existing position
    action=HOLD, position_action=OPEN    → no new trade (all gates passed except final)
    """

    ticker: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Core decision ──────────────────────────────────────────────────────
    action: Literal["BUY", "SELL", "HOLD"]
    position_action: Literal["OPEN", "CLOSE", "REDUCE", "HOLD_POSITION"]
    trade_style: Literal["day_trade", "swing_trade"] | None = None

    # ── Prices ────────────────────────────────────────────────────────────
    entry_price: float | None = None
    stop_price: float | None = None
    target_1: float | None = None
    target_2: float | None = None

    # ── Risk / reward metrics ──────────────────────────────────────────────
    expected_return: float | None = None     # as fraction, e.g. 0.04 = 4%
    reward_risk: float | None = None
    recommended_position_size: float | None = None   # in shares

    # ── Signal scores (pass-through for transparency) ──────────────────────
    forecast_confidence: float | None = None
    fta_score: float | None = None
    probability_of_success: float | None = None

    # ── Context labels ────────────────────────────────────────────────────
    risk_profile: str = "moderate"
    timeframe_mode: str = "daily_only"
    news_mode: str = "disabled"

    # ── Explanation ───────────────────────────────────────────────────────
    rationale: str = ""
    rejection_reason: str | None = None

    @property
    def is_actionable(self) -> bool:
        """True if this recommendation opens or closes a position."""
        return self.position_action in ("OPEN", "CLOSE", "REDUCE")
