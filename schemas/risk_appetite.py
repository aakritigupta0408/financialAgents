"""RiskAppetiteConfig — first-class risk appetite schema.

Supports four modes: conservative | moderate | aggressive | custom.
All fields can be overridden per-instance regardless of mode (custom baseline).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskAppetiteConfig(BaseModel):
    """
    Complete risk appetite specification.

    This is the outer safety envelope; adaptive thresholds may tune within its
    bounds but cannot exceed the hard limits defined here.
    """

    mode: Literal["conservative", "moderate", "aggressive", "custom"] = "moderate"

    # ── Position sizing ────────────────────────────────────────────────────
    risk_per_trade_pct: float = Field(default=0.01, gt=0.0, le=0.10)
    """Fraction of equity risked per trade (stop-loss basis)."""

    day_trade_size_multiplier: float = Field(default=0.75, gt=0.0, le=2.0)
    """Scale factor applied to base size for day trades (intraday)."""

    swing_trade_size_multiplier: float = Field(default=1.0, gt=0.0, le=2.0)
    """Scale factor applied to base size for swing trades (multi-day)."""

    # ── Daily exposure limits ──────────────────────────────────────────────
    max_trades_per_day: int = Field(default=5, ge=1, le=20)
    max_concurrent_positions: int = Field(default=3, ge=1, le=10)
    max_daily_drawdown_pct: float = Field(default=0.03, gt=0.0, le=0.20)
    max_ticker_exposure_pct: float = Field(default=0.10, gt=0.0, le=0.50)
    max_portfolio_exposure_pct: float = Field(default=0.90, gt=0.0, le=1.0)

    # ── Signal quality thresholds ──────────────────────────────────────────
    min_meta_model_probability: float = Field(default=0.60, ge=0.0, le=1.0)
    min_fta_score: float = Field(default=0.60, ge=0.0, le=1.0)
    min_reward_risk: float = Field(default=2.0, gt=0.0)
    min_forecast_confidence: float = Field(default=0.55, ge=0.0, le=1.0)

    # ── News / event risk ─────────────────────────────────────────────────
    event_risk_behavior: Literal["skip", "reduce", "normal"] = "reduce"
    """
    skip   — do not open new trades around major events.
    reduce — open but scale position by event_risk_size_multiplier.
    normal — ignore event risk.
    """
    event_risk_size_multiplier: float = Field(default=0.50, gt=0.0, le=1.0)
    """Applied to position size when event_risk_behavior='reduce'."""

    # ── Streak and cooldown ────────────────────────────────────────────────
    losing_streak_size_reduction: float = Field(default=0.10, ge=0.0, le=0.50)
    """
    Fraction by which position size is reduced per consecutive losing trade.
    E.g. 0.10 → 10% smaller after each loss in a streak; resets on a win.
    """
    cooldown_after_daily_loss: bool = True
    """
    If True, halt new trades for the rest of the day once daily_drawdown
    limit is breached.
    """

    # ── Convenience helpers ────────────────────────────────────────────────
    def to_risk_config(self) -> "RiskConfig":
        """Convert to portfolio RiskConfig (used by Portfolio engine)."""
        from src.portfolio.risk import RiskConfig
        from config.settings import STARTING_CAPITAL
        return RiskConfig(
            starting_capital=STARTING_CAPITAL,
            risk_per_trade_pct=self.risk_per_trade_pct,
            max_trades_per_day=self.max_trades_per_day,
            max_concurrent_positions=self.max_concurrent_positions,
            max_daily_drawdown_pct=self.max_daily_drawdown_pct,
            max_ticker_exposure_pct=self.max_ticker_exposure_pct,
            max_portfolio_exposure_pct=self.max_portfolio_exposure_pct,
        )
