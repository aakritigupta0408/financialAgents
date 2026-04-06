"""Built-in risk appetite presets.

conservative — small size, tight quality gates, skip events, max 3 trades/day
moderate     — balanced defaults (system baseline)
aggressive   — larger size, relaxed gates, max 8 trades/day, ride events
"""
from __future__ import annotations

from schemas.risk_appetite import RiskAppetiteConfig

CONSERVATIVE = RiskAppetiteConfig(
    mode="conservative",
    risk_per_trade_pct=0.005,          # 0.5% of equity per trade
    day_trade_size_multiplier=0.50,
    swing_trade_size_multiplier=0.75,
    max_trades_per_day=3,
    max_concurrent_positions=2,
    max_daily_drawdown_pct=0.02,       # 2% daily drawdown halt
    max_ticker_exposure_pct=0.07,
    max_portfolio_exposure_pct=0.60,
    min_meta_model_probability=0.70,
    min_fta_score=0.70,
    min_reward_risk=2.5,
    min_forecast_confidence=0.70,
    event_risk_behavior="skip",
    event_risk_size_multiplier=0.25,   # irrelevant when behavior=skip, kept for schema completeness
    losing_streak_size_reduction=0.15,
    cooldown_after_daily_loss=True,
)

MODERATE = RiskAppetiteConfig(
    mode="moderate",
    risk_per_trade_pct=0.01,
    day_trade_size_multiplier=0.75,
    swing_trade_size_multiplier=1.0,
    max_trades_per_day=5,
    max_concurrent_positions=3,
    max_daily_drawdown_pct=0.03,
    max_ticker_exposure_pct=0.10,
    max_portfolio_exposure_pct=0.80,
    min_meta_model_probability=0.60,
    min_fta_score=0.60,
    min_reward_risk=2.0,
    min_forecast_confidence=0.55,
    event_risk_behavior="reduce",
    event_risk_size_multiplier=0.50,
    losing_streak_size_reduction=0.10,
    cooldown_after_daily_loss=True,
)

AGGRESSIVE = RiskAppetiteConfig(
    mode="aggressive",
    risk_per_trade_pct=0.02,
    day_trade_size_multiplier=1.00,
    swing_trade_size_multiplier=1.25,
    max_trades_per_day=8,
    max_concurrent_positions=5,
    max_daily_drawdown_pct=0.06,
    max_ticker_exposure_pct=0.20,
    max_portfolio_exposure_pct=0.95,
    min_meta_model_probability=0.50,
    min_fta_score=0.50,
    min_reward_risk=1.5,
    min_forecast_confidence=0.45,
    event_risk_behavior="normal",
    event_risk_size_multiplier=1.0,
    losing_streak_size_reduction=0.05,
    cooldown_after_daily_loss=False,
)

_PRESETS: dict[str, RiskAppetiteConfig] = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
}


def get_preset(mode: str) -> RiskAppetiteConfig:
    """Return a preset by name. Raises ValueError for unknown modes."""
    if mode not in _PRESETS:
        raise ValueError(f"Unknown risk appetite mode '{mode}'. Choose from: {list(_PRESETS)}")
    return _PRESETS[mode]
