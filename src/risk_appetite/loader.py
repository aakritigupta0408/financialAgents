"""Load the active RiskAppetiteConfig from config/settings.py."""
from __future__ import annotations

import logging

from schemas.risk_appetite import RiskAppetiteConfig
from src.risk_appetite.presets import get_preset

log = logging.getLogger(__name__)


def load_risk_appetite(mode: str | None = None) -> RiskAppetiteConfig:
    """
    Return a RiskAppetiteConfig for the given mode.

    mode=None    → read RISK_APPETITE_MODE from config/settings.py
    mode="custom" → read individual field env-vars and build a custom config

    For "custom" mode, individual fields default to MODERATE values and are
    overridable via the same env-vars as config/settings.py risk limits.
    """
    from config import settings

    effective_mode = mode or settings.RISK_APPETITE_MODE

    if effective_mode == "custom":
        log.info("risk_appetite.loader: building custom config from env")
        return RiskAppetiteConfig(
            mode="custom",
            risk_per_trade_pct=settings.RISK_PER_TRADE_PCT,
            max_trades_per_day=settings.MAX_TRADES_PER_DAY,
            max_concurrent_positions=settings.MAX_CONCURRENT_POSITIONS,
            max_daily_drawdown_pct=settings.MAX_DAILY_DRAWDOWN_PCT,
            max_ticker_exposure_pct=settings.MAX_TICKER_EXPOSURE_PCT,
            min_meta_model_probability=settings.META_MODEL_MIN_CONFIDENCE,
            min_reward_risk=settings.FTA_MIN_REWARD_RISK,
        )

    try:
        cfg = get_preset(effective_mode)
        log.info("risk_appetite.loader: loaded preset '%s'", effective_mode)
        return cfg
    except ValueError:
        log.warning(
            "risk_appetite.loader: unknown mode '%s'; falling back to moderate", effective_mode
        )
        return get_preset("moderate")
