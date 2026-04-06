"""
src.backtest.candidate — Temporary candidate trade generator.

This module is a placeholder until the full FTA engine and meta-model are wired
in Phase 8. It translates a ForecastOutput into a raw candidate trade dict using
only ATR-based stop/target placement and a minimum reward:risk sanity check.

It is deliberately NOT an FTA filter. The real FTA layer (market structure,
liquidity, first trouble areas, etc.) will replace or wrap this in Phase 8.

Short selling note
------------------
Short candidates are not generated because the portfolio engine uses a
simplified cash-account model where short margin/collateral is not tracked.
Shorts will be re-enabled in Phase 8 when the margin model is implemented.
See TODO below.
"""

from __future__ import annotations

from schemas.features import VolatilityFeatures
from schemas.forecast import ForecastOutput

# Sanity-check minimum reward:risk ratio (hard lower bound, not FTA).
# FTA uses FTA_MIN_REWARD_RISK from settings.py (default 2.0).
# This placeholder uses a looser 1.5 so more trades are generated during
# early backtesting when no FTA layer is present.
_MIN_RR = 1.5


def generate_candidate(
    forecast: ForecastOutput,
    volatility: VolatilityFeatures,
    current_close: float,
    atr_stop_multiple: float = 1.5,
    atr_target_multiple: float = 3.0,
) -> dict | None:
    """
    Generate a candidate trade dict from a forecast and volatility features.

    Parameters
    ----------
    forecast           : ForecastOutput from the TimesFM wrapper.
    volatility         : VolatilityFeatures containing the current ATR.
    current_close      : The close price at the current bar (entry price proxy).
    atr_stop_multiple  : Stop distance = atr_stop_multiple * ATR below entry.
    atr_target_multiple: Target distance = atr_target_multiple * ATR above entry.

    Returns
    -------
    dict with keys: side, entry, stop, target, reward_risk, forecast_confidence
    None if:
    - direction == "down" (shorts not yet implemented)
    - direction is not "up" or "down"
    - ATR is zero (degenerate series)
    - computed reward:risk < _MIN_RR

    Side effects
    ------------
    None. This function is pure (deterministic given inputs).
    """
    if forecast.direction != "up":
        # TODO (Phase 8): implement short candidates once margin model is ready.
        return None

    atr = volatility.atr
    if atr <= 0.0:
        # Degenerate ATR — cannot size stops or targets.
        return None

    entry = current_close
    stop = entry - atr_stop_multiple * atr
    target = entry + atr_target_multiple * atr

    # Stop must be strictly below entry for a long trade.
    if stop >= entry:
        return None

    risk = entry - stop
    reward = target - entry
    rr = reward / risk

    if rr < _MIN_RR:
        return None

    return {
        "side": "long",
        "entry": entry,
        "stop": stop,
        "target": target,
        "reward_risk": rr,
        "forecast_confidence": forecast.confidence,
    }
