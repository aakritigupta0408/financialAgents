"""
src.reports.trades — Per-trade diagnostic report.

generate_trade_diagnostics(result) returns one dict per closed trade with
all fields needed for manual inspection: timing, prices, PnL, outcome,
and meta-feature values where available.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult

_BREAKEVEN_TOLERANCE = 1e-8


def _compute_outcome(realized_pnl: float) -> str:
    """Return "win", "loss", or "breakeven" for a realized PnL value."""
    if realized_pnl > _BREAKEVEN_TOLERANCE:
        return "win"
    if realized_pnl < -_BREAKEVEN_TOLERANCE:
        return "loss"
    return "breakeven"


def _compute_reward_risk_from_prices(
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> float:
    """
    Compute reward:risk from trade prices when meta_features is not available.

    R:R = |target - entry| / |entry - stop|

    Returns 0.0 if denominator is zero.
    """
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    if risk == 0.0:
        return 0.0
    return reward / risk


def generate_trade_diagnostics(result: "BacktestResult") -> list[dict]:
    """
    Build a list of per-trade diagnostic dicts.

    Parameters
    ----------
    result : BacktestResult produced by BacktestEngine.run().

    Returns
    -------
    list of dicts, one per closed trade, with keys:

        trade_id, ticker, side
        entry_time, exit_time, holding_hours (float, >= 0)
        entry_price, exit_price, stop_price, target_price
        realized_pnl, outcome ("win"/"loss"/"breakeven")
        exit_reason
        reward_risk          (from meta_features if present, else computed)
        forecast_direction   (from meta_features, None if missing)
        forecast_confidence  (from meta_features, None if missing)
        meta_model_probability (forecast_confidence proxy, None if missing)
    """
    journal = result.trade_journal or []
    closed = [t for t in journal if t.get("exit_price") is not None]

    diagnostics: list[dict] = []
    for t in closed:
        entry_time = t.get("entry_time")
        exit_time = t.get("exit_time")

        # Compute holding_hours from timestamps.
        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
            holding_hours = max(0.0, (exit_time - entry_time).total_seconds() / 3600.0)
        else:
            holding_hours = 0.0

        realized_pnl = float(t.get("realized_pnl", 0.0))
        outcome = _compute_outcome(realized_pnl)

        entry_price = float(t.get("entry_price") or 0.0)
        exit_price = float(t.get("exit_price") or 0.0)
        stop_price = float(t.get("stop_price") or 0.0)
        target_price = float(t.get("target_price") or 0.0)
        side = t.get("side", "long")

        # Meta-feature extraction.
        mf = t.get("meta_features") or {}

        # reward_risk: prefer meta_features, fall back to price computation.
        rr_raw = mf.get("fta_reward_risk") or mf.get("reward_risk")
        if rr_raw is not None:
            try:
                reward_risk = float(rr_raw)
            except (TypeError, ValueError):
                reward_risk = _compute_reward_risk_from_prices(side, entry_price, stop_price, target_price)
        else:
            reward_risk = _compute_reward_risk_from_prices(side, entry_price, stop_price, target_price)

        # forecast_direction
        fc_dir = mf.get("forecast_direction_up")
        if fc_dir is not None:
            try:
                forecast_direction = "up" if float(fc_dir) >= 0.5 else "down"
            except (TypeError, ValueError):
                forecast_direction = None
        else:
            forecast_direction = None

        # forecast_confidence
        fc_conf = mf.get("forecast_confidence")
        if fc_conf is not None:
            try:
                forecast_confidence = float(fc_conf)
            except (TypeError, ValueError):
                forecast_confidence = None
        else:
            forecast_confidence = None

        # meta_model_probability: use forecast_confidence as proxy
        meta_model_probability = forecast_confidence

        diagnostics.append({
            "trade_id": t.get("trade_id", ""),
            "ticker": t.get("ticker", "UNKNOWN"),
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "holding_hours": holding_hours,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "realized_pnl": realized_pnl,
            "outcome": outcome,
            "exit_reason": t.get("exit_reason", "unknown"),
            "reward_risk": reward_risk,
            "forecast_direction": forecast_direction,
            "forecast_confidence": forecast_confidence,
            "meta_model_probability": meta_model_probability,
        })

    return diagnostics
