"""
src.adaptive.updater — Apply analysed recommendations to AdaptiveContext.

apply_update(context, analysis, report) → (AdaptiveContext, UpdateSummary)

Safety rules enforced (in order):
1. Minimum sample size: n_trades_analyzed < 10 → suppress all threshold changes.
2. Cap per-session change: ±0.10 for confidence thresholds, ±0.50 for RR.
3. Degradation safeguard: if win_rate drops > 0.15 after ≥ 3 sessions,
   do NOT change thresholds.
4. Clamp all values to allowed ranges.

When not suppressed:
- Update best_thresholds.
- EMA blend recent_performance (alpha=0.3) and increment counters.
- Update feature_importance, per_ticker_stats, regime_stats.
- Increment version and set updated_at.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.adaptive.context import (
    AdaptiveContext,
    BestThresholds,
    RegimeStats,
    TickerStats,
    _now_iso,
)
from src.adaptive.analyzer import AnalysisResult

log = logging.getLogger(__name__)

_EMA_ALPHA = 0.3

# Maximum allowed change per session call
_MAX_CONFIDENCE_DELTA = 0.10
_MAX_RR_DELTA = 0.50
_MAX_FORECAST_CONFIDENCE_DELTA = 0.10

# Bounds (mirrors context.py _BOUNDS for inline clamping)
_BOUNDS = {
    "meta_model_min_confidence": (0.30, 0.90),
    "min_reward_risk": (1.0, 5.0),
    "forecast_confidence_min": (0.10, 0.90),
    "fta_score_min": (0.0, 1.0),
}


@dataclass
class UpdateSummary:
    thresholds_changed: bool
    old_thresholds: dict
    new_thresholds: dict
    performance_delta: dict   # win_rate_delta, return_delta, drawdown_delta
    warnings: list[str]
    update_suppressed: bool
    suppression_reason: str   # empty string if not suppressed


def apply_update(
    context: AdaptiveContext,
    analysis: AnalysisResult,
    report: dict,
) -> tuple[AdaptiveContext, UpdateSummary]:
    """
    Apply analysed recommendations to a copy of the context.

    Parameters
    ----------
    context  : Current AdaptiveContext (will NOT be mutated).
    analysis : AnalysisResult from analyze_report().
    report   : Full report dict from generate_full_report().

    Returns
    -------
    (new_context, update_summary)
    """
    # Deep-copy so we never mutate the caller's context.
    new_ctx = _deep_copy_context(context)

    old_thresholds = dataclasses.asdict(context.best_thresholds)
    portfolio = report.get("portfolio") or {}
    current_win_rate = float(portfolio.get("win_rate") or 0.0)
    current_return = float(portfolio.get("total_return_pct") or 0.0)
    current_dd = float(portfolio.get("max_drawdown_pct") or 0.0)
    current_sharpe = float(portfolio.get("sharpe_ratio") or 0.0)
    current_n_trades = int(portfolio.get("n_trades") or 0)

    warnings: list[str] = list(analysis.warnings)

    # ── Safety rule 1: Minimum sample size ───────────────────────────────────
    if analysis.n_trades_analyzed < 10:
        return new_ctx, UpdateSummary(
            thresholds_changed=False,
            old_thresholds=old_thresholds,
            new_thresholds=old_thresholds,
            performance_delta={"win_rate_delta": 0.0, "return_delta": 0.0, "drawdown_delta": 0.0},
            warnings=warnings,
            update_suppressed=True,
            suppression_reason="insufficient_sample_size",
        )

    # ── Safety rule 3: Degradation safeguard ─────────────────────────────────
    if (
        context.recent_performance.n_sessions >= 3
        and current_win_rate < context.recent_performance.win_rate - 0.15
    ):
        warnings.append("Performance degradation detected — reverting thresholds")
        return new_ctx, UpdateSummary(
            thresholds_changed=False,
            old_thresholds=old_thresholds,
            new_thresholds=old_thresholds,
            performance_delta={
                "win_rate_delta": current_win_rate - context.recent_performance.win_rate,
                "return_delta": 0.0,
                "drawdown_delta": 0.0,
            },
            warnings=warnings,
            update_suppressed=True,
            suppression_reason="performance_degradation",
        )

    # ── Safety rule 2: Cap per-session change ────────────────────────────────
    old_conf = context.best_thresholds.meta_model_min_confidence
    new_conf = _clipped_update(
        old_conf,
        analysis.recommended_confidence_threshold,
        _MAX_CONFIDENCE_DELTA,
    )

    old_rr = context.best_thresholds.min_reward_risk
    new_rr = _clipped_update(
        old_rr,
        analysis.recommended_rr_minimum,
        _MAX_RR_DELTA,
    )

    old_fc = context.best_thresholds.forecast_confidence_min
    new_fc = _clipped_update(
        old_fc,
        analysis.recommended_forecast_confidence,
        _MAX_FORECAST_CONFIDENCE_DELTA,
    )

    # fta_score_min: unchanged (no sweep for it currently)
    new_fta_score = context.best_thresholds.fta_score_min

    # ── Rule 4: Clamp ─────────────────────────────────────────────────────────
    new_conf = _clamp(new_conf, *_BOUNDS["meta_model_min_confidence"])
    new_rr = _clamp(new_rr, *_BOUNDS["min_reward_risk"])
    new_fc = _clamp(new_fc, *_BOUNDS["forecast_confidence_min"])
    new_fta_score = _clamp(new_fta_score, *_BOUNDS["fta_score_min"])

    new_thresholds_obj = BestThresholds(
        meta_model_min_confidence=new_conf,
        min_reward_risk=new_rr,
        forecast_confidence_min=new_fc,
        fta_score_min=new_fta_score,
    )
    new_thresholds_dict = dataclasses.asdict(new_thresholds_obj)

    thresholds_changed = new_thresholds_dict != old_thresholds

    # ── Apply all updates to new_ctx ─────────────────────────────────────────
    new_ctx.best_thresholds = new_thresholds_obj

    # EMA performance blend
    rp = new_ctx.recent_performance
    old_win_rate = rp.win_rate
    old_return = rp.total_return_pct
    old_dd = rp.max_drawdown_pct

    rp.win_rate = _ema(rp.win_rate, current_win_rate)
    rp.total_return_pct = _ema(rp.total_return_pct, current_return)
    rp.max_drawdown_pct = _ema(rp.max_drawdown_pct, current_dd)
    rp.sharpe_ratio = _ema(rp.sharpe_ratio, current_sharpe)
    rp.n_sessions += 1
    rp.total_trades += current_n_trades

    # Feature importance
    new_ctx.feature_importance = dict(analysis.feature_importance)

    # Per-ticker stats
    for ticker, pnl in analysis.per_ticker_pnl.items():
        if ticker in new_ctx.per_ticker_stats:
            ts = new_ctx.per_ticker_stats[ticker]
            ts.total_pnl += pnl
            ts.n_trades += 1  # approximate increment
        else:
            new_ctx.per_ticker_stats[ticker] = TickerStats(
                ticker=ticker,
                n_trades=1,
                win_rate=0.0,
                total_pnl=pnl,
            )

    # Regime stats
    for regime, stats in analysis.regime_stats.items():
        new_ctx.regime_stats[regime] = RegimeStats(
            regime=regime,
            n_trades=int(stats.get("n_trades", 0)),
            win_rate=float(stats.get("win_rate", 0.0)),
            avg_pnl=float(stats.get("avg_pnl", 0.0)),
        )

    # Increment version and update timestamp
    new_ctx.version = context.version + 1
    new_ctx.updated_at = _now_iso()

    return new_ctx, UpdateSummary(
        thresholds_changed=thresholds_changed,
        old_thresholds=old_thresholds,
        new_thresholds=new_thresholds_dict,
        performance_delta={
            "win_rate_delta": current_win_rate - old_win_rate,
            "return_delta": current_return - old_return,
            "drawdown_delta": current_dd - old_dd,
        },
        warnings=warnings,
        update_suppressed=False,
        suppression_reason="",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ema(old: float, latest: float, alpha: float = _EMA_ALPHA) -> float:
    """Exponential moving average: new = alpha * latest + (1-alpha) * old."""
    return alpha * latest + (1.0 - alpha) * old


def _clipped_update(old: float, proposed: float, max_delta: float) -> float:
    """Clip proposed change to ±max_delta from old."""
    delta = proposed - old
    delta = max(-max_delta, min(max_delta, delta))
    return old + delta


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _deep_copy_context(ctx: AdaptiveContext) -> AdaptiveContext:
    """Return a deep copy of an AdaptiveContext using dataclasses.asdict + reconstruction."""
    from src.adaptive.context import _from_dict
    return _from_dict(dataclasses.asdict(ctx))
