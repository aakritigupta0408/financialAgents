"""
src.reports.threshold_tuning — Threshold sensitivity analysis.

sweep_thresholds(result) filters the existing trade_journal by different
confidence / reward-risk thresholds WITHOUT re-running the backtest, then
re-computes metrics on each filtered subset.

This is useful for post-hoc analysis: "if I had required a higher confidence
threshold, would performance have improved?"

max_drawdown_pct on a filtered subset
--------------------------------------
Trades are sorted by entry_time. The cumulative PnL series is built by
accumulating realized_pnl in that order. The drawdown is computed on the
cumulative PnL series (not on the full equity curve).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult

_BREAKEVEN_TOLERANCE = 1e-8

_DEFAULT_CONFIDENCE_THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_DEFAULT_RR_MINIMUMS = [1.0, 1.5, 2.0, 2.5, 3.0]
_DEFAULT_FTA_SCORE_MINIMUMS = [0.0, 0.3, 0.5, 0.7]


def _compute_subset_metrics(trades: list[dict]) -> dict:
    """
    Compute summary metrics for a filtered subset of closed trades.

    Parameters
    ----------
    trades : list of trade dicts from the journal (already filtered).

    Returns
    -------
    dict with n_trades, win_rate, total_pnl, max_drawdown_pct.
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "max_drawdown_pct": 0.0,
        }

    # Sort by entry_time for drawdown computation.
    def _sort_key(t: dict):
        et = t.get("entry_time")
        if isinstance(et, datetime):
            return et
        return datetime.min

    sorted_trades = sorted(trades, key=_sort_key)

    winners = sum(1 for t in sorted_trades if float(t.get("realized_pnl", 0.0)) > _BREAKEVEN_TOLERANCE)
    win_rate = winners / n

    total_pnl = sum(float(t.get("realized_pnl", 0.0)) for t in sorted_trades)

    # Compute max drawdown on cumulative PnL series.
    max_drawdown_pct = _compute_cumulative_pnl_drawdown(sorted_trades)

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown_pct": max_drawdown_pct,
    }


def _compute_cumulative_pnl_drawdown(sorted_trades: list[dict]) -> float:
    """
    Compute max drawdown on the cumulative PnL series of sorted trades.

    The series starts at 0 and accumulates realized_pnl sequentially.
    Peak-to-trough drawdown is expressed as a percentage of the peak value.

    Returns 0.0 if the series never goes positive (no meaningful drawdown).
    """
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for t in sorted_trades:
        cumulative += float(t.get("realized_pnl", 0.0))
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            dd = (peak - cumulative) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    return max_dd


def _get_confidence(t: dict) -> float | None:
    """Extract forecast_confidence from meta_features, or None."""
    mf = t.get("meta_features") or {}
    fc = mf.get("forecast_confidence")
    if fc is not None:
        try:
            return float(fc)
        except (TypeError, ValueError):
            return None
    return None


def _get_rr(t: dict) -> float | None:
    """Extract reward_risk from meta_features (fta_reward_risk key), or None."""
    mf = t.get("meta_features") or {}
    rr = mf.get("fta_reward_risk") or mf.get("reward_risk")
    if rr is not None:
        try:
            return float(rr)
        except (TypeError, ValueError):
            return None
    return None


def _get_fta_score(t: dict) -> float | None:
    """Extract fta_structure_score from meta_features, or None."""
    mf = t.get("meta_features") or {}
    score = mf.get("fta_structure_score")
    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            return None
    return None


def sweep_thresholds(
    result: "BacktestResult",
    confidence_thresholds: list[float] | None = None,
    rr_minimums: list[float] | None = None,
    fta_score_minimums: list[float] | None = None,
) -> dict:
    """
    Run threshold sensitivity analysis on existing trade_journal.

    Does NOT re-run the backtest. Filters trades by different thresholds and
    re-computes metrics on each filtered subset.

    Parameters
    ----------
    result               : BacktestResult to analyse.
    confidence_thresholds: List of forecast_confidence thresholds to test.
                           Default: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rr_minimums          : List of minimum reward:risk ratios to test.
                           Default: [1.0, 1.5, 2.0, 2.5, 3.0]
    fta_score_minimums   : List of minimum fta_structure_score thresholds.
                           Default: [0.0, 0.3, 0.5, 0.7]
                           Returns empty list if no trades have fta_score data.

    Returns
    -------
    dict with keys:
        "confidence_sweep" : list of metric dicts per threshold
        "rr_sweep"         : list of metric dicts per rr_minimum
    """
    if confidence_thresholds is None:
        confidence_thresholds = _DEFAULT_CONFIDENCE_THRESHOLDS
    if rr_minimums is None:
        rr_minimums = _DEFAULT_RR_MINIMUMS
    if fta_score_minimums is None:
        fta_score_minimums = _DEFAULT_FTA_SCORE_MINIMUMS

    journal = result.trade_journal or []
    closed = [t for t in journal if t.get("exit_price") is not None]

    # --- confidence sweep ---
    confidence_sweep = []
    for threshold in confidence_thresholds:
        # Include trades where confidence is None (no meta_features) at threshold=0.0
        # At higher thresholds, trades without confidence are excluded.
        if threshold <= 0.0:
            subset = closed
        else:
            subset = []
            for t in closed:
                fc = _get_confidence(t)
                if fc is not None and fc >= threshold:
                    subset.append(t)

        metrics = _compute_subset_metrics(subset)
        confidence_sweep.append({
            "threshold": threshold,
            **metrics,
        })

    # --- rr_sweep ---
    rr_sweep = []
    for rr_min in rr_minimums:
        if rr_min <= 0.0:
            subset = closed
        else:
            subset = []
            for t in closed:
                rr = _get_rr(t)
                if rr is not None and rr >= rr_min:
                    subset.append(t)

        metrics = _compute_subset_metrics(subset)
        rr_sweep.append({
            "rr_minimum": rr_min,
            **metrics,
        })

    return {
        "confidence_sweep": confidence_sweep,
        "rr_sweep": rr_sweep,
    }
