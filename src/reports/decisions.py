"""
src.reports.decisions — Decision funnel report.

generate_decision_report(result) summarises how many candidates were accepted,
how many were filtered at each stage, and characterises the forecast confidence
distribution.

Because BacktestResult does not persist per-bar rejection counters in the
trade_journal (only accepted trades appear there), rejection counts reported
here are conservative lower bounds derived solely from the journal.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult


def generate_decision_report(result: "BacktestResult") -> dict:
    """
    Build a decision-funnel summary dict.

    Parameters
    ----------
    result : BacktestResult produced by BacktestEngine.run().

    Returns
    -------
    dict with keys:

    Funnel counts
        total_candidates       — n_trades (accepted) as lower bound
        accepted_trades        — n_trades
        fta_rejected           — 0 (not stored in journal by default)
        meta_model_rejected    — 0 (not stored in journal by default)
        portfolio_rejected     — 0 (not stored in journal by default)
        acceptance_rate        — accepted / total_candidates (1.0 if no data)

    Forecast confidence stats (from meta_features where present)
        forecast_confidence_mean
        forecast_confidence_std
        meta_model_prob_mean   — same as forecast_confidence (proxy)

    Breakdown by exit reason
        rejection_breakdown    — dict[str, int]: counts of trades by exit_reason
    """
    journal = result.trade_journal or []
    closed = [t for t in journal if t.get("exit_price") is not None]

    accepted_trades = result.n_trades

    # Rejection counts are not stored in the journal; default to 0.
    fta_rejected = 0
    meta_model_rejected = 0
    portfolio_rejected = 0

    # total_candidates: at minimum all accepted trades
    total_candidates = accepted_trades
    acceptance_rate = 1.0 if total_candidates == 0 else accepted_trades / total_candidates

    # Forecast confidence from meta_features
    confidences: list[float] = []
    for t in closed:
        mf = t.get("meta_features") or {}
        fc = mf.get("forecast_confidence")
        if fc is not None:
            try:
                confidences.append(float(fc))
            except (TypeError, ValueError):
                pass

    if confidences:
        conf_mean = mean(confidences)
        conf_std = stdev(confidences) if len(confidences) >= 2 else 0.0
    else:
        conf_mean = 0.0
        conf_std = 0.0

    # rejection_breakdown: group by exit_reason
    reason_counts: dict[str, int] = defaultdict(int)
    for t in closed:
        reason = t.get("exit_reason", "unknown")
        reason_counts[reason] += 1
    rejection_breakdown = dict(reason_counts)

    return {
        "total_candidates": total_candidates,
        "accepted_trades": accepted_trades,
        "fta_rejected": fta_rejected,
        "meta_model_rejected": meta_model_rejected,
        "portfolio_rejected": portfolio_rejected,
        "acceptance_rate": acceptance_rate,
        "forecast_confidence_mean": conf_mean,
        "forecast_confidence_std": conf_std,
        "meta_model_prob_mean": conf_mean,
        "rejection_breakdown": rejection_breakdown,
    }
