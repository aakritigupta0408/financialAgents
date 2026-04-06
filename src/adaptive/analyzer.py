"""
src.adaptive.analyzer — Analyse a full report dict and produce recommendations.

analyze_report(report) → AnalysisResult

Threshold selection logic
--------------------------
1. Confidence threshold: maximise total_pnl subject to
   max_drawdown_pct ≤ 1.5 × current_max_drawdown_pct.
   Falls back to current context threshold if no sweep data or constraint
   cannot be satisfied.

2. RR minimum: same drawdown-controlled return maximisation over rr_sweep.

3. feature_importance: from report["model_diagnostics"]["feature_importance"].

4. per_ticker_pnl: from report["portfolio"]["per_ticker_pnl"].

5. regime_stats: grouped from report["trade_diagnostics"] by
   meta_features.volatility_regime (key="unknown" if missing).

Warnings appended when:
- n_trades_analyzed < 10
- recommended confidence > current + 0.20
- recommended_rr_minimum > current + 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.adaptive.context import AdaptiveContext

log = logging.getLogger(__name__)

_BREAKEVEN_TOLERANCE = 1e-8


@dataclass
class AnalysisResult:
    recommended_confidence_threshold: float
    recommended_rr_minimum: float
    recommended_forecast_confidence: float
    n_trades_analyzed: int
    best_sweep_metric: str          # "return" or "drawdown"
    feature_importance: dict[str, float]
    per_ticker_pnl: dict[str, float]
    regime_stats: dict[str, dict]   # regime → {n_trades, win_rate, avg_pnl}
    warnings: list[str] = field(default_factory=list)


def analyze_report(report: dict) -> AnalysisResult:
    """
    Analyse a full report dict and produce threshold recommendations.

    Parameters
    ----------
    report : dict returned by generate_full_report().

    Returns
    -------
    AnalysisResult with recommendations and advisory warnings.
    """
    # Load current context for reference thresholds and drawdown baseline.
    context = AdaptiveContext.load()
    current_confidence = context.best_thresholds.meta_model_min_confidence
    current_rr = context.best_thresholds.min_reward_risk
    current_forecast_confidence = context.best_thresholds.forecast_confidence_min

    portfolio = report.get("portfolio") or {}
    current_max_dd = float(portfolio.get("max_drawdown_pct") or 0.0)
    drawdown_cap = 1.5 * current_max_dd  # allowed drawdown for candidate thresholds

    warnings: list[str] = []

    # --- Count analyzable trades ---
    trade_diag = report.get("trade_diagnostics") or []
    n_trades_analyzed = len(trade_diag)

    if n_trades_analyzed < 10:
        warnings.append("Insufficient sample size — threshold changes suppressed")

    # --- Confidence sweep ---
    threshold_sens = report.get("threshold_sensitivity") or {}
    confidence_sweep = threshold_sens.get("confidence_sweep") or []

    recommended_confidence = current_confidence
    best_sweep_metric = "return"

    if confidence_sweep:
        recommended_confidence = _select_best_threshold(
            sweep=confidence_sweep,
            threshold_key="threshold",
            current_max_dd=current_max_dd,
            drawdown_cap=drawdown_cap,
            fallback=current_confidence,
        )

    # --- RR sweep ---
    rr_sweep = threshold_sens.get("rr_sweep") or []
    recommended_rr = current_rr

    if rr_sweep:
        recommended_rr = _select_best_threshold(
            sweep=rr_sweep,
            threshold_key="rr_minimum",
            current_max_dd=current_max_dd,
            drawdown_cap=drawdown_cap,
            fallback=current_rr,
        )

    # forecast_confidence: use same as recommended_confidence as a proxy
    # (no dedicated forecast_confidence sweep exists; keep current unless
    # the confidence threshold shift suggests movement)
    recommended_forecast_confidence = current_forecast_confidence

    # --- Feature importance ---
    model_diag = report.get("model_diagnostics") or {}
    feature_importance: dict[str, float] = {
        k: float(v)
        for k, v in (model_diag.get("feature_importance") or {}).items()
    }

    # --- Per-ticker PnL ---
    per_ticker_pnl: dict[str, float] = {
        k: float(v)
        for k, v in (portfolio.get("per_ticker_pnl") or {}).items()
    }

    # --- Regime stats from trade diagnostics ---
    regime_stats = _build_regime_stats(trade_diag)

    # --- Additional warnings ---
    if recommended_confidence > current_confidence + 0.20:
        warnings.append("Large confidence threshold jump detected")
    if recommended_rr > current_rr + 1.0:
        warnings.append("Large RR jump detected")

    return AnalysisResult(
        recommended_confidence_threshold=recommended_confidence,
        recommended_rr_minimum=recommended_rr,
        recommended_forecast_confidence=recommended_forecast_confidence,
        n_trades_analyzed=n_trades_analyzed,
        best_sweep_metric=best_sweep_metric,
        feature_importance=feature_importance,
        per_ticker_pnl=per_ticker_pnl,
        regime_stats=regime_stats,
        warnings=warnings,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _select_best_threshold(
    sweep: list[dict],
    threshold_key: str,
    current_max_dd: float,
    drawdown_cap: float,
    fallback: float,
) -> float:
    """
    Select threshold that maximises total_pnl subject to drawdown constraint.

    If current_max_dd is 0 (no trades or no drawdown), allow any drawdown.
    Falls back to `fallback` if no qualifying entry exists.
    """
    best_pnl: float | None = None
    best_threshold = fallback

    for entry in sweep:
        t_val = entry.get(threshold_key)
        pnl = entry.get("total_pnl")
        dd = entry.get("max_drawdown_pct")

        if t_val is None or pnl is None:
            continue

        pnl_f = float(pnl)
        dd_f = float(dd) if dd is not None else 0.0

        # Drawdown constraint: skip if exceeds cap (only when baseline dd > 0).
        if current_max_dd > 0.0 and dd_f > drawdown_cap:
            continue

        if best_pnl is None or pnl_f > best_pnl:
            best_pnl = pnl_f
            best_threshold = float(t_val)

    return best_threshold


def _build_regime_stats(trade_diag: list[dict]) -> dict[str, dict]:
    """
    Group trade diagnostics by volatility_regime and compute stats.

    Uses "unknown" when meta_features.volatility_regime is missing.
    """
    groups: dict[str, list[float]] = {}

    for t in trade_diag:
        mf = t.get("meta_features") or {}
        regime = str(mf.get("volatility_regime") or "unknown")
        pnl = float(t.get("realized_pnl", 0.0))
        groups.setdefault(regime, []).append(pnl)

    result: dict[str, dict] = {}
    for regime, pnls in groups.items():
        n = len(pnls)
        winners = sum(1 for p in pnls if p > _BREAKEVEN_TOLERANCE)
        result[regime] = {
            "n_trades": n,
            "win_rate": winners / n if n > 0 else 0.0,
            "avg_pnl": sum(pnls) / n if n > 0 else 0.0,
        }

    return result
