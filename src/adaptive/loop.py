"""
src.adaptive.loop — End-to-end improvement cycle for Phase 10.

run_improvement_cycle(result, retrain_model, save_context) → ImprovementCycleResult

Steps:
1. Load current AdaptiveContext.
2. Generate full report via generate_full_report(result).
3. Analyse via analyze_report(report).
4. Apply update via apply_update(context, analysis, report).
5. Optionally retrain meta-model and compare val_roc_auc.
6. Optionally save context.
7. Return ImprovementCycleResult.

No circular imports: this module does NOT import from src.loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult

log = logging.getLogger(__name__)


@dataclass
class ImprovementCycleResult:
    context_before: "AdaptiveContext"   # type: ignore[name-defined]
    context_after: "AdaptiveContext"    # type: ignore[name-defined]
    analysis: "AnalysisResult"          # type: ignore[name-defined]
    update_summary: "UpdateSummary"     # type: ignore[name-defined]
    model_retrained: bool
    model_improved: bool
    report: dict


def run_improvement_cycle(
    result: "BacktestResult",
    retrain_model: bool = False,
    save_context: bool = True,
) -> ImprovementCycleResult:
    """
    Run one full adaptive improvement cycle.

    Parameters
    ----------
    result         : BacktestResult from BacktestEngine.run().
    retrain_model  : If True and enough trades exist, attempt meta-model retraining.
    save_context   : If True, persist the updated AdaptiveContext to disk.

    Returns
    -------
    ImprovementCycleResult
    """
    from src.adaptive.context import AdaptiveContext
    from src.adaptive.analyzer import analyze_report
    from src.adaptive.updater import apply_update
    from src.reports import generate_full_report

    # Step 1: Load current context.
    context_before = AdaptiveContext.load()

    # Step 2: Generate full report.
    try:
        report = generate_full_report(result)
    except Exception as exc:
        log.warning("run_improvement_cycle: generate_full_report failed: %s", exc)
        report = {
            "portfolio": {},
            "decisions": {},
            "model_diagnostics": {},
            "threshold_sensitivity": {},
            "trade_diagnostics": [],
            "charts": [],
        }

    # Step 3: Analyse.
    try:
        analysis = analyze_report(report)
    except Exception as exc:
        log.warning("run_improvement_cycle: analyze_report failed: %s", exc)
        from src.adaptive.analyzer import AnalysisResult
        analysis = AnalysisResult(
            recommended_confidence_threshold=context_before.best_thresholds.meta_model_min_confidence,
            recommended_rr_minimum=context_before.best_thresholds.min_reward_risk,
            recommended_forecast_confidence=context_before.best_thresholds.forecast_confidence_min,
            n_trades_analyzed=0,
            best_sweep_metric="return",
            feature_importance={},
            per_ticker_pnl={},
            regime_stats={},
            warnings=["analyze_report raised an exception"],
        )

    # Step 4: Apply update.
    context_after, update_summary = apply_update(context_before, analysis, report)

    # Step 5: Optionally retrain meta-model.
    model_retrained = False
    model_improved = False

    if retrain_model and analysis.n_trades_analyzed >= 10:
        try:
            from src.meta_model.pipeline import run_training_pipeline

            # Probe run (save_model=False) to get metrics.
            _, probe_metrics = run_training_pipeline([result], save_model=False)
            probe_auc = _extract_val_auc(probe_metrics)

            # Compare against a baseline (0.5 = random).
            baseline_auc = 0.5
            if probe_auc > baseline_auc:
                # Save model.
                run_training_pipeline([result], save_model=True)
                model_retrained = True
                model_improved = True
                # Reload singleton.
                from src.meta_model.scorer import reset_default_model
                reset_default_model()
                log.info(
                    "run_improvement_cycle: model saved. probe_auc=%.4f > baseline=%.4f",
                    probe_auc,
                    baseline_auc,
                )
            else:
                model_retrained = True
                model_improved = False
                log.info(
                    "run_improvement_cycle: model NOT saved. probe_auc=%.4f <= baseline=%.4f",
                    probe_auc,
                    baseline_auc,
                )
        except Exception as exc:
            log.warning("run_improvement_cycle: model retraining failed: %s", exc)
            model_retrained = False
            model_improved = False

    # Step 6: Save context.
    if save_context:
        context_after.save()

    return ImprovementCycleResult(
        context_before=context_before,
        context_after=context_after,
        analysis=analysis,
        update_summary=update_summary,
        model_retrained=model_retrained,
        model_improved=model_improved,
        report=report,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_val_auc(metrics: dict) -> float:
    """Extract val_roc_auc from pipeline metrics, defaulting to 0.5."""
    val = metrics.get("val") or {}
    if isinstance(val, dict):
        auc = val.get("roc_auc")
        if auc is not None:
            try:
                return float(auc)
            except (TypeError, ValueError):
                pass
    # Try top-level key
    auc = metrics.get("val_roc_auc")
    if auc is not None:
        try:
            return float(auc)
        except (TypeError, ValueError):
            pass
    return 0.5
