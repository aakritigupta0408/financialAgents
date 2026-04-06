"""
src.reports.model_diagnostics — Meta-model diagnostic report.

generate_model_diagnostics(result) reads meta_features from the trade journal
and computes:

- feature_importance : per-feature Pearson correlation with win/loss outcome
- prediction_distribution : summary stats for forecast_confidence
- calibration_summary : fraction of trades with meta_features
- threshold_sensitivity : output of sweep_thresholds()

All computations are done on the existing trade journal without re-running the
backtest.
"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult

# Numeric feature names stored in meta_features dict by the engine.
# These correspond to MetaModelInput fields minus non-numeric fields.
_NUMERIC_META_FEATURES = [
    "forecast_direction_up",
    "forecast_expected_return",
    "forecast_confidence",
    "fta_reward_risk",
    "fta_distance_to_fta_pct",
    "fta_structure_score",
    "fta_liquidity_score",
    "fta_volatility_ok",
    "atr_pct",
    "volatility_regime_encoded",
    "relative_volume",
    "trend_strength",
    "trend_state_encoded",
]

_BREAKEVEN_TOLERANCE = 1e-8


def _pearson(xs: list[float], ys: list[float]) -> float:
    """
    Compute Pearson correlation coefficient between two equal-length lists.

    Returns 0.0 if the series is constant or has fewer than 2 points.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)

    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return cov / (std_x * std_y)


def generate_model_diagnostics(result: "BacktestResult") -> dict:
    """
    Build a meta-model diagnostic dict.

    Parameters
    ----------
    result : BacktestResult produced by BacktestEngine.run().

    Returns
    -------
    dict with keys:

    feature_importance
        dict[str, float] — Pearson correlation of each numeric meta_feature
        with win/loss outcome (1.0 = win, 0.0 = loss/breakeven).
        Empty dict if no trades have meta_features.

    prediction_distribution
        dict with "mean", "std", "min", "max", "pct_above_0.6" for
        forecast_confidence across all trades that have it.

    calibration_summary
        dict with "total_trades" and "labeled_fraction" (fraction of trades
        that have non-empty meta_features).

    threshold_sensitivity
        Result of sweep_thresholds(result) for convenience.
    """
    from src.reports.threshold_tuning import sweep_thresholds

    journal = result.trade_journal or []
    closed = [t for t in journal if t.get("exit_price") is not None]

    total_trades = len(closed)

    # Partition trades by whether meta_features is populated.
    labeled = [t for t in closed if t.get("meta_features")]
    labeled_fraction = len(labeled) / total_trades if total_trades > 0 else 0.0

    # Outcome vector: 1.0 = win, 0.0 = loss/breakeven.
    def _outcome(t: dict) -> float:
        pnl = float(t.get("realized_pnl", 0.0))
        return 1.0 if pnl > _BREAKEVEN_TOLERANCE else 0.0

    # feature_importance: Pearson(feature, outcome) for each numeric meta feature.
    feature_importance: dict[str, float] = {}
    if labeled:
        outcomes = [_outcome(t) for t in labeled]
        for feat in _NUMERIC_META_FEATURES:
            values: list[float] = []
            for t in labeled:
                mf = t.get("meta_features") or {}
                v = mf.get(feat)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        values.append(0.0)
                else:
                    values.append(0.0)
            if len(values) == len(outcomes):
                feature_importance[feat] = _pearson(values, outcomes)

    # prediction_distribution: stats over forecast_confidence.
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
        conf_mean = sum(confidences) / len(confidences)
        conf_std = (
            math.sqrt(sum((c - conf_mean) ** 2 for c in confidences) / len(confidences))
            if len(confidences) >= 2 else 0.0
        )
        conf_min = min(confidences)
        conf_max = max(confidences)
        pct_above_06 = sum(1 for c in confidences if c > 0.6) / len(confidences)
    else:
        conf_mean = 0.0
        conf_std = 0.0
        conf_min = 0.0
        conf_max = 0.0
        pct_above_06 = 0.0

    prediction_distribution = {
        "mean": conf_mean,
        "std": conf_std,
        "min": conf_min,
        "max": conf_max,
        "pct_above_0.6": pct_above_06,
    }

    calibration_summary = {
        "total_trades": total_trades,
        "labeled_fraction": labeled_fraction,
    }

    threshold_sensitivity = sweep_thresholds(result)

    return {
        "feature_importance": feature_importance,
        "prediction_distribution": prediction_distribution,
        "calibration_summary": calibration_summary,
        "threshold_sensitivity": threshold_sensitivity,
    }
