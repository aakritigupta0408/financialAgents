"""
src.meta_model.features — Map rich feature dicts to the flat MetaModelInput schema.

FEATURE_NAMES defines the canonical 1D array order for model training and inference.
This order is fixed and must never change without retraining all saved models.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelInput

# ---------------------------------------------------------------------------
# Canonical feature order — must be stable across all calls.
# Index 0 = forecast_direction_up, index 12 = trend_state_encoded.
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "forecast_direction_up",       # 0
    "forecast_expected_return",    # 1
    "forecast_confidence",         # 2
    "fta_reward_risk",             # 3
    "fta_distance_to_fta_pct",     # 4
    "fta_structure_score",         # 5
    "fta_liquidity_score",         # 6
    "fta_volatility_ok",           # 7
    "atr_pct",                     # 8
    "volatility_regime_encoded",   # 9
    "relative_volume",             # 10
    "trend_strength",              # 11
    "trend_state_encoded",         # 12
]

_VOLATILITY_REGIME_MAP: dict[str, float] = {
    "low": 0.0,
    "normal": 1.0,
    "high": 2.0,
    "extreme": 3.0,
}

_TREND_STATE_MAP: dict[str, float] = {
    "uptrend": 1.0,
    "downtrend": -1.0,
    "ranging": 0.0,
    "unknown": 0.0,
}

_UTC = timezone.utc


def build_feature_vector(
    features: dict,
    forecast: ForecastOutput,
    candidate: dict,
    ticker: str | None = None,
    evaluated_at: datetime | None = None,
) -> MetaModelInput:
    """
    Map compute_all_features() output + ForecastOutput + candidate dict
    to the flat MetaModelInput schema.

    Parameters
    ----------
    features    : dict returned by compute_all_features(). Expected keys:
                  "structure" (StructureFeatures), "volatility" (VolatilityFeatures),
                  "liquidity" (LiquidityFeatures), "levels" (LevelFeatures, unused).
    forecast    : ForecastOutput from the TimesFM wrapper.
    candidate   : dict from generate_candidate(). Must contain "reward_risk".
    ticker      : Optional ticker override. Defaults to forecast.ticker or "UNKNOWN".
    evaluated_at: Optional timestamp override. Defaults to now (UTC).

    Returns
    -------
    MetaModelInput with all 13 numeric fields populated.

    Encoding notes
    --------------
    - forecast_direction_up   : 1.0 if forecast.direction == "up" else 0.0
    - volatility_regime_encoded: {"low": 0, "normal": 1, "high": 2, "extreme": 3}
    - trend_state_encoded     : {"uptrend": 1.0, "downtrend": -1.0, "ranging": 0.0, "unknown": 0.0}
    - fta_distance_to_fta_pct : 0.0 — TODO: wire real FTA output in Phase 8
    - fta_structure_score     : proxy = features["structure"].trend_strength
    - fta_liquidity_score     : min(relative_volume / 2.0, 1.0)
    - fta_volatility_ok       : 1.0 if regime in ("low", "normal") else 0.0
    """
    structure = features.get("structure")
    volatility = features.get("volatility")
    liquidity = features.get("liquidity")

    # --- forecast features ---
    forecast_direction_up = 1.0 if forecast.direction == "up" else 0.0
    forecast_expected_return = float(forecast.expected_return)
    forecast_confidence = float(forecast.confidence)

    # --- FTA proxies (real FTA wired in Phase 8) ---
    fta_reward_risk = float(candidate.get("reward_risk", 0.0))
    fta_distance_to_fta_pct = 0.0  # TODO: wire FTA in Phase 8; 0.0 is safe default

    # structure proxy
    fta_structure_score = float(getattr(structure, "trend_strength", 0.0)) if structure else 0.0

    # liquidity score capped at 1.0
    rel_vol_raw = float(getattr(liquidity, "relative_volume", 0.0)) if liquidity else 0.0
    fta_liquidity_score = min(rel_vol_raw / 2.0, 1.0)

    # volatility ok flag
    vol_regime = getattr(volatility, "volatility_regime", "normal") if volatility else "normal"
    fta_volatility_ok = 1.0 if vol_regime in ("low", "normal") else 0.0

    # --- volatility features ---
    atr_pct = float(getattr(volatility, "atr_pct", 0.0)) if volatility else 0.0
    volatility_regime_encoded = _VOLATILITY_REGIME_MAP.get(vol_regime, 1.0)

    # --- liquidity features ---
    relative_volume = rel_vol_raw

    # --- structure features ---
    trend_strength = float(getattr(structure, "trend_strength", 0.0)) if structure else 0.0
    trend_state_raw = getattr(structure, "trend_state", "unknown") if structure else "unknown"
    trend_state_encoded = _TREND_STATE_MAP.get(trend_state_raw, 0.0)

    resolved_ticker = ticker or getattr(forecast, "ticker", "UNKNOWN") or "UNKNOWN"
    resolved_at = evaluated_at or datetime.now(_UTC)

    return MetaModelInput(
        ticker=resolved_ticker,
        evaluated_at=resolved_at,
        forecast_direction_up=forecast_direction_up,
        forecast_expected_return=forecast_expected_return,
        forecast_confidence=forecast_confidence,
        fta_reward_risk=fta_reward_risk,
        fta_distance_to_fta_pct=fta_distance_to_fta_pct,
        fta_structure_score=fta_structure_score,
        fta_liquidity_score=fta_liquidity_score,
        fta_volatility_ok=fta_volatility_ok,
        atr_pct=atr_pct,
        volatility_regime_encoded=volatility_regime_encoded,
        relative_volume=relative_volume,
        trend_strength=trend_strength,
        trend_state_encoded=trend_state_encoded,
    )


def feature_vector_to_numpy(mmi: MetaModelInput) -> np.ndarray:
    """
    Convert a MetaModelInput to a 1D float64 array in canonical FEATURE_NAMES order.

    Returns
    -------
    np.ndarray of shape (13,), dtype float64.
    The order is defined by FEATURE_NAMES and is stable across calls.
    """
    return np.array(
        [getattr(mmi, name) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
