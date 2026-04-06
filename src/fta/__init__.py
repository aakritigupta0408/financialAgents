"""fta — Financial Technical Analysis engine.

Public API
----------
evaluate(fta_input)         -> FTAOutput
build_fta_input(...)        -> FTAInput
"""

from __future__ import annotations

from schemas.fta import FTACandidate, FTAInput, FTAOutput
from schemas.features import (
    LevelFeatures,
    LiquidityFeatures,
    StructureFeatures,
    VolatilityFeatures,
)
from schemas.forecast import ForecastOutput

from src.fta.engine import evaluate


def build_fta_input(
    features: dict,
    forecast: ForecastOutput,
    candidate: dict,
    ticker: str,
) -> FTAInput:
    """Construct an FTAInput from the raw dicts produced by the feature and
    candidate-generation layers.

    Parameters
    ----------
    features:
        Output of ``compute_all_features()``.  Must contain keys
        "structure", "levels", "volatility", "liquidity".
    forecast:
        A ``ForecastOutput`` instance from the TimesFM wrapper.
    candidate:
        Dict from ``generate_candidate()``.  Expected keys:
        ``side``, ``entry``, ``stop``, ``target``, ``reward_risk``,
        ``forecast_confidence``.
    ticker:
        Ticker symbol string (e.g. "AAPL").

    Returns
    -------
    FTAInput
    """
    fta_candidate = FTACandidate(
        ticker=ticker,
        side=candidate["side"],
        entry_price=float(candidate["entry"]),
        stop_price=float(candidate["stop"]),
    )

    structure: StructureFeatures = features["structure"]
    levels: LevelFeatures = features["levels"]
    volatility: VolatilityFeatures = features["volatility"]
    liquidity: LiquidityFeatures = features["liquidity"]

    return FTAInput(
        candidate=fta_candidate,
        structure=structure,
        levels=levels,
        volatility=volatility,
        liquidity=liquidity,
        forecast=forecast,
    )


__all__ = ["evaluate", "build_fta_input"]
