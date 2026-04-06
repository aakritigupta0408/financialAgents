"""
src.meta_model.scorer — Integration contract for Phase 8.

score_trade() is the single entry point called by the live loop.
It maps features + forecast + candidate to MetaModelOutput.

Model caching
-------------
The default model is loaded once from MODEL_DIR / "meta_model.pkl" and cached
as a module-level singleton.  If the file does not exist, the untrained
HeuristicMetaModel is used as a safe fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import META_MODEL_MIN_CONFIDENCE, MODEL_DIR
from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelOutput
from src.meta_model.features import build_feature_vector, feature_vector_to_numpy
from src.meta_model.model import BaseMetaModel, get_meta_model

log = logging.getLogger(__name__)

# Module-level model singleton — loaded lazily on first score_trade() call.
_default_model: BaseMetaModel | None = None

_MODEL_PATH = Path(MODEL_DIR) / "meta_model.pkl"


def score_trade(
    features: dict,
    forecast: ForecastOutput,
    candidate: dict,
    model: BaseMetaModel | None = None,
    threshold: float | None = None,
) -> MetaModelOutput:
    """
    Score a candidate trade setup and return a MetaModelOutput.

    Parameters
    ----------
    features  : dict from compute_all_features().
    forecast  : ForecastOutput from the TimesFM wrapper.
    candidate : dict from generate_candidate().
    model     : Optional pre-loaded model. If None, the default model is used
                (loaded from disk or HeuristicMetaModel fallback).
    threshold : Override for META_MODEL_MIN_CONFIDENCE.

    Returns
    -------
    MetaModelOutput with probability_of_success, confidence, and should_trade.

    Steps
    -----
    1. Build flat MetaModelInput from (features, forecast, candidate).
    2. Convert to numpy array.
    3. Resolve model (use provided or load default).
    4. Get P(success) from predict_proba.
    5. Apply threshold gate.
    6. Return MetaModelOutput.
    """
    # Step 1: Build feature vector.
    mmi = build_feature_vector(features, forecast, candidate)

    # Step 2: Convert to numpy.
    x = feature_vector_to_numpy(mmi)

    # Step 3: Resolve model.
    if model is None:
        model = _get_or_load_default_model()

    # Step 4: Predict.
    proba = float(model.predict_proba(x.reshape(1, -1))[0, 1])

    # Step 5: Threshold.
    effective_threshold = threshold if threshold is not None else META_MODEL_MIN_CONFIDENCE
    should_trade = proba >= effective_threshold

    # Step 6: Return.
    return MetaModelOutput(
        ticker=mmi.ticker,
        evaluated_at=mmi.evaluated_at,
        probability_of_success=proba,
        confidence=proba,  # extended with uncertainty quantification in a later phase
        should_trade=should_trade,
    )


def _get_or_load_default_model() -> BaseMetaModel:
    """
    Return the cached default model, loading from disk if needed.

    Load priority
    -------------
    1. Module-level singleton (already loaded).
    2. MODEL_DIR / "meta_model.pkl" if it exists.
    3. HeuristicMetaModel() as a safe untrained fallback.

    The loaded model is cached as a module-level singleton so it is only
    deserialised once per process lifetime.
    """
    global _default_model
    if _default_model is not None:
        return _default_model

    if _MODEL_PATH.exists():
        try:
            import pickle

            from src.meta_model.model import HeuristicMetaModel, SklearnMetaModel

            with open(_MODEL_PATH, "rb") as fh:
                state = pickle.load(fh)

            # Determine which class to reconstruct based on saved state keys.
            if "calibrated" in state:
                model = SklearnMetaModel.__new__(SklearnMetaModel)
                model.calibrated = state["calibrated"]
                model.scaler = state["scaler"]
                model._fitted = state.get("_fitted", True)
            else:
                model = HeuristicMetaModel(threshold=state.get("threshold", 0.50))

            _default_model = model
            log.info("Loaded meta-model from %s", _MODEL_PATH)
            return _default_model
        except Exception as exc:
            log.warning("Failed to load model from %s: %s — using heuristic fallback.", _MODEL_PATH, exc)

    log.info("No saved model found at %s; using HeuristicMetaModel fallback.", _MODEL_PATH)
    _default_model = get_meta_model(prefer_sklearn=False)
    return _default_model


def reset_default_model() -> None:
    """
    Clear the cached default model singleton.

    Useful in tests and after retraining so the next score_trade() call
    picks up the freshly saved model.
    """
    global _default_model
    _default_model = None
