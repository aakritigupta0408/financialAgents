"""
src.meta_model — Meta-model training and inference for the paper-trading system.

Phase 7 deliverable.

Public API
----------
score_trade            : Score a candidate trade and return MetaModelOutput.
run_training_pipeline  : Build dataset, train model, walk-forward validate.
build_feature_vector   : Map feature dicts + forecast + candidate to MetaModelInput.
get_meta_model         : Factory — returns SklearnMetaModel or HeuristicMetaModel.
HeuristicMetaModel     : Deterministic weighted-sum scorer (no training required).

Quick start
-----------
    from src.meta_model import score_trade, run_training_pipeline

    # After backtest:
    model, metrics = run_training_pipeline([backtest_result])

    # At trade evaluation time:
    output = score_trade(features, forecast, candidate, model=model)
    if output.should_trade:
        ...
"""

from src.meta_model.features import build_feature_vector
from src.meta_model.model import HeuristicMetaModel, get_meta_model
from src.meta_model.pipeline import run_training_pipeline
from src.meta_model.scorer import score_trade

__all__ = [
    "score_trade",
    "run_training_pipeline",
    "build_feature_vector",
    "get_meta_model",
    "HeuristicMetaModel",
]
