"""
src.meta_model.pipeline — Full end-to-end meta-model training pipeline.

Entry point: run_training_pipeline()

Pipeline steps
--------------
1. Build dataset from backtest results.
2. Guard: if fewer than 20 samples, return HeuristicMetaModel with a warning.
3. Time-based train / val / test split.
4. Train model + evaluate on validation set.
5. Evaluate on held-out test set.
6. Walk-forward validation over the full dataset.
7. Optionally save model to MODEL_DIR / "meta_model.pkl".
8. Print evaluation report.
9. Return (model, metrics_dict).

Re-training policy
------------------
This pipeline is run after market close.  It must NOT be called on every tick.
See CLAUDE.md: "Do not retrain model weights on every tick."
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import MODEL_DIR
from src.meta_model.dataset import build_dataset, time_split
from src.meta_model.model import HeuristicMetaModel, get_meta_model
from src.meta_model.trainer import evaluate, print_report, train
from src.meta_model.walk_forward import walk_forward_validate

log = logging.getLogger(__name__)

_MODEL_SAVE_PATH = Path(MODEL_DIR) / "meta_model.pkl"

# Minimum samples required before we attempt sklearn training.
_MIN_TRAIN_SAMPLES = 20


def run_training_pipeline(
    backtest_results: list,  # list[BacktestResult]
    save_model: bool = True,
    prefer_sklearn: bool = True,
) -> tuple:  # tuple[BaseMetaModel, dict]
    """
    Run the full meta-model training pipeline.

    Parameters
    ----------
    backtest_results : List of BacktestResult objects from BacktestEngine.run().
    save_model       : If True, save the trained model to MODEL_DIR / "meta_model.pkl".
    prefer_sklearn   : If True, prefer SklearnMetaModel when sklearn is available.

    Returns
    -------
    (model, metrics)

    metrics dict keys:
      - "train"          : evaluate() dict on training set (or None)
      - "val"            : evaluate() dict on validation set
      - "test"           : evaluate() dict on test set (or None)
      - "walk_forward"   : list of per-fold metrics dicts
      - "warning"        : str if pipeline fell back due to insufficient data

    Guard conditions
    ----------------
    - If total samples < 20: return (HeuristicMetaModel(), {"warning": "insufficient_data"})
    """
    # Step 1: Build dataset.
    X, y, feature_names = build_dataset(backtest_results)
    n_samples = len(X)

    # Step 2: Insufficient data guard.
    if n_samples < _MIN_TRAIN_SAMPLES:
        log.warning(
            "run_training_pipeline: only %d samples; need >= %d. "
            "Falling back to HeuristicMetaModel.",
            n_samples,
            _MIN_TRAIN_SAMPLES,
        )
        fallback = HeuristicMetaModel()
        return fallback, {"warning": "insufficient_data", "n_samples": n_samples}

    # Step 3: Time-based split.
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)

    log.info(
        "Dataset split: train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # Step 4: Train + validate.
    model, val_metrics = train(X_train, y_train, X_val, y_val, prefer_sklearn=prefer_sklearn)

    # Step 5: Test set evaluation.
    test_metrics = evaluate(model, X_test, y_test) if len(X_test) > 0 else {}

    # Step 6: Walk-forward validation.
    wf_results = walk_forward_validate(X, y, n_splits=3, prefer_sklearn=prefer_sklearn)

    # Step 7: Save model.
    if save_model:
        try:
            model.save(_MODEL_SAVE_PATH)
            log.info("Meta-model saved to %s", _MODEL_SAVE_PATH)
        except Exception as exc:
            log.warning("Failed to save model: %s", exc)

    # Step 8: Print report.
    print_report(test_metrics, feature_names, model.feature_importances_)

    # Step 9: Return.
    all_metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": val_metrics,
        "test": test_metrics,
        "walk_forward": wf_results,
        "n_samples": n_samples,
    }

    return model, all_metrics
