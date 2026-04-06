"""
src.meta_model.walk_forward — Walk-forward (expanding window) cross-validation.

This is the correct validation methodology for time-series data.  Each fold
trains on all data up to that point and evaluates on the next window.  No
random shuffling is ever applied.

Walk-forward design
-------------------
Given n_splits = 3 and N samples, the folds are:

  Fold 1 : train on [0, split_1), evaluate on [split_1, split_2)
  Fold 2 : train on [0, split_2), evaluate on [split_2, split_3)
  Fold 3 : train on [0, split_3), evaluate on [split_3, N)

where split_i = int(N * i / (n_splits + 1)).

Any fold where the training set has fewer than min_train_size samples is
skipped (with a debug log message).
"""

from __future__ import annotations

import logging

import numpy as np

from src.meta_model.model import get_meta_model
from src.meta_model.trainer import evaluate

log = logging.getLogger(__name__)


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    min_train_size: int = 20,
    prefer_sklearn: bool = True,
) -> list[dict]:
    """
    Perform expanding-window walk-forward validation.

    Parameters
    ----------
    X              : Feature matrix, shape (n, 13). Must be in time order.
    y              : Label vector, shape (n,). Must be in time order.
    n_splits       : Number of test folds.
    min_train_size : Minimum training samples required for a fold to run.
    prefer_sklearn : Passed to get_meta_model().

    Returns
    -------
    list of dicts, one per valid fold.  Each dict contains all keys returned
    by evaluate() plus:
      - "fold"          : int, 1-indexed fold number
      - "train_size"    : int, number of training samples
      - "test_size"     : int, number of test samples
      - "train_range"   : (int, int) tuple of start/end indices (exclusive end)
      - "test_range"    : (int, int) tuple of start/end indices (exclusive end)

    Notes
    -----
    Returns fewer than n_splits entries if some folds are skipped due to
    insufficient training data.
    """
    N = len(X)
    if N == 0:
        return []

    # Compute split points that divide data into (n_splits + 1) segments.
    # Segment i = [split[i], split[i+1]).
    # Training window = [0, split[fold]), test window = [split[fold], split[fold+1]).
    splits = [int(N * i / (n_splits + 1)) for i in range(n_splits + 2)]

    results: list[dict] = []

    for fold_idx in range(n_splits):
        train_end = splits[fold_idx + 1]
        test_start = splits[fold_idx + 1]
        test_end = splits[fold_idx + 2]

        train_size = train_end  # training is [0, train_end)
        test_size = test_end - test_start

        if train_size < min_train_size:
            log.debug(
                "walk_forward fold %d: training set too small (%d < %d); skipping.",
                fold_idx + 1,
                train_size,
                min_train_size,
            )
            continue

        if test_size <= 0:
            log.debug("walk_forward fold %d: empty test window; skipping.", fold_idx + 1)
            continue

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test  = X[test_start:test_end]
        y_test  = y[test_start:test_end]

        # Fresh model for each fold.
        model = get_meta_model(prefer_sklearn=prefer_sklearn)
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            log.warning("walk_forward fold %d fit failed: %s; skipping.", fold_idx + 1, exc)
            continue

        fold_metrics = evaluate(model, X_test, y_test)
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["train_size"] = train_size
        fold_metrics["test_size"] = test_size
        fold_metrics["train_range"] = (0, train_end)
        fold_metrics["test_range"] = (test_start, test_end)

        results.append(fold_metrics)

    return results
