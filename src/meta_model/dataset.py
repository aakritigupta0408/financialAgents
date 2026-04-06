"""
src.meta_model.dataset — Build training datasets from BacktestResult objects.

Target labeling rule (transparent)
-----------------------------------
y = 1 if trade["realized_pnl"] > 0 else 0

This is a binary classification: did the trade make money?  Breakeven (pnl ==
0.0) is labelled as 0 (failure) to be conservative.

Data leakage prevention
-----------------------
- build_dataset() only reads fields that were recorded at trade-open time
  (stored under "meta_features" in the journal entry).
- It never reads realized_pnl-derived features into X.
- time_split() is always sequential (time-based), never random.

Missing meta_features
----------------------
If "meta_features" is absent from a journal entry (produced by an older
backtest without Phase 7 augmentation), that row is silently skipped with a
warning at the end.
"""

from __future__ import annotations

import logging

import numpy as np

from src.meta_model.features import FEATURE_NAMES

log = logging.getLogger(__name__)

# Minimum sample count below which training is considered unreliable.
_MIN_RELIABLE_SAMPLES = 30


def build_dataset(
    results: list,  # list[BacktestResult] — avoid circular import
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a training dataset from one or more BacktestResult objects.

    Each trade journal entry that contains a "meta_features" dict is converted
    into one row of X using the canonical FEATURE_NAMES order.  The target y
    is 1 if realized_pnl > 0, else 0.

    Parameters
    ----------
    results : list of BacktestResult instances.

    Returns
    -------
    X            : np.ndarray of shape (n_samples, 13), dtype float64.
    y            : np.ndarray of shape (n_samples,), dtype int32 (0 or 1).
    feature_names: FEATURE_NAMES list (stable reference).

    Raises
    ------
    None — skips rows silently if meta_features is missing.
    Logs a warning if fewer than _MIN_RELIABLE_SAMPLES rows are valid.
    """
    X_rows: list[list[float]] = []
    y_vals: list[int] = []
    skipped = 0

    for result in results:
        for entry in result.trade_journal:
            meta = entry.get("meta_features")
            if not meta:
                skipped += 1
                continue

            # Build feature row in canonical order. Missing keys default to 0.0.
            row = [float(meta.get(name, 0.0)) for name in FEATURE_NAMES]

            pnl = entry.get("realized_pnl", 0.0) or 0.0
            label = 1 if float(pnl) > 0.0 else 0

            X_rows.append(row)
            y_vals.append(label)

    if skipped > 0:
        log.warning(
            "build_dataset: skipped %d journal entries without meta_features", skipped
        )

    n = len(X_rows)
    if n < _MIN_RELIABLE_SAMPLES:
        log.warning(
            "build_dataset: only %d samples available (< %d). "
            "Training results may be unreliable.",
            n,
            _MIN_RELIABLE_SAMPLES,
        )

    if n == 0:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
            FEATURE_NAMES,
        )

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.int32)
    return X, y, FEATURE_NAMES


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Sequential (time-based) train / validation / test split.

    The data is assumed to be in time order (oldest first).
    Splits are made by index — no shuffling is ever applied.

    Parameters
    ----------
    X          : Feature matrix, shape (n, 13).
    y          : Label vector, shape (n,).
    train_frac : Fraction of data used for training (default 0.70).
    val_frac   : Fraction used for validation (default 0.15).
                 Remaining (1 - train_frac - val_frac) goes to test.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test

    All splits are contiguous and non-overlapping.
    """
    n = len(X)
    if n == 0:
        empty = np.empty((0, X.shape[1] if X.ndim > 1 else 0), dtype=X.dtype)
        empty_y = np.empty((0,), dtype=y.dtype)
        return empty, empty_y, empty, empty_y, empty, empty_y

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test
