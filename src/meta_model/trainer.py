"""
src.meta_model.trainer — Train and evaluate meta-model classifiers.

All sklearn imports inside try/except so this module is importable even when
sklearn is unavailable.  Metrics that require sklearn (roc_auc) are set to None
when sklearn is not available.
"""

from __future__ import annotations

import logging

import numpy as np

from src.meta_model.model import BaseMetaModel, get_meta_model

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn metrics
# ---------------------------------------------------------------------------
try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc

    _ROC_AUC_OK = True
except (ImportError, AttributeError, ValueError):
    _ROC_AUC_OK = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    prefer_sklearn: bool = True,
) -> tuple[BaseMetaModel, dict]:
    """
    Train a meta-model and evaluate it on the validation set.

    Parameters
    ----------
    X_train, y_train : Training data.
    X_val,   y_val   : Validation data.
    prefer_sklearn   : If True, prefer SklearnMetaModel when available.

    Returns
    -------
    (model, val_metrics)
    val_metrics is the dict returned by evaluate(model, X_val, y_val).
    """
    model = get_meta_model(prefer_sklearn=prefer_sklearn)

    # Fit — HeuristicMetaModel.fit() is a no-op, SklearnMetaModel trains for real.
    model.fit(X_train, y_train)

    # Evaluate on validation set.
    if len(X_val) > 0:
        metrics = evaluate(model, X_val, y_val)
    else:
        log.warning("train: validation set is empty; skipping evaluation.")
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
        }

    return model, metrics


def evaluate(
    model: BaseMetaModel,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Evaluate a model on (X, y) and return a metrics dict.

    Metrics computed without sklearn
    ---------------------------------
    accuracy, precision, recall, f1, confusion_matrix

    Metrics requiring sklearn
    -------------------------
    roc_auc (None when sklearn unavailable or only one class present)

    Parameters
    ----------
    model : Fitted BaseMetaModel.
    X     : Feature matrix.
    y     : True labels (0 or 1).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    if len(X) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
        }

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    y = np.asarray(y, dtype=int)

    # Confusion matrix components
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    total = len(y)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    roc_auc: float | None = None
    if _ROC_AUC_OK:
        try:
            unique_classes = np.unique(y)
            if len(unique_classes) >= 2:
                roc_auc = float(_sklearn_roc_auc(y, proba))
        except Exception as exc:
            log.debug("roc_auc_score failed: %s", exc)

    confusion = [[tn, fp], [fn, tp]]

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": confusion,
    }


def print_report(
    metrics: dict,
    feature_names: list[str],
    importances: np.ndarray | None,
) -> None:
    """
    Print a human-readable evaluation report to stdout.

    Parameters
    ----------
    metrics       : dict from evaluate().
    feature_names : FEATURE_NAMES list.
    importances   : Feature importances array or None.
    """
    print("=" * 55)
    print("  META-MODEL EVALUATION REPORT")
    print("=" * 55)

    def _fmt(v) -> str:
        if v is None:
            return "N/A"
        return f"{v:.4f}"

    print(f"  Accuracy   : {_fmt(metrics.get('accuracy'))}")
    print(f"  Precision  : {_fmt(metrics.get('precision'))}")
    print(f"  Recall     : {_fmt(metrics.get('recall'))}")
    print(f"  F1 score   : {_fmt(metrics.get('f1'))}")
    print(f"  ROC-AUC    : {_fmt(metrics.get('roc_auc'))}")

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print("  Confusion matrix (rows=actual, cols=pred):")
        print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
        print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    if importances is not None and len(importances) == len(feature_names):
        print("  Feature importances:")
        pairs = sorted(
            zip(feature_names, importances), key=lambda x: -x[1]
        )
        for name, imp in pairs:
            bar = "#" * int(imp * 30)
            print(f"    {name:<35s} {imp:.4f}  {bar}")

    print("=" * 55)
