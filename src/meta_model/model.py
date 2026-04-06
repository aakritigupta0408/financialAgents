"""
src.meta_model.model — Meta-model classifier implementations.

Environment constraint
----------------------
This system runs on Python 3.12.2, arm64 macOS, with a conda environment where
scipy/sklearn compiled extensions were built against NumPy 1.x.  Running
`python script.py` triggers an ABI mismatch ImportError; pytest avoids this via
its import chain.

Design
------
All sklearn imports are inside a try/except ImportError guard.  When sklearn is
unavailable, _SKLEARN_OK is False and get_meta_model() returns HeuristicMetaModel
instead of SklearnMetaModel.

HeuristicMetaModel
------------------
Deterministic weighted-sum scorer requiring no training.

  score = 0.35 * forecast_confidence
        + 0.25 * min(reward_risk / 4.0, 1.0)
        + 0.20 * trend_strength
        + 0.10 * min(relative_volume / 2.0, 1.0)
        + 0.10 * (1.0 - volatility_regime_encoded / 3.0)

Indices into the feature vector match FEATURE_NAMES canonical order.

SklearnMetaModel
----------------
GradientBoostingClassifier + isotonic calibration + StandardScaler.
Requires sklearn. Raises ImportError on __init__ if sklearn is unavailable.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.meta_model.features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# sklearn availability guard
# ---------------------------------------------------------------------------
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_OK = True
except (ImportError, AttributeError, ValueError):
    _SKLEARN_OK = False

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMetaModel(ABC):
    """Abstract interface that all meta-model implementations must satisfy."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on (X, y). May be a no-op for heuristic models."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for each sample.

        Returns
        -------
        np.ndarray of shape (n_samples, 2).
        Column 0 = P(failure), column 1 = P(success).
        """
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 if P(success) >= threshold else 0."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    @abstractmethod
    def feature_importances_(self) -> np.ndarray | None:
        """Feature importances aligned with FEATURE_NAMES, or None."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseMetaModel":
        """Load a previously saved model from disk."""
        ...


# ---------------------------------------------------------------------------
# HeuristicMetaModel — deterministic fallback, no training required
# ---------------------------------------------------------------------------

# FEATURE_NAMES indices used by the heuristic formula
_IDX_FORECAST_CONFIDENCE      = FEATURE_NAMES.index("forecast_confidence")        # 2
_IDX_FTA_REWARD_RISK          = FEATURE_NAMES.index("fta_reward_risk")             # 3
_IDX_TREND_STRENGTH           = FEATURE_NAMES.index("trend_strength")              # 11
_IDX_RELATIVE_VOLUME          = FEATURE_NAMES.index("relative_volume")             # 10
_IDX_VOLATILITY_REGIME_ENCODED = FEATURE_NAMES.index("volatility_regime_encoded")  # 9

# Fixed importance weights aligned with FEATURE_NAMES (13 values).
# Weights reflect contribution of each feature to the heuristic score.
_HEURISTIC_IMPORTANCES: np.ndarray = np.array(
    [
        0.00,  # forecast_direction_up
        0.00,  # forecast_expected_return
        0.35,  # forecast_confidence
        0.25,  # fta_reward_risk
        0.00,  # fta_distance_to_fta_pct
        0.00,  # fta_structure_score
        0.00,  # fta_liquidity_score
        0.00,  # fta_volatility_ok
        0.00,  # atr_pct
        0.10,  # volatility_regime_encoded
        0.10,  # relative_volume
        0.20,  # trend_strength
        0.00,  # trend_state_encoded
    ],
    dtype=np.float64,
)


class HeuristicMetaModel(BaseMetaModel):
    """
    Deterministic rule-based scorer.

    Formula
    -------
    score = 0.35 * forecast_confidence
          + 0.25 * min(reward_risk / 4.0, 1.0)
          + 0.20 * trend_strength
          + 0.10 * min(relative_volume / 2.0, 1.0)
          + 0.10 * (1.0 - volatility_regime_encoded / 3.0)

    score is clipped to [0, 1] and treated as P(success).
    """

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op — heuristic model requires no training."""
        return

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the heuristic formula to each row of X.

        Parameters
        ----------
        X : np.ndarray, shape (n, 13).

        Returns
        -------
        np.ndarray of shape (n, 2).
        Column 0 = 1 - score, column 1 = score.
        """
        X = np.atleast_2d(X).astype(np.float64)
        n = X.shape[0]
        scores = np.empty(n, dtype=np.float64)

        for i in range(n):
            row = X[i]
            fc = row[_IDX_FORECAST_CONFIDENCE]
            rr = row[_IDX_FTA_REWARD_RISK]
            ts = row[_IDX_TREND_STRENGTH]
            rv = row[_IDX_RELATIVE_VOLUME]
            ve = row[_IDX_VOLATILITY_REGIME_ENCODED]

            score = (
                0.35 * float(fc)
                + 0.25 * min(float(rr) / 4.0, 1.0)
                + 0.20 * float(ts)
                + 0.10 * min(float(rv) / 2.0, 1.0)
                + 0.10 * (1.0 - float(ve) / 3.0)
            )
            scores[i] = max(0.0, min(1.0, score))

        result = np.column_stack([1.0 - scores, scores])
        return result

    @property
    def feature_importances_(self) -> np.ndarray:
        return _HEURISTIC_IMPORTANCES.copy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"threshold": self.threshold}, fh)

    @classmethod
    def load(cls, path: Path) -> "HeuristicMetaModel":
        with open(Path(path), "rb") as fh:
            state = pickle.load(fh)
        return cls(threshold=state.get("threshold", 0.50))


# ---------------------------------------------------------------------------
# SklearnMetaModel — calibrated GBM classifier
# ---------------------------------------------------------------------------


class SklearnMetaModel(BaseMetaModel):
    """
    Calibrated GradientBoostingClassifier with StandardScaler preprocessing.

    Requires sklearn. Raises ImportError on __init__ if sklearn is unavailable
    (i.e., _SKLEARN_OK is False).

    Notes
    -----
    Serialisation uses pickle (not joblib) because joblib uses numpy internals
    that may trigger the same ABI issues that affect sklearn imports.
    """

    def __init__(self) -> None:
        if not _SKLEARN_OK:
            raise ImportError(
                "sklearn not available in this environment. "
                "Use HeuristicMetaModel instead."
            )
        base = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit scaler and calibrated classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n, 13).
        y : np.ndarray, shape (n,) with values 0 or 1.

        Raises
        ------
        ValueError if len(X) < 20.
        """
        if len(X) < 20:
            raise ValueError(
                f"Need at least 20 samples to fit SklearnMetaModel, got {len(X)}."
            )
        X_scaled = self.scaler.fit_transform(X)
        self.calibrated.fit(X_scaled, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X)
        return self.calibrated.predict_proba(X_scaled)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        if not self._fitted:
            return None
        try:
            # Access base estimator through CalibratedClassifierCV.
            # The base estimator is stored as calibrated_.estimator in newer sklearn,
            # or as calibrated_classifiers_[0].base_estimator in older versions.
            base = getattr(self.calibrated, "estimator", None)
            if base is None:
                # Try older sklearn API
                ccs = getattr(self.calibrated, "calibrated_classifiers_", [])
                if ccs:
                    base = getattr(ccs[0], "base_estimator", None) or getattr(ccs[0], "estimator", None)
            if base is not None and hasattr(base, "feature_importances_"):
                return np.array(base.feature_importances_, dtype=np.float64)
        except Exception:
            pass
        return None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "calibrated": self.calibrated,
            "scaler": self.scaler,
            "_fitted": self._fitted,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: Path) -> "SklearnMetaModel":
        if not _SKLEARN_OK:
            raise ImportError("sklearn not available; cannot load SklearnMetaModel.")
        with open(Path(path), "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.calibrated = state["calibrated"]
        obj.scaler = state["scaler"]
        obj._fitted = state.get("_fitted", True)
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_meta_model(prefer_sklearn: bool = True) -> BaseMetaModel:
    """
    Return the best available meta-model.

    If prefer_sklearn is True and sklearn is available (_SKLEARN_OK), returns
    a fresh SklearnMetaModel.  Otherwise returns HeuristicMetaModel.

    Parameters
    ----------
    prefer_sklearn : If True, attempt to use sklearn.

    Returns
    -------
    BaseMetaModel instance (untrained).
    """
    if prefer_sklearn and _SKLEARN_OK:
        return SklearnMetaModel()
    return HeuristicMetaModel()
