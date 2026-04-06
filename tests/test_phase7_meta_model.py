"""
tests/test_phase7_meta_model.py — Phase 7: Meta-Model Training Pipeline tests.

All 20 tests use synthetic data only.  No API calls.

Tests
-----
1.  test_build_feature_vector
2.  test_feature_vector_to_numpy_shape
3.  test_feature_vector_canonical_order
4.  test_heuristic_model_predict_proba_shape
5.  test_heuristic_model_proba_sums_to_one
6.  test_heuristic_model_no_fit_required
7.  test_heuristic_high_score
8.  test_heuristic_low_score
9.  test_dataset_build_from_journal
10. test_time_split_ordering
11. test_time_split_fractions
12. test_sklearn_model_or_heuristic_trains
13. test_evaluate_metrics_keys
14. test_score_trade_returns_valid_output
15. test_score_trade_should_trade_gate
16. test_walk_forward_n_folds
17. test_pipeline_insufficient_data_warning
18. test_model_save_load
19. test_backtest_engine_stores_meta_features
20. test_meta_features_keys
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from schemas.features import (
    LiquidityFeatures,
    LevelFeatures,
    StructureFeatures,
    VolatilityFeatures,
)
from schemas.forecast import ForecastOutput
from schemas.meta_model import MetaModelInput, MetaModelOutput
from src.meta_model.features import (
    FEATURE_NAMES,
    build_feature_vector,
    feature_vector_to_numpy,
)
from src.meta_model.model import HeuristicMetaModel, get_meta_model
from src.meta_model.dataset import build_dataset, time_split
from src.meta_model.trainer import evaluate
from src.meta_model.walk_forward import walk_forward_validate
from src.meta_model.pipeline import run_training_pipeline
from src.meta_model.scorer import score_trade

_UTC = timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_features(
    trend_state: str = "uptrend",
    trend_strength: float = 0.7,
    volatility_regime: str = "normal",
    atr_pct: float = 0.01,
    relative_volume: float = 1.5,
) -> dict:
    return {
        "structure": StructureFeatures(
            ticker="SYN",
            timeframe="1h",
            trend_state=trend_state,  # type: ignore[arg-type]
            trend_strength=trend_strength,
        ),
        "levels": LevelFeatures(ticker="SYN", timeframe="1h"),
        "volatility": VolatilityFeatures(
            ticker="SYN",
            timeframe="1h",
            atr=1.0,
            atr_pct=atr_pct,
            volatility_regime=volatility_regime,  # type: ignore[arg-type]
        ),
        "liquidity": LiquidityFeatures(
            ticker="SYN",
            timeframe="1h",
            avg_volume=100_000.0,
            relative_volume=relative_volume,
            spread_estimate=0.0,
        ),
    }


def _make_forecast(
    direction: str = "up",
    expected_return: float = 0.012,
    confidence: float = 0.75,
) -> ForecastOutput:
    return ForecastOutput(
        ticker="SYN",
        timeframe="1h",
        direction=direction,  # type: ignore[arg-type]
        expected_return=expected_return,
        confidence=confidence,
        horizon=10,
    )


def _make_candidate(reward_risk: float = 2.5) -> dict:
    return {
        "side": "long",
        "entry": 100.0,
        "stop": 98.5,
        "target": 103.75,
        "reward_risk": reward_risk,
        "forecast_confidence": 0.75,
    }


def _make_synthetic_X_y(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature matrix and binary labels."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(FEATURE_NAMES))).astype(np.float64)
    y = (rng.random(n) > 0.5).astype(np.int32)
    return X, y


def _make_journal_with_features(n: int = 50, seed: int = 0) -> list[dict]:
    """Generate n fake journal entries with meta_features dicts."""
    rng = np.random.default_rng(seed)
    journal = []
    for i in range(n):
        meta = {name: float(rng.random()) for name in FEATURE_NAMES}
        pnl = float(rng.choice([-50.0, 100.0]))
        journal.append({
            "trade_id": f"T{i:04d}",
            "realized_pnl": pnl,
            "meta_features": meta,
        })
    return journal


# ─────────────────────────────────────────────────────────────────────────────
# Tests 1–3: feature vector construction
# ─────────────────────────────────────────────────────────────────────────────

def test_build_feature_vector():
    """build_feature_vector returns a valid MetaModelInput."""
    features = _make_features()
    forecast = _make_forecast()
    candidate = _make_candidate()

    mmi = build_feature_vector(features, forecast, candidate)

    assert isinstance(mmi, MetaModelInput)
    assert mmi.forecast_direction_up == 1.0
    assert mmi.forecast_confidence == pytest.approx(0.75)
    assert mmi.fta_reward_risk == pytest.approx(2.5)
    assert mmi.volatility_regime_encoded == pytest.approx(1.0)  # normal
    assert mmi.trend_state_encoded == pytest.approx(1.0)  # uptrend
    assert 0.0 <= mmi.fta_liquidity_score <= 1.0


def test_feature_vector_to_numpy_shape():
    """feature_vector_to_numpy returns shape (13,)."""
    features = _make_features()
    forecast = _make_forecast()
    candidate = _make_candidate()
    mmi = build_feature_vector(features, forecast, candidate)
    vec = feature_vector_to_numpy(mmi)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == 13


def test_feature_vector_canonical_order():
    """First element of the numpy array is forecast_direction_up."""
    features = _make_features()
    forecast = _make_forecast(direction="up")
    candidate = _make_candidate()
    mmi = build_feature_vector(features, forecast, candidate)
    vec = feature_vector_to_numpy(mmi)

    assert FEATURE_NAMES[0] == "forecast_direction_up"
    assert vec[0] == pytest.approx(1.0)

    # Also verify "down" maps to 0.0
    forecast_down = _make_forecast(direction="down")
    mmi_down = build_feature_vector(features, forecast_down, candidate)
    vec_down = feature_vector_to_numpy(mmi_down)
    assert vec_down[0] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 4–8: HeuristicMetaModel
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_model_predict_proba_shape():
    """HeuristicMetaModel.predict_proba returns shape (n, 2)."""
    model = HeuristicMetaModel()
    X, _ = _make_synthetic_X_y(n=5)
    proba = model.predict_proba(X)

    assert proba.shape == (5, 2)


def test_heuristic_model_proba_sums_to_one():
    """Each row of predict_proba sums to 1.0."""
    model = HeuristicMetaModel()
    X, _ = _make_synthetic_X_y(n=10)
    proba = model.predict_proba(X)

    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_heuristic_model_no_fit_required():
    """HeuristicMetaModel.fit() is a no-op; predict still works after."""
    model = HeuristicMetaModel()
    X, y = _make_synthetic_X_y(n=10)

    model.fit(X, y)  # should not raise
    proba = model.predict_proba(X)

    assert proba.shape == (10, 2)


def test_heuristic_high_score():
    """All-good features produce probability > 0.6."""
    model = HeuristicMetaModel()
    features = _make_features(
        trend_state="uptrend",
        trend_strength=0.9,
        volatility_regime="normal",
        relative_volume=2.0,
    )
    forecast = _make_forecast(direction="up", confidence=0.90)
    candidate = _make_candidate(reward_risk=4.0)

    mmi = build_feature_vector(features, forecast, candidate)
    x = feature_vector_to_numpy(mmi).reshape(1, -1)
    proba = model.predict_proba(x)

    assert proba[0, 1] > 0.6, f"Expected prob > 0.6, got {proba[0, 1]:.4f}"


def test_heuristic_low_score():
    """All-bad features produce probability < 0.5."""
    model = HeuristicMetaModel()
    features = _make_features(
        trend_state="downtrend",
        trend_strength=0.05,
        volatility_regime="extreme",
        relative_volume=0.1,
    )
    forecast = _make_forecast(direction="up", confidence=0.10)
    candidate = _make_candidate(reward_risk=0.5)

    mmi = build_feature_vector(features, forecast, candidate)
    x = feature_vector_to_numpy(mmi).reshape(1, -1)
    proba = model.predict_proba(x)

    assert proba[0, 1] < 0.5, f"Expected prob < 0.5, got {proba[0, 1]:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests 9–11: Dataset building and splitting
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_build_from_journal():
    """build_dataset produces X with shape (n, 13) from journal entries."""
    from src.backtest.result import BacktestResult

    journal = _make_journal_with_features(n=50)

    # Minimal BacktestResult stub.
    class _FakeResult:
        trade_journal = journal

    X, y, feat_names = build_dataset([_FakeResult()])

    assert X.shape == (50, 13)
    assert y.shape == (50,)
    assert feat_names == FEATURE_NAMES
    assert set(y).issubset({0, 1})


def test_time_split_ordering():
    """train/val/test splits are sequential and non-overlapping."""
    X, y = _make_synthetic_X_y(n=100)
    X_tr, y_tr, X_v, y_v, X_te, y_te = time_split(X, y)

    # Total rows preserved.
    assert len(X_tr) + len(X_v) + len(X_te) == 100

    # No overlap — confirmed by index arithmetic: train is first, then val, then test.
    n_tr = len(X_tr)
    n_v  = len(X_v)
    n_te = len(X_te)

    # Check that train comes first by verifying row identity.
    np.testing.assert_array_equal(X_tr, X[:n_tr])
    np.testing.assert_array_equal(X_v,  X[n_tr : n_tr + n_v])
    np.testing.assert_array_equal(X_te, X[n_tr + n_v :])


def test_time_split_fractions():
    """train/val/test sizes approximately match requested fractions."""
    n = 200
    X, y = _make_synthetic_X_y(n=n)
    X_tr, y_tr, X_v, y_v, X_te, y_te = time_split(X, y, train_frac=0.70, val_frac=0.15)

    assert abs(len(X_tr) / n - 0.70) < 0.02
    assert abs(len(X_v) / n - 0.15) < 0.02
    # Test is the remainder ≈ 0.15
    assert abs(len(X_te) / n - 0.15) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Tests 12–13: Model training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def test_sklearn_model_or_heuristic_trains():
    """get_meta_model().fit(X, y) with 30 samples raises no exception."""
    X, y = _make_synthetic_X_y(n=30)
    model = get_meta_model()  # uses sklearn if available, heuristic otherwise

    model.fit(X, y)  # must not raise

    proba = model.predict_proba(X)
    assert proba.shape == (30, 2)


def test_evaluate_metrics_keys():
    """evaluate() returns dict with required keys."""
    X, y = _make_synthetic_X_y(n=30)
    model = HeuristicMetaModel()
    model.fit(X, y)

    metrics = evaluate(model, X, y)

    required = {"accuracy", "precision", "recall", "f1"}
    assert required.issubset(metrics.keys())
    assert "confusion_matrix" in metrics

    # All numeric metrics are floats or None.
    for key in ("accuracy", "precision", "recall", "f1"):
        assert isinstance(metrics[key], float), f"{key} should be float"


# ─────────────────────────────────────────────────────────────────────────────
# Tests 14–15: score_trade integration
# ─────────────────────────────────────────────────────────────────────────────

def test_score_trade_returns_valid_output():
    """score_trade() returns MetaModelOutput with valid fields."""
    model = HeuristicMetaModel()
    features = _make_features()
    forecast = _make_forecast()
    candidate = _make_candidate()

    out = score_trade(features, forecast, candidate, model=model)

    assert isinstance(out, MetaModelOutput)
    assert 0.0 <= out.probability_of_success <= 1.0
    assert 0.0 <= out.confidence <= 1.0
    assert isinstance(out.should_trade, bool)
    assert out.ticker != ""


def test_score_trade_should_trade_gate():
    """Output with probability above threshold has should_trade=True."""
    model = HeuristicMetaModel()
    # Use all-good features to get high probability.
    features = _make_features(
        trend_state="uptrend",
        trend_strength=0.9,
        volatility_regime="normal",
        relative_volume=2.0,
    )
    forecast = _make_forecast(direction="up", confidence=0.90)
    candidate = _make_candidate(reward_risk=4.0)

    out_high = score_trade(features, forecast, candidate, model=model, threshold=0.1)
    assert out_high.should_trade is True

    out_low = score_trade(features, forecast, candidate, model=model, threshold=0.99)
    assert out_low.should_trade is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: Walk-forward validation
# ─────────────────────────────────────────────────────────────────────────────

def test_walk_forward_n_folds():
    """walk_forward_validate(X, y, n_splits=3) returns <= 3 folds."""
    X, y = _make_synthetic_X_y(n=120)
    results = walk_forward_validate(X, y, n_splits=3, min_train_size=20)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert len(results) >= 1  # at least one fold should succeed with 120 samples

    for fold in results:
        assert "accuracy" in fold
        assert "fold" in fold
        assert "train_size" in fold
        assert "test_size" in fold


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: Pipeline with insufficient data
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_insufficient_data_warning():
    """run_training_pipeline with < 20 samples returns HeuristicMetaModel + warning."""
    from src.backtest.result import BacktestResult

    journal = _make_journal_with_features(n=5)

    class _FakeResult:
        trade_journal = journal

    model, metrics = run_training_pipeline([_FakeResult()], save_model=False)

    assert isinstance(model, HeuristicMetaModel)
    assert "warning" in metrics
    assert metrics["warning"] == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: Model save / load
# ─────────────────────────────────────────────────────────────────────────────

def test_model_save_load(tmp_path: Path):
    """Save HeuristicMetaModel to disk and reload; predictions match."""
    model = HeuristicMetaModel(threshold=0.55)
    save_path = tmp_path / "test.pkl"

    X, _ = _make_synthetic_X_y(n=10)
    proba_before = model.predict_proba(X)

    model.save(save_path)
    loaded = HeuristicMetaModel.load(save_path)

    proba_after = loaded.predict_proba(X)
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-9)
    assert loaded.threshold == pytest.approx(0.55)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 19–20: BacktestEngine integration
# ─────────────────────────────────────────────────────────────────────────────

def test_backtest_engine_stores_meta_features():
    """BacktestEngine on 300-bar series produces at least one journal entry with meta_features."""
    from src.backtest import BacktestEngine, make_synthetic_ohlcv

    series = make_synthetic_ohlcv(n_bars=300, seed=42, trend=0.0003)
    result = BacktestEngine(starting_capital=100_000, verbose=False).run(series)

    entries_with_features = [
        e for e in result.trade_journal if e.get("meta_features")
    ]

    assert len(entries_with_features) >= 1, (
        f"Expected at least one journal entry with meta_features, "
        f"got {len(result.trade_journal)} total trades"
    )


def test_meta_features_keys():
    """meta_features dict in journal entries contains all 13 FEATURE_NAMES keys."""
    from src.backtest import BacktestEngine, make_synthetic_ohlcv

    series = make_synthetic_ohlcv(n_bars=300, seed=42, trend=0.0003)
    result = BacktestEngine(starting_capital=100_000, verbose=False).run(series)

    entries_with_features = [
        e for e in result.trade_journal if e.get("meta_features")
    ]

    assert entries_with_features, "Need at least one trade with meta_features"

    for entry in entries_with_features:
        mf = entry["meta_features"]
        for name in FEATURE_NAMES:
            assert name in mf, f"Missing feature key '{name}' in meta_features"
        assert len(mf) == 13
