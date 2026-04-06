"""
src.phase15.dataset_expansion — Expand the meta-model training dataset.

Runs multi-ticker 1h backtests and collects trade journal entries with
meta_features for downstream meta-model training.

Model decision thresholds:
  - total_trades >= 50 AND label_balance in [0.2, 0.8]: "trained_model"
  - total_trades >= 20 but < 50, or imbalanced: "heuristic_fallback"
  - total_trades < 20: "insufficient_data"
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine
from src.meta_model.dataset import build_dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetExpansionResult:
    total_trades: int
    positive_labels: int
    negative_labels: int
    label_balance: float          # positive / total (or 0.0 if no trades)
    per_ticker_counts: dict[str, int] = field(default_factory=dict)
    sufficient_for_training: bool = False
    training_result: dict | None = None
    model_decision: str = "insufficient_data"   # "trained_model" | "heuristic_fallback" | "insufficient_data"


# ---------------------------------------------------------------------------
# FTA patching helper (same approach as dual_calibration.py)
# ---------------------------------------------------------------------------

@contextmanager
def _patch_fta_params(rr: float, dist_pct: float) -> Iterator[None]:
    """Temporarily patch both FTA params. Restores in finally block."""
    import config.settings as _cfg
    import src.fta.engine as _fta_engine

    orig_rr_cfg = _cfg.FTA_MIN_REWARD_RISK
    orig_dist_cfg = _cfg.FTA_MIN_DISTANCE_TO_FTA_PCT
    orig_rr_eng = getattr(_fta_engine, "FTA_MIN_REWARD_RISK", None)
    orig_dist_eng = getattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT", None)

    try:
        _cfg.FTA_MIN_REWARD_RISK = rr
        _cfg.FTA_MIN_DISTANCE_TO_FTA_PCT = dist_pct
        if hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = rr
        if hasattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT"):
            _fta_engine.FTA_MIN_DISTANCE_TO_FTA_PCT = dist_pct
        yield
    finally:
        _cfg.FTA_MIN_REWARD_RISK = orig_rr_cfg
        _cfg.FTA_MIN_DISTANCE_TO_FTA_PCT = orig_dist_cfg
        if orig_rr_eng is not None and hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = orig_rr_eng
        elif orig_rr_eng is None and hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = orig_rr_cfg
        if orig_dist_eng is not None and hasattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT"):
            _fta_engine.FTA_MIN_DISTANCE_TO_FTA_PCT = orig_dist_eng
        elif orig_dist_eng is None and hasattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT"):
            _fta_engine.FTA_MIN_DISTANCE_TO_FTA_PCT = orig_dist_cfg


# ---------------------------------------------------------------------------
# Main expansion function
# ---------------------------------------------------------------------------

def expand_dataset(
    series_map: dict,              # ticker -> OHLCVSeries
    calibrated_rr: float,
    calibrated_distance_pct: float,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 50,
    min_trades_for_training: int = 50,
    save_model: bool = False,
) -> DatasetExpansionResult:
    """
    Run BacktestEngine with fta_enabled=True, meta_model_enabled=False
    on each ticker in series_map using the calibrated FTA params.

    Collects all BacktestResult objects, builds dataset, evaluates label balance,
    and optionally trains a meta-model.

    Parameters
    ----------
    series_map              : ticker -> OHLCVSeries
    calibrated_rr           : FTA_MIN_REWARD_RISK from dual calibration
    calibrated_distance_pct : FTA_MIN_DISTANCE_TO_FTA_PCT from dual calibration
    starting_capital        : Capital for each backtest run
    min_bars_required       : Min bars before trading
    min_trades_for_training : Threshold for "trained_model" decision (default 50)
    save_model              : If True, save trained model to disk

    Returns
    -------
    DatasetExpansionResult
    """
    all_backtest_results = []
    per_ticker_counts: dict[str, int] = {}

    with _patch_fta_params(calibrated_rr, calibrated_distance_pct):
        for ticker, series in series_map.items():
            try:
                engine = BacktestEngine(
                    starting_capital=starting_capital,
                    fta_enabled=True,
                    meta_model_enabled=False,
                    min_bars_required=min_bars_required,
                    verbose=False,
                )
                result = engine.run(series)
                all_backtest_results.append(result)
                per_ticker_counts[ticker] = result.n_trades
            except Exception as exc:
                log.warning("expand_dataset: backtest failed for %s: %s", ticker, exc)
                per_ticker_counts[ticker] = 0

    # Build dataset from all results
    try:
        X, y, feature_names = build_dataset(all_backtest_results)
    except Exception as exc:
        log.warning("expand_dataset: build_dataset failed: %s", exc)
        X, y = [], []

    total_trades = int(len(y)) if hasattr(y, "__len__") else 0
    positive_labels = int(sum(1 for lbl in y if lbl == 1))
    negative_labels = total_trades - positive_labels
    label_balance = positive_labels / total_trades if total_trades > 0 else 0.0

    # Determine sufficient_for_training
    balance_ok = 0.2 <= label_balance <= 0.8
    sufficient_for_training = total_trades >= min_trades_for_training and balance_ok

    # Model decision
    if total_trades >= min_trades_for_training and balance_ok:
        model_decision = "trained_model"
    elif total_trades >= 20:
        model_decision = "heuristic_fallback"
    else:
        model_decision = "insufficient_data"

    # Optionally train model
    training_result: dict | None = None
    if sufficient_for_training and model_decision == "trained_model":
        try:
            from src.meta_model.pipeline import run_training_pipeline
            _model, metrics = run_training_pipeline(
                all_backtest_results, save_model=save_model
            )
            training_result = metrics
            log.info(
                "expand_dataset: trained meta-model on %d samples", total_trades
            )
        except Exception as exc:
            log.warning("expand_dataset: training failed: %s", exc)
            training_result = {"error": str(exc)}
            model_decision = "heuristic_fallback"

    return DatasetExpansionResult(
        total_trades=total_trades,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        label_balance=label_balance,
        per_ticker_counts=per_ticker_counts,
        sufficient_for_training=sufficient_for_training,
        training_result=training_result,
        model_decision=model_decision,
    )
