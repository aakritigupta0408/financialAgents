"""
src.backtest — Historical backtest engine for the paper-trading research system.

Phase 6 deliverable.

Public API
----------
BacktestEngine   : Orchestrates the full simulation loop over an OHLCVSeries.
BacktestResult   : Immutable dataclass holding all metrics and the equity curve.
compute_metrics  : Standalone function to compute metrics from curve + journal.
make_synthetic_ohlcv : Generate reproducible synthetic OHLCV data for tests.

Quick start
-----------
    from src.backtest import BacktestEngine, make_synthetic_ohlcv

    series = make_synthetic_ohlcv(n_bars=300, trend=0.0003, seed=42)
    engine = BacktestEngine(starting_capital=100_000, verbose=True)
    result = engine.run(series)
    print(result.summary())
"""

from src.backtest.data_utils import make_synthetic_ohlcv
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.backtest.result import BacktestResult

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "compute_metrics",
    "make_synthetic_ohlcv",
]
