"""
src.features — feature engineering layer for the paper-trading research system.

Public API
----------
compute_all_features(snapshot, primary_tf) -> dict
    Master pipeline entry point.  Runs all feature modules and returns a dict
    with keys: "structure", "levels", "volatility", "liquidity".

Individual functions are also exported for unit testing.
"""

from __future__ import annotations

from src.features.levels import compute_levels
from src.features.liquidity import compute_liquidity
from src.features.pipeline import compute_all_features
from src.features.resampling import resample_ohlcv
from src.features.structure import compute_structure
from src.features.volatility import compute_volatility

__all__ = [
    "compute_all_features",
    "compute_structure",
    "compute_levels",
    "compute_volatility",
    "compute_liquidity",
    "resample_ohlcv",
]
