"""
src.phase15.dual_calibration — Dual FTA parameter calibration.

Sweeps FTA_MIN_REWARD_RISK x FTA_MIN_DISTANCE_TO_FTA_PCT to find the
combination that yields healthy trade frequency.

Phase 13 identified that FTA_MIN_DISTANCE_TO_FTA_PCT=0.005 (default) blocks
almost all trades. This sweep tests lower distance_pct values including
0.001, 0.002, 0.003, and the problematic 0.005.

Patching strategy (matches fta_calibration.py):
  Both FTA_MIN_REWARD_RISK and FTA_MIN_DISTANCE_TO_FTA_PCT are imported at
  module level in src.fta.engine, so we must patch both:
    config.settings.FTA_MIN_REWARD_RISK / FTA_MIN_DISTANCE_TO_FTA_PCT
    src.fta.engine.FTA_MIN_REWARD_RISK  / FTA_MIN_DISTANCE_TO_FTA_PCT
"""
from __future__ import annotations

import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine

_DEFAULT_RR_THRESHOLDS: list[float] = [1.25, 1.50, 1.75, 2.00]
_DEFAULT_DISTANCE_PCTS: list[float] = [0.001, 0.002, 0.003, 0.005]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DualCalibrationPoint:
    """Metrics for one (ticker, rr_threshold, distance_pct) combination."""
    rr_threshold: float
    distance_pct: float
    ticker: str
    n_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    expectancy: float
    trades_per_100_bars: float
    verdict: str    # "zero", "too_few", "good", "too_many"


@dataclass
class DualCalibrationResult:
    """Full dual calibration result for one ticker."""
    ticker: str
    grid: list[DualCalibrationPoint] = field(default_factory=list)
    recommended_rr: float = 1.25
    recommended_distance_pct: float = 0.001
    recommended_reason: str = "default"


# ---------------------------------------------------------------------------
# Patching context manager
# ---------------------------------------------------------------------------

@contextmanager
def _patch_fta_params(rr: float, dist_pct: float) -> Iterator[None]:
    """
    Temporarily patch both FTA params in config.settings and src.fta.engine.
    Restores originals in finally block regardless of exceptions.
    """
    import config.settings as _cfg
    import src.fta.engine as _fta_engine

    orig_rr_cfg = _cfg.FTA_MIN_REWARD_RISK
    orig_dist_cfg = _cfg.FTA_MIN_DISTANCE_TO_FTA_PCT
    orig_rr_eng = getattr(_fta_engine, "FTA_MIN_REWARD_RISK", None)
    orig_dist_eng = getattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT", None)

    try:
        # Patch config module
        _cfg.FTA_MIN_REWARD_RISK = rr
        _cfg.FTA_MIN_DISTANCE_TO_FTA_PCT = dist_pct

        # Patch engine module (bound at import time)
        if hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = rr
        if hasattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT"):
            _fta_engine.FTA_MIN_DISTANCE_TO_FTA_PCT = dist_pct

        yield

    finally:
        # Restore config module
        _cfg.FTA_MIN_REWARD_RISK = orig_rr_cfg
        _cfg.FTA_MIN_DISTANCE_TO_FTA_PCT = orig_dist_cfg

        # Restore engine module
        if orig_rr_eng is not None and hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = orig_rr_eng
        elif orig_rr_eng is None and hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = orig_rr_cfg

        if orig_dist_eng is not None and hasattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT"):
            _fta_engine.FTA_MIN_DISTANCE_TO_FTA_PCT = orig_dist_eng
        elif orig_dist_eng is None and hasattr(_fta_engine, "FTA_MIN_DISTANCE_TO_FTA_PCT"):
            _fta_engine.FTA_MIN_DISTANCE_TO_FTA_PCT = orig_dist_cfg


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_dual_calibration(
    series_map: dict,        # ticker -> OHLCVSeries
    rr_thresholds: list[float] | None = None,
    distance_pcts: list[float] | None = None,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 50,
) -> dict[str, DualCalibrationResult]:
    """
    Run BacktestEngine with fta_enabled=True, meta_model_enabled=False
    for every (ticker, rr_threshold, distance_pct) combination.

    Returns dict: ticker -> DualCalibrationResult

    Verdict rules:
      "zero"     : n_trades == 0
      "too_few"  : 0 < trades_per_100_bars < 2.0
      "good"     : 2.0 <= trades_per_100_bars <= 20.0
      "too_many" : trades_per_100_bars > 20.0

    Recommendation logic:
      1. Filter to "good" points, maximise expectancy, prefer higher distance_pct
         (more conservative = fewer false signals).
      2. If no "good" points: pick lowest distance_pct + lowest rr (least restrictive).
      3. If still zero trades everywhere: fallback to (rr=1.25, dist=0.001).
    """
    if rr_thresholds is None:
        rr_thresholds = _DEFAULT_RR_THRESHOLDS
    if distance_pcts is None:
        distance_pcts = _DEFAULT_DISTANCE_PCTS

    results: dict[str, DualCalibrationResult] = {}

    for ticker, series in series_map.items():
        grid: list[DualCalibrationPoint] = []

        for dist_pct in distance_pcts:
            for rr in rr_thresholds:
                with _patch_fta_params(rr, dist_pct):
                    engine = BacktestEngine(
                        starting_capital=starting_capital,
                        fta_enabled=True,
                        meta_model_enabled=False,
                        min_bars_required=min_bars_required,
                        verbose=False,
                    )
                    try:
                        bt = engine.run(series)
                    except Exception:
                        bt = None

                if bt is None:
                    point = DualCalibrationPoint(
                        rr_threshold=rr,
                        distance_pct=dist_pct,
                        ticker=ticker,
                        n_trades=0,
                        win_rate=0.0,
                        total_return_pct=0.0,
                        max_drawdown_pct=0.0,
                        expectancy=0.0,
                        trades_per_100_bars=0.0,
                        verdict="zero",
                    )
                    grid.append(point)
                    continue

                n_trades = bt.n_trades
                n_bars = bt.n_bars
                t_per_100 = (n_trades / n_bars * 100.0) if n_bars > 0 else 0.0

                if n_trades == 0:
                    verdict = "zero"
                elif t_per_100 < 2.0:
                    verdict = "too_few"
                elif t_per_100 > 20.0:
                    verdict = "too_many"
                else:
                    verdict = "good"

                avg_win = bt.avg_winner if bt.avg_winner is not None else 0.0
                avg_los = bt.avg_loser if bt.avg_loser is not None else 0.0
                wr = bt.win_rate if bt.win_rate is not None else 0.0
                expectancy = avg_win * wr + avg_los * (1.0 - wr)

                grid.append(DualCalibrationPoint(
                    rr_threshold=rr,
                    distance_pct=dist_pct,
                    ticker=ticker,
                    n_trades=n_trades,
                    win_rate=wr,
                    total_return_pct=bt.total_return_pct,
                    max_drawdown_pct=bt.max_drawdown_pct,
                    expectancy=expectancy,
                    trades_per_100_bars=t_per_100,
                    verdict=verdict,
                ))

        # Build recommendation
        rec_rr, rec_dist, rec_reason = _choose_recommendation(
            grid, rr_thresholds, distance_pcts
        )

        results[ticker] = DualCalibrationResult(
            ticker=ticker,
            grid=grid,
            recommended_rr=rec_rr,
            recommended_distance_pct=rec_dist,
            recommended_reason=rec_reason,
        )

    return results


def _choose_recommendation(
    grid: list[DualCalibrationPoint],
    rr_thresholds: list[float],
    distance_pcts: list[float],
) -> tuple[float, float, str]:
    """Return (recommended_rr, recommended_distance_pct, reason)."""

    # 1. Good verdict: maximise expectancy, then prefer higher distance_pct
    good_points = [p for p in grid if p.verdict == "good"]
    if good_points:
        good_points.sort(key=lambda p: (-p.expectancy, -p.distance_pct))
        best = good_points[0]
        return best.rr_threshold, best.distance_pct, "good_verdict_max_expectancy"

    # 2. Any non-zero trades: pick least restrictive
    nonzero = [p for p in grid if p.n_trades > 0]
    if nonzero:
        nonzero.sort(key=lambda p: (p.distance_pct, p.rr_threshold))
        best = nonzero[0]
        return best.rr_threshold, best.distance_pct, "nonzero_least_restrictive"

    # 3. Everything zero: fallback
    min_dist = min(distance_pcts)
    min_rr = min(rr_thresholds)
    return min_rr, min_dist, "zero_trades_fallback"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_dual_calibration_table(results: dict[str, DualCalibrationResult]) -> None:
    """
    Print a 2D grid table for each ticker showing trades/100bars at
    each (dist_pct, rr_threshold) combination.
    """
    rr_values = _DEFAULT_RR_THRESHOLDS
    dist_values = _DEFAULT_DISTANCE_PCTS

    for ticker, cal in results.items():
        # Build lookup: (dist_pct, rr) -> point
        lookup: dict[tuple, DualCalibrationPoint] = {}
        for p in cal.grid:
            lookup[(p.distance_pct, p.rr_threshold)] = p

        # Detect actual rr values in grid
        actual_rrs = sorted({p.rr_threshold for p in cal.grid})
        actual_dists = sorted({p.distance_pct for p in cal.grid})
        if not actual_rrs:
            actual_rrs = rr_values
        if not actual_dists:
            actual_dists = dist_values

        col_w = 9
        rr_col_w = 8

        # Header
        print(f"\n{ticker} — Dual FTA Calibration (trades per 100 bars)")
        rr_header = "  ".join(f"{rr:.2f}".center(rr_col_w) for rr in actual_rrs)
        print(f"{'dist_pct':>10}  {rr_header}")
        print("-" * (12 + len(actual_rrs) * (rr_col_w + 2)))

        for dist in actual_dists:
            row_parts = [f"{dist:.3f}".rjust(10)]
            for rr in actual_rrs:
                pt = lookup.get((dist, rr))
                if pt is None:
                    row_parts.append("  N/A    ")
                else:
                    val_str = f"{pt.trades_per_100_bars:.1f}".center(rr_col_w)
                    # Mark the problematic 0.005 row
                    if dist == 0.005:
                        val_str = val_str.rstrip() + "*"
                        val_str = val_str.center(rr_col_w)
                    row_parts.append(val_str)
            print("  ".join(row_parts))

        print(
            f"  Recommended: RR={cal.recommended_rr:.2f}, "
            f"dist_pct={cal.recommended_distance_pct:.3f} "
            f"({cal.recommended_reason})"
        )

    print()
    print("* dist_pct=0.005 is Phase 13's default — identified as main blocker")
