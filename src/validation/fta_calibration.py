"""
src.validation.fta_calibration — FTA R:R threshold sweep and calibration.

Runs BacktestEngine with fta_enabled=True across RR thresholds [1.25, 1.50,
1.75, 2.00] to find the threshold that gives a healthy trade frequency.

Monkeypatching strategy
-----------------------
src.fta.engine imports FTA_MIN_REWARD_RISK at the TOP of the function (it uses
  `from config.settings import FTA_MIN_REWARD_RISK`
which binds the name at module load time).  Therefore we must patch both
  config.settings.FTA_MIN_REWARD_RISK  (for any future imports)
  src.fta.engine.FTA_MIN_REWARD_RISK   (for the already-bound module-level name)
The finally block restores both.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine

_DEFAULT_RR_THRESHOLDS: list[float] = [1.25, 1.50, 1.75, 2.00]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FTACalibrationResult:
    """Metrics for one (ticker, rr_threshold) combination."""
    rr_threshold: float
    ticker: str
    n_bars: int
    n_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    expectancy: float          # avg_winner*win_rate + avg_loser*(1-win_rate)
    profit_factor: float
    trades_per_100_bars: float # n_trades / n_bars * 100
    verdict: str               # "too_few" | "good" | "too_many"


@dataclass
class CalibrationSummary:
    """Aggregated calibration result across all RR thresholds for one ticker."""
    ticker: str
    results: list[FTACalibrationResult] = field(default_factory=list)
    recommended_rr: float = 1.25
    recommended_reason: str = "default"
    all_zero_trades: bool = True


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_fta_calibration(
    series_map: dict[str, OHLCVSeries],
    rr_thresholds: Optional[list[float]] = None,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 50,
) -> dict[str, CalibrationSummary]:
    """
    For each ticker in series_map, sweep RR thresholds and collect metrics.

    Temporarily overrides FTA_MIN_REWARD_RISK by patching config.settings and
    src.fta.engine at runtime, then restores original value.

    Parameters
    ----------
    series_map        : ticker -> OHLCVSeries mapping.
    rr_thresholds     : List of R:R thresholds to test. Default: [1.25,1.50,1.75,2.00].
    starting_capital  : Capital for each backtest run.
    min_bars_required : Minimum bars before attempting trades.

    Returns
    -------
    dict: ticker -> CalibrationSummary
    """
    if rr_thresholds is None:
        rr_thresholds = _DEFAULT_RR_THRESHOLDS

    import config.settings as _cfg
    import src.fta.engine as _fta_engine

    orig_rr = _cfg.FTA_MIN_REWARD_RISK
    orig_fta_rr = getattr(_fta_engine, "FTA_MIN_REWARD_RISK", None)

    summaries: dict[str, CalibrationSummary] = {}

    try:
        for ticker, series in series_map.items():
            cal_results: list[FTACalibrationResult] = []

            for rr in rr_thresholds:
                # Patch both locations
                _cfg.FTA_MIN_REWARD_RISK = rr
                if hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
                    _fta_engine.FTA_MIN_REWARD_RISK = rr

                result = _run_single(
                    series=series,
                    starting_capital=starting_capital,
                    min_bars_required=min_bars_required,
                )

                n_trades = result.n_trades
                n_bars = result.n_bars
                t_per_100 = (n_trades / n_bars * 100.0) if n_bars > 0 else 0.0

                # Verdict
                if t_per_100 < 2.0:
                    verdict = "too_few"
                elif t_per_100 > 20.0:
                    verdict = "too_many"
                else:
                    verdict = "good"

                # Expectancy
                avg_win = result.avg_winner if result.avg_winner is not None else 0.0
                avg_los = result.avg_loser if result.avg_loser is not None else 0.0
                wr = result.win_rate if result.win_rate is not None else 0.0
                expectancy = avg_win * wr + avg_los * (1.0 - wr)

                cal_results.append(
                    FTACalibrationResult(
                        rr_threshold=rr,
                        ticker=ticker,
                        n_bars=n_bars,
                        n_trades=n_trades,
                        win_rate=wr,
                        total_return_pct=result.total_return_pct,
                        max_drawdown_pct=result.max_drawdown_pct,
                        expectancy=expectancy,
                        profit_factor=result.profit_factor if result.profit_factor is not None else 0.0,
                        trades_per_100_bars=t_per_100,
                        verdict=verdict,
                    )
                )

            # Build summary
            summary = _build_summary(ticker, cal_results)
            summaries[ticker] = summary

    finally:
        # Always restore original values
        _cfg.FTA_MIN_REWARD_RISK = orig_rr
        if orig_fta_rr is not None and hasattr(_fta_engine, "FTA_MIN_REWARD_RISK"):
            _fta_engine.FTA_MIN_REWARD_RISK = orig_fta_rr
        elif hasattr(_fta_engine, "FTA_MIN_REWARD_RISK") and orig_fta_rr is None:
            # It was there but we patched it; restore from config
            _fta_engine.FTA_MIN_REWARD_RISK = orig_rr

    return summaries


def _run_single(
    series: OHLCVSeries,
    starting_capital: float,
    min_bars_required: int,
):
    """Run BacktestEngine with fta_enabled=True, meta_model_enabled=False."""
    engine = BacktestEngine(
        starting_capital=starting_capital,
        fta_enabled=True,
        meta_model_enabled=False,
        min_bars_required=min_bars_required,
        verbose=False,
    )
    return engine.run(series)


def _build_summary(
    ticker: str,
    cal_results: list[FTACalibrationResult],
) -> CalibrationSummary:
    """Choose the recommended RR threshold from calibration results."""
    all_zero = all(r.n_trades == 0 for r in cal_results)

    if all_zero:
        return CalibrationSummary(
            ticker=ticker,
            results=cal_results,
            recommended_rr=1.25,
            recommended_reason="no_trades_all_thresholds",
            all_zero_trades=True,
        )

    # Prefer "good" verdict (2–20 trades/100 bars) with highest expectancy
    good_results = [r for r in cal_results if r.verdict == "good"]
    if good_results:
        # Sort by expectancy desc, then by rr_threshold asc (lower is more selective)
        good_results.sort(key=lambda r: (-r.expectancy, r.rr_threshold))
        best = good_results[0]
        reason = "good_verdict_best_expectancy"
        if len(good_results) > 1:
            # Check for ties: same expectancy → prefer lower RR
            reason = "good_verdict_best_expectancy"
        return CalibrationSummary(
            ticker=ticker,
            results=cal_results,
            recommended_rr=best.rr_threshold,
            recommended_reason=reason,
            all_zero_trades=False,
        )

    # No "good" verdict: pick lowest threshold (least restrictive)
    non_zero = [r for r in cal_results if r.n_trades > 0]
    if non_zero:
        non_zero.sort(key=lambda r: r.rr_threshold)
        best = non_zero[0]
        return CalibrationSummary(
            ticker=ticker,
            results=cal_results,
            recommended_rr=best.rr_threshold,
            recommended_reason="no_good_verdict_least_restrictive",
            all_zero_trades=False,
        )

    # All thresholds give 0 trades — shouldn't reach here but safety fallback
    return CalibrationSummary(
        ticker=ticker,
        results=cal_results,
        recommended_rr=1.25,
        recommended_reason="no_trades_all_thresholds",
        all_zero_trades=True,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_calibration_table(summaries: dict[str, CalibrationSummary]) -> None:
    """
    Print a formatted calibration table to stdout.

    Shows trades/100 bars at each RR threshold, recommended RR, and the
    full per-threshold metrics for each ticker.
    """
    thresholds = _DEFAULT_RR_THRESHOLDS

    header = f"\n{'FTA R:R CALIBRATION TABLE':^80}"
    sep = "-" * 80
    print(header)
    print(sep)

    # Summary row: ticker x threshold → trades_per_100_bars
    col_w = 9
    hdr_parts = ["Ticker   "] + [f"RR={rr:.2f}  " for rr in thresholds] + ["Recommended"]
    print("  ".join(hdr_parts))
    sub_parts = [" " * 9] + ["t/100b   " for _ in thresholds] + [""]
    print("  ".join(sub_parts))
    print(sep)

    for ticker, summary in summaries.items():
        rr_map = {r.rr_threshold: r for r in summary.results}
        row_parts = [f"{ticker:<9}"]
        for rr in thresholds:
            r = rr_map.get(rr)
            if r:
                row_parts.append(f"{r.trades_per_100_bars:>6.1f}   ")
            else:
                row_parts.append(f"{'N/A':>6}   ")
        row_parts.append(
            f"RR={summary.recommended_rr:.2f} ({summary.recommended_reason})"
        )
        print("  ".join(row_parts))

    print(sep)
    print()

    # Detailed per-ticker tables
    for ticker, summary in summaries.items():
        print(f"  Ticker: {ticker}")
        print(f"  {'RR':>6}  {'n_trades':>8}  {'t/100b':>7}  {'win%':>6}  "
              f"{'ret%':>7}  {'dd%':>7}  {'expect':>8}  {'PF':>6}  {'verdict'}")
        print("  " + "-" * 75)
        for r in summary.results:
            print(
                f"  {r.rr_threshold:>6.2f}  {r.n_trades:>8d}  "
                f"{r.trades_per_100_bars:>7.2f}  {r.win_rate:>6.1%}  "
                f"{r.total_return_pct:>7.2f}  {r.max_drawdown_pct:>7.2f}  "
                f"{r.expectancy:>8.2f}  {r.profit_factor:>6.2f}  {r.verdict}"
            )
        print(f"  Recommended: RR={summary.recommended_rr:.2f} — {summary.recommended_reason}")
        print()
