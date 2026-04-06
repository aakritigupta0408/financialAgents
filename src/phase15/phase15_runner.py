"""
src.phase15.phase15_runner — Phase 15 orchestrator.

Full pipeline:
1. Populate 1h store via populate_1h_store()
2. Load series from store via ReplayLoader (NOT direct synthetic)
3. Run dual FTA calibration
4. Choose global (rr, distance_pct): median across tickers
5. Expand meta-model dataset at calibrated params
6. Run all 7 benchmark configs at calibrated params
7. Determine verdict
8. Return Phase15Result
"""
from __future__ import annotations

import logging
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase15Result:
    # Data
    tickers_loaded: list[str]
    inventory_summary: list[dict]
    total_1h_bars_stored: int

    # Calibration
    dual_calibration: dict[str, Any]    # ticker -> DualCalibrationResult
    global_rr: float
    global_distance_pct: float

    # Dataset
    dataset_expansion: Any              # DatasetExpansionResult

    # Benchmark
    benchmark_results: dict[str, list[Any]]  # ticker -> list[BenchmarkResult]

    # Final verdict
    verdict: str                        # "READY_FOR_PAPER_TRADING" | ...
    verdict_reasons: list[str]
    recommended_config: dict
    remaining_todos: list[str]


# ---------------------------------------------------------------------------
# FTA patching helper
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
# Runner
# ---------------------------------------------------------------------------

def run_phase15(
    tickers: list[str] | None = None,
    n_bars: int = 800,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 50,
    rr_thresholds: list[float] | None = None,
    distance_pcts: list[float] | None = None,
    save_model: bool = False,
) -> Phase15Result:
    """
    Full Phase 15 pipeline.

    Parameters
    ----------
    tickers          : Tickers to run. Default: all 6 TICKER_PROXIES.
    n_bars           : Bars per synthetic series.
    starting_capital : Capital for each backtest.
    min_bars_required: Min bars before attempting trades.
    rr_thresholds    : RR sweep list. Default [1.25, 1.50, 1.75, 2.00].
    distance_pcts    : Distance sweep list. Default [0.001, 0.002, 0.003, 0.005].
    save_model       : Whether to persist trained meta-model.

    Returns
    -------
    Phase15Result
    """
    from src.data_store.inventory import DataInventory
    from src.data_store.replay import ReplayLoader
    from src.data_store.store import DataStore
    from src.phase15.dataset_expansion import expand_dataset
    from src.phase15.dual_calibration import run_dual_calibration
    from src.phase15.intraday_ingest import (
        TICKER_PROXIES,
        get_1h_inventory_summary,
        populate_1h_store,
    )
    from src.validation.benchmark import BenchmarkRunner

    if tickers is None:
        tickers = [p["ticker"] for p in TICKER_PROXIES]

    # ------------------------------------------------------------------
    # Step 1: Populate 1h store with synthetic data
    # ------------------------------------------------------------------
    store = DataStore()
    inventory = DataInventory()

    rows_written = populate_1h_store(
        store=store,
        inventory=inventory,
        n_bars=n_bars,
        tickers=tickers,
    )
    total_1h_bars_stored = sum(rows_written.values())

    # ------------------------------------------------------------------
    # Step 2: Load series from store via ReplayLoader
    # ------------------------------------------------------------------
    loader = ReplayLoader(store=store)
    series_map = {}
    tickers_loaded = []

    for ticker in tickers:
        try:
            series = loader.load(ticker, timeframe="1h", min_bars=min_bars_required)
            series_map[ticker] = series
            tickers_loaded.append(ticker)
        except Exception as exc:
            log.warning("Phase15: failed to load %s from store: %s", ticker, exc)

    # Inventory summary
    inventory_summary = get_1h_inventory_summary(inventory=inventory, tickers=tickers)

    # ------------------------------------------------------------------
    # Step 3: Dual FTA calibration
    # ------------------------------------------------------------------
    dual_calibration = run_dual_calibration(
        series_map=series_map,
        rr_thresholds=rr_thresholds,
        distance_pcts=distance_pcts,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    # ------------------------------------------------------------------
    # Step 4: Global recommended (rr, distance_pct) = median across tickers
    # ------------------------------------------------------------------
    rr_values = [r.recommended_rr for r in dual_calibration.values()]
    dist_values = [r.recommended_distance_pct for r in dual_calibration.values()]

    valid_rrs = [1.25, 1.50, 1.75, 2.00]
    if rr_values:
        median_rr = statistics.median(rr_values)
        global_rr = min(valid_rrs, key=lambda v: abs(v - median_rr))
    else:
        global_rr = 1.25

    if dist_values:
        median_dist = statistics.median(dist_values)
        # Round to nearest valid distance_pct
        valid_dists = distance_pcts or [0.001, 0.002, 0.003, 0.005]
        global_distance_pct = min(valid_dists, key=lambda v: abs(v - median_dist))
    else:
        global_distance_pct = 0.001

    # ------------------------------------------------------------------
    # Step 5: Dataset expansion at calibrated params
    # ------------------------------------------------------------------
    dataset_expansion = expand_dataset(
        series_map=series_map,
        calibrated_rr=global_rr,
        calibrated_distance_pct=global_distance_pct,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
        save_model=save_model,
    )

    # ------------------------------------------------------------------
    # Step 6: Benchmark at calibrated params
    # ------------------------------------------------------------------
    benchmark_results: dict[str, list] = {}
    runner = BenchmarkRunner(
        n_bars=n_bars,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    with _patch_fta_params(global_rr, global_distance_pct):
        for ticker, series in series_map.items():
            try:
                br = runner.run_all_configs(series)
                benchmark_results[ticker] = br
            except Exception as exc:
                log.warning("Phase15: benchmark failed for %s: %s", ticker, exc)
                benchmark_results[ticker] = []

    # ------------------------------------------------------------------
    # Step 7: Determine verdict
    # ------------------------------------------------------------------
    verdict, verdict_reasons = _determine_verdict(
        tickers_loaded=tickers_loaded,
        dual_calibration=dual_calibration,
        dataset_expansion=dataset_expansion,
        benchmark_results=benchmark_results,
        global_rr=global_rr,
        global_distance_pct=global_distance_pct,
    )

    # Recommended config
    recommended_config = {
        "FTA_MIN_REWARD_RISK": global_rr,
        "FTA_MIN_DISTANCE_TO_FTA_PCT": global_distance_pct,
        "META_MODEL_MIN_CONFIDENCE": 0.55,
        "MIN_BARS_REQUIRED": min_bars_required,
        "TIMEFRAME": "1h",
        "NOTE": (
            f"Phase 15 calibrated on {len(tickers_loaded)} synthetic 1h tickers. "
            "Validate on real intraday data before live deployment."
        ),
    }

    # Remaining TODOs
    remaining_todos = _build_todos(verdict, dataset_expansion, tickers_loaded)

    return Phase15Result(
        tickers_loaded=tickers_loaded,
        inventory_summary=inventory_summary,
        total_1h_bars_stored=total_1h_bars_stored,
        dual_calibration=dual_calibration,
        global_rr=global_rr,
        global_distance_pct=global_distance_pct,
        dataset_expansion=dataset_expansion,
        benchmark_results=benchmark_results,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        recommended_config=recommended_config,
        remaining_todos=remaining_todos,
    )


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _determine_verdict(
    tickers_loaded: list[str],
    dual_calibration: dict,
    dataset_expansion: Any,
    benchmark_results: dict,
    global_rr: float,
    global_distance_pct: float,
) -> tuple[str, list[str]]:
    """Determine Phase 15 acceptance verdict."""
    reasons: list[str] = []
    n_tickers = len(tickers_loaded)

    # Count tickers with >0 full_system trades at calibrated params
    tickers_with_trades = 0
    tickers_high_dd = 0
    for ticker in tickers_loaded:
        brs = benchmark_results.get(ticker, [])
        full_sys = next((r for r in brs if r.config_name == "full_system"), None)
        if full_sys and full_sys.n_trades > 0:
            tickers_with_trades += 1
            if full_sys.max_drawdown_pct > 25.0:
                tickers_high_dd += 1

    # Count tickers with "good" calibration verdict
    tickers_good_cal = sum(
        1 for cal in dual_calibration.values()
        if any(p.verdict == "good" for p in cal.grid)
    )

    total_trades = dataset_expansion.total_trades
    model_decision = dataset_expansion.model_decision

    reasons.append(f"tickers_loaded={n_tickers}")
    reasons.append(f"tickers_with_full_system_trades={tickers_with_trades}/{n_tickers}")
    reasons.append(f"tickers_good_calibration={tickers_good_cal}/{n_tickers}")
    reasons.append(f"total_dataset_trades={total_trades}")
    reasons.append(f"model_decision={model_decision}")
    reasons.append(
        f"calibrated_params=RR={global_rr:.2f}, dist_pct={global_distance_pct:.3f}"
    )

    # FAILS_CURRENT_ACCEPTANCE: all zero even at most permissive settings
    all_zero_everywhere = all(
        all(p.n_trades == 0 for p in cal.grid)
        for cal in dual_calibration.values()
    )
    if all_zero_everywhere and n_tickers > 0:
        reasons.append(
            "ALL tickers produce 0 trades even at (rr=1.25, distance_pct=0.001)."
        )
        return "FAILS_CURRENT_ACCEPTANCE", reasons

    # READY_FOR_PAPER_TRADING: all criteria met
    ready = (
        tickers_with_trades >= 3
        and tickers_good_cal >= 1
        and model_decision != "insufficient_data"
        and tickers_high_dd == 0
    )
    if ready:
        reasons.append("All Phase 15 readiness criteria satisfied.")
        return "READY_FOR_PAPER_TRADING", reasons

    # NEEDS_MORE_INTRADAY_DATA: some trades but not enough coverage
    if tickers_with_trades > 0 and (total_trades < 20 or tickers_loaded.__len__() < 3):
        reasons.append(
            f"Trades exist on {tickers_with_trades} ticker(s) but total_trades={total_trades} "
            "or coverage < 3 tickers. Need more intraday data."
        )
        return "NEEDS_MORE_INTRADAY_DATA", reasons

    # NEEDS_CALIBRATION: trades exist but thresholds or model need work
    if tickers_with_trades > 0 or tickers_good_cal > 0:
        if tickers_with_trades < 3:
            reasons.append(
                f"Only {tickers_with_trades}/3 tickers have trades at calibrated params."
            )
        if model_decision == "heuristic_fallback":
            reasons.append(
                "Model fell back to heuristic due to insufficient/imbalanced data."
            )
        if tickers_high_dd > 0:
            reasons.append(
                f"{tickers_high_dd} ticker(s) exceed 25% max drawdown."
            )
        return "NEEDS_CALIBRATION", reasons

    # Fallback
    reasons.append(
        "Insufficient evidence to reach READY. System needs tuning."
    )
    return "NEEDS_CALIBRATION", reasons


def _build_todos(
    verdict: str,
    dataset_expansion: Any,
    tickers_loaded: list[str],
) -> list[str]:
    todos = [
        "TODO: Acquire real 1h intraday data (Alpha Vantage premium or Polygon.io) "
        "and re-run Phase 15 calibration on real data.",
        "TODO: Validate FTA filter on >= 200 real 1h bars per ticker before enabling "
        "paper trading.",
        "TODO: Replace synthetic 1h data with real data once Alpha Vantage premium is "
        "available — re-run full pipeline.",
        "TODO: Monitor meta-model confidence calibration after first 50 live trades.",
        "TODO: Add slippage and commission model for more realistic simulation.",
    ]

    if verdict == "FAILS_CURRENT_ACCEPTANCE":
        todos.insert(0,
            "URGENT: Zero trades on all tickers. Check FTA_MIN_DISTANCE_TO_FTA_PCT "
            "and ensure structured synthetic data has enough swing structure."
        )
    elif verdict == "NEEDS_MORE_INTRADAY_DATA":
        todos.insert(0,
            f"Need more 1h intraday data. Currently covering {len(tickers_loaded)} "
            "tickers with synthetic data. Acquire real intraday data."
        )
    elif verdict == "NEEDS_CALIBRATION":
        todos.insert(0,
            "Calibration incomplete. Consider using distance_pct=0.001 or 0.002 "
            "as production default until real data confirms better threshold."
        )

    if dataset_expansion.model_decision == "insufficient_data":
        todos.append(
            "TODO: Meta-model has insufficient training data. Run with more tickers "
            "or more bars to reach >= 50 trades for reliable training."
        )

    return todos


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_phase15_report(result: Phase15Result) -> None:
    """
    Print complete Phase 15 report to stdout.
    """
    sep = "=" * 80
    thin = "-" * 80

    print()
    print(sep)
    print("  PHASE 15 REPORT — 1h Intraday Data Store + Dual FTA Calibration")
    print("               + Dataset Expansion + Final Acceptance Verdict")
    print(sep)

    # Section 1: Data Inventory
    print()
    print("  1. DATA INVENTORY SUMMARY")
    print(thin)
    print(f"  Tickers loaded : {', '.join(result.tickers_loaded)}")
    print(f"  Total 1h bars  : {result.total_1h_bars_stored:,}")
    print()
    print(f"  {'Ticker':<12} {'Source':<12} {'First':<12} {'Last':<12} {'Rows':>7} {'Fresh'}")
    print("  " + thin)
    for item in result.inventory_summary:
        first = str(item.get("first_date", "N/A"))
        last = str(item.get("last_date", "N/A"))
        fresh = "yes" if item.get("is_fresh") else "no"
        src = item.get("source", "?")
        print(
            f"  {item['ticker']:<12} {src:<12} {first:<12} {last:<12} "
            f"{item.get('row_count', 0):>7} {fresh}"
        )

    # Section 2: Dual Calibration Tables
    print()
    print("  2. DUAL FTA CALIBRATION TABLES")
    print(thin)
    from src.phase15.dual_calibration import print_dual_calibration_table
    print_dual_calibration_table(result.dual_calibration)
    print(f"  Global Recommended: RR={result.global_rr:.2f}, "
          f"dist_pct={result.global_distance_pct:.3f}")

    # Section 3: Dataset Expansion
    print()
    print("  3. DATASET EXPANSION SUMMARY")
    print(thin)
    de = result.dataset_expansion
    print(f"  Total trades collected : {de.total_trades}")
    print(f"  Positive labels (wins) : {de.positive_labels}")
    print(f"  Negative labels (loss) : {de.negative_labels}")
    print(f"  Label balance          : {de.label_balance:.1%}")
    print(f"  Sufficient for training: {de.sufficient_for_training}")
    print(f"  Model decision         : {de.model_decision}")
    print()
    print(f"  {'Ticker':<14} {'Trades':>7}")
    print("  " + thin[:40])
    for ticker, count in de.per_ticker_counts.items():
        print(f"  {ticker:<14} {count:>7}")

    # Section 4: Benchmark Results
    print()
    print("  4. BENCHMARK RESULTS AT CALIBRATED PARAMS "
          f"(RR={result.global_rr:.2f}, dist={result.global_distance_pct:.3f})")
    print(thin)
    configs_to_show = [
        "buy_and_hold", "forecast_only", "full_system_no_meta_model", "full_system"
    ]
    header = (
        f"  {'Ticker':<12} {'Config':<28} {'Trades':>6} "
        f"{'Ret%':>7} {'WinR':>6} {'DD%':>7} {'Sharpe':>7}"
    )
    print(header)
    print("  " + thin)
    for ticker, bench_list in result.benchmark_results.items():
        for br in bench_list:
            if br.config_name in configs_to_show:
                print(
                    f"  {ticker:<12} {br.config_name:<28} {br.n_trades:>6} "
                    f"{br.total_return_pct:>7.2f} {br.win_rate:>6.1%} "
                    f"{br.max_drawdown_pct:>7.2f} {br.sharpe_ratio:>7.3f}"
                )

    # Section 5: Final Verdict
    print()
    print("  5. FINAL ACCEPTANCE VERDICT")
    print(thin)
    verdict_labels = {
        "READY_FOR_PAPER_TRADING": "READY FOR PAPER TRADING",
        "NEEDS_MORE_INTRADAY_DATA": "NEEDS MORE INTRADAY DATA",
        "NEEDS_CALIBRATION": "NEEDS CALIBRATION",
        "FAILS_CURRENT_ACCEPTANCE": "FAILS CURRENT ACCEPTANCE",
    }
    print(f"  Verdict: {verdict_labels.get(result.verdict, result.verdict)}")
    print()
    print("  Reasons:")
    for reason in result.verdict_reasons:
        print(f"    - {reason}")

    # Section 6: Recommended Config
    print()
    print("  6. RECOMMENDED PRODUCTION CONFIG")
    print(thin)
    for k, v in result.recommended_config.items():
        if k != "NOTE":
            print(f"  {k:<40}: {v}")
    note = result.recommended_config.get("NOTE", "")
    if note:
        print(f"\n  Note: {note}")

    # Section 7: Remaining TODOs
    print()
    print("  7. REMAINING TODOs")
    print(thin)
    for todo in result.remaining_todos:
        print(f"  - {todo}")

    print()
    print(sep)
    print()
