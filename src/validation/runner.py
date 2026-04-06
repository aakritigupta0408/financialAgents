"""
src.validation.runner — Orchestrates the full Phase 11 validation pipeline.

run_full_validation() is the single entry-point that calls all sub-runners,
merges results, and optionally saves a validation_summary.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.validation.summary import ValidationSummary, generate_validation_summary

log = logging.getLogger(__name__)


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj).__name__}")


def run_full_validation(
    n_bars: int = 400,
    starting_capital: float = 100_000.0,
    min_bars_required: int = 50,
    n_tickers: int = 3,
    save_json: bool = False,
    output_dir: str | Path | None = None,
) -> ValidationSummary:
    """
    Orchestrate the full Phase 11 validation pipeline.

    Steps
    -----
    1.  Generate primary series (AAPL proxy, trend=0.0003, vol=0.012, seed=1).
    2.  Run BenchmarkRunner.run_all_configs(primary_series).
    3.  Run BenchmarkRunner on first n_tickers TICKER_UNIVERSE entries.
    4.  Run AblationRunner.run_ablation(primary_series).
    5.  Run full_system BacktestEngine on primary_series (for robustness).
    6.  Run RobustnessRunner.run_all(primary_series).
    7.  Run DataSplitValidator.run() and .walk_forward().
    8.  Run sweep_thresholds on the primary result.
    9.  Merge ticker benchmark results.
    10. Call generate_validation_summary(...).
    11. Optionally write validation_summary.json.
    12. Return ValidationSummary.
    """
    from src.backtest.data_utils import make_synthetic_ohlcv
    from src.backtest.engine import BacktestEngine
    from src.reports.threshold_tuning import sweep_thresholds
    from src.validation.benchmark import BenchmarkRunner
    from src.validation.ablation import AblationRunner
    from src.validation.robustness import RobustnessRunner
    from src.validation.data_split import DataSplitValidator
    from src.validation.configs import TICKER_UNIVERSE

    # Step 1 — primary series
    primary_series = make_synthetic_ohlcv(
        n_bars=n_bars,
        ticker="AAPL",
        trend=0.0003,
        volatility=0.012,
        seed=1,
    )

    benchmark_runner = BenchmarkRunner(
        n_bars=n_bars,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )

    # Step 2 — benchmark on primary
    primary_benchmark = benchmark_runner.run_all_configs(primary_series)

    # Step 3 — benchmark across tickers
    ticker_benchmark_results: dict[str, list] = {}
    # Include primary ticker
    ticker_benchmark_results["AAPL"] = primary_benchmark

    for spec in TICKER_UNIVERSE[:n_tickers]:
        ticker = spec["ticker"]
        if ticker == "AAPL":
            continue  # already done
        series = make_synthetic_ohlcv(
            n_bars=n_bars,
            ticker=ticker,
            trend=spec["trend"],
            volatility=spec["volatility"],
            seed=spec["seed"],
        )
        ticker_benchmark_results[ticker] = benchmark_runner.run_all_configs(series)

    # Step 4 — ablation
    ablation_runner = AblationRunner(
        n_bars=n_bars,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )
    ablation_results = ablation_runner.run_ablation(primary_series)

    # Step 5 — full system result for robustness
    primary_result = BacktestEngine(
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
        fta_enabled=True,
        meta_model_enabled=True,
    ).run(primary_series)

    # Step 6 — robustness
    robustness_runner = RobustnessRunner(
        n_bars=n_bars,
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )
    robustness_results = robustness_runner.run_all(primary_series)

    # Step 7 — data splits
    split_validator = DataSplitValidator(
        starting_capital=starting_capital,
        min_bars_required=min_bars_required,
    )
    split_results = split_validator.run(primary_series)
    wf_results = split_validator.walk_forward(primary_series)
    all_split_results = split_results + wf_results

    # Step 8 — threshold sweep
    threshold_sweep = sweep_thresholds(primary_result)

    # Step 10 — generate summary
    summary = generate_validation_summary(
        benchmark_results=ticker_benchmark_results,
        ablation_results=ablation_results,
        robustness_results=robustness_results,
        data_split_results=all_split_results,
        threshold_sweep=threshold_sweep,
    )

    # Step 11 — optional JSON output
    if save_json and output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / "validation_summary.json"
        import dataclasses
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(dataclasses.asdict(summary), fh, default=_json_default, indent=2)
            log.info("Validation summary saved to %s", json_path)
        except Exception as exc:
            log.warning("Could not save validation summary: %s", exc)

    return summary
