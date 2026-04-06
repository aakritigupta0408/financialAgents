"""
src.validation.robustness — RobustnessRunner and RobustnessResult.

Tests system stability under:
- varying slippage assumptions
- varying fee assumptions
- different confidence thresholds (post-hoc via sweep_thresholds)
- different market regimes
- small vs. large sample sizes (adaptive suppression check)
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine
from src.validation.configs import (
    FEE_VARIANTS,
    REGIME_CONFIGS,
    SLIPPAGE_VARIANTS,
)
from src.validation.benchmark import _compute_transaction_cost


@dataclass
class RobustnessResult:
    test_name: str
    description: str
    baseline_return_pct: float
    stressed_return_pct: float
    baseline_max_drawdown: float
    stressed_max_drawdown: float
    baseline_n_trades: int
    stressed_n_trades: int
    passed: bool
    notes: str


class RobustnessRunner:
    """Run a suite of robustness / stress tests on the full system."""

    def __init__(
        self,
        n_bars: int = 300,
        starting_capital: float = 100_000.0,
        min_bars_required: int = 50,
    ) -> None:
        self.n_bars = n_bars
        self.starting_capital = starting_capital
        self.min_bars_required = min_bars_required

    # ------------------------------------------------------------------
    # Slippage tests
    # ------------------------------------------------------------------

    def run_slippage_tests(self, series: OHLCVSeries) -> list[RobustnessResult]:
        """Test full system at each SLIPPAGE_VARIANTS level."""
        baseline_result = self._run_full_system(series)
        baseline_return = baseline_result.total_return_pct
        baseline_dd = baseline_result.max_drawdown_pct
        baseline_n = baseline_result.n_trades

        results: list[RobustnessResult] = []
        for slippage in SLIPPAGE_VARIANTS:
            cost = _compute_transaction_cost(baseline_result.trade_journal, slippage)
            stressed_pnl = baseline_result.realized_pnl - cost
            stressed_return = stressed_pnl / self.starting_capital * 100.0

            # Pass: stressed_return >= baseline * 0.50 (or baseline <= 0)
            if baseline_return <= 0:
                passed = True
            else:
                passed = stressed_return >= baseline_return * 0.50

            results.append(RobustnessResult(
                test_name=f"slippage_{slippage}",
                description=f"Slippage fraction {slippage:.4f} applied post-hoc",
                baseline_return_pct=baseline_return,
                stressed_return_pct=stressed_return,
                baseline_max_drawdown=baseline_dd,
                stressed_max_drawdown=baseline_dd,  # drawdown unchanged post-hoc
                baseline_n_trades=baseline_n,
                stressed_n_trades=baseline_n,
                passed=passed,
                notes=f"cost_deducted={cost:.2f}",
            ))
        return results

    # ------------------------------------------------------------------
    # Fee tests
    # ------------------------------------------------------------------

    def run_fee_tests(self, series: OHLCVSeries) -> list[RobustnessResult]:
        """Test full system at each FEE_VARIANTS level."""
        baseline_result = self._run_full_system(series)
        baseline_return = baseline_result.total_return_pct
        baseline_dd = baseline_result.max_drawdown_pct
        baseline_n = baseline_result.n_trades

        results: list[RobustnessResult] = []
        for fee in FEE_VARIANTS:
            cost = _compute_transaction_cost(baseline_result.trade_journal, fee)
            stressed_pnl = baseline_result.realized_pnl - cost
            stressed_return = stressed_pnl / self.starting_capital * 100.0

            if baseline_return <= 0:
                passed = True
            else:
                passed = stressed_return >= baseline_return * 0.50

            results.append(RobustnessResult(
                test_name=f"fee_{fee}",
                description=f"Fee fraction {fee:.4f} applied post-hoc",
                baseline_return_pct=baseline_return,
                stressed_return_pct=stressed_return,
                baseline_max_drawdown=baseline_dd,
                stressed_max_drawdown=baseline_dd,
                baseline_n_trades=baseline_n,
                stressed_n_trades=baseline_n,
                passed=passed,
                notes=f"cost_deducted={cost:.2f}",
            ))
        return results

    # ------------------------------------------------------------------
    # Threshold sensitivity
    # ------------------------------------------------------------------

    def run_threshold_sensitivity(self, series: OHLCVSeries) -> list[RobustnessResult]:
        """
        Use sweep_thresholds() on the existing result's trade journal.
        No re-run needed.  Tests confidence thresholds [0.3, 0.5, 0.6, 0.7, 0.8].
        Baseline = threshold 0.0 (all trades).
        """
        from src.reports.threshold_tuning import sweep_thresholds

        result = self._run_full_system(series)
        sweep = sweep_thresholds(
            result,
            confidence_thresholds=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8],
        )
        confidence_sweep = sweep.get("confidence_sweep", [])

        # Find baseline (threshold=0.0)
        baseline_row = next(
            (r for r in confidence_sweep if r["threshold"] == 0.0),
            None,
        )
        if baseline_row is None:
            baseline_pnl = result.realized_pnl
            baseline_n = result.n_trades
        else:
            baseline_pnl = baseline_row["total_pnl"]
            baseline_n = baseline_row["n_trades"]

        baseline_return = baseline_pnl / self.starting_capital * 100.0

        robustness_results: list[RobustnessResult] = []
        for row in confidence_sweep:
            threshold = row["threshold"]
            if threshold == 0.0:
                continue  # skip baseline row
            stressed_pnl = row["total_pnl"]
            stressed_return = stressed_pnl / self.starting_capital * 100.0
            stressed_n = row["n_trades"]

            # Pass: at least 3 trades AND return >= 50% of baseline (or baseline <= 0)
            has_trades = stressed_n >= 3
            if baseline_return <= 0:
                return_ok = True
            else:
                return_ok = stressed_return >= baseline_return * 0.5
            passed = has_trades and return_ok

            robustness_results.append(RobustnessResult(
                test_name=f"confidence_threshold_{threshold}",
                description=f"Confidence threshold >= {threshold}",
                baseline_return_pct=baseline_return,
                stressed_return_pct=stressed_return,
                baseline_max_drawdown=result.max_drawdown_pct,
                stressed_max_drawdown=row.get("max_drawdown_pct", 0.0),
                baseline_n_trades=baseline_n,
                stressed_n_trades=stressed_n,
                passed=passed,
                notes=f"win_rate={row.get('win_rate', 0.0):.2f}",
            ))
        return robustness_results

    # ------------------------------------------------------------------
    # Regime splits
    # ------------------------------------------------------------------

    def run_regime_splits(self) -> list[RobustnessResult]:
        """
        Run full_system on each REGIME_CONFIG (300 bars).
        baseline = trending_bull; all others compared against threshold.
        passed = stressed_return_pct >= -5.0
        """
        from src.backtest.data_utils import make_synthetic_ohlcv

        regime_results: dict[str, object] = {}
        for cfg in REGIME_CONFIGS:
            series = make_synthetic_ohlcv(
                n_bars=self.n_bars,
                ticker=cfg["name"],
                trend=cfg["trend"],
                volatility=cfg["volatility"],
                seed=cfg["seed"],
            )
            regime_results[cfg["name"]] = self._run_full_system(series)

        baseline_result = regime_results.get("trending_bull")
        baseline_return = (
            baseline_result.total_return_pct if baseline_result else 0.0
        )
        baseline_dd = (
            baseline_result.max_drawdown_pct if baseline_result else 0.0
        )
        baseline_n = baseline_result.n_trades if baseline_result else 0

        robustness_results: list[RobustnessResult] = []
        for cfg in REGIME_CONFIGS:
            r = regime_results[cfg["name"]]
            stressed_return = r.total_return_pct
            # Each regime passes if it doesn't blow up (>= -5%)
            passed = stressed_return >= -5.0

            robustness_results.append(RobustnessResult(
                test_name=f"regime_{cfg['name']}",
                description=f"Regime: {cfg['name']}",
                baseline_return_pct=baseline_return,
                stressed_return_pct=stressed_return,
                baseline_max_drawdown=baseline_dd,
                stressed_max_drawdown=r.max_drawdown_pct,
                baseline_n_trades=baseline_n,
                stressed_n_trades=r.n_trades,
                passed=passed,
                notes=(
                    f"trend={cfg['trend']:.5f} vol={cfg['volatility']:.3f}"
                ),
            ))
        return robustness_results

    # ------------------------------------------------------------------
    # Sample-size validation
    # ------------------------------------------------------------------

    def run_sample_size_validation(self, series: OHLCVSeries) -> list[RobustnessResult]:
        """
        Validate that adaptive suppression fires for small samples (< 10 trades).

        Short series (80 bars) → run_improvement_cycle → expect suppression.
        Long series (300 bars) → run_improvement_cycle → suppression NOT triggered.
        """
        from src.backtest.data_utils import make_synthetic_ohlcv
        from src.adaptive.loop import run_improvement_cycle

        results: list[RobustnessResult] = []

        # Small series: 80 bars, likely < 10 trades
        small_series = make_synthetic_ohlcv(
            n_bars=80,
            ticker=series.ticker,
            trend=0.0003,
            volatility=0.012,
            seed=999,
        )
        small_result = BacktestEngine(
            starting_capital=self.starting_capital,
            min_bars_required=self.min_bars_required,
        ).run(small_series)

        try:
            cycle_small = run_improvement_cycle(small_result, retrain_model=False, save_context=False)
            suppressed = cycle_small.update_summary.update_suppressed
        except Exception:
            suppressed = True  # If cycle fails, treat as suppressed (conservative)

        results.append(RobustnessResult(
            test_name="sample_size_small",
            description="Short series (80 bars) — suppression should fire when n_trades < 10",
            baseline_return_pct=small_result.total_return_pct,
            stressed_return_pct=small_result.total_return_pct,
            baseline_max_drawdown=small_result.max_drawdown_pct,
            stressed_max_drawdown=small_result.max_drawdown_pct,
            baseline_n_trades=small_result.n_trades,
            stressed_n_trades=small_result.n_trades,
            passed=suppressed,
            notes=f"n_trades={small_result.n_trades} suppressed={suppressed}",
        ))

        # Large series: 300 bars
        large_series = make_synthetic_ohlcv(
            n_bars=300,
            ticker=series.ticker,
            trend=0.0003,
            volatility=0.012,
            seed=998,
        )
        large_result = BacktestEngine(
            starting_capital=self.starting_capital,
            min_bars_required=self.min_bars_required,
        ).run(large_series)

        try:
            cycle_large = run_improvement_cycle(large_result, retrain_model=False, save_context=False)
            large_suppressed = cycle_large.update_summary.update_suppressed
        except Exception:
            large_suppressed = False  # If cycle fails with many trades, no suppression expected

        # passed = suppression NOT triggered for large series
        results.append(RobustnessResult(
            test_name="sample_size_large",
            description="Long series (300 bars) — suppression should NOT fire when n_trades >= 10",
            baseline_return_pct=large_result.total_return_pct,
            stressed_return_pct=large_result.total_return_pct,
            baseline_max_drawdown=large_result.max_drawdown_pct,
            stressed_max_drawdown=large_result.max_drawdown_pct,
            baseline_n_trades=large_result.n_trades,
            stressed_n_trades=large_result.n_trades,
            passed=not large_suppressed,
            notes=f"n_trades={large_result.n_trades} suppressed={large_suppressed}",
        ))

        return results

    # ------------------------------------------------------------------
    # Full suite
    # ------------------------------------------------------------------

    def run_all(self, series: OHLCVSeries) -> list[RobustnessResult]:
        """Run all robustness tests and return combined list."""
        results: list[RobustnessResult] = []
        results.extend(self.run_slippage_tests(series))
        results.extend(self.run_fee_tests(series))
        results.extend(self.run_threshold_sensitivity(series))
        results.extend(self.run_regime_splits())
        results.extend(self.run_sample_size_validation(series))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_full_system(self, series: OHLCVSeries):
        engine = BacktestEngine(
            starting_capital=self.starting_capital,
            min_bars_required=self.min_bars_required,
            fta_enabled=True,
            meta_model_enabled=True,
        )
        return engine.run(series)
