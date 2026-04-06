"""
src.validation.benchmark — BenchmarkRunner and BenchmarkResult.

Runs each named benchmark configuration (defined in configs.py) against one
or more OHLCVSeries and collects key performance metrics.  Slippage and fees
are applied post-hoc to keep the engine implementation clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine
from src.validation.configs import BENCHMARK_CONFIGS, TICKER_UNIVERSE


@dataclass
class BenchmarkResult:
    config_name: str
    ticker: str
    n_bars: int
    n_trades: int
    total_return_pct: float
    realized_pnl: float
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    slippage_applied: float
    fee_applied: float


class BenchmarkRunner:
    """Run multiple BacktestEngine configurations on synthetic OHLCVSeries."""

    def __init__(
        self,
        n_bars: int = 300,
        starting_capital: float = 100_000.0,
        min_bars_required: int = 50,
        verbose: bool = False,
    ) -> None:
        self.n_bars = n_bars
        self.starting_capital = starting_capital
        self.min_bars_required = min_bars_required
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def run_config(
        self,
        config_name: str,
        series: OHLCVSeries,
        slippage: float = 0.0,
        fee: float = 0.0,
    ) -> BenchmarkResult:
        """
        Run one named config on one series.

        "buy_and_hold" is computed analytically.  All other configs
        instantiate BacktestEngine with the relevant kwargs and call .run().

        Slippage and fee are deducted post-hoc:
            total_cost = sum(entry_price * quantity * (slippage + fee))
        realized_pnl and total_return_pct are adjusted accordingly.
        """
        if config_name == "buy_and_hold":
            return self._run_buy_and_hold(series, slippage, fee)

        engine_kwargs = BENCHMARK_CONFIGS.get(config_name, {})
        engine = BacktestEngine(
            starting_capital=self.starting_capital,
            min_bars_required=self.min_bars_required,
            verbose=self.verbose,
            **engine_kwargs,
        )
        result = engine.run(series)

        # Post-hoc slippage + fee
        total_cost = _compute_transaction_cost(result.trade_journal, slippage + fee)
        adjusted_pnl = result.realized_pnl - total_cost
        adjusted_return = adjusted_pnl / self.starting_capital * 100.0

        return BenchmarkResult(
            config_name=config_name,
            ticker=result.ticker,
            n_bars=result.n_bars,
            n_trades=result.n_trades,
            total_return_pct=adjusted_return,
            realized_pnl=adjusted_pnl,
            win_rate=result.win_rate if result.win_rate is not None else 0.0,
            max_drawdown_pct=result.max_drawdown_pct,
            sharpe_ratio=result.sharpe_ratio if result.sharpe_ratio is not None else 0.0,
            profit_factor=result.profit_factor if result.profit_factor is not None else 0.0,
            avg_winner=result.avg_winner if result.avg_winner is not None else 0.0,
            avg_loser=result.avg_loser if result.avg_loser is not None else 0.0,
            slippage_applied=slippage,
            fee_applied=fee,
        )

    def run_all_configs(self, series: OHLCVSeries) -> list[BenchmarkResult]:
        """Run every config in BENCHMARK_CONFIGS on a single series."""
        results = []
        for name in BENCHMARK_CONFIGS:
            try:
                r = self.run_config(name, series)
                results.append(r)
            except Exception as exc:
                # Gracefully skip a failing config rather than aborting the sweep.
                import logging
                logging.getLogger(__name__).warning(
                    "BenchmarkRunner: config %s failed: %s", name, exc
                )
        return results

    def run_ticker_universe(self, n_bars: int = 300) -> dict[str, list[BenchmarkResult]]:
        """
        Run all BENCHMARK_CONFIGS x first-3 TICKER_UNIVERSE entries.

        Returns dict: ticker -> list[BenchmarkResult]
        """
        from src.backtest.data_utils import make_synthetic_ohlcv

        ticker_results: dict[str, list[BenchmarkResult]] = {}
        for spec in TICKER_UNIVERSE[:3]:
            ticker = spec["ticker"]
            series = make_synthetic_ohlcv(
                n_bars=n_bars,
                ticker=ticker,
                trend=spec["trend"],
                volatility=spec["volatility"],
                seed=spec["seed"],
            )
            ticker_results[ticker] = self.run_all_configs(series)
        return ticker_results

    def comparison_table(self, results: list[BenchmarkResult]) -> list[dict]:
        """Return list of dicts sorted by total_return_pct descending."""
        rows = [
            {
                "config_name": r.config_name,
                "ticker": r.ticker,
                "n_bars": r.n_bars,
                "n_trades": r.n_trades,
                "total_return_pct": r.total_return_pct,
                "realized_pnl": r.realized_pnl,
                "win_rate": r.win_rate,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "profit_factor": r.profit_factor,
            }
            for r in results
        ]
        rows.sort(key=lambda x: x["total_return_pct"], reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Buy-and-hold (analytical, no engine)
    # ------------------------------------------------------------------

    def _run_buy_and_hold(
        self,
        series: OHLCVSeries,
        slippage: float = 0.0,
        fee: float = 0.0,
    ) -> BenchmarkResult:
        bars = series.bars
        entry_bar = bars[self.min_bars_required]
        exit_bar = bars[-1]

        entry_price = entry_bar.close
        exit_price = exit_bar.close
        quantity = math.floor(self.starting_capital / entry_price)

        gross_pnl = (exit_price - entry_price) * quantity
        cost = entry_price * quantity * (slippage + fee)
        realized_pnl = gross_pnl - cost
        total_return_pct = realized_pnl / self.starting_capital * 100.0

        win_rate = 1.0 if realized_pnl > 0 else 0.0
        profit_factor = (
            max(realized_pnl, 0.0) / max(-realized_pnl, 1e-9)
        )

        return BenchmarkResult(
            config_name="buy_and_hold",
            ticker=series.ticker,
            n_bars=len(bars),
            n_trades=1,
            total_return_pct=total_return_pct,
            realized_pnl=realized_pnl,
            win_rate=win_rate,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=profit_factor,
            avg_winner=realized_pnl if realized_pnl > 0 else 0.0,
            avg_loser=realized_pnl if realized_pnl <= 0 else 0.0,
            slippage_applied=slippage,
            fee_applied=fee,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_transaction_cost(
    trade_journal: list[dict],
    cost_rate: float,
) -> float:
    """
    Sum entry_price * quantity * cost_rate over all closed trades.

    Falls back gracefully if journal entries are missing fields.
    """
    if cost_rate == 0.0:
        return 0.0
    total = 0.0
    for trade in trade_journal:
        entry_price = float(trade.get("entry_price") or 0.0)
        quantity = float(trade.get("quantity") or 0.0)
        total += entry_price * quantity * cost_rate
    return total
