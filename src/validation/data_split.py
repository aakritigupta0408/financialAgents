"""
src.validation.data_split — DataSplitValidator and DataSplitResult.

Provides time-based train/val/test splits and walk-forward validation
to confirm no data leakage across splits.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine


@dataclass
class DataSplitResult:
    split_name: str         # "train", "val", "test", "wf_test_0", ...
    n_bars: int
    n_trades: int
    total_return_pct: float
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float


class DataSplitValidator:
    """Validate the system on chronologically-split data to detect leakage."""

    def __init__(
        self,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        starting_capital: float = 100_000.0,
        min_bars_required: int = 50,
    ) -> None:
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.starting_capital = starting_capital
        self.min_bars_required = min_bars_required

    def run(self, series: OHLCVSeries) -> list[DataSplitResult]:
        """
        Split series.bars by position (time-based, no shuffle):
          train: bars[0 : int(n*0.70)]
          val:   bars[int(n*0.70) : int(n*0.85)]
          test:  bars[int(n*0.85) :]

        Run BacktestEngine (fta_enabled=False, meta_model_enabled=False) on each.
        Skips splits with < min_bars_required bars.
        """
        n = len(series.bars)
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.val_pct))

        splits = [
            ("train", series.bars[:train_end]),
            ("val",   series.bars[train_end:val_end]),
            ("test",  series.bars[val_end:]),
        ]

        results: list[DataSplitResult] = []
        for split_name, bars in splits:
            if len(bars) < self.min_bars_required:
                continue
            sub_series = OHLCVSeries(
                ticker=series.ticker,
                timeframe=series.timeframe,
                bars=list(bars),
                fetched_at=series.fetched_at,
            )
            result = self._run_engine(sub_series)
            results.append(DataSplitResult(
                split_name=split_name,
                n_bars=len(bars),
                n_trades=result.n_trades,
                total_return_pct=result.total_return_pct,
                win_rate=result.win_rate if result.win_rate is not None else 0.0,
                max_drawdown_pct=result.max_drawdown_pct,
                sharpe_ratio=result.sharpe_ratio if result.sharpe_ratio is not None else 0.0,
            ))
        return results

    def walk_forward(
        self,
        series: OHLCVSeries,
        n_splits: int = 3,
    ) -> list[DataSplitResult]:
        """
        Expanding-window walk-forward.

        For n_splits=3 with total n bars:
          split_size = n // (n_splits + 1) approximately
          window 0: train=bars[0:1/3*n], test=bars[1/3*n:2/3*n]
          window 1: train=bars[0:2/3*n], test=bars[2/3*n:3/3*n]
          (etc. for larger n_splits)

        Returns only test-window DataSplitResults, labelled "wf_test_0", "wf_test_1", ...
        """
        bars = series.bars
        n = len(bars)
        # Divide total bars into (n_splits + 1) roughly equal chunks.
        chunk = max(n // (n_splits + 1), 1)

        results: list[DataSplitResult] = []
        for i in range(n_splits):
            train_end = chunk * (i + 1)
            test_end = chunk * (i + 2)
            if test_end > n:
                test_end = n

            test_bars = bars[train_end:test_end]
            if len(test_bars) < self.min_bars_required:
                continue

            sub_series = OHLCVSeries(
                ticker=series.ticker,
                timeframe=series.timeframe,
                bars=list(test_bars),
                fetched_at=series.fetched_at,
            )
            result = self._run_engine(sub_series)
            results.append(DataSplitResult(
                split_name=f"wf_test_{i}",
                n_bars=len(test_bars),
                n_trades=result.n_trades,
                total_return_pct=result.total_return_pct,
                win_rate=result.win_rate if result.win_rate is not None else 0.0,
                max_drawdown_pct=result.max_drawdown_pct,
                sharpe_ratio=result.sharpe_ratio if result.sharpe_ratio is not None else 0.0,
            ))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_engine(self, sub_series: OHLCVSeries):
        engine = BacktestEngine(
            starting_capital=self.starting_capital,
            min_bars_required=self.min_bars_required,
            fta_enabled=False,
            meta_model_enabled=False,
        )
        return engine.run(sub_series)
