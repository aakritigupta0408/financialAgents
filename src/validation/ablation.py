"""
src.validation.ablation — AblationRunner and AblationResult.

Measures the marginal contribution of each system component (FTA, meta-model,
adaptation) by comparing a full-system baseline against ablated variants.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.market_data import OHLCVSeries
from src.backtest.engine import BacktestEngine


@dataclass
class AblationResult:
    removed_component: str        # "fta", "meta_model", "adaptation", "heuristic_meta"
    baseline_return_pct: float
    ablated_return_pct: float
    return_delta_pct: float       # ablated - baseline (negative => component adds value)
    baseline_win_rate: float
    ablated_win_rate: float
    baseline_max_drawdown: float
    ablated_max_drawdown: float
    baseline_n_trades: int
    ablated_n_trades: int
    marginal_contribution: str    # "positive", "negative", "neutral"


class AblationRunner:
    """Compare full-system vs. ablated variants to measure component value."""

    def __init__(
        self,
        n_bars: int = 300,
        starting_capital: float = 100_000.0,
        min_bars_required: int = 50,
    ) -> None:
        self.n_bars = n_bars
        self.starting_capital = starting_capital
        self.min_bars_required = min_bars_required

    def run_ablation(self, series: OHLCVSeries) -> list[AblationResult]:
        """
        Baseline = full_system (fta_enabled=True, meta_model_enabled=True).

        Variants tested:
        - fta            : remove FTA filter (fta_enabled=False, meta_model_enabled=True)
        - meta_model     : remove meta-model (fta_enabled=True, meta_model_enabled=False)
        - heuristic_meta : no FTA, with meta-model (fta_enabled=False, meta_model_enabled=True)
        - adaptation     : same as full_system; adaptation is post-hoc, not in engine
        """
        baseline_result = self._run_engine(series, fta_enabled=True, meta_model_enabled=True)
        baseline = _extract(baseline_result)

        variants: list[tuple[str, dict]] = [
            ("fta",            {"fta_enabled": False, "meta_model_enabled": True}),
            ("meta_model",     {"fta_enabled": True,  "meta_model_enabled": False}),
            ("heuristic_meta", {"fta_enabled": False, "meta_model_enabled": True}),
            ("adaptation",     {"fta_enabled": True,  "meta_model_enabled": True}),
        ]

        results: list[AblationResult] = []
        for component, kwargs in variants:
            ablated_result = self._run_engine(series, **kwargs)
            ablated = _extract(ablated_result)

            return_delta = ablated["total_return_pct"] - baseline["total_return_pct"]
            if return_delta > 0.5:
                contribution = "positive"
            elif return_delta < -0.5:
                contribution = "negative"
            else:
                contribution = "neutral"

            results.append(AblationResult(
                removed_component=component,
                baseline_return_pct=baseline["total_return_pct"],
                ablated_return_pct=ablated["total_return_pct"],
                return_delta_pct=return_delta,
                baseline_win_rate=baseline["win_rate"],
                ablated_win_rate=ablated["win_rate"],
                baseline_max_drawdown=baseline["max_drawdown_pct"],
                ablated_max_drawdown=ablated["max_drawdown_pct"],
                baseline_n_trades=baseline["n_trades"],
                ablated_n_trades=ablated["n_trades"],
                marginal_contribution=contribution,
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_engine(self, series: OHLCVSeries, **engine_kwargs) -> object:
        """Instantiate a BacktestEngine with given kwargs and run it."""
        engine = BacktestEngine(
            starting_capital=self.starting_capital,
            min_bars_required=self.min_bars_required,
            **engine_kwargs,
        )
        return engine.run(series)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract(result) -> dict:
    return {
        "total_return_pct": result.total_return_pct,
        "win_rate": result.win_rate if result.win_rate is not None else 0.0,
        "max_drawdown_pct": result.max_drawdown_pct,
        "n_trades": result.n_trades,
    }
