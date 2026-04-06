"""
src.backtest.metrics — Performance metric computation.

All metric computation is centralised here so the engine and result objects
stay thin. Metrics are computed from the equity curve and trade journal only —
no portfolio state is accessed directly.

Sharpe ratio formula
--------------------
returns     = [equity[i+1] / equity[i] - 1 for i in range(len-1)]
mean_r      = mean(returns)
std_r       = stdev(returns, ddof=1)
annualised  = (mean_r / std_r) * sqrt(periods_per_year)
periods_per_year = 252 * 24  for hourly bars (default)
             = 252            for daily bars

The result is clamped to [-10, 10] to suppress noise on very short series.
"""

from __future__ import annotations

import math
from datetime import datetime
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult

# Annualisation factors.
_PERIODS_PER_YEAR_DAILY = 252
_PERIODS_PER_YEAR_HOURLY = 252 * 24
_SHARPE_CLAMP = 10.0


def compute_metrics(
    equity_curve: list[tuple[datetime, float]],
    trade_journal: list[dict],
    starting_capital: float,
    timeframe: str = "1h",
) -> dict:
    """
    Compute all scalar backtest metrics from the equity curve and trade journal.

    Parameters
    ----------
    equity_curve     : List of (timestamp, equity) tuples, chronologically ordered.
    trade_journal    : List of trade dicts (output of Trade.to_dict()).
    starting_capital : Initial cash balance used to compute total return.
    timeframe        : Bar timeframe; used to choose annualisation factor for Sharpe.

    Returns
    -------
    Flat dict with keys matching BacktestResult scalar fields.
    """
    final_equity = equity_curve[-1][1] if equity_curve else starting_capital
    total_return_pct = (final_equity - starting_capital) / starting_capital * 100.0

    max_drawdown_pct = _compute_max_drawdown(equity_curve)
    sharpe = _compute_sharpe(equity_curve, timeframe)

    # Trade-level statistics — filter to closed trades only (exit_price not None).
    closed = [t for t in trade_journal if t.get("exit_price") is not None]
    n_trades = len(closed)

    winners = [t for t in closed if (t.get("realized_pnl") or 0.0) > 0]
    losers = [t for t in closed if (t.get("realized_pnl") or 0.0) <= 0]

    n_winners = len(winners)
    n_losers = len(losers)
    win_rate = n_winners / n_trades if n_trades > 0 else None

    avg_winner = (
        mean(t["realized_pnl"] for t in winners) if winners else None
    )
    avg_loser = (
        mean(t["realized_pnl"] for t in losers) if losers else None
    )

    gross_profit = sum(t["realized_pnl"] for t in winners) if winners else 0.0
    gross_loss = abs(sum(t["realized_pnl"] for t in losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else None

    realized_pnl = sum(t.get("realized_pnl", 0.0) for t in closed)

    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe,
        "n_trades": n_trades,
        "n_winners": n_winners,
        "n_losers": n_losers,
        "win_rate": win_rate,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "profit_factor": profit_factor,
        "realized_pnl": realized_pnl,
    }


def print_summary(result: "BacktestResult") -> None:
    """
    Print a formatted one-page backtest summary to stdout.

    Delegates to result.summary() for the actual formatting so the two outputs
    are always consistent.
    """
    print(result.summary())


# ── Internal helpers ──────────────────────────────────────────────────────────


def _compute_max_drawdown(equity_curve: list[tuple[datetime, float]]) -> float:
    """
    Compute the maximum peak-to-trough drawdown as a percentage (0–100).

    Scans the equity curve once, tracking the running peak. Returns 0.0 if
    the curve has fewer than 2 points.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0][1]
    max_dd = 0.0

    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    return max_dd


def _compute_sharpe(
    equity_curve: list[tuple[datetime, float]],
    timeframe: str = "1h",
) -> float | None:
    """
    Compute an annualised Sharpe ratio from per-bar equity returns.

    Formula
    -------
    returns[i] = equity[i+1] / equity[i] - 1
    sharpe     = (mean(returns) / stdev(returns, ddof=1)) * sqrt(periods_per_year)

    Clamped to [-10, 10] to avoid noise on very short series.
    Returns None if fewer than 2 equity points or if std_r == 0.
    """
    if len(equity_curve) < 2:
        return None

    equities = [e for _, e in equity_curve]
    returns = []
    for i in range(len(equities) - 1):
        prev = equities[i]
        curr = equities[i + 1]
        if prev > 0:
            returns.append(curr / prev - 1.0)

    if len(returns) < 2:
        return None

    mean_r = mean(returns)
    try:
        std_r = stdev(returns)
    except Exception:
        return None

    if std_r == 0.0:
        return None

    # Annualisation: use hourly factor for 1h bars, daily for all others.
    if timeframe == "1h":
        ann_factor = math.sqrt(_PERIODS_PER_YEAR_HOURLY)
    else:
        ann_factor = math.sqrt(_PERIODS_PER_YEAR_DAILY)

    sharpe = (mean_r / std_r) * ann_factor
    # Clamp to suppress extreme values on tiny samples.
    return max(-_SHARPE_CLAMP, min(_SHARPE_CLAMP, sharpe))
