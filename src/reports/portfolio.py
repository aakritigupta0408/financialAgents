"""
src.reports.portfolio — Portfolio-level performance report.

generate_portfolio_report(result) aggregates all scalar metrics from a
BacktestResult plus derived breakdowns (per-ticker, per-exit-reason, daily PnL,
average holding duration).

All monetary values share the currency of result.starting_capital.
Percentages use the suffix _pct and are expressed as 0–100.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.result import BacktestResult


def generate_portfolio_report(result: "BacktestResult") -> dict:
    """
    Build a portfolio-level performance summary dict.

    Parameters
    ----------
    result : BacktestResult produced by BacktestEngine.run().

    Returns
    -------
    dict with the following keys:

    Identity
        ticker, timeframe, start_date, end_date, n_bars

    Capital
        starting_capital, final_equity, total_return_pct

    PnL
        realized_pnl, unrealized_pnl_at_close

    Trade stats
        n_trades, n_winners, n_losers, win_rate,
        avg_winner, avg_loser, profit_factor

    Risk
        max_drawdown_pct, sharpe_ratio

    Derived
        avg_holding_bars   — mean timedelta between entry_time and exit_time;
                             expressed as fractional hours for 1h bars
        per_ticker_pnl     — dict[str, float]
        per_exit_reason_pnl — dict[str, float]
        daily_pnl          — dict[YYYY-MM-DD, float]
    """
    journal = result.trade_journal or []

    # Only closed trades (exit_price not None).
    closed = [t for t in journal if t.get("exit_price") is not None]

    # --- avg_holding_bars (expressed as hours for consistency) ---
    holding_deltas: list[float] = []
    for t in closed:
        entry_time = t.get("entry_time")
        exit_time = t.get("exit_time")
        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
            delta_hours = (exit_time - entry_time).total_seconds() / 3600.0
            holding_deltas.append(max(0.0, delta_hours))

    avg_holding_bars = mean(holding_deltas) if holding_deltas else 0.0

    # --- per_ticker_pnl ---
    per_ticker: dict[str, float] = defaultdict(float)
    for t in closed:
        ticker = t.get("ticker", "UNKNOWN")
        per_ticker[ticker] += float(t.get("realized_pnl", 0.0))
    per_ticker_pnl = dict(per_ticker)

    # --- per_exit_reason_pnl ---
    per_reason: dict[str, float] = defaultdict(float)
    for t in closed:
        reason = t.get("exit_reason", "unknown")
        per_reason[reason] += float(t.get("realized_pnl", 0.0))
    per_exit_reason_pnl = dict(per_reason)

    # --- daily_pnl ---
    daily: dict[str, float] = defaultdict(float)
    for t in closed:
        exit_time = t.get("exit_time")
        if isinstance(exit_time, datetime):
            day_str = exit_time.strftime("%Y-%m-%d")
        else:
            day_str = "unknown"
        daily[day_str] += float(t.get("realized_pnl", 0.0))
    daily_pnl = dict(sorted(daily.items()))

    return {
        # Identity
        "ticker": result.ticker,
        "timeframe": result.timeframe,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "n_bars": result.n_bars,
        # Capital
        "starting_capital": result.starting_capital,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        # PnL
        "realized_pnl": result.realized_pnl,
        "unrealized_pnl_at_close": result.unrealized_pnl_at_close,
        # Trade stats
        "n_trades": result.n_trades,
        "n_winners": result.n_winners,
        "n_losers": result.n_losers,
        "win_rate": result.win_rate,
        "avg_winner": result.avg_winner,
        "avg_loser": result.avg_loser,
        "profit_factor": result.profit_factor,
        # Risk
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        # Derived
        "avg_holding_bars": avg_holding_bars,
        "per_ticker_pnl": per_ticker_pnl,
        "per_exit_reason_pnl": per_exit_reason_pnl,
        "daily_pnl": daily_pnl,
    }
