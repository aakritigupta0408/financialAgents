"""
src.backtest.result — BacktestResult dataclass.

This module defines the immutable result produced by BacktestEngine.run().
All scalar metrics are stored as plain Python types so the result is trivially
serialisable. The equity_curve and trade_journal are stored as lists for easy
post-processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Optional


@dataclass
class BacktestResult:
    """
    Immutable result produced by BacktestEngine.run().

    All monetary values share the currency of starting_capital.
    Percentages use the suffix _pct and are expressed as 0–100.
    Ratios (win_rate, profit_factor) are dimensionless floats.
    """

    # Identity
    ticker: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    n_bars: int

    # Capital
    starting_capital: float
    final_equity: float
    total_return_pct: float

    # PnL
    realized_pnl: float
    unrealized_pnl_at_close: float

    # Trade stats
    n_trades: int
    n_winners: int
    n_losers: int
    win_rate: Optional[float]          # 0–1; None if no trades
    avg_winner: Optional[float]        # mean realized PnL of winning trades
    avg_loser: Optional[float]         # mean realized PnL of losing trades
    profit_factor: Optional[float]     # gross_profit / abs(gross_loss); None if no losers

    # Drawdown
    max_drawdown_pct: float            # 0–100

    # Risk-adjusted
    sharpe_ratio: Optional[float]      # annualised; None if fewer than 2 equity points

    # Full curves (not included in to_dict())
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trade_journal: list[dict] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return all scalar fields as a flat dict.

        equity_curve and trade_journal are excluded — they are available as
        direct attributes for callers that need them.
        """
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "n_bars": self.n_bars,
            "starting_capital": self.starting_capital,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl_at_close": self.unrealized_pnl_at_close,
            "n_trades": self.n_trades,
            "n_winners": self.n_winners,
            "n_losers": self.n_losers,
            "win_rate": self.win_rate,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
        }

    # ── Human-readable report ─────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Return a one-page human-readable backtest summary string.

        This is designed to be printed to stdout. All numbers are rounded for
        readability. None values are shown as "N/A".
        """
        def _fmt(val, fmt=".2f", suffix="") -> str:
            if val is None:
                return "N/A"
            return f"{val:{fmt}}{suffix}"

        lines = [
            "=" * 60,
            f"  BACKTEST SUMMARY  —  {self.ticker} / {self.timeframe}",
            "=" * 60,
            f"  Period         : {self.start_date.strftime('%Y-%m-%d')} → {self.end_date.strftime('%Y-%m-%d')}",
            f"  Bars simulated : {self.n_bars}",
            "",
            "  CAPITAL",
            f"    Starting      : ${self.starting_capital:>12,.2f}",
            f"    Final equity  : ${self.final_equity:>12,.2f}",
            f"    Total return  : {_fmt(self.total_return_pct, '.2f', '%')}",
            "",
            "  PnL",
            f"    Realized PnL  : ${self.realized_pnl:>12,.2f}",
            f"    Unrealized    : ${self.unrealized_pnl_at_close:>12,.2f}",
            "",
            "  TRADES",
            f"    Total trades  : {self.n_trades}",
            f"    Winners       : {self.n_winners}",
            f"    Losers        : {self.n_losers}",
            f"    Win rate      : {_fmt(self.win_rate, '.1%') if self.win_rate is not None else 'N/A'}",
            f"    Avg winner    : {_fmt(self.avg_winner, '.2f', '')}",
            f"    Avg loser     : {_fmt(self.avg_loser, '.2f', '')}",
            f"    Profit factor : {_fmt(self.profit_factor)}",
            "",
            "  RISK",
            f"    Max drawdown  : {_fmt(self.max_drawdown_pct, '.2f', '%')}",
            f"    Sharpe ratio  : {_fmt(self.sharpe_ratio)}",
            "=" * 60,
        ]
        return "\n".join(lines)
