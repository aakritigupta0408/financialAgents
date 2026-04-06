"""LiveLoopResult — structured output from the live loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LiveLoopResult:
    ticker: str
    timeframe: str
    n_bars_processed: int
    starting_capital: float
    final_equity: float
    trade_journal: list
    decision_log: list
    equity_curve: list

    # Computed in __post_init__
    n_trades: int = 0
    n_fta_rejections: int = 0
    n_meta_rejections: int = 0
    n_portfolio_rejections: int = 0
    total_return_pct: float = 0.0

    def __post_init__(self):
        self.n_trades = len(self.trade_journal)
        self.n_fta_rejections = sum(
            1 for d in self.decision_log if d.get("fta_accepted") is False
        )
        self.n_meta_rejections = sum(
            1 for d in self.decision_log if d.get("meta_model_accepted") is False
        )
        self.n_portfolio_rejections = sum(
            1 for d in self.decision_log
            if isinstance(d.get("rejection_reason"), str)
            and d["rejection_reason"].startswith("portfolio")
        )
        if self.starting_capital > 0:
            self.total_return_pct = (
                (self.final_equity - self.starting_capital) / self.starting_capital * 100
            )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  LIVE LOOP RESULT  —  {self.ticker} / {self.timeframe}",
            "=" * 60,
            f"  Bars processed    : {self.n_bars_processed}",
            f"  Starting capital  : ${self.starting_capital:>12,.2f}",
            f"  Final equity      : ${self.final_equity:>12,.2f}",
            f"  Total return      : {self.total_return_pct:+.2f}%",
            "",
            "  FILTERING FUNNEL",
            f"    FTA rejections  : {self.n_fta_rejections}",
            f"    Meta rejections : {self.n_meta_rejections}",
            f"    Port rejections : {self.n_portfolio_rejections}",
            f"    Trades opened   : {self.n_trades}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_backtest_result(self):
        """Convert to BacktestResult for meta-model retraining."""
        from src.backtest.result import BacktestResult
        from datetime import timezone
        now = datetime.now(timezone.utc)
        ec = self.equity_curve or [(now, self.starting_capital), (now, self.final_equity)]
        return BacktestResult(
            ticker=self.ticker,
            timeframe=self.timeframe,
            start_date=ec[0][0] if ec else now,
            end_date=ec[-1][0] if ec else now,
            n_bars=self.n_bars_processed,
            starting_capital=self.starting_capital,
            final_equity=self.final_equity,
            total_return_pct=self.total_return_pct,
            realized_pnl=sum(
                t.get("realized_pnl", 0)
                for t in self.trade_journal
                if isinstance(t.get("realized_pnl"), (int, float))
            ),
            unrealized_pnl_at_close=0.0,
            n_trades=self.n_trades,
            n_winners=sum(
                1 for t in self.trade_journal
                if isinstance(t.get("realized_pnl"), (int, float)) and t["realized_pnl"] > 0
            ),
            n_losers=sum(
                1 for t in self.trade_journal
                if isinstance(t.get("realized_pnl"), (int, float)) and t["realized_pnl"] <= 0
            ),
            win_rate=0.0,
            avg_winner=None,
            avg_loser=None,
            profit_factor=None,
            max_drawdown_pct=0.0,
            sharpe_ratio=None,
            equity_curve=self.equity_curve,
            trade_journal=self.trade_journal,
        )
