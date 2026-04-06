"""Portfolio and risk schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TradeOrder(BaseModel):
    """A paper-trade order submitted to the portfolio engine."""

    order_id: str
    ticker: str
    side: Literal["long", "short"]
    quantity: float  # shares / contracts
    entry_price: float
    stop_price: float
    target_price: float | None = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    source: Literal["meta_model", "backtest", "manual"] = "meta_model"

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def risk_amount(self) -> float:
        return self.risk_per_share * self.quantity


class Position(BaseModel):
    """An open or closed paper-trade position."""

    position_id: str
    ticker: str
    side: Literal["long", "short"]
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float | None = None
    opened_at: datetime
    closed_at: datetime | None = None
    exit_price: float | None = None

    status: Literal["open", "closed", "stopped"] = "open"

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def unrealized_pnl(self, current_price: float | None = None) -> float:
        # Caller must pass current_price; here we just define the formula
        # Use PortfolioState.compute_unrealized_pnl() in practice
        return 0.0

    def realized_pnl(self) -> float | None:
        if self.exit_price is None:
            return None
        sign = 1 if self.side == "long" else -1
        return sign * (self.exit_price - self.entry_price) * self.quantity


class PortfolioState(BaseModel):
    """Snapshot of the paper portfolio at a point in time."""

    snapshot_at: datetime = Field(default_factory=datetime.utcnow)

    starting_capital: float
    cash: float
    equity: float  # cash + unrealized value of open positions

    open_positions: list[Position] = Field(default_factory=list)
    closed_positions: list[Position] = Field(default_factory=list)

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    trades_today: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def total_return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return (self.equity - self.starting_capital) / self.starting_capital * 100

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100
