"""
src.portfolio.trade — Mutable internal trade record.

This is a plain dataclass (not Pydantic) because trades are updated in-place
throughout their lifecycle: last_price, status, exit_price, etc. are mutated
after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from schemas.portfolio import Position


@dataclass
class Trade:
    """
    Internal mutable representation of a paper trade.

    Lifecycle
    ---------
    open  → stopped   (stop-loss hit)
    open  → target_hit (take-profit hit)
    open  → closed    (manual exit)
    open  → manual_exit (engine-level forced exit)
    """

    trade_id: str
    ticker: str
    side: Literal["long", "short"]
    quantity: float
    entry_price: float
    stop_price: float
    entry_time: datetime
    target_price: float | None = None
    status: Literal["open", "closed", "stopped", "target_hit", "manual_exit"] = "open"
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None
    source: str = "meta_model"
    last_price: float | None = None
    last_update: datetime | None = None

    # ── PnL helpers ──────────────────────────────────────────────────────────

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Mark-to-market gain/loss on the open trade.

        Formula:
            sign = +1 (long) or -1 (short)
            unrealized_pnl = sign * (current_price - entry_price) * quantity
        """
        sign = 1.0 if self.side == "long" else -1.0
        return sign * (current_price - self.entry_price) * self.quantity

    def realized_pnl(self) -> float:
        """
        Locked-in gain/loss once the trade is closed.

        Formula:
            sign = +1 (long) or -1 (short)
            realized_pnl = sign * (exit_price - entry_price) * quantity
        Returns 0.0 if the trade has not been closed (exit_price is None).
        """
        if self.exit_price is None:
            return 0.0
        sign = 1.0 if self.side == "long" else -1.0
        return sign * (self.exit_price - self.entry_price) * self.quantity

    def cost_basis(self) -> float:
        """
        Capital tied up in this position.

        Formula:
            cost_basis = entry_price * quantity

        Note: for shorts, cost_basis represents the notional value used for
        exposure tracking. Cash is NOT debited on short opens in this
        simplified paper-trading model (TODO: margin model in Phase 8).
        """
        return self.entry_price * self.quantity

    # ── Schema bridge ─────────────────────────────────────────────────────────

    def to_position(self) -> Position:
        """
        Return an immutable Pydantic Position snapshot for PortfolioState.

        The Position.unrealized_pnl property in schemas/portfolio.py is a stub
        that always returns 0.0. Actual unrealized PnL is computed in the
        Portfolio engine using Trade.unrealized_pnl(current_price).
        """
        # Map internal status to the narrower set Position allows
        position_status: Literal["open", "closed", "stopped"]
        if self.status == "open":
            position_status = "open"
        elif self.status in ("target_hit", "manual_exit", "closed"):
            position_status = "closed"
        else:
            position_status = "stopped"

        return Position(
            position_id=self.trade_id,
            ticker=self.ticker,
            side=self.side,
            quantity=self.quantity,
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            target_price=self.target_price,
            opened_at=self.entry_time,
            closed_at=self.exit_time,
            exit_price=self.exit_price,
            status=position_status,
        )

    def to_dict(self) -> dict:
        """
        Flat dictionary representation for journal serialisation.
        All datetime fields are ISO-8601 strings; None values are preserved.
        """
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "status": self.status,
            "source": self.source,
            "last_price": self.last_price,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "realized_pnl": self.realized_pnl(),
            "cost_basis": self.cost_basis(),
        }
