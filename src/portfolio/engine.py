"""
src.portfolio.engine — Paper portfolio engine.

Paper trading fill assumptions
-------------------------------
- Fills execute at the exact price provided (zero slippage).
- No commissions or fees are charged.
- No partial fills: the full computed quantity transacts at once.
- Stop-loss fills at exactly trade.stop_price (no gap risk modelled).
- Take-profit fills at exactly trade.target_price.
- Short positions: cost_basis is tracked for exposure purposes, but cash is
  NOT debited on open (simplified cash-account model). Shorts receive
  proceeds equal to cost_basis on open and pay them back on close.
  TODO: proper margin/collateral model in Phase 8.
- TODO: configurable slippage model (Phase 8) — e.g. fixed bps or
  half-spread based on bid/ask.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any

from config.settings import (
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_TICKER_EXPOSURE_PCT,
    MAX_TRADES_PER_DAY,
    RISK_PER_TRADE_PCT,
    STARTING_CAPITAL,
)
from schemas.portfolio import PortfolioState
from src.portfolio.risk import RiskConfig, RiskManager
from src.portfolio.sizing import compute_position_size, required_capital
from src.portfolio.trade import Trade

log = logging.getLogger(__name__)

_UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(_UTC)


class Portfolio:
    """
    Mutable paper portfolio engine.

    State is kept in plain Python attributes. Use portfolio_snapshot() to
    produce an immutable PortfolioState Pydantic object at any point.

    Lifecycle
    ---------
    1. Construct → initialize() (called automatically in __init__)
    2. open_trade() to enter positions
    3. update_positions(price_map) on every price tick to mark-to-market
       and auto-close stops/targets
    4. close_trade() for manual exits
    5. portfolio_snapshot() / get_metrics() for reporting
    6. export_trade_journal() for audit trail
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        if config is None:
            config = RiskConfig(
                starting_capital=STARTING_CAPITAL,
                risk_per_trade_pct=RISK_PER_TRADE_PCT,
                max_trades_per_day=MAX_TRADES_PER_DAY,
                max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
                max_daily_drawdown_pct=MAX_DAILY_DRAWDOWN_PCT,
                max_ticker_exposure_pct=MAX_TICKER_EXPOSURE_PCT,
            )
        self.config: RiskConfig = config
        self.risk_manager: RiskManager = RiskManager(config)

        # These are all reset by initialize(); defined here for type visibility.
        self.cash: float = 0.0
        self.starting_capital: float = 0.0
        self.peak_equity: float = 0.0
        self.day_start_equity: float = 0.0
        self.open_trades: dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades_today: int = 0
        self.current_date: date | None = None
        self._price_cache: dict[str, float] = {}

        self.initialize()

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, starting_capital: float | None = None) -> None:
        """
        Full reset. Accepts an optional capital override.

        Calling this mid-session wipes all trade history and resets cash.
        Use with care outside of backtests.
        """
        capital = starting_capital if starting_capital is not None else self.config.starting_capital
        self.starting_capital = capital
        self.cash = capital
        self.peak_equity = capital
        self.day_start_equity = capital
        self.open_trades = {}
        self.closed_trades = []
        self.equity_curve = []
        self.trades_today = 0
        self.current_date = None
        self._price_cache = {}
        self.risk_manager.reset_daily_state(capital)
        log.info("Portfolio initialised. starting_capital=%.2f", capital)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        """
        Total portfolio value = cash + market value of all open positions.

        For each open trade, if a last_price is known the market value is
        cost_basis + unrealized_pnl; otherwise entry_price is used (zero
        unrealized PnL at open).

        Formula:
            equity = cash + Σ (cost_basis + unrealized_pnl(last_price))
                   = cash + Σ (entry_price * qty + sign * (last_price - entry_price) * qty)
                   = cash + Σ (last_price * qty)   [long]
        """
        total = self.cash
        for trade in self.open_trades.values():
            price = trade.last_price if trade.last_price is not None else trade.entry_price
            total += trade.unrealized_pnl(price) + trade.cost_basis()
        return total

    def gross_exposure(self) -> float:
        """
        Sum of cost_basis across all open trades (notional capital deployed).
        """
        return sum(t.cost_basis() for t in self.open_trades.values())

    def ticker_exposure_pct(self, ticker: str) -> float:
        """
        Fraction of current equity in open positions for a given ticker.

        Returns 0.0 if equity is zero (avoids division by zero).
        """
        eq = self.equity
        if eq == 0:
            return 0.0
        notional = sum(
            t.cost_basis() for t in self.open_trades.values() if t.ticker == ticker
        )
        return notional / eq

    def daily_drawdown_pct(self) -> float:
        """
        Fraction of day-start equity lost today.

        Formula:
            daily_drawdown_pct = (day_start_equity - equity) / day_start_equity

        Returns 0.0 if equity has risen (no drawdown) or day_start_equity is zero.
        The value is positive when equity has FALLEN from the day's open.
        """
        if self.day_start_equity <= 0:
            return 0.0
        dd = (self.day_start_equity - self.equity) / self.day_start_equity
        return max(0.0, dd)

    def realized_pnl_total(self) -> float:
        """Sum of realized PnL across all closed trades."""
        return sum(t.realized_pnl() for t in self.closed_trades)

    def unrealized_pnl_total(self) -> float:
        """Sum of unrealized PnL across all open trades."""
        return sum(
            t.unrealized_pnl(t.last_price if t.last_price is not None else t.entry_price)
            for t in self.open_trades.values()
        )

    # ── Trade entry ───────────────────────────────────────────────────────────

    def can_open_trade(
        self,
        ticker: str,
        entry_price: float,
        stop_price: float,
        quantity: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check whether a new trade can be opened without actually opening it.

        Useful for external callers (e.g. backtest loop) that want to
        pre-validate before committing.
        """
        if quantity is None:
            quantity = compute_position_size(
                equity=self.equity,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_pct=self.config.risk_per_trade_pct,
                max_ticker_exposure_pct=self.config.max_ticker_exposure_pct,
                available_cash=self.cash,
            )
        cap = required_capital(entry_price, quantity)
        return self.risk_manager.can_open_trade(self, ticker, cap)

    def open_trade(
        self,
        ticker: str,
        side: str = "long",
        entry_price: float = 0.0,
        stop_price: float = 0.0,
        target_price: float | None = None,
        quantity: float | None = None,
        confidence: float = 1.0,
        source: str = "meta_model",
        timestamp: datetime | None = None,
    ) -> Trade | None:
        """
        Open a new paper trade.

        Steps:
        1. Auto-size quantity if not provided.
        2. Run risk checks — return None if rejected.
        3. Advance the day counter if the calendar date has rolled.
        4. Allocate a unique trade_id and debit cash.
        5. Register the trade and return it.

        Returns None (and logs a warning) if risk checks reject the trade.
        """
        ts = timestamp or _now_utc()
        self._maybe_advance_day(ts)

        # 1. Size
        if quantity is None:
            quantity = compute_position_size(
                equity=self.equity,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_pct=self.config.risk_per_trade_pct,
                confidence=confidence,
                max_ticker_exposure_pct=self.config.max_ticker_exposure_pct,
                available_cash=self.cash,
            )

        cap = required_capital(entry_price, quantity)

        # 2. Risk gate
        allowed, reason = self.risk_manager.can_open_trade(self, ticker, cap)
        if not allowed:
            log.warning(
                "Trade rejected: ticker=%s reason=%s entry=%.2f qty=%.0f",
                ticker,
                reason,
                entry_price,
                quantity,
            )
            return None

        # 4. Create trade
        trade_id = uuid.uuid4().hex[:8]
        trade = Trade(
            trade_id=trade_id,
            ticker=ticker,
            side=side,  # type: ignore[arg-type]
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            entry_time=ts,
            target_price=target_price,
            source=source,
            last_price=entry_price,
        )

        # 5. Debit cash
        self.cash -= cap

        # 6. Register
        self.open_trades[trade_id] = trade
        self.trades_today += 1
        self._price_cache[ticker] = entry_price

        self._update_equity_and_peak(ts)

        log.info(
            "Trade opened: id=%s ticker=%s side=%s qty=%.0f entry=%.2f stop=%.2f target=%s",
            trade_id,
            ticker,
            side,
            quantity,
            entry_price,
            stop_price,
            target_price,
        )
        return trade

    # ── Price updates ─────────────────────────────────────────────────────────

    def update_positions(
        self,
        price_map: dict[str, float],
        timestamp: datetime | None = None,
    ) -> list[Trade]:
        """
        Mark all open positions to market and auto-close any stops/targets.

        Parameters
        ----------
        price_map:
            {ticker: current_price} for all tickers you have data for.
            Tickers not in the map are skipped (last known price retained).
        timestamp:
            The bar/tick time. Defaults to now(UTC).

        Returns
        -------
        List of Trade objects that were closed during this update.
        """
        ts = timestamp or _now_utc()
        self._maybe_advance_day(ts)

        closed_this_update: list[Trade | None] = []

        for trade_id, trade in list(self.open_trades.items()):
            price = price_map.get(trade.ticker)
            if price is None:
                continue

            trade.last_price = price
            trade.last_update = ts
            self._price_cache[trade.ticker] = price

            if self.risk_manager.check_stop_hit(trade, price):
                closed_this_update.append(
                    self.close_trade(trade_id, trade.stop_price, "stop_loss", ts)
                )
            elif self.risk_manager.check_target_hit(trade, price):
                closed_this_update.append(
                    self.close_trade(trade_id, trade.target_price, "take_profit", ts)  # type: ignore[arg-type]
                )

        self._update_equity_and_peak(ts)
        return [t for t in closed_this_update if t is not None]

    # ── Trade exit ────────────────────────────────────────────────────────────

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: str = "manual",
        timestamp: datetime | None = None,
    ) -> Trade | None:
        """
        Close an open trade at the given price.

        Cash accounting:
            cash += cost_basis + realized_pnl
            = entry_price * qty + sign * (exit_price - entry_price) * qty
            = exit_price * qty   [for long]

        This correctly returns the initial capital plus any profit, or
        initial capital minus any loss.

        Returns None if trade_id is not found in open_trades.
        """
        trade = self.open_trades.pop(trade_id, None)
        if trade is None:
            log.warning("close_trade: trade_id=%s not found in open_trades", trade_id)
            return None

        trade.exit_price = exit_price
        trade.exit_time = timestamp or _now_utc()
        trade.exit_reason = reason
        trade.last_price = exit_price

        if reason == "stop_loss":
            trade.status = "stopped"
        elif reason == "take_profit":
            trade.status = "target_hit"
        else:
            trade.status = "closed"

        # Return capital + PnL to cash
        self.cash += trade.cost_basis() + trade.realized_pnl()

        self.closed_trades.append(trade)

        log.info(
            "Trade closed: id=%s ticker=%s reason=%s exit=%.2f pnl=%.2f",
            trade_id,
            trade.ticker,
            reason,
            exit_price,
            trade.realized_pnl(),
        )
        return trade

    # ── Snapshot and reporting ────────────────────────────────────────────────

    def portfolio_snapshot(self, timestamp: datetime | None = None) -> PortfolioState:
        """
        Produce an immutable PortfolioState Pydantic snapshot.

        Note: PortfolioState.unrealized_pnl is populated here from the engine's
        computed value, not from Position.unrealized_pnl (which is a stub).
        """
        ts = timestamp or _now_utc()
        return PortfolioState(
            snapshot_at=ts,
            starting_capital=self.starting_capital,
            cash=self.cash,
            equity=self.equity,
            open_positions=[t.to_position() for t in self.open_trades.values()],
            closed_positions=[t.to_position() for t in self.closed_trades],
            realized_pnl=self.realized_pnl_total(),
            unrealized_pnl=self.unrealized_pnl_total(),
            trades_today=self.trades_today,
            max_drawdown=(
                (self.peak_equity - self.equity) / self.peak_equity * 100
                if self.peak_equity > 0
                else 0.0
            ),
            peak_equity=self.peak_equity,
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Compute and return a flat metrics dictionary.

        All monetary values are in the same currency as starting_capital.
        Percentages are expressed as fractions (0–1) unless the key ends
        in _pct, in which case they are 0–100.
        """
        eq = self.equity
        winners = [t for t in self.closed_trades if t.realized_pnl() > 0]
        losers = [t for t in self.closed_trades if t.realized_pnl() <= 0]
        n_closed = len(self.closed_trades)

        win_rate = len(winners) / n_closed if n_closed > 0 else None
        avg_winner = (
            sum(t.realized_pnl() for t in winners) / len(winners) if winners else None
        )
        avg_loser = (
            sum(t.realized_pnl() for t in losers) / len(losers) if losers else None
        )

        gross_exp = self.gross_exposure()

        return {
            "starting_capital": self.starting_capital,
            "current_equity": eq,
            "cash": self.cash,
            "total_return_pct": (eq - self.starting_capital) / self.starting_capital * 100,
            "realized_pnl": self.realized_pnl_total(),
            "unrealized_pnl": self.unrealized_pnl_total(),
            "peak_equity": self.peak_equity,
            "max_drawdown_pct": (
                (self.peak_equity - eq) / self.peak_equity * 100 if self.peak_equity > 0 else 0.0
            ),
            "current_drawdown_pct": self.daily_drawdown_pct(),
            "trades_today": self.trades_today,
            "total_trades": len(self.open_trades) + len(self.closed_trades),
            "open_trades_count": len(self.open_trades),
            "closed_trades_count": n_closed,
            "win_rate": win_rate,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "gross_exposure": gross_exp,
            # TODO: subtract short notional when short selling is implemented (Phase 8)
            "net_exposure": gross_exp,
        }

    def export_trade_journal(self) -> list[dict]:
        """
        Return a list of flat dicts for all closed trades.

        Each entry is the output of Trade.to_dict(), which includes
        realized_pnl and cost_basis as computed fields.
        """
        return [t.to_dict() for t in self.closed_trades]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_equity_and_peak(self, timestamp: datetime | None = None) -> None:
        """Update peak_equity and append a point to the equity curve."""
        eq = self.equity
        self.peak_equity = max(self.peak_equity, eq)
        self.equity_curve.append((timestamp or _now_utc(), eq))

    def _maybe_advance_day(self, timestamp: datetime | None = None) -> None:
        """
        Roll the daily counters forward when the calendar date changes.

        On the first call (current_date is None) or whenever the date in
        timestamp is later than current_date:
        - Reset trades_today to 0
        - Snapshot day_start_equity to current equity
        - Notify RiskManager
        """
        ts = timestamp or _now_utc()
        date_now = ts.date() if hasattr(ts, "date") else ts

        if self.current_date is None or date_now > self.current_date:
            self.day_start_equity = self.equity
            self.trades_today = 0
            self.current_date = date_now
            self.risk_manager.reset_daily_state(self.day_start_equity)
            log.debug("Day advanced to %s. day_start_equity=%.2f", date_now, self.day_start_equity)
