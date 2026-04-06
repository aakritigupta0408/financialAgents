"""
src.portfolio.risk — Risk configuration and pre-trade risk checks.

RiskConfig holds all risk limits (mirrors config/settings.py defaults but is
independently configurable per Portfolio instance).

RiskManager enforces those limits before any trade is opened and detects
stop/target hits on every price update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.portfolio.engine import Portfolio
    from src.portfolio.trade import Trade


@dataclass
class RiskConfig:
    """
    Risk parameters for a single Portfolio instance.

    Defaults match config/settings.py values so the engine works out-of-the-box.

    Attributes
    ----------
    starting_capital:
        Initial cash balance.
    risk_per_trade_pct:
        Fraction of equity risked per trade (e.g. 0.01 = 1%).
    max_trades_per_day:
        Maximum number of new trades that may be opened on a single calendar day.
    max_concurrent_positions:
        Maximum number of simultaneously open trades.
    max_daily_drawdown_pct:
        If today's equity loss reaches this fraction of day-start equity, no
        new trades are allowed for the rest of that day.
    max_ticker_exposure_pct:
        Maximum fraction of equity allocated to any single ticker (notional).
    max_portfolio_exposure_pct:
        Maximum fraction of equity deployed across all open positions.
    """

    starting_capital: float = 100_000.0
    risk_per_trade_pct: float = 0.01
    max_trades_per_day: int = 5
    max_concurrent_positions: int = 3
    max_daily_drawdown_pct: float = 0.03
    max_ticker_exposure_pct: float = 0.10
    max_portfolio_exposure_pct: float = 0.90
    # TODO: max_sector_exposure_pct — requires a ticker→sector mapping.
    #       Planned for Phase 8 when a sector reference table is available.


class RiskManager:
    """
    Stateless (except for reset_daily_state) enforcer of risk limits.

    All check methods are deterministic given Portfolio state.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        # Mirrors the day_start_equity held in Portfolio; kept here so
        # reset_daily_state has a local copy for any future per-manager logic.
        self._day_start_equity: float = config.starting_capital

    # ── Pre-trade gate ────────────────────────────────────────────────────────

    def can_open_trade(
        self,
        portfolio: "Portfolio",
        ticker: str,
        required_capital: float,
    ) -> tuple[bool, str]:
        """
        Run ordered pre-trade checks. Returns on the first failure.

        Checks (in order):
        1. Daily trade count
        2. Concurrent position count
        3. Daily drawdown limit
        4. Per-ticker exposure cap
        5. Portfolio-level gross exposure cap
        6. Sufficient cash

        Returns
        -------
        (True, "ok") if all checks pass.
        (False, reason_string) where reason_string is the name of the
        first failed check.
        """
        cfg = self.config

        # 1. Daily trade limit
        if portfolio.trades_today >= cfg.max_trades_per_day:
            return False, "max_trades_per_day reached"

        # 2. Concurrent position limit
        if len(portfolio.open_trades) >= cfg.max_concurrent_positions:
            return False, "max_concurrent_positions reached"

        # 3. Daily drawdown limit — stop trading if today's loss is too large
        if portfolio.daily_drawdown_pct() >= cfg.max_daily_drawdown_pct:
            return False, "daily_drawdown_limit_breached"

        # 4. Per-ticker exposure cap
        equity = portfolio.equity
        new_ticker_exposure = portfolio.ticker_exposure_pct(ticker) + (
            required_capital / equity if equity > 0 else 0.0
        )
        if new_ticker_exposure > cfg.max_ticker_exposure_pct:
            return False, "max_ticker_exposure exceeded"

        # 5. Portfolio gross exposure cap
        if portfolio.gross_exposure() + required_capital > cfg.max_portfolio_exposure_pct * equity:
            return False, "max_portfolio_exposure exceeded"

        # 6. Liquidity check
        if portfolio.cash < required_capital:
            return False, "insufficient_cash"

        return True, "ok"

    # ── Per-trade exit triggers ───────────────────────────────────────────────

    def check_stop_hit(self, trade: "Trade", current_price: float) -> bool:
        """
        True when current_price has reached or crossed the stop level.

        Long:  stop is below entry — triggered when price falls to stop.
        Short: stop is above entry — triggered when price rises to stop.
        """
        if trade.side == "long":
            return current_price <= trade.stop_price
        else:
            return current_price >= trade.stop_price

    def check_target_hit(self, trade: "Trade", current_price: float) -> bool:
        """
        True when current_price has reached or crossed the target level.

        Returns False immediately if no target was set.

        Long:  target is above entry — triggered when price rises to target.
        Short: target is below entry — triggered when price falls to target.
        """
        if trade.target_price is None:
            return False
        if trade.side == "long":
            return current_price >= trade.target_price
        else:
            return current_price <= trade.target_price

    # ── Daily state reset ─────────────────────────────────────────────────────

    def reset_daily_state(self, current_equity: float) -> None:
        """
        Call at the start of each new trading day (or on first trade).

        Saves current_equity as the reference for today's drawdown calculation.
        """
        self._day_start_equity = current_equity
