"""
src.portfolio — Paper portfolio and risk engine.

Public API
----------
Portfolio           : mutable engine; the primary entry-point.
RiskConfig          : dataclass of risk parameters.
RiskManager         : pre-trade and per-tick risk enforcement.
TradeJournal        : persistent JSONL trade log.
compute_position_size : risk-based position sizer.
create_portfolio    : convenience factory.

Quick start
-----------
    from src.portfolio import create_portfolio

    p = create_portfolio(starting_capital=100_000)
    trade = p.open_trade("AAPL", side="long", entry_price=150.0,
                          stop_price=147.0, target_price=156.0)
    p.update_positions({"AAPL": 153.0})
    print(p.get_metrics())
"""

from config.settings import STARTING_CAPITAL
from src.portfolio.engine import Portfolio
from src.portfolio.journal import TradeJournal
from src.portfolio.risk import RiskConfig, RiskManager
from src.portfolio.sizing import compute_position_size

__all__ = [
    "Portfolio",
    "RiskConfig",
    "RiskManager",
    "TradeJournal",
    "compute_position_size",
    "create_portfolio",
]


def create_portfolio(starting_capital: float | None = None, **kwargs) -> Portfolio:
    """
    Convenience factory for creating a configured Portfolio.

    Parameters
    ----------
    starting_capital:
        Override for the default STARTING_CAPITAL from settings.
    **kwargs:
        Any other RiskConfig field, e.g. max_trades_per_day=10.

    Returns
    -------
    Portfolio
        A fresh, initialised Portfolio instance.
    """
    config = RiskConfig(
        starting_capital=starting_capital if starting_capital is not None else STARTING_CAPITAL,
        **kwargs,
    )
    return Portfolio(config=config)
