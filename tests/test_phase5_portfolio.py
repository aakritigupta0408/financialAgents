"""
tests/test_phase5_portfolio.py — Phase 5: Portfolio and Risk Engine tests.

All tests use synthetic data only. No API calls, no filesystem side-effects
(except test_trade_journal_append_and_load which uses tmp_path).

Run:
    cd /Users/aakritigupta/trading-system && python -m pytest tests/test_phase5_portfolio.py -v
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.portfolio import Portfolio, RiskConfig, TradeJournal, compute_position_size, create_portfolio
from src.portfolio.engine import Portfolio
from src.portfolio.risk import RiskConfig, RiskManager
from src.portfolio.sizing import compute_position_size, required_capital
from src.portfolio.trade import Trade
from schemas.portfolio import PortfolioState

_UTC = timezone.utc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts(day_offset: int = 0) -> datetime:
    """Return a deterministic UTC datetime for testing."""
    return datetime(2026, 1, 2, 10, 0, 0, tzinfo=_UTC) + timedelta(days=day_offset)


def _make_portfolio(**kwargs) -> Portfolio:
    """Create a Portfolio with test-friendly defaults."""
    defaults = dict(
        starting_capital=100_000.0,
        max_trades_per_day=5,
        max_concurrent_positions=3,
        max_daily_drawdown_pct=0.03,
        max_ticker_exposure_pct=0.10,
    )
    defaults.update(kwargs)
    return create_portfolio(**defaults)


# ── 1. Initialisation ──────────────────────────────────────────────────────────

def test_portfolio_init():
    p = _make_portfolio(starting_capital=50_000.0)
    assert p.cash == 50_000.0
    assert p.starting_capital == 50_000.0
    assert p.equity == 50_000.0
    assert p.peak_equity == 50_000.0
    assert p.open_trades == {}
    assert p.closed_trades == []
    assert p.trades_today == 0
    assert p.equity_curve == []


# ── 2. Position sizing: basic ──────────────────────────────────────────────────

def test_position_sizing_basic():
    # equity=100_000, risk_pct=0.01, entry=150, stop=147 → risk_per_share=3
    # raw_size = 1000/3 = 333.33 → floor = 333
    # max_by_exposure = 10_000/150 = 66.66 → floor = 66 (cap binds first)
    size = compute_position_size(
        equity=100_000.0,
        entry_price=150.0,
        stop_price=147.0,
        risk_pct=0.01,
        max_ticker_exposure_pct=0.10,
    )
    expected = max(1.0, math.floor(min(333.33, 10_000 / 150)))
    assert size == expected


def test_position_sizing_confidence_scaling():
    # With confidence=0.5, raw_size is scaled by 0.5
    size_full = compute_position_size(
        equity=100_000.0, entry_price=100.0, stop_price=95.0,
        risk_pct=0.01, confidence_scaling=False,
        max_ticker_exposure_pct=0.50,
    )
    size_half = compute_position_size(
        equity=100_000.0, entry_price=100.0, stop_price=95.0,
        risk_pct=0.01, confidence=0.5, confidence_scaling=True,
        max_ticker_exposure_pct=0.50,
    )
    assert size_half <= size_full
    assert size_half >= 1.0


def test_position_sizing_cash_cap():
    # available_cash=200, entry=100 → at most 2 shares regardless of equity math
    size = compute_position_size(
        equity=100_000.0,
        entry_price=100.0,
        stop_price=95.0,
        risk_pct=0.01,
        max_ticker_exposure_pct=0.50,
        available_cash=200.0,
    )
    assert size <= 2.0


# ── 3. Open trade succeeds ─────────────────────────────────────────────────────

def test_open_trade_succeeds():
    p = _make_portfolio()
    trade = p.open_trade(
        "AAPL", side="long", entry_price=150.0, stop_price=147.0,
        target_price=156.0, timestamp=_ts(),
    )
    assert trade is not None
    assert trade.ticker == "AAPL"
    assert trade.side == "long"
    assert trade.entry_price == 150.0
    assert trade.stop_price == 147.0
    assert trade.target_price == 156.0
    assert trade.status == "open"
    assert trade.trade_id in p.open_trades
    assert p.trades_today == 1


# ── 4. Opening a trade updates equity ─────────────────────────────────────────

def test_open_trade_updates_equity():
    p = _make_portfolio()
    equity_before = p.equity
    trade = p.open_trade(
        "AAPL", side="long", entry_price=150.0, stop_price=147.0,
        timestamp=_ts(),
    )
    assert trade is not None
    # Equity should remain the same immediately after open (no slippage)
    assert abs(p.equity - equity_before) < 0.01
    # Cash has been reduced
    cost = trade.cost_basis()
    assert abs(p.cash - (equity_before - cost)) < 0.01


# ── 5. Rejected: max concurrent positions ─────────────────────────────────────

def test_open_trade_rejected_max_positions():
    p = _make_portfolio(max_concurrent_positions=2)
    ts = _ts()
    p.open_trade("AAPL", "long", 150.0, 147.0, timestamp=ts)
    p.open_trade("MSFT", "long", 380.0, 375.0, timestamp=ts)
    # Third trade must be rejected
    trade3 = p.open_trade("NVDA", "long", 500.0, 490.0, timestamp=ts)
    assert trade3 is None
    assert len(p.open_trades) == 2


# ── 6. Rejected: max trades per day ───────────────────────────────────────────

def test_open_trade_rejected_max_daily():
    p = _make_portfolio(
        max_trades_per_day=2,
        max_concurrent_positions=10,
        max_ticker_exposure_pct=0.50,
        max_portfolio_exposure_pct=0.99,
    )
    ts = _ts()
    p.open_trade("AAPL", "long", 150.0, 145.0, timestamp=ts)
    p.open_trade("MSFT", "long", 380.0, 370.0, timestamp=ts)
    trade3 = p.open_trade("NVDA", "long", 500.0, 490.0, timestamp=ts)
    assert trade3 is None
    assert p.trades_today == 2


# ── 7. update_positions: mark-to-market ───────────────────────────────────────

def test_update_positions_mark_to_market():
    p = _make_portfolio()
    trade = p.open_trade("AAPL", "long", 150.0, 147.0, target_price=160.0, timestamp=_ts())
    assert trade is not None

    p.update_positions({"AAPL": 153.0}, timestamp=_ts())
    assert trade.last_price == 153.0
    # Unrealized PnL should be positive
    upnl = trade.unrealized_pnl(153.0)
    assert upnl > 0


# ── 8. Stop loss triggered ────────────────────────────────────────────────────

def test_stop_loss_triggered():
    p = _make_portfolio()
    trade = p.open_trade("AAPL", "long", 150.0, 147.0, target_price=160.0, timestamp=_ts())
    assert trade is not None
    trade_id = trade.trade_id

    closed = p.update_positions({"AAPL": 146.0}, timestamp=_ts())
    assert len(closed) == 1
    assert closed[0].trade_id == trade_id
    assert closed[0].status == "stopped"
    assert closed[0].exit_price == 147.0
    assert closed[0].exit_reason == "stop_loss"
    assert trade_id not in p.open_trades


# ── 9. Take-profit triggered ──────────────────────────────────────────────────

def test_take_profit_triggered():
    p = _make_portfolio()
    trade = p.open_trade("AAPL", "long", 150.0, 147.0, target_price=156.0, timestamp=_ts())
    assert trade is not None
    trade_id = trade.trade_id

    closed = p.update_positions({"AAPL": 157.0}, timestamp=_ts())
    assert len(closed) == 1
    assert closed[0].status == "target_hit"
    assert closed[0].exit_price == 156.0
    assert closed[0].exit_reason == "take_profit"
    assert trade_id not in p.open_trades


# ── 10. Manual close ──────────────────────────────────────────────────────────

def test_close_trade_manual():
    p = _make_portfolio()
    trade = p.open_trade("AAPL", "long", 150.0, 147.0, timestamp=_ts())
    assert trade is not None

    closed = p.close_trade(trade.trade_id, exit_price=154.0, reason="manual", timestamp=_ts())
    assert closed is not None
    assert closed.status == "closed"
    assert closed.exit_price == 154.0
    assert closed.exit_reason == "manual"
    assert trade.trade_id not in p.open_trades
    assert closed in p.closed_trades


# ── 11. Realized PnL — winner ─────────────────────────────────────────────────

def test_realized_pnl_positive_winner():
    p = _make_portfolio()
    trade = p.open_trade("AAPL", "long", 150.0, 147.0, quantity=10.0, timestamp=_ts())
    assert trade is not None
    p.close_trade(trade.trade_id, exit_price=155.0, reason="manual", timestamp=_ts())

    assert trade.realized_pnl() == pytest.approx(50.0)  # (155-150) * 10
    assert p.realized_pnl_total() == pytest.approx(50.0)


# ── 12. Realized PnL — loser ──────────────────────────────────────────────────

def test_realized_pnl_negative_loser():
    p = _make_portfolio()
    trade = p.open_trade("AAPL", "long", 150.0, 147.0, quantity=10.0, timestamp=_ts())
    assert trade is not None
    p.close_trade(trade.trade_id, exit_price=147.0, reason="stop_loss", timestamp=_ts())

    assert trade.realized_pnl() == pytest.approx(-30.0)  # (147-150) * 10
    assert p.realized_pnl_total() == pytest.approx(-30.0)


# ── 13. Daily drawdown limit blocks new trades ────────────────────────────────

def test_daily_drawdown_limit():
    """
    Set max_daily_drawdown_pct=0.01 (1%).
    Open a long at 200.0 with qty=100 (cost_basis = 20_000).
    Update price to 190.0 → unrealized loss = -1_000 on 100_000 equity
    → drawdown = 1_000/100_000 = 1% which equals the limit, so it should be breached.
    Then assert that a new trade is rejected.
    """
    # max_ticker_exposure_pct=0.50 allows qty=100@200 ($20k = 20% of equity)
    p = _make_portfolio(
        starting_capital=100_000.0,
        max_daily_drawdown_pct=0.01,
        max_concurrent_positions=10,
        max_ticker_exposure_pct=0.50,
    )
    ts = _ts()
    trade = p.open_trade(
        "AAPL", "long", entry_price=200.0, stop_price=150.0,
        quantity=100, timestamp=ts,
    )
    assert trade is not None

    # Price drops $10.50/share → loss = 100 * 10.50 = $1,050 > 1% of $100k
    p.update_positions({"AAPL": 189.50}, timestamp=ts)

    # Verify drawdown is now >= 1%
    assert p.daily_drawdown_pct() >= 0.01

    # Attempt to open a new trade — must be rejected
    new_trade = p.open_trade(
        "MSFT", "long", entry_price=380.0, stop_price=370.0, timestamp=ts,
    )
    assert new_trade is None


# ── 14. Portfolio snapshot matches Pydantic schema ────────────────────────────

def test_portfolio_snapshot_schema():
    p = _make_portfolio()
    p.open_trade("AAPL", "long", 150.0, 147.0, target_price=156.0, timestamp=_ts())
    snap = p.portfolio_snapshot(timestamp=_ts())

    assert isinstance(snap, PortfolioState)
    assert snap.cash < snap.starting_capital  # cash reduced
    assert len(snap.open_positions) == 1
    assert snap.open_positions[0].ticker == "AAPL"
    assert snap.equity == pytest.approx(p.equity)


# ── 15. Win rate in metrics ───────────────────────────────────────────────────

def test_metrics_win_rate():
    p = _make_portfolio()
    ts = _ts()

    # Winner: +5 per share
    t1 = p.open_trade("AAPL", "long", 100.0, 95.0, quantity=10, timestamp=ts)
    p.close_trade(t1.trade_id, 105.0, "manual", ts)

    # Loser: -3 per share
    t2 = p.open_trade("MSFT", "long", 200.0, 195.0, quantity=10, timestamp=ts)
    p.close_trade(t2.trade_id, 197.0, "manual", ts)

    metrics = p.get_metrics()
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["avg_winner"] == pytest.approx(50.0)
    assert metrics["avg_loser"] == pytest.approx(-30.0)
    assert metrics["closed_trades_count"] == 2


# ── 16. Trade journal append and load ─────────────────────────────────────────

def test_trade_journal_append_and_load(tmp_path: Path):
    journal_path = tmp_path / "test_trades.jsonl"
    journal = TradeJournal(path=journal_path)

    trade = Trade(
        trade_id="abc12345",
        ticker="AAPL",
        side="long",
        quantity=10.0,
        entry_price=150.0,
        stop_price=147.0,
        entry_time=_ts(),
        target_price=156.0,
        exit_price=154.0,
        exit_time=_ts(),
        exit_reason="manual",
        status="closed",
    )

    journal.append(trade)

    records = journal.load_all()
    assert len(records) == 1
    assert records[0]["trade_id"] == "abc12345"
    assert records[0]["ticker"] == "AAPL"
    assert records[0]["realized_pnl"] == pytest.approx(40.0)  # (154-150)*10


# ── 17. Equity curve populated ────────────────────────────────────────────────

def test_equity_curve_populated():
    p = _make_portfolio()
    ts = _ts()
    p.open_trade("AAPL", "long", 150.0, 147.0, target_price=160.0, timestamp=ts)
    p.update_positions({"AAPL": 152.0}, timestamp=ts + timedelta(hours=1))
    p.update_positions({"AAPL": 154.0}, timestamp=ts + timedelta(hours=2))

    assert len(p.equity_curve) >= 2
    # Equity curve entries are (datetime, float) tuples
    for dt, eq in p.equity_curve:
        assert isinstance(dt, datetime)
        assert isinstance(eq, float)
        assert eq > 0


# ── 18. Day advance resets daily trade count ──────────────────────────────────

def test_day_advance_resets_daily_count():
    p = _make_portfolio(max_trades_per_day=2, max_concurrent_positions=10,
                        max_ticker_exposure_pct=0.50)
    day1 = _ts(0)
    day2 = _ts(1)

    p.open_trade("AAPL", "long", 150.0, 145.0, quantity=1, timestamp=day1)
    p.open_trade("MSFT", "long", 380.0, 370.0, quantity=1, timestamp=day1)
    assert p.trades_today == 2

    # Day 2: counter must reset
    p.open_trade("NVDA", "long", 500.0, 490.0, quantity=1, timestamp=day2)
    assert p.trades_today == 1
    assert p.current_date == day2.date()
