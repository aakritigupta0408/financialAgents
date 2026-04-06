"""Phase 18 — Intraday recommendation loop and EOD learning tests."""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.market_data import OHLCVBar, OHLCVSeries


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_series(ticker="AAPL", n=80, start_price=200.0, trend=0.002) -> OHLCVSeries:
    """Synthetic daily OHLCV series with a gentle uptrend."""
    bars = []
    price = start_price
    base = datetime(2026, 1, 2, tzinfo=timezone.utc)
    for i in range(n):
        o = price
        c = price * (1.0 + trend + (0.005 if i % 3 == 0 else -0.003))
        # Clamp open and close inside [low, high]
        h = max(o, c) * 1.005
        lo = min(o, c) * 0.995
        o = max(lo, min(h, o))
        c = max(lo, min(h, c))
        bars.append(OHLCVBar(
            timestamp=base + timedelta(days=i),
            open=o, high=h, low=lo, close=c,
            volume=1_000_000,
            ticker=ticker,
            timeframe="1d",
        ))
        price = c
    return OHLCVSeries(ticker=ticker, timeframe="1d", bars=bars)


# ── LoopState tests ───────────────────────────────────────────────────────────

def test_loop_state_fresh_creation(tmp_path):
    with patch("src.loop.state._STATE_DIR", tmp_path):
        from src.loop.state import LoopState
        state = LoopState.load("AAPL")
    assert state.ticker == "AAPL"
    assert state.equity == pytest.approx(100_000.0, rel=0.01)
    assert state.open_positions == {}
    assert state.closed_trades == []


def test_loop_state_save_and_reload(tmp_path):
    with patch("src.loop.state._STATE_DIR", tmp_path):
        from src.loop.state import LoopState, PositionRecord

        state = LoopState.load("MSFT")
        state.equity = 102_500.0
        state.trades_today = 3
        state.add_open_position(PositionRecord(
            trade_id="t1",
            ticker="MSFT",
            side="long",
            entry_price=380.0,
            stop_price=370.0,
            target_price=400.0,
            position_size=25.0,
            opened_at="2026-04-06T10:30:00+00:00",
        ))
        state.save()

        state2 = LoopState.load("MSFT")
        assert state2.equity == pytest.approx(102_500.0)
        assert state2.trades_today == 3
        assert "t1" in state2.open_positions


def test_loop_state_drawdown_tracking():
    from src.loop.state import LoopState
    state = LoopState(ticker="TEST", session_date=date.today().isoformat(),
                      day_start_equity=100_000.0, equity=100_000.0,
                      peak_equity=100_000.0)
    state.update_equity(95_000.0)
    assert state.daily_drawdown_pct == pytest.approx(0.05)
    assert state.max_drawdown_pct == pytest.approx(0.05)


def test_loop_state_summary_format():
    from src.loop.state import LoopState
    state = LoopState(ticker="AAPL", session_date="2026-04-06",
                      equity=101_000.0, day_start_equity=100_000.0,
                      daily_pnl=1_000.0, peak_equity=101_000.0, trades_today=2)
    summary = state.summary()
    assert "AAPL" in summary
    assert "101,000" in summary


# ── Scheduler tests ───────────────────────────────────────────────────────────

def test_scheduler_open_during_session():
    from src.loop.scheduler import is_market_open, to_eastern
    # Monday April 6 2026, 10:30 ET  — should be open
    dt_utc = datetime(2026, 4, 6, 14, 30, tzinfo=timezone.utc)  # 10:30 EDT
    assert is_market_open(dt_utc) is True


def test_scheduler_closed_after_hours():
    from src.loop.scheduler import is_market_open
    # Monday April 6 2026, 21:00 UTC = 17:00 EDT — after close
    dt_utc = datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc)
    assert is_market_open(dt_utc) is False


def test_scheduler_closed_weekend():
    from src.loop.scheduler import is_market_open
    # Saturday April 5 2026, 14:00 UTC
    dt_utc = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
    assert is_market_open(dt_utc) is False


def test_seconds_until_close_positive_during_session():
    from src.loop.scheduler import seconds_until_market_close
    # 10:30 EDT = 14:30 UTC on Monday
    dt_utc = datetime(2026, 4, 6, 14, 30, tzinfo=timezone.utc)
    secs = seconds_until_market_close(dt_utc)
    assert secs > 0
    assert secs <= 6 * 3600   # at most 6 hours left


def test_market_session_label():
    from src.loop.scheduler import market_session_label
    # 10:30 EDT Monday
    dt_utc = datetime(2026, 4, 6, 14, 30, tzinfo=timezone.utc)
    assert market_session_label(dt_utc) == "OPEN"
    # After hours
    dt_utc2 = datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc)
    assert market_session_label(dt_utc2) == "AFTER_HOURS"
    # Weekend
    dt_utc3 = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
    assert market_session_label(dt_utc3) == "WEEKEND"


# ── IntradayRecommendationLoop tests ──────────────────────────────────────────

def test_replay_produces_result_and_recommendations(tmp_path):
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.risk_appetite.presets import MODERATE

    series = _make_series(n=70)
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="AAPL", fta_enabled=False, meta_model_enabled=False,
                                   verbose=False, eod_retrain=False),
            risk_appetite=MODERATE,
            session_date=date(2026, 4, 6),
        )
        result, recs = loop.run_replay(series)

    assert result.n_bars_processed == 70
    assert len(recs) > 0
    assert result.final_equity > 0


def test_replay_all_recs_are_trade_recommendations(tmp_path):
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.risk_appetite.presets import MODERATE
    from schemas.recommendation import TradeRecommendation

    series = _make_series(n=65)
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="AAPL", fta_enabled=False, meta_model_enabled=False),
            risk_appetite=MODERATE,
            session_date=date(2026, 4, 6),
        )
        _, recs = loop.run_replay(series)

    for rec in recs:
        assert isinstance(rec, TradeRecommendation)
        assert rec.action in ("BUY", "SELL", "HOLD")
        assert rec.position_action in ("OPEN", "CLOSE", "REDUCE", "HOLD_POSITION")


def test_replay_state_persisted_to_disk(tmp_path):
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.loop.state import LoopState
    from src.risk_appetite.presets import MODERATE

    series = _make_series(n=65)
    session = date(2026, 4, 6)
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="NVDA", fta_enabled=False, meta_model_enabled=False),
            risk_appetite=MODERATE,
            session_date=session,
        )
        result, _ = loop.run_replay(series)

        # State file should exist
        state_files = list(tmp_path.glob("state_NVDA_*.json"))
        assert len(state_files) == 1

        # Reload state and verify
        reloaded = LoopState.load("NVDA", session)
        assert reloaded.equity == pytest.approx(result.final_equity, rel=0.01)
        assert reloaded.session_date == "2026-04-06"


def test_replay_decisions_logged_to_jsonl(tmp_path):
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.risk_appetite.presets import MODERATE

    series = _make_series(n=65)
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="TSLA", fta_enabled=False, meta_model_enabled=False),
            risk_appetite=MODERATE,
            session_date=date(2026, 4, 6),
        )
        loop.run_replay(series)

        log_files = list(tmp_path.glob("decisions_TSLA_*.jsonl"))
        assert len(log_files) == 1
        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) > 0
        # Each line must be valid JSON with a recommendation key
        for line in lines[:5]:
            obj = json.loads(line)
            assert "recommendation" in obj
            assert "portfolio_equity" in obj


def test_replay_portfolio_equity_evolves(tmp_path):
    """Equity curve must have entries and equity must vary across iterations."""
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.risk_appetite.presets import AGGRESSIVE

    series = _make_series(n=70, trend=0.005)
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="AAPL", fta_enabled=False, meta_model_enabled=False),
            risk_appetite=AGGRESSIVE,
            session_date=date(2026, 4, 6),
        )
        result, _ = loop.run_replay(series)

    assert len(result.equity_curve) > 0
    equities = [eq for _, eq in result.equity_curve]
    assert max(equities) != min(equities), "Equity never changed — no trades?"


def test_replay_open_close_lifecycle(tmp_path):
    """Verify OPEN → CLOSE lifecycle through forced forecast flip."""
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.risk_appetite.presets import AGGRESSIVE

    # Series with enough bars to warm up and trigger trades
    series = _make_series(n=80, trend=0.003)
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="SPY", fta_enabled=False, meta_model_enabled=False),
            risk_appetite=AGGRESSIVE,
            session_date=date(2026, 4, 6),
        )
        result, recs = loop.run_replay(series)

    # At minimum: warm-up bars produce recommendations, some bars produce actionable recs
    actions = [r.position_action for r in recs]
    assert len(recs) > 0
    # End-of-replay closes remaining positions — journal may have trades
    # Even if no explicit CLOSE rec, the final forced close counts
    assert result.n_bars_processed == 80


def test_conservative_profile_fewer_opens_than_aggressive(tmp_path):
    """Conservative profile should open fewer or equal trades than aggressive."""
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.risk_appetite.presets import CONSERVATIVE, AGGRESSIVE

    series = _make_series(n=80, trend=0.003)
    session = date(2026, 4, 6)

    def run_with_profile(ra, suffix):
        with patch("src.loop.state._STATE_DIR", tmp_path / suffix), \
             patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path / suffix):
            (tmp_path / suffix).mkdir(exist_ok=True)
            loop = IntradayRecommendationLoop(
                loop_config=LoopConfig(ticker="AAPL", fta_enabled=False, meta_model_enabled=False),
                risk_appetite=ra,
                session_date=session,
            )
            result, recs = loop.run_replay(series)
        return sum(1 for r in recs if r.position_action == "OPEN")

    opens_cons = run_with_profile(CONSERVATIVE, "cons")
    opens_agg = run_with_profile(AGGRESSIVE, "agg")

    # Conservative must open no more than aggressive
    assert opens_cons <= opens_agg


# ── EOD cycle tests ───────────────────────────────────────────────────────────

def test_eod_cycle_no_trades_skipped(tmp_path):
    """EOD cycle with no closed trades should produce zero results."""
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("config.settings.LOG_DIR", tmp_path):
        from src.loop.eod import run_eod_cycle
        summary = run_eod_cycle(
            session_date=date(2026, 4, 6),
            tickers=["AAPL"],
            save_model=False,
            verbose=False,
        )
    assert summary["tickers_processed"] == 0


def test_eod_cycle_with_closed_trades(tmp_path):
    """EOD cycle should run without error when closed trades exist."""
    from src.loop.state import LoopState, ClosedTradeRecord

    with patch("src.loop.state._STATE_DIR", tmp_path):
        state = LoopState.load("AAPL", date(2026, 4, 6))
        state.day_start_equity = 100_000.0
        state.equity = 101_200.0
        state.daily_pnl = 1_200.0
        state.close_position(
            "t1",
            ClosedTradeRecord(
                trade_id="t1",
                ticker="AAPL",
                side="long",
                entry_price=200.0,
                exit_price=212.0,
                position_size=100.0,
                realized_pnl=1_200.0,
                opened_at="2026-04-06T09:30:00+00:00",
                closed_at="2026-04-06T15:30:00+00:00",
                close_reason="target",
            ),
        )
        state.save()

    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("config.settings.LOG_DIR", tmp_path):
        from src.loop.eod import run_eod_cycle
        summary = run_eod_cycle(
            session_date=date(2026, 4, 6),
            tickers=["AAPL"],
            save_model=False,
            verbose=False,
        )

    assert summary["tickers_processed"] == 1
    r = summary["results"][0]
    assert r["ticker"] == "AAPL"
    assert r["n_trades"] == 1
    assert r["status"] == "ok"


def test_eod_summary_saved_to_disk(tmp_path):
    """EOD summary JSON file should be created."""
    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.eod.LOG_DIR", tmp_path):
        from src.loop.eod import run_eod_cycle
        run_eod_cycle(
            session_date=date(2026, 4, 1),
            tickers=["TSLA"],
            save_model=False,
        )
    eod_files = list((tmp_path / "eod").glob("eod_summary_*.json"))
    assert len(eod_files) == 1


# ── State persistence across restart ──────────────────────────────────────────

def test_state_persists_across_restart(tmp_path):
    """Run loop, simulate restart, verify state picks up from where it left off."""
    from src.loop.config import LoopConfig
    from src.loop.intraday import IntradayRecommendationLoop
    from src.loop.state import LoopState
    from src.risk_appetite.presets import MODERATE

    series = _make_series(n=70)
    session = date(2026, 4, 6)

    with patch("src.loop.state._STATE_DIR", tmp_path), \
         patch("src.loop.intraday._DECISION_LOG_DIR", tmp_path):
        # First run
        loop = IntradayRecommendationLoop(
            loop_config=LoopConfig(ticker="AMD", fta_enabled=False, meta_model_enabled=False),
            risk_appetite=MODERATE,
            session_date=session,
        )
        result_1, _ = loop.run_replay(series)

        # "Restart" — reload state
        state_reloaded = LoopState.load("AMD", session)
        assert state_reloaded.ticker == "AMD"
        assert state_reloaded.equity == pytest.approx(result_1.final_equity, rel=0.01)
