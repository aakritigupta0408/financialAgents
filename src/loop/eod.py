"""End-of-day processing — retrain meta-model after market close."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings import LOG_DIR
from src.adaptive.loop import ImprovementCycleResult

if TYPE_CHECKING:
    from src.loop.result import LiveLoopResult

log = logging.getLogger(__name__)


def end_of_day_process(result: "LiveLoopResult", save_model: bool = True) -> ImprovementCycleResult:
    """Retrain meta-model on today's trades after market close."""
    from src.adaptive.loop import run_improvement_cycle
    cycle = run_improvement_cycle(result, retrain_model=save_model, save_context=True)
    return cycle


def run_eod_cycle(
    session_date: date | None = None,
    tickers: list[str] | None = None,
    save_model: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Standalone EOD cycle — loads stored state and decisions from today,
    runs adaptive learning, and persists updated context.

    Steps:
    1. Load LoopState for each ticker
    2. Build synthetic BacktestResult from closed trades
    3. Run improvement cycle (adaptive context + optional meta-model retrain)
    4. Persist updated context
    5. Return summary dict

    Parameters
    ----------
    session_date : date to process (defaults to today)
    tickers      : tickers to process (defaults to config SCAN_TICKERS)
    save_model   : whether to save retrained meta-model
    verbose      : print progress
    """
    from config.settings import SCAN_TICKERS
    from src.loop.state import LoopState

    session_date = session_date or date.today()
    tickers = tickers or SCAN_TICKERS

    results = []
    for ticker in tickers:
        state = LoopState.load(ticker, session_date)
        if not state.closed_trades:
            log.info("eod_cycle.no_trades: %s %s", ticker, session_date)
            if verbose:
                print(f"  {ticker}: no closed trades today — skipping")
            continue

        # Build a minimal LiveLoopResult from stored state
        loop_result = _state_to_loop_result(state)

        if verbose:
            print(f"  {ticker}: {len(state.closed_trades)} closed trades, "
                  f"daily_pnl=${state.daily_pnl:+,.2f}")

        try:
            cycle = end_of_day_process(loop_result, save_model=save_model)
            result_entry = {
                "ticker": ticker,
                "session_date": session_date.isoformat(),
                "n_trades": len(state.closed_trades),
                "daily_pnl": state.daily_pnl,
                "model_retrained": cycle.model_retrained,
                "model_improved": cycle.model_improved,
                "status": "ok",
            }
            if verbose:
                print(f"    → retrained={cycle.model_retrained} improved={cycle.model_improved}")
        except Exception as e:
            log.warning("eod_cycle.error %s: %s", ticker, e)
            result_entry = {
                "ticker": ticker,
                "session_date": session_date.isoformat(),
                "status": "error",
                "error": str(e),
            }

        results.append(result_entry)

    # Persist EOD summary
    _save_eod_summary(session_date, results)

    return {
        "session_date": session_date.isoformat(),
        "tickers_processed": len(results),
        "results": results,
    }


def _state_to_loop_result(state: "LoopState") -> "LiveLoopResult":
    """Convert a LoopState into a minimal LiveLoopResult for adaptive retraining."""
    from src.loop.result import LiveLoopResult
    from config.settings import STARTING_CAPITAL

    n_trades = len(state.closed_trades)
    total_return = (
        (state.equity - state.day_start_equity) / state.day_start_equity * 100
        if state.day_start_equity > 0 else 0.0
    )

    # Build a minimal trade_journal from closed trades
    trade_journal = [
        {
            "trade_id": ct.get("trade_id", ""),
            "ticker": ct.get("ticker", state.ticker),
            "side": ct.get("side", "long"),
            "entry_price": ct.get("entry_price", 0.0),
            "exit_price": ct.get("exit_price", 0.0),
            "position_size": ct.get("position_size", 0.0),
            "realized_pnl": ct.get("realized_pnl", 0.0),
            "close_reason": ct.get("close_reason", "unknown"),
            "meta_features": {},
        }
        for ct in state.closed_trades
    ]

    now = datetime.now(timezone.utc)
    return LiveLoopResult(
        ticker=state.ticker,
        timeframe="1d",
        n_bars_processed=state.iteration + 1,
        starting_capital=state.day_start_equity or STARTING_CAPITAL,
        final_equity=state.equity,
        trade_journal=trade_journal,
        decision_log=[],
        equity_curve=[(now, state.day_start_equity), (now, state.equity)],
    )


def _save_eod_summary(session_date: date, results: list[dict]) -> None:
    eod_dir = LOG_DIR / "eod"
    eod_dir.mkdir(parents=True, exist_ok=True)
    path = eod_dir / f"eod_summary_{session_date.isoformat()}.json"
    try:
        with path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("eod_cycle.summary_saved: %s", path)
    except Exception as e:
        log.warning("eod_cycle.save_failed: %s", e)
