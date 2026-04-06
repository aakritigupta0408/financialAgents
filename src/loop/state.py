"""LoopState — persistent portfolio state across loop iterations and restarts.

Stored as a single JSON file:
    data/logs/loop_state_{ticker}_{date}.json

The state captures enough to reconstruct the paper portfolio on restart.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import LOG_DIR

log = logging.getLogger(__name__)

_STATE_DIR = LOG_DIR / "loop_state"


def _state_path(ticker: str, session_date: date) -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / f"state_{ticker.upper()}_{session_date.isoformat()}.json"


@dataclass
class PositionRecord:
    """Lightweight serialisable position record."""
    trade_id: str
    ticker: str
    side: str
    entry_price: float
    stop_price: float
    target_price: float | None
    position_size: float      # shares
    opened_at: str            # ISO datetime string
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTradeRecord:
    trade_id: str
    ticker: str
    side: str
    entry_price: float
    exit_price: float
    position_size: float
    realized_pnl: float
    opened_at: str
    closed_at: str
    close_reason: str


@dataclass
class LoopState:
    """
    Mutable session state — persists across loop iterations.

    Load with LoopState.load() / persist with .save().
    """

    ticker: str
    session_date: str            # ISO date
    cash: float = 100_000.0
    equity: float = 100_000.0
    day_start_equity: float = 100_000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_equity: float = 100_000.0
    max_drawdown_pct: float = 0.0
    trades_today: int = 0
    decisions_today: int = 0
    last_tick_at: str | None = None
    iteration: int = 0
    open_positions: dict[str, Any] = field(default_factory=dict)
    closed_trades: list[Any] = field(default_factory=list)
    # key: trade_id → recommendation dict snapshot
    recommendation_log: list[dict] = field(default_factory=list)

    # ── Derived helpers ────────────────────────────────────────────────────

    @property
    def daily_drawdown_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)

    def update_equity(self, equity: float) -> None:
        self.equity = equity
        self.daily_pnl = equity - self.day_start_equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

    def add_open_position(self, pos: PositionRecord) -> None:
        self.open_positions[pos.trade_id] = asdict(pos)

    def close_position(self, trade_id: str, close_rec: ClosedTradeRecord) -> None:
        self.open_positions.pop(trade_id, None)
        self.closed_trades.append(asdict(close_rec))
        self.realized_pnl += close_rec.realized_pnl

    def log_recommendation(self, rec_dict: dict) -> None:
        self.recommendation_log.append(rec_dict)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self) -> Path:
        path = _state_path(self.ticker, date.fromisoformat(self.session_date))
        try:
            with path.open("w") as f:
                json.dump(asdict(self), f, indent=2, default=str)
            log.debug("loop_state.saved: %s", path)
        except Exception as e:
            log.warning("loop_state.save_failed: %s", e)
        return path

    @classmethod
    def load(cls, ticker: str, session_date: date | None = None) -> "LoopState":
        """Load state for today (or given date). Returns fresh state if not found."""
        d = session_date or date.today()
        path = _state_path(ticker, d)
        if path.exists():
            try:
                with path.open() as f:
                    data = json.load(f)
                log.info("loop_state.loaded: %s (%d positions, %d closed)",
                         path, len(data.get("open_positions", {})), len(data.get("closed_trades", [])))
                return cls(**data)
            except Exception as e:
                log.warning("loop_state.load_failed: %s — starting fresh", e)
        log.info("loop_state.new_session: %s %s", ticker, d)
        from config.settings import STARTING_CAPITAL
        return cls(
            ticker=ticker.upper(),
            session_date=d.isoformat(),
            cash=STARTING_CAPITAL,
            equity=STARTING_CAPITAL,
            day_start_equity=STARTING_CAPITAL,
            peak_equity=STARTING_CAPITAL,
        )

    def summary(self) -> str:
        lines = [
            f"── Loop State  {self.ticker}  {self.session_date} ──────────────",
            f"  Equity      : ${self.equity:>12,.2f}",
            f"  Daily PnL   : ${self.daily_pnl:>+12,.2f}  ({self.daily_drawdown_pct:.2%} drawdown)",
            f"  Open pos    : {len(self.open_positions)}",
            f"  Closed      : {len(self.closed_trades)}",
            f"  Decisions   : {self.decisions_today}  |  Trades: {self.trades_today}",
            f"  Iteration   : {self.iteration}",
        ]
        return "\n".join(lines)
