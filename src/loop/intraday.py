"""IntradayRecommendationLoop — Phase 18 continuous recommendation engine.

Two operating modes
───────────────────
1. Replay mode  — processes a pre-loaded OHLCVSeries bar-by-bar (for testing
                  and backtesting). Calls run_replay(series).
2. Live mode    — polls data at configurable intervals during market hours.
                  Calls run_live(tickers, interval_seconds).

Both modes drive the Phase 17 RecommendationEngine and emit
TradeRecommendation objects. Decisions are appended to a JSONL decision log
and portfolio state is persisted via LoopState after each iteration.

Integration with Phase 8 LiveLoop
───────────────────────────────────
IntradayRecommendationLoop reuses the existing infrastructure:
  - src.backtest.data_utils.build_snapshot_from_series   (no-lookahead)
  - src.features.pipeline.compute_all_features           (feature engineering)
  - src.features.volatility.compute_volatility           (ATR)
  - src.timesfm.run_forecast                             (TimesFM / fallback)
  - src.fta.{build_fta_input, evaluate}                  (structural filter)
  - src.meta_model.scorer.score_trade                    (meta-model gate)
  - src.portfolio.engine.Portfolio                       (paper execution)
  - src.recommendation.engine.RecommendationEngine       (Phase 17 layer)

The key addition vs LiveLoop: every candidate that passes FTA + meta-model
is also evaluated by RecommendationEngine before any trade is opened.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from config.settings import LOG_DIR, STARTING_CAPITAL
from schemas.fta import FTAOutput, FTAVerdict
from schemas.forecast import ForecastOutput
from schemas.market_data import OHLCVSeries
from schemas.meta_model import MetaModelOutput
from schemas.recommendation import TradeRecommendation
from schemas.risk_appetite import RiskAppetiteConfig
from src.backtest.candidate import generate_candidate
from src.backtest.data_utils import build_snapshot_from_series
from src.features.pipeline import compute_all_features
from src.features.volatility import compute_volatility
from src.loop.config import LoopConfig
from src.loop.decision_log import BarDecision, DecisionLog
from src.loop.result import LiveLoopResult
from src.loop.state import ClosedTradeRecord, LoopState, PositionRecord
from src.portfolio import create_portfolio
from src.recommendation.engine import RecommendationContext, RecommendationEngine
from src.timesfm import run_forecast
from src.risk_appetite.loader import load_risk_appetite

log = logging.getLogger(__name__)
_UTC = timezone.utc

_DECISION_LOG_DIR = LOG_DIR / "decisions"


def _decision_log_path(ticker: str, session_date: date) -> Path:
    _DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _DECISION_LOG_DIR / f"decisions_{ticker.upper()}_{session_date.isoformat()}.jsonl"


def _append_decision_jsonl(path: Path, record: dict) -> None:
    try:
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        log.warning("decision_log.write_failed: %s", e)


class IntradayRecommendationLoop:
    """
    Continuously running recommendation engine.

    Parameters
    ----------
    loop_config    : LoopConfig (existing phase-8 config)
    risk_appetite  : RiskAppetiteConfig; if None, loads from settings
    session_date   : date for state file namespacing (defaults to today)
    verbose        : print per-bar summary lines
    """

    def __init__(
        self,
        loop_config: LoopConfig | None = None,
        risk_appetite: RiskAppetiteConfig | None = None,
        session_date: date | None = None,
        verbose: bool = False,
    ) -> None:
        self.cfg = loop_config or LoopConfig()
        self.ra = risk_appetite or load_risk_appetite()
        self.session_date = session_date or date.today()
        self.verbose = verbose
        self._engine = RecommendationEngine()
        self._shutdown = False

    # ── Public API ─────────────────────────────────────────────────────────

    def run_replay(
        self,
        series: OHLCVSeries,
        state: LoopState | None = None,
    ) -> tuple[LiveLoopResult, list[TradeRecommendation]]:
        """
        Process all bars in `series` sequentially (replay / backtest mode).

        Returns
        -------
        (LiveLoopResult, recommendations)
          LiveLoopResult   — compatible with adaptive loop / EOD retraining
          recommendations  — list of per-bar TradeRecommendation objects
        """
        cfg = self.cfg
        ticker = cfg.ticker or series.ticker
        timeframe = cfg.timeframe or series.timeframe

        # Fresh paper portfolio
        portfolio = create_portfolio(
            starting_capital=cfg.starting_capital,
            max_concurrent_positions=self.ra.max_concurrent_positions,
        )

        # Load or create persistent state
        state = state or LoopState.load(ticker, self.session_date)
        state.cash = cfg.starting_capital
        state.equity = cfg.starting_capital
        state.day_start_equity = cfg.starting_capital
        state.peak_equity = cfg.starting_capital

        decision_log = DecisionLog()
        equity_curve: list = []
        recommendations: list[TradeRecommendation] = []
        meta_features_store: dict = {}
        log_path = _decision_log_path(ticker, self.session_date)

        for t in range(len(series.bars)):
            if self._shutdown:
                log.info("intraday_loop.shutdown_at_bar_%d", t)
                break

            bar = series.bars[t]
            current_close = bar.close
            current_ts = bar.timestamp

            decision = BarDecision(
                bar_idx=t,
                timestamp=current_ts,
                ticker=ticker,
                close=current_close,
            )

            # Always mark-to-market first
            portfolio.update_positions({ticker: current_close}, timestamp=current_ts)
            state.update_equity(portfolio.equity)

            equity_curve.append((current_ts, portfolio.equity))

            # Skip warm-up period
            if t < cfg.min_bars_required:
                decision.equity = portfolio.equity
                decision_log.append(decision)
                continue

            # Build no-lookahead snapshot
            snapshot = build_snapshot_from_series(series, t, cfg.context_bars)
            primary_tf = "1h" if snapshot.tf_1h is not None and len(snapshot.tf_1h.bars) >= 10 else "1d"
            primary_series = snapshot.tf_1h if primary_tf == "1h" else snapshot.tf_1d

            if primary_series is None or len(primary_series.bars) < 10:
                decision.equity = portfolio.equity
                decision_log.append(decision)
                continue

            # Compute features
            try:
                df_slice = primary_series.to_dataframe()
                volatility = compute_volatility(df_slice, ticker, timeframe)
                features = compute_all_features(snapshot, primary_tf=primary_tf)
            except Exception as exc:
                log.debug("[t=%d] feature_error: %s", t, exc)
                decision.equity = portfolio.equity
                decision_log.append(decision)
                continue

            # Run forecast
            try:
                forecast = run_forecast(
                    series=primary_series,
                    horizon=cfg.forecast_horizon,
                    ticker=ticker,
                    timeframe=timeframe,
                )
                decision.forecast_direction = forecast.direction
                decision.forecast_confidence = forecast.confidence
            except Exception as exc:
                log.debug("[t=%d] forecast_error: %s", t, exc)
                decision.equity = portfolio.equity
                decision_log.append(decision)
                continue

            # Generate ATR-based candidate
            candidate = generate_candidate(
                forecast=forecast,
                volatility=volatility,
                current_close=current_close,
                atr_stop_multiple=cfg.atr_stop_multiple,
                atr_target_multiple=cfg.atr_target_multiple,
            )
            if candidate is None:
                decision.equity = portfolio.equity
                decision_log.append(decision)
                continue

            decision.candidate_generated = True

            # Run FTA
            fta_output: FTAOutput | None = None
            if cfg.fta_enabled:
                try:
                    from src.fta import build_fta_input, evaluate as fta_evaluate
                    fta_input = build_fta_input(
                        features=features,
                        forecast=forecast,
                        candidate=candidate,
                        ticker=ticker,
                    )
                    fta_output = fta_evaluate(fta_input)
                    decision.fta_evaluated = True
                    decision.fta_accepted = fta_output.verdict.accepted
                    decision.fta_score = fta_output.verdict.score
                    decision.fta_rejection_reasons = [r.code for r in fta_output.rejection_reasons]
                except Exception as exc:
                    log.warning("[t=%d] fta_error: %s", t, exc)

            # Run meta-model
            meta_output: MetaModelOutput | None = None
            if cfg.meta_model_enabled:
                try:
                    from src.meta_model.scorer import score_trade
                    meta_output = score_trade(
                        features=features,
                        forecast=forecast,
                        candidate=candidate,
                        threshold=cfg.meta_model_threshold,
                    )
                    decision.meta_model_evaluated = True
                    decision.meta_model_prob = meta_output.probability_of_success
                    decision.meta_model_accepted = meta_output.should_trade
                except Exception as exc:
                    log.warning("[t=%d] meta_error: %s", t, exc)

            # Build synthetic FTAOutput / MetaModelOutput if not computed
            if fta_output is None:
                fta_output = _synthetic_fta(ticker, candidate, forecast)
            if meta_output is None:
                meta_output = _synthetic_meta(ticker, forecast.confidence)

            # Determine open position size for exit-signal checks
            open_size = 0.0
            if ticker in portfolio.open_trades:
                open_size = sum(
                    t_obj.position_size for t_obj in portfolio.open_trades.values()
                    if t_obj.ticker == ticker
                )

            # Phase 17 Recommendation Engine
            ctx = RecommendationContext(
                ticker=ticker,
                current_price=current_close,
                forecast=forecast,
                fta_output=fta_output,
                meta_output=meta_output,
                risk_appetite=self.ra,
                news_features=None,
                timeframe_mode="daily_only",
                news_mode="disabled",
                portfolio_equity=portfolio.equity,
                open_position_size=open_size,
                losing_streak=_count_losing_streak(state),
                trades_today=state.trades_today,
            )
            rec = self._engine.evaluate(ctx)
            recommendations.append(rec)

            # Log recommendation to JSONL
            rec_dict = rec.model_dump()
            state.log_recommendation(rec_dict)
            _append_decision_jsonl(log_path, {
                "bar_idx": t,
                "timestamp": current_ts.isoformat() if hasattr(current_ts, "isoformat") else str(current_ts),
                "recommendation": rec_dict,
                "portfolio_equity": portfolio.equity,
            })
            state.decisions_today += 1

            # Act on recommendation
            if rec.action in ("BUY", "SELL") and rec.position_action == "OPEN":
                if rec.action == "BUY":
                    ok, reason = portfolio.can_open_trade(
                        ticker=ticker,
                        entry_price=candidate["entry"],
                        stop_price=candidate["stop"],
                    )
                    if ok:
                        trade = portfolio.open_trade(
                            ticker=ticker,
                            side="long",
                            entry_price=candidate["entry"],
                            stop_price=candidate["stop"],
                            target_price=candidate["target"],
                            confidence=forecast.confidence,
                            source="intraday_loop",
                            timestamp=current_ts,
                        )
                        if trade is not None:
                            decision.trade_opened = True
                            decision.trade_id = trade.trade_id
                            state.trades_today += 1
                            pos = PositionRecord(
                                trade_id=trade.trade_id,
                                ticker=ticker,
                                side="long",
                                entry_price=candidate["entry"],
                                stop_price=candidate["stop"],
                                target_price=candidate["target"],
                                position_size=trade.quantity,
                                opened_at=str(current_ts),
                            )
                            state.add_open_position(pos)
                            if self.verbose:
                                print(f"  [t={t}] OPEN LONG  entry={candidate['entry']:.2f} "
                                      f"stop={candidate['stop']:.2f} size={trade.position_size:.0f}")

            elif rec.position_action in ("CLOSE", "REDUCE"):
                # Close / reduce open trades for this ticker
                to_close = [tid for tid, tr in portfolio.open_trades.items()
                            if tr.ticker == ticker]
                for tid in to_close:
                    if rec.position_action == "REDUCE":
                        # Simplified: close half by closing at current price
                        # (full partial-fill support requires margin model changes)
                        portfolio.close_trade(tid, current_close, "recommendation_reduce", current_ts)
                    else:
                        portfolio.close_trade(tid, current_close, "recommendation_close", current_ts)
                    if tid in state.open_positions:
                        pos_rec = PositionRecord(**state.open_positions[tid])
                        entry_p = pos_rec.entry_price
                        realized = (current_close - entry_p) * pos_rec.position_size
                        state.close_position(
                            tid,
                            ClosedTradeRecord(
                                trade_id=tid,
                                ticker=ticker,
                                side="long",
                                entry_price=entry_p,
                                exit_price=current_close,
                                position_size=pos_rec.position_size,
                                realized_pnl=realized,
                                opened_at=pos_rec.opened_at,
                                closed_at=str(current_ts),
                                close_reason=rec.position_action.lower(),
                            ),
                        )
                    if self.verbose and to_close:
                        print(f"  [t={t}] {rec.position_action} exit={current_close:.2f}")

            decision.rejection_reason = rec.rejection_reason
            decision.equity = portfolio.equity
            decision_log.append(decision)
            state.update_equity(portfolio.equity)
            state.iteration = t

            if self.verbose:
                print(f"  [t={t}] {rec.action}/{rec.position_action} "
                      f"eq=${portfolio.equity:,.0f} "
                      f"{'REJECTED: '+rec.rejection_reason if rec.rejection_reason else ''}")

        # Close all remaining positions at final bar
        last_close = series.bars[-1].close
        last_ts = series.bars[-1].timestamp
        for tid in list(portfolio.open_trades.keys()):
            portfolio.close_trade(tid, last_close, "end_of_replay", last_ts)

        # Persist final state
        state.update_equity(portfolio.equity)
        state.save()

        # Build raw journal
        raw_journal = portfolio.export_trade_journal()
        for entry in raw_journal:
            entry["meta_features"] = meta_features_store.get(entry.get("trade_id", ""), {})

        result = LiveLoopResult(
            ticker=ticker,
            timeframe=timeframe,
            n_bars_processed=len(series.bars),
            starting_capital=cfg.starting_capital,
            final_equity=portfolio.equity,
            trade_journal=raw_journal,
            decision_log=decision_log.to_list(),
            equity_curve=equity_curve,
        )

        if cfg.eod_retrain:
            from src.loop.eod import end_of_day_process
            eod_report = end_of_day_process(result, save_model=True)
            if self.verbose:
                print(f"\n  EOD retraining: model_improved={eod_report.model_improved}")

        return result, recommendations

    def register_shutdown_handler(self) -> None:
        """Register SIGINT/SIGTERM handlers for graceful shutdown."""
        def _handler(signum, frame):
            log.info("intraday_loop.shutdown_signal: %s", signum)
            self._shutdown = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def run_live(
        self,
        fetch_fn: Callable[[str], OHLCVSeries],
        tickers: list[str],
        interval_seconds: float = 3600.0,
        max_iterations: int | None = None,
    ) -> None:
        """
        Continuous live loop — polls data at `interval_seconds` intervals.

        Parameters
        ----------
        fetch_fn         : callable(ticker) → OHLCVSeries (from local store)
        tickers          : list of tickers to scan each iteration
        interval_seconds : seconds between polling cycles (3600 = hourly)
        max_iterations   : if set, stop after this many cycles (for testing)
        """
        from src.loop.scheduler import is_market_open, seconds_until_market_open, market_session_label

        self.register_shutdown_handler()
        iteration = 0
        log.info("intraday_loop.start: tickers=%s interval=%ds", tickers, interval_seconds)

        while not self._shutdown:
            if max_iterations is not None and iteration >= max_iterations:
                log.info("intraday_loop.max_iterations_reached: %d", max_iterations)
                break

            now = datetime.now(_UTC)
            label = market_session_label(now)

            if not is_market_open(now):
                wait = seconds_until_market_open(now)
                log.info("intraday_loop.market_%s: waiting %.0fs for open", label, wait)
                if self._shutdown:
                    break
                time.sleep(min(60.0, wait))
                continue

            log.info("intraday_loop.iteration_%d: market=%s", iteration, label)
            for ticker in tickers:
                if self._shutdown:
                    break
                try:
                    series = fetch_fn(ticker)
                    if series and len(series.bars) >= self.cfg.min_bars_required:
                        self.cfg.ticker = ticker
                        result, recs = self.run_replay(series)
                        latest = recs[-1] if recs else None
                        if latest:
                            log.info("  %s → %s/%s %s",
                                     ticker, latest.action, latest.position_action,
                                     f"size={latest.recommended_position_size:.0f}"
                                     if latest.recommended_position_size else "")
                except Exception as e:
                    log.warning("intraday_loop.ticker_error %s: %s", ticker, e)

            iteration += 1
            if not self._shutdown:
                log.info("intraday_loop.sleep: %.0fs", interval_seconds)
                # Sleep in small chunks so shutdown signal is responsive
                slept = 0.0
                while slept < interval_seconds and not self._shutdown:
                    time.sleep(min(5.0, interval_seconds - slept))
                    slept += 5.0

        log.info("intraday_loop.stopped: %d iterations", iteration)


# ── Internal helpers ───────────────────────────────────────────────────────

def _synthetic_fta(ticker: str, candidate: dict, forecast: ForecastOutput) -> FTAOutput:
    """Build a minimal FTAOutput when FTA is disabled — always accepts."""
    from schemas.fta import FTAVerdict
    rr = candidate.get("reward_risk", 2.0)
    return FTAOutput(
        ticker=ticker,
        verdict=FTAVerdict(accepted=True, score=0.65, summary="fta_disabled"),
        reward_risk=rr,
        structure_score=0.65,
        liquidity_score=0.65,
        volatility_ok=True,
    )


def _synthetic_meta(ticker: str, confidence: float) -> MetaModelOutput:
    """Build a minimal MetaModelOutput when meta-model is disabled."""
    prob = min(0.99, max(0.0, confidence))
    return MetaModelOutput(
        ticker=ticker,
        probability_of_success=prob,
        confidence=prob,
        should_trade=prob >= 0.50,
    )


def _count_losing_streak(state: LoopState) -> int:
    """Count consecutive losing trades from the end of closed_trades."""
    streak = 0
    for ct in reversed(state.closed_trades):
        pnl = ct.get("realized_pnl", 0)
        if isinstance(pnl, (int, float)) and pnl <= 0:
            streak += 1
        else:
            break
    return streak
