"""
src.loop.engine — Live intraday paper-trading loop.

Decision pipeline per bar
-------------------------
1. Update open positions (mark-to-market, auto-close stops/targets).
2. Skip if < min_bars_required.
3. Build no-lookahead snapshot.
4. Compute features (structure, levels, volatility, liquidity).
5. Run TimesFM forecast.
6. Generate candidate trade (ATR stop/target).
7. FTA filter (if fta_enabled).
8. Meta-model filter (if meta_model_enabled).
9. Portfolio risk gate.
10. Open trade.
11. Record decision + equity.

After the loop: close all open positions at the last bar's close.
Optional: EOD retraining if cfg.eod_retrain=True.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.loop.config import LoopConfig
from src.loop.decision_log import BarDecision, DecisionLog
from src.loop.result import LiveLoopResult
from src.loop.eod import end_of_day_process
from src.backtest.data_utils import build_snapshot_from_series
from src.backtest.candidate import generate_candidate
from src.features.pipeline import compute_all_features
from src.features.volatility import compute_volatility
from src.portfolio import create_portfolio, RiskConfig
from src.timesfm import run_forecast
from schemas.market_data import OHLCVSeries
from config.settings import META_MODEL_MIN_CONFIDENCE

log = logging.getLogger(__name__)
_UTC = timezone.utc


class LiveLoop:
    """
    Live intraday paper-trading loop.

    Parameters
    ----------
    config : LoopConfig | None
        Loop configuration. Defaults to LoopConfig() if not provided.
    model : BaseMetaModel | None
        Pre-loaded meta-model. If None, the scorer auto-loads or uses
        the HeuristicMetaModel fallback.
    """

    def __init__(
        self,
        config: LoopConfig | None = None,
        model=None,  # BaseMetaModel | None
    ):
        self.config = config or LoopConfig()
        self.model = model  # None = use scorer's auto-loaded singleton

    def run(self, series: OHLCVSeries) -> LiveLoopResult:
        cfg = self.config
        ticker = cfg.ticker or series.ticker
        timeframe = cfg.timeframe or series.timeframe
        n_bars = len(series.bars)

        portfolio = create_portfolio(
            starting_capital=cfg.starting_capital,
            max_concurrent_positions=3,
        )
        decision_log = DecisionLog()
        equity_curve: list = []
        meta_features_store: dict = {}

        for t in range(n_bars):
            bar = series.bars[t]
            current_close = bar.close
            current_ts = bar.timestamp
            decision = BarDecision(
                bar_idx=t,
                timestamp=current_ts,
                ticker=ticker,
                close=current_close,
            )

            # Step 1: Always update positions first (stops/targets).
            portfolio.update_positions({ticker: current_close}, timestamp=current_ts)

            # Step 2: Skip early bars — not enough history.
            if t < cfg.min_bars_required:
                decision.equity = portfolio.equity
                equity_curve.append((current_ts, portfolio.equity))
                decision_log.append(decision)
                continue

            if cfg.bar_sleep_seconds > 0:
                time.sleep(cfg.bar_sleep_seconds)

            # Step 3: Build no-lookahead snapshot.
            snapshot = build_snapshot_from_series(series, t, cfg.context_bars)
            if snapshot.tf_1h is None or len(snapshot.tf_1h.bars) < 10:
                decision.equity = portfolio.equity
                equity_curve.append((current_ts, portfolio.equity))
                decision_log.append(decision)
                continue

            # Step 4: Compute features.
            try:
                df_slice = snapshot.tf_1h.to_dataframe()
                volatility = compute_volatility(df_slice, ticker, timeframe)
                features = compute_all_features(snapshot, primary_tf="1h")
            except Exception as exc:
                log.debug("[t=%d] Feature error: %s", t, exc)
                decision.equity = portfolio.equity
                equity_curve.append((current_ts, portfolio.equity))
                decision_log.append(decision)
                continue

            # Step 5: Run forecast.
            try:
                forecast = run_forecast(
                    series=snapshot.tf_1h,
                    horizon=cfg.forecast_horizon,
                    ticker=ticker,
                    timeframe=timeframe,
                )
                decision.forecast_direction = forecast.direction
                decision.forecast_confidence = forecast.confidence
            except Exception as exc:
                log.debug("[t=%d] Forecast error: %s", t, exc)
                decision.equity = portfolio.equity
                equity_curve.append((current_ts, portfolio.equity))
                decision_log.append(decision)
                continue

            # Step 6: Generate candidate.
            candidate = generate_candidate(
                forecast=forecast,
                volatility=volatility,
                current_close=current_close,
                atr_stop_multiple=cfg.atr_stop_multiple,
                atr_target_multiple=cfg.atr_target_multiple,
            )
            if candidate is None:
                decision.equity = portfolio.equity
                equity_curve.append((current_ts, portfolio.equity))
                decision_log.append(decision)
                continue

            decision.candidate_generated = True

            # Step 7: FTA filter.
            if cfg.fta_enabled:
                try:
                    from src.fta import build_fta_input
                    from src.fta import evaluate as fta_evaluate
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
                    if not fta_output.verdict.accepted:
                        decision.rejection_reason = "fta_rejected"
                        if cfg.verbose:
                            print(f"  [t={t}] FTA REJECTED: {fta_output.verdict.summary}")
                        decision.equity = portfolio.equity
                        equity_curve.append((current_ts, portfolio.equity))
                        decision_log.append(decision)
                        continue
                except Exception as exc:
                    log.warning("[t=%d] FTA error: %s", t, exc)

            # Step 8: Meta-model filter.
            if cfg.meta_model_enabled:
                try:
                    from src.meta_model.scorer import score_trade
                    mm_output = score_trade(
                        features=features,
                        forecast=forecast,
                        candidate=candidate,
                        model=self.model,
                        threshold=cfg.meta_model_threshold,
                    )
                    decision.meta_model_evaluated = True
                    decision.meta_model_prob = mm_output.probability_of_success
                    decision.meta_model_accepted = mm_output.should_trade
                    if not mm_output.should_trade:
                        decision.rejection_reason = "meta_model_rejected"
                        if cfg.verbose:
                            print(
                                f"  [t={t}] META REJECTED: prob={mm_output.probability_of_success:.3f}"
                            )
                        decision.equity = portfolio.equity
                        equity_curve.append((current_ts, portfolio.equity))
                        decision_log.append(decision)
                        continue
                except Exception as exc:
                    log.warning("[t=%d] Meta-model error: %s", t, exc)

            # Step 9: Portfolio risk gate.
            ok, reason = portfolio.can_open_trade(
                ticker=ticker,
                entry_price=candidate["entry"],
                stop_price=candidate["stop"],
            )
            if not ok:
                decision.rejection_reason = f"portfolio_rejected:{reason}"
                if cfg.verbose:
                    print(f"  [t={t}] PORTFOLIO REJECTED: {reason}")
                decision.equity = portfolio.equity
                equity_curve.append((current_ts, portfolio.equity))
                decision_log.append(decision)
                continue

            # Step 10: Open trade.
            trade = portfolio.open_trade(
                ticker=ticker,
                side=candidate["side"],
                entry_price=candidate["entry"],
                stop_price=candidate["stop"],
                target_price=candidate["target"],
                confidence=candidate["forecast_confidence"],
                source="live_loop",
                timestamp=current_ts,
            )
            if trade is not None:
                decision.trade_opened = True
                decision.trade_id = trade.trade_id
                # Store meta_features keyed by trade_id for later journaling.
                try:
                    from src.meta_model.features import build_feature_vector, FEATURE_NAMES
                    mmi = build_feature_vector(features, forecast, candidate)
                    meta_features_store[trade.trade_id] = {
                        name: getattr(mmi, name) for name in FEATURE_NAMES
                    }
                except Exception:
                    pass
                if cfg.verbose:
                    print(
                        f"  [t={t}] TRADE OPENED: id={trade.trade_id} "
                        f"entry={candidate['entry']:.2f} stop={candidate['stop']:.2f} "
                        f"target={candidate['target']:.2f}"
                    )

            # Step 11: Record equity.
            decision.equity = portfolio.equity
            equity_curve.append((current_ts, portfolio.equity))
            decision_log.append(decision)

        # Close all remaining open positions at the last bar's close.
        last_close = series.bars[-1].close
        last_ts = series.bars[-1].timestamp
        for tid in list(portfolio.open_trades.keys()):
            portfolio.close_trade(tid, last_close, "end_of_loop", last_ts)

        # Build journal with meta_features attached.
        raw_journal = portfolio.export_trade_journal()
        for entry in raw_journal:
            entry["meta_features"] = meta_features_store.get(entry.get("trade_id", ""), {})

        result = LiveLoopResult(
            ticker=ticker,
            timeframe=timeframe,
            n_bars_processed=n_bars,
            starting_capital=cfg.starting_capital,
            final_equity=portfolio.equity,
            trade_journal=raw_journal,
            decision_log=decision_log.to_list(),
            equity_curve=equity_curve,
        )

        if cfg.eod_retrain:
            eod_report = end_of_day_process(result, save_model=True)
            if cfg.verbose:
                print(f"\n  EOD retraining: {eod_report}")

        return result
