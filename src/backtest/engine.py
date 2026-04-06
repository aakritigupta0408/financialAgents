"""
src.backtest.engine — Historical backtest engine.

No-lookahead guarantee
-----------------------
At every timestep t the engine calls:

    snapshot = build_snapshot_from_series(series, t_idx=t, context_bars=context_bars)

build_snapshot_from_series() slices series.bars[:t+1] (inclusive of the current
bar) and then takes the last context_bars bars. Future bars (t+1 onward) are
NEVER included. All downstream computations — compute_all_features(), run_forecast(),
generate_candidate() — receive only this truncated snapshot or its derived series.

Fill assumptions (inherited from Phase 5 portfolio engine)
-----------------------------------------------------------
- Zero slippage: all fills at exact prices.
- No commissions or fees.
- Stop fills at exactly trade.stop_price (no gap modelling).
- Take-profit fills at exactly trade.target_price.
- Short positions: not generated in this phase (see candidate.py).

Rejection tracking
------------------
Every bar's candidate and rejection are logged. Verbose mode prints them.
The rejection_counts dict in the BacktestResult trade_journal entries carries
the reason string from can_open_trade().
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import STARTING_CAPITAL
from schemas.market_data import OHLCVSeries
from src.backtest.candidate import generate_candidate
from src.backtest.data_utils import build_snapshot_from_series
from src.backtest.metrics import compute_metrics
from src.backtest.result import BacktestResult
from src.features.pipeline import compute_all_features
from src.features.volatility import compute_volatility
from src.portfolio import RiskConfig, create_portfolio
from src.timesfm import run_forecast

log = logging.getLogger(__name__)

_UTC = timezone.utc


class BacktestEngine:
    """
    Simulates historical trading using the system's full decision flow.

    Decision flow at each bar t
    ---------------------------
    1. Build no-lookahead snapshot (only bars 0..t visible).
    2. Update open positions — auto-close stops/targets hit by current close.
    3. Compute features (structure, levels, volatility, liquidity).
    4. Run TimesFM forecast on the truncated series.
    5. Generate candidate trade (ATR-based stop/target + R:R check).
    6. Check portfolio/risk constraints via portfolio.can_open_trade().
    7. Open trade if allowed.
    8. Record equity snapshot.

    After the loop all remaining open positions are closed at the last bar's
    close price with reason="end_of_backtest".
    """

    def __init__(
        self,
        starting_capital: float | None = None,
        risk_config: RiskConfig | None = None,
        atr_stop_multiple: float = 1.5,
        atr_target_multiple: float = 3.0,
        forecast_horizon: int = 10,
        context_bars: int = 100,
        min_bars_required: int = 50,
        verbose: bool = False,
        fta_enabled: bool = False,
        meta_model_enabled: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        starting_capital    : Initial cash. Defaults to config/settings STARTING_CAPITAL.
        risk_config         : Full RiskConfig override. If provided, starting_capital
                              in this object is used for the portfolio. If both
                              starting_capital and risk_config are given, risk_config
                              takes precedence for risk limits; starting_capital is
                              passed to create_portfolio().
        atr_stop_multiple   : Stop distance in ATR units (default 1.5).
        atr_target_multiple : Target distance in ATR units (default 3.0).
        forecast_horizon    : Bars ahead for TimesFM forecast (default 10).
        context_bars        : Maximum bars of history fed to features/forecast (default 100).
        min_bars_required   : Skip trade attempts for the first min_bars_required bars
                              (ensures enough history for ATR and structure computation).
        verbose             : Print per-bar log lines to stdout when True.
        fta_enabled         : When True, run FTA filter after candidate generation.
                              Defaults to False to preserve all existing tests.
        meta_model_enabled  : When True, run meta-model filter after FTA pass.
                              Defaults to False to preserve all existing tests.
        """
        self.starting_capital = float(starting_capital) if starting_capital is not None else STARTING_CAPITAL
        self.risk_config = risk_config
        self.atr_stop_multiple = atr_stop_multiple
        self.atr_target_multiple = atr_target_multiple
        self.forecast_horizon = forecast_horizon
        self.context_bars = context_bars
        self.min_bars_required = min_bars_required
        self.verbose = verbose
        self.fta_enabled = fta_enabled
        self.meta_model_enabled = meta_model_enabled

    def run(
        self,
        series: OHLCVSeries,
        ticker: str | None = None,
    ) -> BacktestResult:
        """
        Run a full backtest over the provided OHLCVSeries.

        Parameters
        ----------
        series : OHLCVSeries — full historical series (ascending timestamps).
        ticker : str | None — ticker label. Defaults to series.ticker.

        Returns
        -------
        BacktestResult with all metrics and the full equity curve + journal.
        """
        resolved_ticker = ticker or series.ticker
        timeframe = series.timeframe
        n_bars = len(series.bars)

        if n_bars == 0:
            raise ValueError("series has no bars; cannot run backtest")

        # Initialise a fresh portfolio.
        if self.risk_config is not None:
            portfolio = create_portfolio(
                starting_capital=self.risk_config.starting_capital,
                risk_per_trade_pct=self.risk_config.risk_per_trade_pct,
                max_trades_per_day=self.risk_config.max_trades_per_day,
                max_concurrent_positions=self.risk_config.max_concurrent_positions,
                max_daily_drawdown_pct=self.risk_config.max_daily_drawdown_pct,
                max_ticker_exposure_pct=self.risk_config.max_ticker_exposure_pct,
            )
        else:
            portfolio = create_portfolio(starting_capital=self.starting_capital)

        equity_curve: list[tuple[datetime, float]] = []
        rejection_counts: dict[str, int] = {}

        # Meta-features keyed by trade_id; merged into journal at the end.
        self._meta_features: dict[str, dict] = {}

        start_date = series.bars[0].timestamp
        end_date = series.bars[-1].timestamp

        for t in range(self.min_bars_required, n_bars):
            bar = series.bars[t]
            current_close = bar.close
            current_ts = bar.timestamp

            # Step 2: Update open positions (stop/target auto-close).
            closed_trades = portfolio.update_positions(
                {resolved_ticker: current_close},
                timestamp=current_ts,
            )
            if self.verbose and closed_trades:
                for ct in closed_trades:
                    log.info(
                        "[t=%d] Auto-closed trade %s reason=%s exit=%.4f pnl=%.2f",
                        t, ct.trade_id, ct.exit_reason, ct.exit_price or 0.0, ct.realized_pnl(),
                    )
                    if self.verbose:
                        print(
                            f"  [t={t}] CLOSED trade={ct.trade_id} "
                            f"reason={ct.exit_reason} exit={ct.exit_price:.4f} "
                            f"pnl={ct.realized_pnl():.2f}"
                        )

            # Step 3: Build no-lookahead snapshot.
            snapshot = build_snapshot_from_series(series, t, self.context_bars)

            # Step 4: Compute features.
            try:
                df_slice = snapshot.tf_1h.to_dataframe()
                volatility = compute_volatility(df_slice, resolved_ticker, timeframe)
                _features = compute_all_features(snapshot, primary_tf="1h")
            except Exception as exc:
                log.debug("[t=%d] Feature computation failed: %s", t, exc)
                equity_curve.append((current_ts, portfolio.equity))
                continue

            # Step 5: Run forecast on the truncated (no-lookahead) series.
            try:
                forecast = run_forecast(
                    series=snapshot.tf_1h,
                    horizon=self.forecast_horizon,
                    ticker=resolved_ticker,
                    timeframe=timeframe,
                )
            except Exception as exc:
                log.debug("[t=%d] Forecast failed: %s", t, exc)
                equity_curve.append((current_ts, portfolio.equity))
                continue

            # Step 6: Generate candidate trade.
            candidate = generate_candidate(
                forecast=forecast,
                volatility=volatility,
                current_close=current_close,
                atr_stop_multiple=self.atr_stop_multiple,
                atr_target_multiple=self.atr_target_multiple,
            )

            if candidate is not None:
                # Optional Step 6a: FTA filter (only when fta_enabled=True).
                if self.fta_enabled:
                    try:
                        from src.fta import build_fta_input, evaluate as fta_evaluate
                        fta_input = build_fta_input(
                            features=_features,
                            forecast=forecast,
                            candidate=candidate,
                            ticker=resolved_ticker,
                        )
                        fta_output = fta_evaluate(fta_input)
                        if not fta_output.verdict.accepted:
                            rejection_counts["fta_rejected"] = (
                                rejection_counts.get("fta_rejected", 0) + 1
                            )
                            if self.verbose:
                                print(
                                    f"  [t={t}] FTA REJECTED: {fta_output.verdict.summary}"
                                )
                            equity_curve.append((current_ts, portfolio.equity))
                            continue
                    except Exception as _fta_exc:
                        log.warning("[t=%d] FTA error: %s", t, _fta_exc)

                # Optional Step 6b: Meta-model filter (only when meta_model_enabled=True).
                if self.meta_model_enabled:
                    try:
                        from src.meta_model.scorer import score_trade
                        mm_output = score_trade(
                            features=_features,
                            forecast=forecast,
                            candidate=candidate,
                        )
                        if not mm_output.should_trade:
                            rejection_counts["meta_model_rejected"] = (
                                rejection_counts.get("meta_model_rejected", 0) + 1
                            )
                            if self.verbose:
                                print(
                                    f"  [t={t}] META REJECTED: "
                                    f"prob={mm_output.probability_of_success:.3f}"
                                )
                            equity_curve.append((current_ts, portfolio.equity))
                            continue
                    except Exception as _mm_exc:
                        log.warning("[t=%d] Meta-model error: %s", t, _mm_exc)

                # Step 7: Portfolio risk gate.
                ok, reason = portfolio.can_open_trade(
                    ticker=resolved_ticker,
                    entry_price=candidate["entry"],
                    stop_price=candidate["stop"],
                )
                if ok:
                    trade = portfolio.open_trade(
                        ticker=resolved_ticker,
                        side=candidate["side"],
                        entry_price=candidate["entry"],
                        stop_price=candidate["stop"],
                        target_price=candidate["target"],
                        confidence=candidate["forecast_confidence"],
                        source="backtest",
                        timestamp=current_ts,
                    )

                    # Store meta_features for this trade so the dataset builder
                    # can read them back from the journal.  Uses lazy import to
                    # avoid circular imports at module level.
                    if trade is not None:
                        try:
                            from src.meta_model.features import (
                                FEATURE_NAMES,
                                build_feature_vector,
                            )

                            mmi = build_feature_vector(
                                _features, forecast, candidate
                            )
                            self._meta_features[trade.trade_id] = {
                                name: getattr(mmi, name) for name in FEATURE_NAMES
                            }
                        except Exception as _mf_exc:
                            log.debug(
                                "[t=%d] meta_features build failed: %s", t, _mf_exc
                            )

                    if self.verbose and trade is not None:
                        print(
                            f"  [t={t}] OPEN  trade={trade.trade_id} "
                            f"entry={candidate['entry']:.4f} "
                            f"stop={candidate['stop']:.4f} "
                            f"target={candidate['target']:.4f} "
                            f"rr={candidate['reward_risk']:.2f}"
                        )
                else:
                    # Record rejection.
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    if self.verbose:
                        print(f"  [t={t}] REJECTED candidate: {reason}")

            # Step 8: Record equity.
            equity_curve.append((current_ts, portfolio.equity))

        # End-of-backtest: close all remaining open positions at last close.
        last_close = series.bars[-1].close
        last_ts = series.bars[-1].timestamp
        for trade_id in list(portfolio.open_trades.keys()):
            portfolio.close_trade(trade_id, last_close, "end_of_backtest", last_ts)

        if self.verbose:
            print(f"\n  Rejections by reason: {rejection_counts}")

        # Build final metrics.
        trade_journal = self._augment_journal(portfolio.export_trade_journal())
        metrics = compute_metrics(equity_curve, trade_journal, portfolio.starting_capital, timeframe)

        # Unrealized PnL at close is 0 because all positions were force-closed above.
        unrealized_at_close = float(portfolio.unrealized_pnl_total())

        result = BacktestResult(
            ticker=resolved_ticker,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            n_bars=n_bars,
            starting_capital=portfolio.starting_capital,
            final_equity=portfolio.equity,
            total_return_pct=metrics["total_return_pct"],
            realized_pnl=metrics["realized_pnl"],
            unrealized_pnl_at_close=unrealized_at_close,
            n_trades=metrics["n_trades"],
            n_winners=metrics["n_winners"],
            n_losers=metrics["n_losers"],
            win_rate=metrics["win_rate"],
            avg_winner=metrics["avg_winner"],
            avg_loser=metrics["avg_loser"],
            profit_factor=metrics["profit_factor"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            sharpe_ratio=metrics["sharpe_ratio"],
            equity_curve=equity_curve,
            trade_journal=trade_journal,
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _augment_journal(self, journal: list[dict]) -> list[dict]:
        """
        Merge stored meta_features into each trade journal entry.

        For each entry, looks up the trade_id in self._meta_features and attaches
        the dict under the key "meta_features".  Entries without a stored record
        receive an empty dict so downstream code can always call .get("meta_features").

        Parameters
        ----------
        journal : list of trade dicts from portfolio.export_trade_journal().

        Returns
        -------
        The same list with "meta_features" injected into each entry (in-place).
        """
        for entry in journal:
            trade_id = entry.get("trade_id", "")
            entry["meta_features"] = self._meta_features.get(trade_id, {})
        return journal
