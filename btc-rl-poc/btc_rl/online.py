"""Always-on experiment runner: control + treatments predicting BTC price.

Every arm commits integer price predictions each 5 minutes for the +5/+15/
+30-min targets; the objective is pure prediction quality. Arms never share
model state, so metric gaps are attributable (see README for the ladder).

Learning happens at two speeds:
  per prediction  the moment a prediction is scored, its (state, action,
                  reward) does an immediate Q-update — the agent improves
                  with every single evaluated prediction
  per hour        experience replay over the last 24h consolidates, guarded
                  by a hold-out no-regression gate (bad updates revert)

Usage:  python -m btc_rl.online            # run forever
        python -m btc_rl.online --once     # backfill + one pass, then exit
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from . import metrics as M
from .history import append_history
from .agents import (BinaryLogit, DistDQNAgent, LinearQAgent, LinUCBAgent,
                     LSTMDistAgent, TabularQAgent)

BANDIT_TYPES = (LinUCBAgent, LinearQAgent, DistDQNAgent)  # shared select API
from .env import build_episodes, reward
from .features import (BOOK_DIM, KALSHI_DIM, LIVE_DIM, LLM_DIM, OFI_DIM,
                       TREND_DIM, book_feature_vector, compute_features,
                       discretize, feature_vector, kalshi_feature_vector,
                       live_feature_vector, llm_feature_vector,
                       ofi_feature_vector, trend_feature_vector)
from .llm_sentiment import sentiment_snapshot
from .sources import (fetch_book_stats, fetch_brti_composite,
                      fetch_kalshi_btc15,
                      fetch_deribit_mark, fetch_fear_greed, fetch_mempool_fee,
                      fetch_okx_funding_rate, fetch_range, fetch_recent_trades)

FEATURE_DIM = 10  # len(feature_vector(...)) — intercept + 9 signals
SNAP_FILE_NAME = "live_snapshots.jsonl"  # streamed-feature history — feeds
                                         # the live-feature arms (t6/t10)
SNAP_MAX_AGE_S = 600   # a snapshot older than 10 min doesn't describe a slot

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PRED_LOG = RESULTS_DIR / "prediction_log.jsonl"
STATUS = RESULTS_DIR / "online_status.json"
BATCH_QTABLE = RESULTS_DIR / "q_table.json"

# Experiment arms: one stream per horizon, each predicting at its own natural
# cadence (cadence == horizon), each with its own Q-table file. The h15/h30
# tables warm-start from the original (control) model's batch tables; add new
# dicts here for future treatments.
# POLICY: a new feature-bandit treatment must pass scripts/offline_gate.py
# (MAE gate + duplicate check) BEFORE being added here; deep arms (t8/t9)
# and live-only-signal arms (t6/t10/t11) are gated by batch-vs-persistence
# comparisons recorded in metrics_history instead. t3/t4/t5 were retired as
# offline-proven duplicates of t2/t6 (see results/offline_gate.json).
# All arms predict every 5 minutes (9:00, 9:05, 9:10…). CONTROL arms (h5/h15/
# h30, tabular Q) are the original tabular baseline: their CONFIG is frozen,
# but like every arm they learn online per scored prediction and in the
# hourly replay. TREATMENT 2 (t2-*) is a LinUCB contextual bandit over the
# 19 vol-scaled K_FACTORS arms; every learner uses the prediction reward:
# +1 within the vol band max($5, 0.1·σ_h), else -|error|/100.
VARIANTS: dict[str, dict] = {
    # +1-minute horizon: the one where the noise floor ($13.9 over 60 days)
    # permits MAE < $20 — every arm family runs there too.
    "h1": {"predict_every": 300, "horizons": [1], "agent": "tabular"},
    "rp-h1": {"predict_every": 300, "horizons": [1], "agent": "replay"},
    "t2-h1": {"predict_every": 300, "horizons": [1], "agent": "linucb"},
    "t6-h1": {"predict_every": 300, "horizons": [1], "agent": "linucb",
              "live": True, "trend": True, "book": True, "llm": True,
              "ofi": True},
    "t7-h1": {"predict_every": 300, "horizons": [1], "agent": "linearq",
              "trend": True},
    "t8-h1": {"predict_every": 300, "horizons": [1], "agent": "dqn",
              "trend": True},
    "t9-h1": {"predict_every": 300, "horizons": [1], "agent": "seq"},
    "h5": {"predict_every": 300, "horizons": [5], "agent": "tabular"},
    "h15": {"predict_every": 300, "horizons": [15], "agent": "tabular"},
    "h30": {"predict_every": 300, "horizons": [30], "agent": "tabular"},
    # RP = chart-replay baseline: copy the last h-minute move forward —
    # pred(t+h) = price(t) + (price(t) - price(t-h)). No learning; exists so
    # "better than copying the past graph?" is answered on every chart.
    "rp-h5": {"predict_every": 300, "horizons": [5], "agent": "replay"},
    "rp-h15": {"predict_every": 300, "horizons": [15], "agent": "replay"},
    "rp-h30": {"predict_every": 300, "horizons": [30], "agent": "replay"},
    "t2-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb"},
    "t2-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb"},
    "t2-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb"},
    # TREATMENT 7 = the roadmap's L2 rung: linear function approximation,
    # SGD on Q(s,a) over continuous bar-derived features (base + trend, 13
    # dims — no snapshot dependency), vol-scaled actions. Batch-trained on
    # 60 days (scripts/train_l2.py), then online like every other arm.
    # RETIRED 2026-08-20 (evidence in metrics_history kind="retire"):
    # t7-h5 (MASE 1.115 worsening, replay gate 1/8 kept), t7-h30 (direction
    # 45% PT z -2.3 significantly wrong-way, -2627 bps paper P&L), t6-h5
    # (MASE 1.135, PT z -2.4 anti-directional), t8-h30 (MASE 1.252
    # worsening, 64% band coverage), t9-h5 (MASE 1.126, 44% direction).
    # Criterion: >=250 scored slots, MASE >= 1.10, no significant positive
    # direction/P&L signal, no marked improvement trend.
    "t7-h15": {"predict_every": 300, "horizons": [15], "agent": "linearq",
               "trend": True},
    # TREATMENT 6 = t5 + ORDER-FLOW IMBALANCE — the research-backed
    # short-horizon signal (Cont et al.; crypto order-flow literature):
    # signed taker-volume imbalance over 1/5/15 min + trade intensity,
    # computed live from the Coinbase trades stream.
    # TREATMENT 8 = the roadmap's L3 rung: small distributional network —
    # predict the vol-normalized delta DISTRIBUTION over the action bins,
    # act on its mode. Batch-trained by scripts/train_l3.py.
    "t8-h5": {"predict_every": 300, "horizons": [5], "agent": "dqn",
              "trend": True},
    "t8-h15": {"predict_every": 300, "horizons": [15], "agent": "dqn",
               "trend": True},
    # TREATMENT 9 = the L4 rung: LSTM over the raw 1m return stream,
    # distributional output, mode action (scripts/train_l4.py).
    "t9-h15": {"predict_every": 300, "horizons": [15], "agent": "seq"},
    "t9-h30": {"predict_every": 300, "horizons": [30], "agent": "seq"},
    # TREATMENT 10 = t2 + the KALSHI BTC-15-MIN PREDICTION MARKET — the
    # crowd's own P(up), strike gap, window clock, and quote spread as
    # context (see kalshi_feature_vector). Same bandit/base features as t2,
    # so any metric gap isolates what the prediction market adds. Live-only
    # signal — offline_gate is not runnable (no historical Kalshi feed),
    # same standing as the other snapshot-fed features.
    "t10-h1": {"predict_every": 300, "horizons": [1], "agent": "linucb",
               "kalshi": True},
    "t10-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb",
               "kalshi": True},
    "t10-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb",
                "kalshi": True},
    "t10-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb",
                "kalshi": True},
    # TREATMENT 11 = RLHF: same bandit/base features as t2, but its ONLINE
    # reward is blended with human feedback — a directional view recorded
    # via scripts/feedback.py earns agreement credit (±HF_WEIGHT) on
    # predictions committed within the next 30 min. With no feedback it
    # degenerates to exactly t2, so the t11-vs-t2 standings gap is a
    # measured price of the human in the loop.
    "t11-h1": {"predict_every": 300, "horizons": [1], "agent": "linucb",
               "hf": True},
    "t11-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb",
               "hf": True},
    "t11-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb",
                "hf": True},
    "t11-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb",
                "hf": True},
    "t6-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb",
               "live": True, "trend": True, "book": True, "llm": True,
               "ofi": True},
    "t6-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb",
               "live": True, "trend": True, "book": True, "llm": True,
               "ofi": True},
}


def _ctx_dim(spec: dict) -> int:
    if spec.get("agent") == "seq":
        return 60  # raw 1m return sequence
    return (FEATURE_DIM + (TREND_DIM if spec.get("trend") else 0)
            + (LIVE_DIM if spec.get("live") else 0)
            + (BOOK_DIM if spec.get("book") else 0)
            + (LLM_DIM if spec.get("llm") else 0)
            + (OFI_DIM if spec.get("ofi") else 0)
            + (KALSHI_DIM if spec.get("kalshi") else 0))


def _context(spec: dict, feat: dict, snap: dict | None) -> list[float]:
    if spec.get("agent") == "seq":
        return list(feat["ret_seq"])
    x = feature_vector(feat)
    if spec.get("trend"):
        x += trend_feature_vector(feat)
    if spec.get("live"):
        x += live_feature_vector(snap)
    if spec.get("book"):
        x += book_feature_vector(snap)
    if spec.get("llm"):
        x += llm_feature_vector(snap)
    if spec.get("ofi"):
        x += ofi_feature_vector(snap)
    if spec.get("kalshi"):
        x += kalshi_feature_vector(snap)
    return x


def _ofi_stats(trades: dict[int, dict], now_ts: float) -> dict:
    """Signed taker-volume imbalance over 1/5/15 min + trade intensity."""
    import math
    out: dict = {}
    n_1m = 0
    for label, secs in (("ofi_1m", 60), ("ofi_5m", 300), ("ofi_15m", 900)):
        buy = sell = 0.0
        n = 0
        for t in trades.values():
            if t["ts"] >= now_ts - secs:
                n += 1
                if t["taker_buy"]:
                    buy += t["size"]
                else:
                    sell += t["size"]
        out[label] = ((buy - sell) / (buy + sell)) if buy + sell else None
        if secs == 60:
            n_1m = n
        if secs == 900:
            baseline = n / 15.0
            out["tr_int"] = (math.log(max(n_1m, 1) / max(baseline, 1)) / 2
                             if baseline else None)
    return out

RETRAIN_EVERY = 3600           # 1 hour
RETRAIN_WINDOW_H = 24          # replay the last day of bars
VAL_HOLDOUT_H = 3              # newest hours validate, never train
RETRAIN_EPOCHS = 3
POLL_SECONDS = 30
BACKFILL_HOURS = 6             # seed the charts with recent history on first run
ONLINE_EPSILON = 0.05          # exploration during learning only
ONLINE_ALPHA = 0.3             # LinUCB bonus — reward gaps are ~0.1, so a big
                               # alpha buys exploration long past its value
DIR_BONUS = 0.1                # reward credit for a correct-sign deviation —
                               # cashes the measured 60%+ direction skill
SIGMA_FLOOR = 5.0              # $ floor for the per-horizon vol estimate
BAND_CAP = 200.0               # max published 80%-band width in dollars —
                               # wider tells a viewer nothing actionable;
                               # when the true spread exceeds it, coverage
                               # will honestly read below 80%


def _cap_band(pred: float, lo: float, hi: float,
              horizon: int = 5) -> tuple[int, int]:
    """Shrink a band toward its prediction, keeping its asymmetry ratio.
    Horizon-aware cap: $200 at +1/+5m (statistically free there), $400 at
    +15/+30m — a flat $200 crushed long-horizon coverage to ~43% because
    honest 80% bands there need $260-330."""
    cap = BAND_CAP if horizon <= 5 else 2 * BAND_CAP
    width = hi - lo
    if width <= cap:
        return int(lo), int(hi)
    k = cap / width
    return int(pred - (pred - lo) * k), int(pred + (hi - pred) * k)
CAL_HORIZONS = (1, 5, 15, 30)  # calibrated-winner meta-arm: every horizon
CAL_WINDOW = 30                # trailing scored residuals the calibrator sees
CAL_MIN = 10                   # need this many residuals before correcting
CAL_TRAIL = 50                 # scored rows per arm used to pick the winner
KB_LOG_NAME = "kalshi_binary_log.jsonl"  # binary-call arm: own log — a
                               # probability row doesn't fit the ledger schema
KB_MAX_ROWS = 20_000
KB_BET_LOG_NAME = "kb_bets.jsonl"  # one-shot paper bets on KXBTC15M
KB_BET_MAX_PRICE_C = 80        # entries only below 80 cents (fee+spread
                               # make higher entries -EV; tightened from
                               # 85 on ledger evidence, 2026-08-21)
KB_BET_EDGE_C = 3              # edge (cents) that triggers an EARLY strike —
                               # tuned on a leakage-free chronological
                               # backtest (tests/tune_bet_thresholds.py):
                               # under exactly-one-bet, higher thresholds
                               # only convert chosen strikes into forced
                               # late longshots; 3c beat 5c on held-out
                               # windows (58% vs 52% win, +11.7 vs +7.5
                               # cents/bet)
KB_BET_FORCE_S = 180           # exactly-one-bet rule: if no edge appeared,
                               # forced entry in the last 3 min of the window
KB_BET_DOOR_C = 75             # closing-door strike: if the called side's
                               # price reaches 75c it is about to cross the
                               # 85c legality cap — take it while it's legal
HF_LOG_NAME = "human_feedback.jsonl"  # scripts/feedback.py appends here
HF_WEIGHT = 0.15               # RLHF blend: comparable to DIR_BONUS so the
                               # human tilts learning but can't drown the
                               # realized-price reward (typically O(1))
HF_LOOKBACK_S = 1800           # a view guides predictions made in the next
                               # 30 min, then expires


def _horizon_sigma(feat: dict, horizon: int) -> float:
    """Live dollar volatility of an h-minute move (per-min sigma x sqrt(h))."""
    return max(feat["vol_30m"] * math.sqrt(horizon) * feat["price"], SIGMA_FLOOR)


def _k_of(agent, arm: int) -> float:
    """Action multiple for an arm index — DQN carries its own bin support."""
    return agent.bins[arm] if hasattr(agent, "bins") else config.K_FACTORS[arm]


def _vol_delta(k: float, feat: dict, horizon: int) -> int:
    return int(round(k * _horizon_sigma(feat, horizon)))


def _hit_band(sigma: float | None) -> float:
    """Vol-scaled hit tolerance: max($5 floor, 10% of the horizon sigma)."""
    if not sigma:
        return config.HIT_BAND
    return max(config.HIT_BAND, config.HIT_BAND_VOL * sigma)


def _bandit_reward(pred: float, actual: float, price_now: float,
                   delta: int, band: float | None = None) -> float:
    """Prediction reward + direction credit for correct-sign deviations."""
    r = reward(pred, actual, shaped=True, band=band)
    if delta and (delta > 0) == (actual > price_now) and actual != price_now:
        r += DIR_BONUS
    return r


def _band_map(ledger: list[dict]) -> dict:
    """Rolling 80% conformal band per arm x horizon: 10th/90th percentiles
    of the last 100 scored residuals (actual - pred). Reporting layer only —
    it never alters any model's point prediction."""
    res: dict = {}
    for row in ledger:
        if row["actual"] is None:
            continue
        res.setdefault((row["variant"], row["horizon"]), []).append(
            row["actual"] - row["pred"])
    out = {}
    for k, v in res.items():
        v = sorted(v[-100:])
        if len(v) >= 20:
            out[k] = (int(v[int(0.1 * len(v))]),
                      int(v[min(len(v) - 1, int(0.9 * len(v)))]))
    return out


def _pm_view(arms: dict, feat: dict, snap: dict | None,
             brti: dict | None, m: dict | None = None) -> dict | None:
    """The Robinhood/Kalshi BTC-15-min market + our model's P(up) beside it.

    Their contract: BRTI 60s-avg at window close >= at window open. Our
    estimate: t8-h15's predicted delta distribution, thresholded at the
    strike (approximate — full-15-min sigma vs the remaining window)."""
    m = m or fetch_kalshi_btc15()
    if not m:
        return None
    out = dict(m)
    out["brti_now"] = brti["price"] if brti else None
    try:
        agents = arms.get("t8-h15")
        agent = agents.get(15) if agents else None
        base_price = out["brti_now"] or feat["price"]
        if agent is not None and m.get("strike") and base_price:
            z0 = (m["strike"] - base_price) / _horizon_sigma(feat, 15)
            probs = agent.probs(_context(VARIANTS["t8-h15"], feat, snap))
            out["model_p_up"] = round(sum(
                pb for k, pb in zip(agent.bins, probs) if k >= z0), 3)
    except Exception:
        pass
    return out


def _load_hf() -> list[dict]:
    path = RESULTS_DIR / HF_LOG_NAME
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-500:]


def _hf_bonus(hf: list[dict], row: dict) -> float:
    """RLHF shaping for t11: the latest human view recorded in the 30 min
    before this prediction was committed earns agreement credit on the
    committed delta's SIGN. Zero when there's no view, the view expired,
    or the arm predicted flat — so with no feedback t11 is exactly t2."""
    if not row["delta"]:
        return 0.0
    view = next((v for v in reversed(hf)
                 if row["made_ts"] - HF_LOOKBACK_S <= v["ts"] <= row["made_ts"]),
                None)
    if view is None:
        return 0.0
    return HF_WEIGHT * view["view"] * (1 if row["delta"] > 0 else -1)


def _kb_p_up(arms: dict, feat: dict, snap: dict | None, strike: float,
             base_price: float, mins_left: float) -> float | None:
    """Our P(window close >= strike): t8-h15's delta distribution above the
    strike, with sigma scaled to the time REMAINING in the window (the
    pm-panel version uses the full 15-min sigma, which drags P toward 0.5
    mid-window)."""
    agents = arms.get("t8-h15")
    agent = agents.get(15) if agents else None
    if agent is None or not strike or not base_price:
        return None
    try:
        z0 = ((strike - base_price)
              / _horizon_sigma(feat, max(mins_left, 1.0)))
        probs = agent.probs(_context(VARIANTS["t8-h15"], feat, snap))
        return round(sum(pb for k, pb in zip(agent.bins, probs) if k >= z0), 4)
    except Exception:
        return None


KB_LOGIT_PATH_NAME = "kb_logit.json"
KB_LOGIT_DIM = 12
KB_LOGIT_MIN_UPDATES = 400     # graduate to publishing once trained this far


def _kb_logit_features(feat: dict, snap: dict | None, strike: float,
                       base: float, mins_left: float,
                       k_pup: float | None) -> list[float]:
    """Direct binary-task features: the market's own opinion, the physical
    strike geometry, and the microstructure the t8 path never saw."""
    z0 = (strike - base) / _horizon_sigma(feat, max(mins_left, 1.0))
    s = snap or {}
    return [
        1.0,                                          # intercept
        (k_pup - 0.5) * 2 if k_pup is not None else 0.0,   # market mid
        1.0 if k_pup is not None else 0.0,            # quote present
        max(-4.0, min(4.0, -z0)),                     # above-strike z
        mins_left / 15.0,                             # window phase
        max(-4.0, min(4.0, -z0)) * (mins_left / 15.0),  # z x phase
        s.get("ofi_1m") or 0.0,
        s.get("ofi_5m") or 0.0,
        s.get("imb") or 0.0,
        feat.get("ret_5m", 0.0) * 1e4 / 30,
        feat.get("ret_15m", 0.0) * 1e4 / 60,
        _m_log_vol(feat),
    ]


def _m_log_vol(feat: dict) -> float:
    import math as _m
    return _m.log(max(feat.get("vol_ratio", 1.0), 1e-6)) / 2


SEL_POLICY_NAME = "kb_sel_policy.json"


def _sel_features(side: str, price_c: float, p_model: float,
                  mins_left: float, forced: bool) -> list[float]:
    """Selector features — every one knowable at the instant the mandatory
    bet is placed (no fixed T-x anchor, nothing from the future): the taken
    side's model prob, its entry price, model-vs-price edge, how deep into
    the window the entry landed, and whether it was a forced entry."""
    p_side = p_model if side == "yes" else 1.0 - p_model
    return [1.0, p_side, price_c / 100.0,
            (100.0 * p_side - price_c) / 100.0,
            min(max(mins_left, 0.0), 15.0) / 15.0,
            1.0 if forced else 0.0,
            1.0 if side == "yes" else 0.0]


def _sel_predict(w: list[float], x: list[float]) -> float:
    z = max(-30.0, min(30.0, sum(wi * xi for wi, xi in zip(w, x))))
    return 1.0 / (1.0 + math.exp(-z))


SEL_TARGET = 0.90   # selector precision target (RA benchmark chase)
SEL_DIM = 10        # 7 bet features + kb3 centered prob, conf, presence


def _sel_kb3_feats(k3_idx: dict, ticker: str, mins_left: float,
                   side: str) -> list[float]:
    """kb3 (online logit over 12 engineered inputs) prob of the taken side
    at the same minute — same-timestamp join, no future info. Centered at
    zero with an explicit presence flag so the majority of rows that
    predate kb3 contribute nothing (a raw 0.5 default let the kb3 weight
    drag the global bias down and depress every old prediction)."""
    p3 = k3_idx.get((ticker, round(mins_left)))
    if p3 is None:
        return [0.0, 0.0, 0.0]
    p3s = p3 if side == "yes" else 1.0 - p3
    return [p3s - 0.5, abs(p3s - 0.5) * 2.0, 1.0]


def _sel_training_set(kb: list[dict], bets: list[dict]) -> list[tuple]:
    """(features, win, weight, price_c, is_real_bet) samples, leakage-free:
    every settled real bet (weight 3 — the deployment distribution), plus
    counterfactual bets derived from settled per-minute kb rows (bet the
    called side at that minute's market quote; features frozen at that
    minute, outcome from the window close — standard supervised, no
    look-ahead). Hindsight enters only through settled outcomes."""
    k3_idx = {(r["ticker"], round(r["mins_left"])): r["p_up"]
              for r in kb if r.get("variant") == "kb3"}
    out = []
    for b in bets:
        if b.get("win") is None:
            continue
        x = _sel_features(b["side"], b["price_c"], b["p_model"],
                          b["mins_left"], bool(b.get("forced")))
        x += _sel_kb3_feats(k3_idx, b["ticker"], b["mins_left"], b["side"])
        out.append((x, b["win"], 3.0, b["price_c"], True))
    for r in kb:
        if (r.get("variant", "kb") != "kb" or r.get("hit") is None
                or r.get("mkt_p_up") is None):
            continue
        side = "yes" if r["call"] else "no"
        price_c = 100.0 * (r["mkt_p_up"] if side == "yes"
                           else 1.0 - r["mkt_p_up"])
        if not 1.0 <= price_c <= 99.0:
            continue
        x = _sel_features(side, price_c, r["p_up"], r["mins_left"], False)
        x += _sel_kb3_feats(k3_idx, r["ticker"], r["mins_left"], side)
        out.append((x, int(r["hit"]), 1.0, price_c, False))
    return out


def _train_sel_model(samples: list[tuple], epochs: int = 40) -> list[float]:
    """Weighted SGD logistic regression, deterministic (fixed order, decaying
    lr) so a retrain is reproducible from the same ledgers."""
    d = len(samples[0][0])
    w = [0.0] * d
    for ep in range(epochs):
        lr = 0.2 / (1.0 + 0.15 * ep)
        for x, y, wt, _, _ in samples:
            g = (_sel_predict(w, x) - y) * wt * lr
            for i in range(d):
                w[i] -= g * x[i]
    return [round(v, 5) for v in w]


def _sel_operating_point(w: list[float], samples: list[tuple],
                         target: float = SEL_TARGET) -> dict:
    """Tune the keep-threshold theta on settled bets: keep iff predicted
    win prob >= max(theta, break-even (price+fee)/100). Among thetas with
    precision >= target pick the max simulated profit; if none reaches the
    target, best (precision, profit) with met_target=False."""
    # scan the FULL pool (real + counterfactual): the real-bet subset is
    # too small/noisy to place theta, and the validated chronological
    # backtest (tests/selector93_backtest.py) tuned on the full pool
    pool = samples
    scored = []
    for x, y, _, price_c, _ in pool:
        fee = math.ceil(7.0 * (price_c / 100.0) * (1.0 - price_c / 100.0))
        scored.append((_sel_predict(w, x), y, price_c, fee))
    cands = []
    for t in range(50, 96):
        th = t / 100.0
        kept = [(y, pc, f) for p, y, pc, f in scored
                if p >= max(th, (pc + f) / 100.0)]
        if len(kept) < 15:
            break
        prec = sum(y for y, _, _ in kept) / len(kept)
        pnl = sum((100.0 - pc - f) if y else (-pc - f) for y, pc, f in kept)
        cands.append({"theta": th, "precision": round(prec, 3),
                      "coverage": round(len(kept) / len(scored), 3),
                      "profit_c": round(pnl, 1), "n_kept": len(kept)})
    if not cands:
        return {"theta": 0.5, "precision": None, "coverage": None,
                "profit_c": None, "n_kept": 0, "met_target": False}
    qual = [c for c in cands if c["precision"] >= target]
    if qual:
        best = max(qual, key=lambda c: c["profit_c"])
        return {**best, "met_target": True}
    best = max(cands, key=lambda c: (c["precision"], c["profit_c"]))
    return {**best, "met_target": False}


def _maybe_retrain_selector(kb: list[dict], bets: list[dict],
                            policy: dict | None, now) -> dict | None:
    """Selector = a bet-level EV model: at the instant the bidding model
    places the mandatory bet (edge strike, closing door, or forced — no
    fixed T-x), it predicts THAT bet's win probability and keeps the bet
    iff the prediction clears both the tuned precision-0.8 threshold and
    the break-even line. Retrains once daily at 19:00 PT and is FROZEN in
    between (a missing model — e.g. schema upgrade — retrains at once)."""
    due_at = now.replace(hour=19, minute=0, second=0, microsecond=0)
    if now < due_at:
        due_at -= timedelta(days=1)
    if (policy and len(policy.get("w") or []) == SEL_DIM
            and policy.get("tuned_ts", 0) >= due_at.timestamp()):
        return policy
    samples = _sel_training_set(kb, bets)
    if len(samples) < 100:
        return policy
    w = _train_sel_model(samples)
    op = _sel_operating_point(w, samples)
    policy = {"tuned_ts": int(now.timestamp()),
              "tuned_at": now.isoformat(),
              "kind": "bet_ev_logit", "w": w, **op,
              "n_train": len(samples),
              "n_bets": sum(1 for s in samples if s[4]),
              "taus": {v: _kb_conf_threshold(kb, v)
                       for v in ("kb", "kb2", "kb3", "kbf")}}
    path = RESULTS_DIR / SEL_POLICY_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(policy))
    tmp.replace(path)
    print(f"{now:%H:%M:%S} selector bet-EV model (re)trained "
          f"(n={len(samples)}, theta={policy['theta']}, "
          f"train precision={policy['precision']}) — frozen until "
          f"next 19:00 PT")
    return policy


def _kb_conf_threshold(kb: list[dict], variant: str,
                       target: float = 0.8) -> dict | None:
    """Precision-targeted operating point: the smallest confidence
    threshold tau such that calls with max(p, 1-p) >= tau hit >= target
    precision on this variant's trailing settled rows (maximizing coverage
    subject to the precision floor). None until 25 qualifying samples."""
    rows = [r for r in kb if r.get("variant", "kb") == variant
            and r["actual"] is not None][-600:]
    if len(rows) < 50:
        return None
    best = None
    for t100 in range(50, 96):
        tau = t100 / 100
        sel = [r for r in rows if max(r["p_up"], 1 - r["p_up"]) >= tau]
        if len(sel) < 25:
            break
        prec = sum(r["hit"] for r in sel) / len(sel)
        if prec >= target:
            best = {"tau": tau, "precision": round(prec, 3),
                    "coverage": round(len(sel) / len(rows), 3)}
            break
    return best


def _kb_blend_weights(kb: list[dict]) -> dict:
    """Per-phase least-squares weight for p_final = w*p_model + (1-w)*mkt,
    fit on settled quoted calls (w = cov(pm-mk, y-mk)/var(pm-mk), clamped
    [0,1]). A data-chosen blend is >= the better input on the fit set —
    'beat the market' becomes 'learn the residual corrections'."""
    out = {}
    for ph in ("early", "mid", "late"):
        rows = [r for r in kb
                if r["actual"] is not None and r.get("mkt_p_up") is not None
                and r.get("variant", "kb") == "kb"
                and _kb_phase(r["mins_left"]) == ph][-500:]
        if len(rows) < 30:
            out[ph] = 0.5
            continue
        num = den = 0.0
        for r in rows:
            pm = r.get("p_cal", r["p_up"])
            d = pm - r["mkt_p_up"]
            num += d * (r["actual"] - r["mkt_p_up"])
            den += d * d
        out[ph] = max(0.0, min(1.0, num / den)) if den > 1e-9 else 0.5
    return out


def _kb_phase(mins_left: float) -> str:
    return "early" if mins_left >= 10 else "mid" if mins_left >= 5 else "late"


def _kb_cal_weights(kb: list[dict]) -> dict:
    """Per-phase calibration of kb's P(up): shrink toward 0.5 by a weight
    fit on the phase's own settled calls (least squares in centered coords,
    w = cov(p-.5, y-.5) / var(p-.5), clamped to [0, 1.2]). Eval showed the
    early window at Brier 0.32 — worse than always saying 50% — while late
    calls were sharp; this learns exactly how much early confidence to keep.
    w=1 (no change) until a phase has 20 settled calls; raw p is preserved
    per row so refits always see uncalibrated inputs."""
    out = {}
    for ph in ("early", "mid", "late"):
        # rows without p_raw predate the raw/calibrated split — fitting on
        # their calibrated p would bias the weight, so exclude them
        rows = [r for r in kb if r["actual"] is not None
                and r.get("p_raw") is not None
                and _kb_phase(r["mins_left"]) == ph][-400:]
        if len(rows) < 20:
            out[ph] = 1.0
            continue
        num = sum((r["p_raw"] - 0.5) * (r["actual"] - 0.5) for r in rows)
        den = sum((r["p_raw"] - 0.5) ** 2 for r in rows)
        out[ph] = max(0.0, min(1.2, num / den)) if den > 1e-9 else 1.0
    return out


def _load_kb() -> list[dict]:
    path = RESULTS_DIR / KB_LOG_NAME
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


KB_SEL_BET_LOG_NAME = "kb_bets_sel.jsonl"  # selector-gated A/B shadow bets


def _class_prf(kb: list[dict], variant: str = "kbf") -> dict:
    """Per-class precision/recall for a binary variant (the deliverable's
    headline metric: every window called, no abstention)."""
    rows = [r for r in kb if r.get("variant") == variant
            and r["actual"] is not None]
    out: dict = {"n": len(rows)}
    for cls, want in (("up", 1), ("down", 0)):
        called = [r for r in rows if r["call"] == want]
        actual = [r for r in rows if r["actual"] == want]
        out[cls] = {
            "prec": round(sum(r["hit"] for r in called) / len(called), 4)
                    if called else None,
            "recall": round(sum(1 for r in actual if r["call"] == want)
                            / len(actual), 4) if actual else None,
            "n_called": len(called), "n_actual": len(actual)}
    return out


def _load_kb_bets(name: str = KB_BET_LOG_NAME) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _kb_bets_summary(bets: list[dict]) -> dict | None:
    if not bets:
        return None
    done = [b for b in bets if b["actual"] is not None]
    out = {"n": len(bets), "settled": len(done), "last": bets[-1]}
    if done:
        out["wins"] = sum(b["win"] for b in done)
        out["pnl_c"] = round(sum(b["pnl_c"] for b in done), 1)
    return out


def _kb_summary(kb: list[dict], variant: str = "kb") -> dict | None:
    """Status-page view: last call + trailing accuracy/Brier, ours vs the
    market's own implied probability on the same settled windows."""
    kb = [r for r in kb if r.get("variant", "kb") == variant]
    if not kb:
        return None
    sc = [r for r in kb if r["actual"] is not None][-100:]
    out = {"last": kb[-1], "scored": len(sc)}
    if sc:
        out["acc"] = round(sum(r["hit"] for r in sc) / len(sc), 3)
        out["brier"] = round(sum(r["brier"] for r in sc) / len(sc), 4)
        mk = [r["mkt_brier"] for r in sc if r.get("mkt_brier") is not None]
        if mk:
            out["mkt_brier"] = round(sum(mk) / len(mk), 4)
    return out


def _calibration_map(ledger: list[dict]) -> dict:
    """Dual-window agreement-median calibration for EVERY model arm x
    horizon: full strength when the trailing-30 and trailing-10 residual
    medians agree on sign, zero when they disagree (a trend turn) — the
    design validated by the cal meta-arm, replacing the old half-strength
    treatment-only intercept. rp is excluded (it IS the chart-replay
    definition) and meta rows never self-calibrate. Capped at apply time."""
    res: dict = {}
    for row in ledger:
        if (row["actual"] is None
                or row["variant"].startswith(("rp", "consensus", "cal"))):
            continue
        res.setdefault((row["variant"], row["horizon"]), []).append(
            row["actual"] - row["pred"])
    out = {}
    for k, v in res.items():
        if len(v) >= CAL_MIN:
            adj = _calibration_adj(v[-CAL_WINDOW:])
            if adj:
                out[k] = adj
    return out


def _history_snapshot(ledger: list[dict], kb: list[dict],
                      kb_bets: list[dict], now_ts: int) -> dict:
    """Trailing-6h online metrics per arm x horizon + binary/bet summaries —
    the payload appended to metrics_history.jsonl at every retrain."""
    cut = now_ts - 6 * 3600
    sc = [r for r in ledger if r["actual"] is not None and r["made_ts"] >= cut]
    base = {(r["horizon"], r["made_ts"]): abs(r["actual"] - r["price_now"])
            for r in sc}
    arms_out: dict = {}
    for r in sc:
        arms_out.setdefault(r["variant"], {}).setdefault(
            r["horizon"], []).append(r)
    snap_arms = {}
    for v, hs in arms_out.items():
        snap_arms[v] = {}
        for h, rows in hs.items():
            if len(rows) < 10:
                continue
            naive = [base[(h, r["made_ts"])] for r in rows
                     if (h, r["made_ts"]) in base]
            moved = [r for r in rows if r["delta"]]
            banded = [r for r in rows if r.get("in_band") is not None]
            snap_arms[v][f"h{h}"] = {
                "n": len(rows),
                "mae": round(sum(r["abs_err"] for r in rows) / len(rows), 2),
                "rmse": round(M.rmse([r["err"] for r in rows]) or 0, 2),
                "mase": round(M.mase([r["abs_err"] for r in rows], naive)
                              or 0, 3) if naive else None,
                "dir": round(sum(
                    1 for r in moved
                    if (r["pred"] > r["price_now"]) == (r["actual"] > r["price_now"])
                ) / len(moved), 3) if moved else None,
                "cov": round(sum(r["in_band"] for r in banded) / len(banded), 3)
                       if banded else None,
                "sharp": round(M.sharpness(
                    [r["lo"] for r in banded], [r["hi"] for r in banded])
                    or 0, 1) if banded else None,
            }
    out = {"arms": snap_arms}
    kb_done = [r for r in kb if r["actual"] is not None and r["made_ts"] >= cut]
    if kb_done:
        br = sum(r["brier"] for r in kb_done) / len(kb_done)
        mk = [r["mkt_brier"] for r in kb_done if r.get("mkt_brier") is not None]
        out["kb"] = {"n": len(kb_done),
                     "acc": round(sum(r["hit"] for r in kb_done) / len(kb_done), 3),
                     "brier": round(br, 4),
                     "mkt_brier": round(sum(mk) / len(mk), 4) if mk else None,
                     "bss_mkt": round(M.brier_skill(br, sum(mk) / len(mk)), 3)
                                if mk else None,
                     "bss_clim": round(M.brier_skill(br, 0.25), 3)}
    bets_done = [b for b in kb_bets if b["actual"] is not None]
    if bets_done:
        net = [b["pnl_c"] - M.kalshi_fee_c(b["price_c"]) for b in bets_done]
        cum = []
        s = 0.0
        for x in net:
            s += x
            cum.append(s)
        out["bets"] = {"n": len(bets_done),
                       "wins": sum(b["win"] for b in bets_done),
                       "pnl_c": round(sum(b["pnl_c"] for b in bets_done), 1),
                       "pnl_net_c": round(sum(net), 1),
                       "max_dd_c": round(M.max_drawdown(cum), 1)}
    return out


def _winner_variant(ledger: list[dict], horizon: int) -> str | None:
    """The arm with the best trailing MAE at this horizon (last CAL_TRAIL
    scored rows, min 20) — the model the calibrated meta-arm shadows."""
    errs: dict[str, list[float]] = {}
    for row in ledger:
        if (row["actual"] is None or row["horizon"] != horizon
                or row["variant"].startswith(("consensus", "cal-"))
                or row["variant"] not in VARIANTS):  # retired arms' fossil
            continue                                 # rows can't win
        errs.setdefault(row["variant"], []).append(row["abs_err"])
    mae = {v: sum(e[-CAL_TRAIL:]) / len(e[-CAL_TRAIL:])
           for v, e in errs.items() if len(e) >= 20}
    return min(mae, key=mae.get) if mae else None


def _calibration_adj(residuals: list[float]) -> int | None:
    """Dollar correction added to the winner's +15m prediction, fit on the
    winner's own trailing residuals (actual - pred; positive = it ran low).

    This is the meta-arm's entire edge over the winner it shadows, and the
    A/B against the shrunk (x0.5, ±0.5 sigma-capped) bias intercept the
    treatment arms already receive. Returns a whole-dollar adjustment, or
    None/0 for "leave the winner's prediction alone".

    Design: dual-window agreement median. Full window (30) and recent
    window (10) must agree on sign — disagreement marks a trend turn, the
    exact regime where the original full-strength intercept doubled the
    miss, so we stand down there. When they agree, apply the smaller of
    the two magnitudes UNSHRUNK: persistent bias gets corrected at full
    strength, turns get zero.
    """
    long_med = statistics.median(residuals)
    short_med = statistics.median(residuals[-10:])
    if long_med * short_med <= 0:
        return 0
    return int(long_med if abs(long_med) < abs(short_med) else short_med)


LEDGER_MAX_ROWS = 60_000       # ~2 weeks of both arms; keeps rewrites cheap


def _heartbeat() -> None:
    """Touch alive_at without a full status build — called on the network
    retry path and between retrain arms, so the watchdog doesn't shoot a
    daemon that is alive but mid-retrain or riding out an outage."""
    try:
        st = json.loads(STATUS.read_text())
        st["alive_at"] = time.time()
        STATUS.write_text(json.dumps(st))
    except Exception:
        pass


def _qtable_path(variant: str) -> Path:
    return RESULTS_DIR / f"q_table_online_{variant}.json"


def _bandit_path(variant: str) -> Path:
    return RESULTS_DIR / f"linucb_{variant}.json"


def _load_dqn(variant: str, horizons: list[int], dim: int,
              kind: str = "dqn"):
    cls = LSTMDistAgent if kind == "seq" else DistDQNAgent
    prefix = "lstm" if kind == "seq" else "dqn"
    agents = {}
    for h in horizons:
        own = RESULTS_DIR / f"{prefix}_{variant}.pt"
        batch = RESULTS_DIR / f"{prefix}_h{h}.pt"
        try:
            if own.exists():
                agents[h] = cls.load(own)
            elif batch.exists():
                agents[h] = cls.load(batch)
            else:
                agents[h] = cls(dim)
        except Exception:
            agents[h] = cls(dim)
        if agents[h].dim != dim:
            agents[h] = cls(dim)
    return agents


def _load_bandits(variant: str, horizons: list[int], dim: int,
                  n_arms: int | None = None, kind: str = "linucb"):
    cls = LinearQAgent if kind == "linearq" else LinUCBAgent
    path = _bandit_path(variant)
    want_arms = n_arms or len(config.K_FACTORS)  # bandits act on K_FACTORS
    agents = {h: cls(dim, n_arms=want_arms) for h in horizons}
    ok = lambda d: d["dim"] == dim and d.get("n_arms", want_arms) == want_arms
    if path.exists():
        raw = json.loads(path.read_text())
        loaded = {h: cls.from_dict(raw[f"h{h}"]) for h in horizons
                  if f"h{h}" in raw and ok(raw[f"h{h}"])}
        if loaded:
            agents = loaded
    elif kind == "linearq" and (RESULTS_DIR / "linear_q.json").exists():
        raw = json.loads((RESULTS_DIR / "linear_q.json").read_text())
        for h in horizons:  # warm-start from the 60-day batch weights
            if f"h{h}" in raw and ok(raw[f"h{h}"]):
                agents[h] = cls.from_dict(raw[f"h{h}"])
    for a in agents.values():
        if isinstance(a, LinUCBAgent):
            a.alpha = ONLINE_ALPHA
    return agents


def _load_snapshots() -> list[dict]:
    path = RESULTS_DIR / SNAP_FILE_NAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[-3000:]


def _nearest_snap(snaps: list[dict], ts: int) -> dict | None:
    best = None
    for s in snaps:  # snaps are appended in time order
        if s["ts"] <= ts:
            best = s
        else:
            break
    if best and ts - best["ts"] <= SNAP_MAX_AGE_S:
        return best
    return None


def _load_agents(variant: str, horizons: list[int]) -> dict[int, TabularQAgent]:
    """This arm's own tables if checkpointed, else warm-start from batch."""
    own = _qtable_path(variant)
    raw = json.loads(own.read_text()) if own.exists() else (
        json.loads(BATCH_QTABLE.read_text()) if BATCH_QTABLE.exists() else {})
    agents: dict[int, TabularQAgent] = {}
    for horizon in horizons:
        agent = TabularQAgent(seed=int(horizon))
        agent.epsilon = ONLINE_EPSILON
        key = f"h{horizon}" if own.exists() else f"tabular-q-shaped_h{horizon}"
        for state_str, qs in raw.get(key, {}).items():
            agent.q[tuple(map(int, state_str.split("|")))] = list(qs)
        agents[horizon] = agent
    return agents


def _checkpoint(variant: str, agents: dict) -> None:
    if any(isinstance(a, DistDQNAgent) for a in agents.values()):
        for a in agents.values():
            prefix = "lstm" if isinstance(a, LSTMDistAgent) else "dqn"
            a.save(RESULTS_DIR / f"{prefix}_{variant}.pt")
        return
    if any(isinstance(a, BANDIT_TYPES) for a in agents.values()):
        payload = {f"h{h}": a.to_dict() for h, a in agents.items()}
        target = _bandit_path(variant)
    else:
        payload = {f"h{h}": {"|".join(map(str, s)): qs for s, qs in a.q.items()}
                   for h, a in agents.items()}
        target = _qtable_path(variant)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(target)


def _load_ledger() -> list[dict]:
    if not PRED_LOG.exists():
        return []
    rows = []
    for line in PRED_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _save_ledger(rows: list[dict]) -> None:
    tmp = PRED_LOG.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows[-LEDGER_MAX_ROWS:]))
    tmp.replace(PRED_LOG)


def _predict_at(variant: str, agents: dict[int, TabularQAgent],
                bars_upto: list[dict], fng, slot_ts: int,
                horizons: list[int], spec: dict | None = None,
                snap: dict | None = None,
                bias: dict | None = None,
                bands: dict | None = None) -> list[dict]:
    """Greedy (no exploration) integer predictions committed at slot_ts.

    Treatment bandits act in vol-scaled units (arm k -> k * sigma_h dollars)
    and get an adaptive-intercept bias correction; the frozen control keeps
    its fixed delta grid, untouched.
    """
    feat = compute_features(bars_upto, fng)
    state = discretize(feat)
    spec = spec or {}
    rows = []
    for horizon in horizons:
        # every row carries its commit-time sigma_h: the vol-scaled hit
        # band and DQN z-scores both need it at scoring time
        row_extra: dict = {"sigma": round(_horizon_sigma(feat, horizon), 2)}
        adj = 0
        if spec.get("agent") == "replay":
            # copy the past chart segment forward, verbatim
            agent = None
            delta = (int(round(feat["price"] - bars_upto[-1 - horizon]["close"]))
                     if len(bars_upto) > horizon else 0)
        else:
            agent = agents[horizon]
        if isinstance(agent, BANDIT_TYPES):
            x = _context(spec, feat, snap)
            # commitments are GREEDY — exploration (UCB bonus / epsilon)
            # belongs in warm-up and hourly replay, never in the published
            # forecast (it was leaking ±3-sigma probes into predictions)
            arm = agent.select(x, greedy=True)
            delta = _vol_delta(_k_of(agent, arm), feat, horizon)
            row_extra["x"] = [round(v, 5) for v in x]
            row_extra["arm"] = arm
        elif agent is not None:
            qs = agent.q.get(state)
            delta = (config.ACTION_DELTAS[max(range(len(qs)), key=lambda i: qs[i])]
                     if qs else 0)
        if agent is not None:
            # every model arm is calibrated: dual-window agreement median
            # at full strength (it zeroes itself at turns), capped ±0.5σ_h
            raw_adj = (bias or {}).get((variant, horizon), 0)
            cap = 0.5 * _horizon_sigma(feat, horizon)
            adj = int(max(-cap, min(cap, raw_adj)))
            if adj:
                row_extra["bias_adj"] = adj
        pred = int(feat["price"] + delta + adj)
        if isinstance(agent, DistDQNAgent):
            # native 80% interval from the model's own predicted distribution
            probs = agent.probs(x)
            sig = _horizon_sigma(feat, horizon)
            cum = 0.0
            k10 = agent.bins[0]
            k90 = agent.bins[-1]
            for k, pb in zip(agent.bins, probs):
                if cum < 0.10 <= cum + pb:
                    k10 = k
                if cum < 0.90 <= cum + pb:
                    k90 = k
                cum += pb
            row_extra["lo"], row_extra["hi"] = _cap_band(
                pred, pred + k10 * sig, pred + k90 * sig, horizon)
            row_extra["band_src"] = "native"
        else:
            band = (bands or {}).get((variant, horizon))
            if band:
                row_extra["lo"], row_extra["hi"] = _cap_band(
                    pred, pred + band[0], pred + band[1], horizon)
        rows.append({
            "variant": variant,
            "made_ts": slot_ts, "target_ts": slot_ts + horizon * 60,
            "horizon": horizon, "price_now": feat["price"],
            "pred": pred, "delta": delta,
            "state": list(state), "actual": None, "abs_err": None, "hit": None,
            **row_extra,
        })
    return rows


def _greedy_mae(agent: TabularQAgent, episodes: list) -> float:
    errs = [abs(e.price_now + agent.delta_for(
        agent.act(e.state, e.price_now, explore=False)) - e.price_future)
        for e in episodes]
    return sum(errs) / len(errs) if errs else 0.0


def _bandit_val_mae(agent: LinUCBAgent, spec: dict, episodes: list,
                    snaps: list[dict]) -> float:
    errs = []
    for e in episodes:
        x = _context(spec, e.features, _nearest_snap(snaps, e.minute_ts))
        a = agent.select(x, greedy=True)
        d = _vol_delta(_k_of(agent, a), e.features, e.horizon_min)
        errs.append(abs(e.price_now + d - e.price_future))
    return sum(errs) / len(errs) if errs else 0.0


def retrain_all(arms: dict[str, dict[int, TabularQAgent]],
                snaps: list[dict], fng=None) -> dict:
    """Hourly retrain of every arm's OWN tables on the same replay window,
    each guarded by the hold-out no-regression gate. Bandits replay too —
    12 live updates/hour is a starvation diet next to control's replay."""
    now = datetime.now(tz=config.PACIFIC)
    bars = fetch_range(now - timedelta(hours=RETRAIN_WINDOW_H), now)
    cut_ts = int((now - timedelta(hours=VAL_HOLDOUT_H)).timestamp())
    # same fng the live commits see — a None here was a train/serve mismatch
    train_eps = build_episodes({"replay": [b for b in bars if b["ts"] < cut_ts]},
                               {"replay": fng})
    val_eps = build_episodes({"replay": [b for b in bars if b["ts"] >= cut_ts]},
                             {"replay": fng})
    info: dict = {"at": now.isoformat(), "train_episodes": len(train_eps),
                  "val_episodes": len(val_eps), "epochs": RETRAIN_EPOCHS,
                  "arms": {}}
    for variant, agents in arms.items():
        _heartbeat()  # a full-fleet retrain can outlast the watchdog window
        if not agents:
            continue  # replay baseline has no model
        if any(isinstance(a, BANDIT_TYPES) for a in agents.values()):
            spec = VARIANTS[variant]
            gate = {}
            for h, agent in agents.items():
                eps = [e for e in train_eps if e.horizon_min == h]
                veps = [e for e in val_eps if e.horizon_min == h]
                if isinstance(agent, DistDQNAgent):
                    import copy as _copy
                    before = _copy.deepcopy(agent.net.state_dict())
                elif isinstance(agent, LinearQAgent):
                    before = ([w.copy() for w in agent.w], list(agent.pulls))
                else:
                    before = ([a.copy() for a in agent.A],
                              [b.copy() for b in agent.b], list(agent.pulls))
                before_mae = _bandit_val_mae(agent, spec, veps, snaps)
                for e in eps:
                    x = _context(spec, e.features,
                                 _nearest_snap(snaps, e.minute_ts))
                    if isinstance(agent, DistDQNAgent):
                        agent.learn_dist(x, (e.price_future - e.price_now)
                                         / _horizon_sigma(e.features, h))
                        continue
                    a = agent.select(x)
                    d = _vol_delta(config.K_FACTORS[a], e.features, h)
                    agent.update(x, a, _bandit_reward(
                        e.price_now + d, e.price_future, e.price_now, d,
                        band=_hit_band(_horizon_sigma(e.features, h))))
                after_mae = _bandit_val_mae(agent, spec, veps, snaps)
                reverted = after_mae > before_mae
                if reverted:
                    if isinstance(agent, DistDQNAgent):
                        agent.net.load_state_dict(before)
                    elif isinstance(agent, LinearQAgent):
                        agent.w, agent.pulls = before
                    else:
                        agent.A, agent.b, agent.pulls = before
                gate[f"h{h}"] = {"val_mae_before": round(before_mae, 2),
                                 "val_mae_after": round(after_mae, 2),
                                 "reverted": reverted}
            _checkpoint(variant, agents)
            info["arms"][variant] = gate
            continue
        horizons = list(agents)
        eps = [e for e in train_eps if e.horizon_min in horizons]
        before = {h: {s: list(qs) for s, qs in agents[h].q.items()}
                  for h in horizons}
        before_mae = {h: _greedy_mae(agents[h],
                                     [e for e in val_eps if e.horizon_min == h])
                      for h in horizons}
        rng = random.Random(int(now.timestamp()) // RETRAIN_EVERY)
        for _ in range(RETRAIN_EPOCHS):
            rng.shuffle(eps)
            for e in eps:
                agent = agents[e.horizon_min]
                a = agent.act(e.state, e.price_now, explore=True)
                pred = e.price_now + agent.delta_for(a)
                agent.learn(e.state, a, reward(
                    pred, e.price_future, shaped=True,
                    band=_hit_band(_horizon_sigma(e.features, e.horizon_min))))
        gate = {}
        for h in horizons:
            after_mae = _greedy_mae(agents[h],
                                    [e for e in val_eps if e.horizon_min == h])
            reverted = after_mae > before_mae[h]
            if reverted:  # chased noise — roll this horizon back
                agents[h].q.clear()
                agents[h].q.update(before[h])
            gate[f"h{h}"] = {"val_mae_before": round(before_mae[h], 2),
                             "val_mae_after": round(after_mae, 2),
                             "reverted": reverted}
        _checkpoint(variant, agents)
        info["arms"][variant] = gate
    return info


def run(once: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    arms = {v: (_load_bandits(v, spec["horizons"], _ctx_dim(spec),
                              kind=spec["agent"])
                if spec.get("agent") in ("linucb", "linearq")
                else _load_dqn(v, spec["horizons"], _ctx_dim(spec),
                               kind=spec["agent"])
                if spec.get("agent") in ("dqn", "seq")
                else {} if spec.get("agent") == "replay"
                else _load_agents(v, spec["horizons"]))
            for v, spec in VARIANTS.items()}
    snaps = _load_snapshots()
    trades: dict[int, dict] = {}  # rolling taker-flow store (last ~20 min)
    cold_bandits = {v for v, spec in VARIANTS.items()
                    if spec.get("agent") in ("linucb", "linearq", "dqn", "seq")
                    and all(a.total_pulls == 0 for a in arms[v].values())}
    ledger = _load_ledger()
    made = {(r.get("variant", "control"), r["made_ts"], r["horizon"])
            for r in ledger}
    kb = _load_kb()
    kb_made = {(r.get("variant", "kb"), r["ticker"], r["made_ts"]) for r in kb}
    kb_bets = _load_kb_bets()
    kb_bet_tickers = {b["ticker"] for b in kb_bets}
    kb_sel_bets = _load_kb_bets(KB_SEL_BET_LOG_NAME)
    kb_sel_tickers = {b["ticker"] for b in kb_sel_bets}
    try:
        kb_policy = json.loads((RESULTS_DIR / SEL_POLICY_NAME).read_text())
    except Exception:
        kb_policy = None
    logit_path = RESULTS_DIR / KB_LOGIT_PATH_NAME
    try:
        kb_logit = (BinaryLogit.from_dict(json.loads(logit_path.read_text()))
                    if logit_path.exists() else BinaryLogit(KB_LOGIT_DIM))
        if kb_logit.dim != KB_LOGIT_DIM:
            kb_logit = BinaryLogit(KB_LOGIT_DIM)
    except Exception:
        kb_logit = BinaryLogit(KB_LOGIT_DIM)
    last_retrain_slot = int(time.time()) // RETRAIN_EVERY
    retrain_info: dict = {}
    retrains = 0
    online_updates = 0
    warmed_this_session: set[str] = set()
    started = time.time()
    print(f"experiment runner up — arms: {', '.join(VARIANTS)}")

    last_bars: list[dict] = []
    last_bars_ts = 0.0
    while True:
        try:
            now = datetime.now(tz=config.PACIFIC)
            now_ts = int(now.timestamp())
            try:
                bars = fetch_range(now - timedelta(hours=BACKFILL_HOURS + 1),
                                   now)
                last_bars, last_bars_ts = bars, time.time()
            except Exception:
                # one flaky vendor must not stall kb/bets/scoring: reuse the
                # last good bars if they're recent enough to trust
                if not last_bars or time.time() - last_bars_ts > 900:
                    raise
                bars = last_bars
            by_ts = {b["ts"]: b for b in bars}
            fng = fetch_fear_greed().get(now.date().isoformat())

            # 0a. stream a live-feature snapshot (t3's extra context)
            brti = fetch_brti_composite()
            spot = bars[-1]["close"] if bars else None
            mark = fetch_deribit_mark()
            book = fetch_book_stats()
            # Kalshi BTC-15-min market: fetched once per poll, feeds both
            # the t10 context features and the status page's pm panel
            pm_mkt = fetch_kalshi_btc15()
            k_pup = k_spread = k_tleft = k_dist_bp = k_close_ts = None
            if pm_mkt:
                yb, ya = pm_mkt.get("yes_bid"), pm_mkt.get("yes_ask")
                if yb and ya:
                    k_pup = (yb + ya) / 200
                    k_spread = (ya - yb) / 100
                elif pm_mkt.get("last_price"):
                    k_pup = pm_mkt["last_price"] / 100
                if pm_mkt.get("strike") and spot:
                    k_dist_bp = ((spot - pm_mkt["strike"])
                                 / pm_mkt["strike"] * 1e4)
                if pm_mkt.get("close_time"):
                    try:
                        k_close_ts = int(datetime.fromisoformat(
                            pm_mkt["close_time"].replace("Z", "+00:00")
                        ).timestamp())
                        k_tleft = max(0.0, min(1.0, (k_close_ts - now_ts) / 900))
                    except ValueError:
                        pass
            snap = {
                "ts": now_ts,
                "funding": fetch_okx_funding_rate(),
                "basis_bp": ((mark - spot) / spot * 1e4
                             if mark and spot else None),
                "disp_bp": None, "gap_bp": None,
                "fee": fetch_mempool_fee(),
                "imb": book["imb"] if book else None,
                "spread_bp": book["spread_bp"] if book else None,
                "k_pup": k_pup, "k_dist_bp": k_dist_bp,
                "k_tleft": k_tleft, "k_spread": k_spread,
            }
            for t in fetch_recent_trades():   # order-flow store (t6)
                trades[t["id"]] = t
            trades = {i: t for i, t in trades.items()
                      if t["ts"] >= now_ts - 1200}
            snap.update(_ofi_stats(trades, now_ts))
            try:  # frozen crypto-LLM reads the news tape (cached per headline)
                senti = sentiment_snapshot()
            except Exception:
                senti = {"sent": None, "news_n": None}
            snap["sent"] = senti["sent"]
            snap["news_n"] = senti["news_n"]
            hour_ago = next((s0 for s0 in reversed(snaps)
                             if s0["ts"] <= now_ts - 3300
                             and s0.get("sent") is not None), None)
            snap["sent_mom"] = (round(snap["sent"] - hour_ago["sent"], 4)
                                if snap["sent"] is not None and hour_ago
                                else None)
            if brti and spot:
                cons_prices = list(brti["constituents"].values())
                mean_p = sum(cons_prices) / len(cons_prices)
                var = sum((p - mean_p) ** 2 for p in cons_prices) / len(cons_prices)
                snap["disp_bp"] = (var ** 0.5) / mean_p * 1e4
                snap["gap_bp"] = (brti["price"] - spot) / spot * 1e4
            snaps.append(snap)
            snaps = snaps[-3000:]
            with (RESULTS_DIR / SNAP_FILE_NAME).open("a") as f:
                f.write(json.dumps(snap) + "\n")

            # 0b. cold-start: warm a fresh bandit up on recent history so its
            #     first real bets aren't pure exploration
            if cold_bandits:
                warmed_this_session.update(cold_bandits)
                warm_eps = build_episodes({"warm": bars}, {"warm": fng})
                for v in sorted(cold_bandits):
                    vspec = VARIANTS[v]
                    for h, agent in arms[v].items():
                        for e in (e for e in warm_eps if e.horizon_min == h):
                            x = _context(vspec, e.features,
                                         _nearest_snap(snaps, e.minute_ts))
                            if isinstance(agent, DistDQNAgent):
                                agent.learn_dist(
                                    x, (e.price_future - e.price_now)
                                    / _horizon_sigma(e.features, h))
                                continue
                            a = agent.select(x)
                            d = _vol_delta(config.K_FACTORS[a], e.features, h)
                            agent.update(x, a, _bandit_reward(
                                e.price_now + d, e.price_future,
                                e.price_now, d,
                                band=_hit_band(_horizon_sigma(e.features, h))))
                    _checkpoint(v, arms[v])
                    print(f"warmed up {v} on {len(warm_eps)} recent episodes")
                cold_bandits.clear()

            # 1. commit predictions per arm, at each arm's own cadence
            #    (uniformly covers live boundaries and first-run backfill)
            bias = _calibration_map(ledger)
            bands = _band_map(ledger)
            new_preds = 0
            for variant, spec in VARIANTS.items():
                step = spec["predict_every"]
                first = ((now_ts - BACKFILL_HOURS * 3600) // step + 1) * step
                if variant in warmed_this_session:
                    # LEAKAGE GUARD: this arm warm-trained on the recent
                    # window — backfilling those same slots would be
                    # train-on-test. Live slots only.
                    first = max(first, (int(started) // step + 1) * step)
                for slot_ts in range(first, now_ts + 1, step):
                    if ((variant, slot_ts, spec["horizons"][0]) in made
                            or slot_ts not in by_ts):
                        continue
                    upto = [b for b in bars if b["ts"] <= slot_ts]
                    if len(upto) < config.LOOKBACK_MIN:
                        continue
                    live_slot = slot_ts >= int(started)
                    rows = _predict_at(variant, arms[variant], upto, fng,
                                       slot_ts, spec["horizons"], spec=spec,
                                       snap=_nearest_snap(snaps, slot_ts),
                                       # LEAKAGE GUARD: bias/band maps are
                                       # built from the ledger as of NOW, so
                                       # they only apply to live commits
                                       bias=bias if live_slot else None,
                                       bands=bands if live_slot else None)
                    ledger.extend(rows)
                    made.update((r["variant"], r["made_ts"], r["horizon"])
                                for r in rows)
                    new_preds += len(rows)

            # 1b. consensus: for every 5-min slot where all +5 predictors have
            #     committed, poll them (ctl-h5, t2-h5, persistence) and take
            #     the median as OUR final level for t+5. Scored like any
            #     predictor; never bets, never learns.
            CONS = {1: "consensus-h1", 5: "consensus",
                    15: "consensus-h15", 30: "consensus-h30"}
            have_consensus = {(r["variant"], r["made_ts"]) for r in ledger
                              if r["variant"].startswith("consensus")}
            by_slot: dict[tuple, dict] = {}
            for r in ledger:
                ch = CONS.get(r["horizon"])
                if ch is None or r.get("state") is None and not r["variant"].startswith(("h", "rp", "t")):
                    continue
                fam = ("h" if r["variant"] == f"h{r['horizon']}"
                       else r["variant"].split("-")[0])
                if r["variant"] in (f"h{r['horizon']}", f"rp-h{r['horizon']}",
                                    f"t2-h{r['horizon']}", f"t6-h{r['horizon']}",
                                    f"t7-h{r['horizon']}", f"t8-h{r['horizon']}") \
                        and (ch, r["made_ts"]) not in have_consensus:
                    by_slot.setdefault((ch, r["made_ts"], r["horizon"]),
                                       {})[fam] = r
            # skill-weighted poll per horizon: drop the worst trailing voter
            for (ch, slot_ts, hz), votes in sorted(by_slot.items()):
                if "h" not in votes or "t2" not in votes:
                    continue  # need at least control + one treatment
                # LEAKAGE GUARD: the trailing worst-voter drop + bands are
                # computed from the ledger as of NOW — live slots only,
                # never backfill (same rule as the other meta-arms)
                if slot_ts < int(started):
                    continue
                base_fam = "h"
                trail: dict[str, float] = {}
                for fam in votes:
                    v = f"{fam}-h{hz}" if fam != "h" else f"h{hz}"
                    errs = [r["abs_err"] for r in ledger
                            if r["variant"] == v and r["actual"] is not None][-50:]
                    if len(errs) >= 20:
                        trail[fam] = sum(errs) / len(errs)
                worst = max(trail, key=trail.get) if len(trail) >= 3 else None
                base = votes[base_fam]
                polled = sorted([v["pred"] for name, v in votes.items()
                                 if name != worst]
                                + [int(base["price_now"])])
                mid = len(polled) // 2
                final = (polled[mid] if len(polled) % 2
                         else (polled[mid - 1] + polled[mid]) // 2)
                crow = {
                    "variant": ch, "made_ts": slot_ts,
                    "target_ts": slot_ts + hz * 60, "horizon": hz,
                    "price_now": base["price_now"],
                    "pred": int(final),  # median of the surviving voters
                    "delta": int(final) - int(base["price_now"]),
                    "votes": polled, "state": None,
                    # base arm's sigma so scoring uses the vol-scaled band
                    "sigma": base.get("sigma"),
                    "actual": None, "abs_err": None, "hit": None,
                }
                cband = bands.get((ch, hz))
                if cband:
                    crow["lo"], crow["hi"] = _cap_band(
                        crow["pred"], crow["pred"] + cband[0],
                        crow["pred"] + cband[1], hz)
                ledger.append(crow)
                new_preds += 1

            # 1c. calibrated winner: shadow whichever arm currently leads on
            #     trailing MAE at each horizon and re-center its prediction
            #     with a full-strength correction fit on that arm's own
            #     trailing residuals. Reporting-layer meta-arm like the
            #     consensus — no model state, never learns.
            #     LEAKAGE GUARD: the residual window is built from the
            #     ledger as of NOW, so only unscored live commits are
            #     shadowed (backfill would calibrate on its own future).
            for cal_h in CAL_HORIZONS:
                cal_v = f"cal-h{cal_h}"
                have_cal = {r["made_ts"] for r in ledger
                            if r["variant"] == cal_v}
                win = _winner_variant(ledger, cal_h)
                if not win:
                    continue
                res = [r["actual"] - r["pred"] for r in ledger
                       if r["variant"] == win and r["horizon"] == cal_h
                       and r["actual"] is not None][-CAL_WINDOW:]
                adj = (_calibration_adj(res) or 0) if len(res) >= CAL_MIN else 0
                shadow = [r for r in ledger
                          if r["variant"] == win and r["horizon"] == cal_h
                          and r["actual"] is None
                          and r["made_ts"] >= int(started)
                          and r["made_ts"] not in have_cal]
                for src in shadow:
                    crow = {
                        "variant": cal_v, "made_ts": src["made_ts"],
                        "target_ts": src["target_ts"], "horizon": cal_h,
                        "price_now": src["price_now"],
                        "pred": int(src["pred"] + adj),
                        "delta": int(src["pred"] + adj - src["price_now"]),
                        "src": win, "cal_adj": int(adj), "state": None,
                        # source arm's sigma so scoring uses the vol-scaled band
                        "sigma": src.get("sigma"),
                        "actual": None, "abs_err": None, "hit": None,
                    }
                    cband = bands.get((cal_v, cal_h))
                    if cband:
                        crow["lo"], crow["hi"] = _cap_band(
                            crow["pred"], crow["pred"] + cband[0],
                            crow["pred"] + cband[1], cal_h)
                    ledger.append(crow)
                    new_preds += 1

            # 1d. kalshi-binary arm (kb): our YES/NO call on the live
            #     KXBTC15M contract itself — P(close >= strike) from
            #     t8-h15's distribution, sigma scaled to the remaining
            #     window. One call per MINUTE per contract (the probability
            #     path across the whole window), market's implied P logged
            #     beside ours, settled at the quarter-hour window close
            #     (9:00, 9:15, ...) against the bar closing there (honest
            #     proxy for the official BRTI 60s-average settle). Never
            #     bets, never learns; lives in its own log (binary schema).
            # BINARY TREATMENT SET (one row per variant per minute):
            #   kb  — control: t8's distribution + per-phase calibration
            #   kb2 — market blend: w*p_cal + (1-w)*market mid, w fit per
            #         phase on settled history
            #   kb3 — direct online logistic on the binary task (market
            #         mid + strike z + phase + order flow/book/momentum),
            #         one SGD step per settled call
            # Each row also carries the precision-0.8 operating point:
            # conf=1 when max(p,1-p) clears the variant's auto-tuned tau.
            kb_changed = False
            slot1 = now_ts // 60 * 60
            if (pm_mkt and pm_mkt.get("strike") and k_close_ts
                    and ("kb", pm_mkt["ticker"], slot1) not in kb_made
                    and k_close_ts - now_ts >= 60):
                kfeat = compute_features(bars, fng)
                base = (brti["price"] if brti else None) or kfeat["price"]
                mins_left = (k_close_ts - now_ts) / 60
                p = _kb_p_up(arms, kfeat, snap,
                             pm_mkt["strike"], base, mins_left)
                if p is not None:
                    w = _kb_cal_weights(kb)[_kb_phase(mins_left)]
                    # clamp: a calibration weight >1 can push p past 1.0
                    p_cal = round(min(0.99, max(0.01, 0.5 + w * (p - 0.5))), 4)
                    bw = _kb_blend_weights(kb)[_kb_phase(mins_left)]
                    p_blend = (round(min(0.99, max(0.01,
                               bw * p_cal + (1 - bw) * k_pup)), 4)
                               if k_pup is not None else p_cal)
                    bx = _kb_logit_features(kfeat, snap, pm_mkt["strike"],
                                            base, mins_left, k_pup)
                    p_logit = round(kb_logit.predict(bx), 4)
                    common = {
                        "ticker": pm_mkt["ticker"], "made_ts": slot1,
                        "close_ts": k_close_ts, "strike": pm_mkt["strike"],
                        "base": round(base, 2),
                        "mins_left": round(mins_left, 1),
                        "mkt_p_up": k_pup, "actual": None, "hit": None,
                    }
                    for variant, pv, extra in (
                            ("kb", p_cal, {"p_raw": p, "cal_w": round(w, 3)}),
                            ("kb2", p_blend, {"w_mkt": round(bw, 3),
                                              "p_cal": p_cal}),
                            ("kb3", p_logit,
                             {"bx": [round(v, 5) for v in bx],
                              "trained": kb_logit.updates})):
                        tau = (kb_policy.get("taus") or {}).get(variant) \
                            if kb_policy else None
                        row = {**common, "variant": variant, "p_up": pv,
                               "call": int(pv >= 0.5)}
                        if tau:
                            row["tau"] = tau["tau"]
                            row["conf"] = int(max(pv, 1 - pv) >= tau["tau"])
                        row.update(extra)
                        kb.append(row)
                        kb_made.add((variant, pm_mkt["ticker"], slot1))
                    # kbf — THE deliverable: one definitive call per window
                    # at T-3 min (every window called; no abstention), the
                    # operating point where per-class precision/recall
                    # cleared 80/80 in backtest (tests/window_call_eval.py)
                    if (mins_left <= 3.4
                            and ("kbf", pm_mkt["ticker"], 0) not in kb_made):
                        kb.append({**common, "variant": "kbf", "p_up": p_cal,
                                   "call": int(p_cal >= 0.5),
                                   "decide_at": round(mins_left, 1)})
                        kb_made.add(("kbf", pm_mkt["ticker"], 0))
                    kb_changed = True
            logit_changed = False
            for r in kb:
                if r["actual"] is not None or now_ts < r["close_ts"]:
                    continue
                # bars are keyed by bucket START: the final minute of the
                # window is bucket close_ts-60, whose close lands exactly at
                # the close — bar[close_ts] would settle a minute late
                settle_bar = by_ts.get(r["close_ts"] - 60)
                if settle_bar is None:
                    continue
                outcome = int(settle_bar["close"] >= r["strike"])
                r["actual"] = outcome
                r["hit"] = int(r["call"] == outcome)
                r["brier"] = round((r["p_up"] - outcome) ** 2, 4)
                if r.get("mkt_p_up") is not None:
                    r["mkt_brier"] = round((r["mkt_p_up"] - outcome) ** 2, 4)
                if r.get("bx") and len(r["bx"]) == kb_logit.dim:
                    kb_logit.update(r["bx"], outcome)
                    logit_changed = True
                kb_changed = True
            if logit_changed:
                tmp = (RESULTS_DIR / KB_LOGIT_PATH_NAME).with_suffix(".tmp")
                tmp.write_text(json.dumps(kb_logit.to_dict()))
                tmp.replace(RESULTS_DIR / KB_LOGIT_PATH_NAME)
            if kb_changed:
                kb = kb[-KB_MAX_ROWS:]
                tmp = (RESULTS_DIR / KB_LOG_NAME).with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in kb))
                tmp.replace(RESULTS_DIR / KB_LOG_NAME)

            # 1e. EXACTLY one paper bet per window (at most one, at least
            #     one), entry only under 85c, settled at close. The arm
            #     picks its moment: strike early the first minute the
            #     called side shows >= KB_BET_EDGE_C cents of edge; if no
            #     edge ever appears, forced entry in the final 3 minutes —
            #     called side if it's under 85c, else the only legal side.
            bets_changed = False
            sel_mirrored = False
            if (pm_mkt and pm_mkt.get("strike") and k_close_ts
                    and pm_mkt["ticker"] not in kb_bet_tickers
                    and k_close_ts - now_ts >= 60):
                yb, ya = pm_mkt.get("yes_bid"), pm_mkt.get("yes_ask")
                p_now = next((r["p_up"] for r in reversed(kb)
                              if r["ticker"] == pm_mkt["ticker"]
                              and r.get("variant", "kb") == "kb"), None)
                if p_now is not None and yb and ya:
                    # bet the side the model calls — value-betting the other
                    # side degenerates into buying longshots (calibration
                    # shrinks p toward 0.5, so our p is systematically less
                    # extreme than a late market)
                    yes_price, no_price = ya, 100 - yb
                    called = ("yes", yes_price, 100 * p_now - yes_price) \
                        if p_now >= 0.5 else \
                        ("no", no_price, 100 * (1 - p_now) - no_price)
                    other = ("no", no_price, 100 * (1 - p_now) - no_price) \
                        if called[0] == "yes" else \
                        ("yes", yes_price, 100 * p_now - yes_price)
                    forced = k_close_ts - now_ts <= KB_BET_FORCE_S
                    door = called[1] >= KB_BET_DOOR_C  # about to be priced out
                    best = None
                    if called[1] < KB_BET_MAX_PRICE_C \
                            and (called[2] >= KB_BET_EDGE_C or forced or door):
                        best = called
                    elif forced and other[1] < KB_BET_MAX_PRICE_C:
                        best = other  # called side priced out (>=85c)
                    if best:
                        kb_bets.append({
                            "ticker": pm_mkt["ticker"], "made_ts": now_ts,
                            "close_ts": k_close_ts,
                            "strike": pm_mkt["strike"],
                            "side": best[0], "price_c": round(best[1], 1),
                            "edge_c": round(best[2], 1),
                            "forced": forced and best[2] < KB_BET_EDGE_C,
                            "p_model": p_now,
                            "mins_left": round((k_close_ts - now_ts) / 60, 1),
                            "actual": None, "win": None, "pnl_c": None,
                        })
                        kb_bet_tickers.add(pm_mkt["ticker"])
                        bets_changed = True
                        # SELECTOR = same-tick judge of this SAME bet:
                        # whenever the bidding model strikes (edge, door,
                        # or forced — no fixed T-x), a bet-level EV model
                        # scores THIS bet from placement-time features
                        # only and keeps it iff predicted win prob clears
                        # both the tuned precision threshold and the
                        # break-even price+fee. Verdict stamped on the
                        # control row either way, so skips are auditable.
                        nb = kb_bets[-1]
                        if (pm_mkt["ticker"] not in kb_sel_tickers
                                and kb_policy
                                and len(kb_policy.get("w") or [])
                                == SEL_DIM):
                            lk3 = next(
                                (r for r in reversed(kb)
                                 if r.get("variant") == "kb3"
                                 and r["ticker"] == pm_mkt["ticker"]), None)
                            k3i = ({(nb["ticker"],
                                     round(nb["mins_left"])): lk3["p_up"]}
                                   if lk3 else {})
                            pw = _sel_predict(
                                kb_policy["w"],
                                _sel_features(nb["side"], nb["price_c"],
                                              nb["p_model"],
                                              nb["mins_left"],
                                              bool(nb.get("forced")))
                                + _sel_kb3_feats(k3i, nb["ticker"],
                                                 nb["mins_left"],
                                                 nb["side"]))
                            fee = math.ceil(
                                7.0 * (nb["price_c"] / 100.0)
                                * (1.0 - nb["price_c"] / 100.0))
                            keep = pw >= max(
                                kb_policy.get("theta", 0.5),
                                (nb["price_c"] + fee) / 100.0)
                            nb["sel_p_win"] = round(pw, 4)
                            nb["sel_keep"] = int(keep)
                            if keep:
                                kb_sel_bets.append(dict(nb))
                                kb_sel_tickers.add(pm_mkt["ticker"])
                                sel_mirrored = True
            for b in kb_bets:
                if b["actual"] is not None or now_ts < b["close_ts"]:
                    continue
                settle_bar = by_ts.get(b["close_ts"] - 60)
                if settle_bar is None:
                    continue
                outcome = int(settle_bar["close"] >= b["strike"])
                b["actual"] = outcome
                b["win"] = int((b["side"] == "yes") == bool(outcome))
                b["pnl_c"] = round((100 - b["price_c"]) if b["win"]
                                   else -b["price_c"], 1)
                bets_changed = True
            # 1e-B. settle the selector's kept bets (entries are mirrored
            #     from the bidding model at placement time, above)
            sel_changed = sel_mirrored
            for b in kb_sel_bets:
                if b["actual"] is not None or now_ts < b["close_ts"]:
                    continue
                settle_bar = by_ts.get(b["close_ts"] - 60)
                if settle_bar is None:
                    continue
                outcome = int(settle_bar["close"] >= b["strike"])
                b["actual"] = outcome
                b["win"] = int((b["side"] == "yes") == bool(outcome))
                b["pnl_c"] = round((100 - b["price_c"]) if b["win"]
                                   else -b["price_c"], 1)
                sel_changed = True
            if sel_changed:
                tmp = (RESULTS_DIR / KB_SEL_BET_LOG_NAME).with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(b) + "\n"
                                       for b in kb_sel_bets))
                tmp.replace(RESULTS_DIR / KB_SEL_BET_LOG_NAME)
            if bets_changed:
                tmp = (RESULTS_DIR / KB_BET_LOG_NAME).with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(b) + "\n" for b in kb_bets))
                tmp.replace(RESULTS_DIR / KB_BET_LOG_NAME)

            # 2. score matured predictions (all arms alike) and LEARN from
            #    each one: an immediate Q-update on the committed (s, a)
            kb_policy = _maybe_retrain_selector(kb, kb_bets, kb_policy, now)
            hf_rows = _load_hf()  # human views for t11's blended reward
            scored = 0
            for row in ledger:
                if row["actual"] is not None:
                    continue
                # Coinbase buckets are [ts, ts+60): only settle once the
                # target bucket is COMPLETE, else "actual" is a mid-minute
                # price that depends on poll phase
                if row["target_ts"] + 60 > now_ts:
                    continue
                bar = by_ts.get(row["target_ts"])
                if bar is None:
                    continue
                row["actual"] = bar["close"]
                row["err"] = round(row["pred"] - bar["close"], 2)  # + = predicted high
                row["abs_err"] = abs(row["err"])
                band = _hit_band(row.get("sigma"))
                row["hit"] = row["abs_err"] <= band
                row["hit_band"] = round(band, 1)
                if row.get("lo") is not None:
                    row["in_band"] = row["lo"] <= bar["close"] <= row["hi"]
                scored += 1
                agents = arms.get(row["variant"])
                agent = agents.get(row["horizon"]) if agents else None
                if agent is None:
                    continue
                if isinstance(agent, DistDQNAgent) and row.get("x") \
                        and row.get("sigma"):
                    z = (row["actual"] - row["price_now"]) / row["sigma"]
                    if len(row["x"]) == agent.dim:
                        agent.learn_dist(row["x"], z)
                        online_updates += 1
                elif isinstance(agent, BANDIT_TYPES) and row.get("x") \
                        and row.get("arm") is not None:
                    if row["arm"] < agent.n_arms and len(row["x"]) == agent.dim:
                        r = _bandit_reward(row["pred"], row["actual"],
                                           row["price_now"], row["delta"],
                                           band=band)
                        if row["variant"].startswith("t11"):
                            r += _hf_bonus(hf_rows, row)
                        agent.update(row["x"], row["arm"], r)
                        online_updates += 1
                elif isinstance(agent, TabularQAgent) and row.get("state") \
                        and row["delta"] in config.ACTION_DELTAS:
                    r = reward(row["pred"], row["actual"], shaped=True,
                               band=band)
                    agent.learn(tuple(row["state"]),
                                config.ACTION_DELTAS.index(row["delta"]), r)
                    online_updates += 1
            if scored:  # persist what was just learned
                for variant, agents in arms.items():
                    if agents:
                        _checkpoint(variant, agents)
            if new_preds or scored:
                ledger = ledger[-LEDGER_MAX_ROWS:]
                _save_ledger(ledger)

            # 3. hourly retrain, each arm separately
            hour_slot = now_ts // RETRAIN_EVERY
            if hour_slot > last_retrain_slot:
                print(f"{now:%H:%M:%S} hourly retrain (all arms)...")
                retrain_info = retrain_all(arms, snaps, fng)
                retrains += 1
                last_retrain_slot = hour_slot
                try:  # persist gate outcomes + trailing snapshot forever
                    append_history("retrain", {
                        "gate": retrain_info.get("arms", {}),
                        "online": _history_snapshot(ledger, kb, kb_bets,
                                                    now_ts)})
                except Exception:
                    pass
                # hourly trim of the append-only logs (last 5000 lines)
                for log_name in (SNAP_FILE_NAME, "learning_log.jsonl"):
                    lp = RESULTS_DIR / log_name
                    try:
                        lines = lp.read_text().splitlines()
                        if len(lines) > 5000:
                            tmp = lp.with_suffix(".tmp")
                            tmp.write_text("\n".join(lines[-5000:]) + "\n")
                            tmp.replace(lp)
                    except OSError:
                        pass

            # 4. status + actual-price series for the charts
            feat = compute_features(bars, fng)
            recent = [{"ts": b["ts"], "c": b["close"]}
                      for b in bars if b["ts"] >= now_ts - BACKFILL_HOURS * 3600]
            (RESULTS_DIR / "recent_prices.json").write_text(json.dumps(recent))
            STATUS.write_text(json.dumps({
                "alive_at": time.time(), "started_at": started,
                "price_now": feat["price"],
                "brti": brti,
                "pm": _pm_view(arms, feat, snap, brti, pm_mkt),
                "kalshi_binary": _kb_summary(kb),
                "kb_treatments": {
                    v: {"scored": s.get("scored"), "acc": s.get("acc"),
                        "brier": s.get("brier"),
                        "mkt_brier": s.get("mkt_brier"),
                        "prec80": _kb_conf_threshold(kb, v)}
                    for v in ("kb", "kb2", "kb3")
                    if (s := _kb_summary(kb, v)) is not None},
                "kb_logit_updates": kb_logit.updates,
                "kbf": _class_prf(kb),
                "sel_policy": {k: kb_policy.get(k) for k in
                               ("tuned_at", "kind", "theta", "precision",
                                "coverage", "profit_c", "n_kept",
                                "met_target", "n_train", "n_bets", "taus")}
                              if kb_policy else None,
                "kb_bets": _kb_bets_summary(kb_bets),
                "kb_bets_sel": _kb_bets_summary(kb_sel_bets),
                "human_feedback": {"n": len(hf_rows),
                                   "last": hf_rows[-1] if hf_rows else None},
                "live_features": snap,
                "variants": {v: {"predict_every_min": s["predict_every"] // 60,
                                 "horizons": s["horizons"],
                                 "agent": s.get("agent", "tabular"),
                                 "states_known": {
                                     h: (a.total_pulls if isinstance(a, BANDIT_TYPES)
                                         else len(a.q))
                                     for h, a in arms[v].items()}}
                             for v, s in VARIANTS.items()},
                "consensus": next(
                    (dict(r) for r in reversed(ledger)
                     if r["variant"] == "consensus" and r["actual"] is None),
                    None),
                "online_updates_session": online_updates,
                "retrain_every_min": RETRAIN_EVERY // 60,
                "retrains_this_session": retrains,
                "last_retrain": retrain_info or None,
                "predictions_total": len(ledger),
            }))
            # learning telemetry for the live "is it learning?" chart
            with (RESULTS_DIR / "learning_log.jsonl").open("a") as lf:
                lf.write(json.dumps({"ts": now_ts,
                                     "updates": online_updates,
                                     "retrains": retrains}) + "\n")
            if new_preds or scored:
                print(f"{now:%H:%M:%S} +{new_preds} predictions, {scored} scored "
                      f"({len(ledger)} total)")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"poll error (will retry): {exc}")
            _heartbeat()  # alive and retrying — an outage isn't a hang
        if once:
            for variant, agents in arms.items():
                if agents:  # replay baseline has no model
                    _checkpoint(variant, agents)
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
