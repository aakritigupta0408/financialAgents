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
from .agents import (DistDQNAgent, LinearQAgent, LinUCBAgent,
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
SNAP_FILE_NAME = "live_snapshots.jsonl"  # streamed-feature history (t3)
SNAP_MAX_AGE_S = 600   # a snapshot older than 10 min doesn't describe a slot

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PRED_LOG = RESULTS_DIR / "prediction_log.jsonl"
STATUS = RESULTS_DIR / "online_status.json"
BATCH_QTABLE = RESULTS_DIR / "q_table.json"

# Experiment arms: one stream per horizon, each predicting at its own natural
# cadence (cadence == horizon), each with its own Q-table file. The h15/h30
# tables warm-start from the original (control) model's batch tables; add new
# dicts here for future treatments.
# POLICY: a new treatment must pass scripts/offline_gate.py (MAE gate +
# duplicate check) BEFORE being added here. t3/t4/t5 were retired as
# offline-proven duplicates of t2/t6 (see results/offline_gate.json).
# All arms predict every 5 minutes (9:00, 9:05, 9:10…). CONTROL arms (h5/h15/
# h30, tabular Q) are frozen — do not touch. TREATMENT 2 (t2-*) is a LinUCB
# contextual bandit over the same 21 integer-delta arms; every learner uses
# the prediction reward: +1 exact integer match, else -|error|/100.
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
    "t7-h5": {"predict_every": 300, "horizons": [5], "agent": "linearq",
              "trend": True},
    "t7-h15": {"predict_every": 300, "horizons": [15], "agent": "linearq",
               "trend": True},
    "t7-h30": {"predict_every": 300, "horizons": [30], "agent": "linearq",
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
    "t8-h30": {"predict_every": 300, "horizons": [30], "agent": "dqn",
               "trend": True},
    # TREATMENT 9 = the L4 rung: LSTM over the raw 1m return stream,
    # distributional output, mode action (scripts/train_l4.py).
    "t9-h5": {"predict_every": 300, "horizons": [5], "agent": "seq"},
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
    "t6-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb",
              "live": True, "trend": True, "book": True, "llm": True,
              "ofi": True},
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
BIAS_WINDOW = 30               # trailing scored rows for online bias correction
                               # (~2.5h at 5-min cadence — reacts to trends
                               # the 50-row window lagged badly)
CAL_VARIANT = "cal-h15"        # calibrated-winner meta-arm (+15m only)
CAL_WINDOW = 30                # trailing scored residuals the calibrator sees
CAL_MIN = 10                   # need this many residuals before correcting
CAL_TRAIL = 50                 # scored rows per arm used to pick the winner
KB_LOG_NAME = "kalshi_binary_log.jsonl"  # binary-call arm: own log — a
                               # probability row doesn't fit the ledger schema
KB_MAX_ROWS = 20_000
KB_BET_LOG_NAME = "kb_bets.jsonl"  # one-shot paper bets on KXBTC15M
KB_BET_MAX_PRICE_C = 85        # broker rule: entries only below 85 cents
KB_BET_EDGE_C = 5              # min model-vs-market edge (cents) to spend
                               # the window's single allowed bet
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


def _bandit_reward(pred: float, actual: float, price_now: float,
                   delta: int) -> float:
    """Prediction reward + direction credit for correct-sign deviations."""
    r = reward(pred, actual, shaped=True)
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
        rows = [r for r in kb if r["actual"] is not None
                and _kb_phase(r["mins_left"]) == ph][-400:]
        if len(rows) < 20:
            out[ph] = 1.0
            continue
        num = sum((r.get("p_raw", r["p_up"]) - 0.5) * (r["actual"] - 0.5)
                  for r in rows)
        den = sum((r.get("p_raw", r["p_up"]) - 0.5) ** 2 for r in rows)
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


def _load_kb_bets() -> list[dict]:
    path = RESULTS_DIR / KB_BET_LOG_NAME
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


def _kb_summary(kb: list[dict]) -> dict | None:
    """Status-page view: last call + trailing accuracy/Brier, ours vs the
    market's own implied probability on the same settled windows."""
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


def _bias_map(ledger: list[dict]) -> dict:
    """Trailing median residual (actual - pred) per treatment arm x horizon —
    an adaptive intercept that removes conditional trend bias."""
    res: dict = {}
    for row in ledger:
        if row["actual"] is None or not row["variant"].startswith("t"):
            continue
        res.setdefault((row["variant"], row["horizon"]), []).append(
            row["actual"] - row["pred"])
    return {k: int(statistics.median(v[-BIAS_WINDOW:]))
            for k, v in res.items() if len(v) >= 10}


def _winner_variant(ledger: list[dict], horizon: int) -> str | None:
    """The arm with the best trailing MAE at this horizon (last CAL_TRAIL
    scored rows, min 20) — the model the calibrated meta-arm shadows."""
    errs: dict[str, list[float]] = {}
    for row in ledger:
        if (row["actual"] is None or row["horizon"] != horizon
                or row["variant"].startswith(("consensus", "cal-"))):
            continue
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
        row_extra: dict = {}
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
            # shrink (x0.5) and cap (±0.5 sigma_h) the bias intercept — the
            # uncapped version chased rallies and doubled the miss at turns
            raw_adj = (bias or {}).get((variant, horizon), 0)
            cap = 0.5 * _horizon_sigma(feat, horizon)
            adj = int(max(-cap, min(cap, 0.5 * raw_adj)))
            row_extra["x"] = [round(v, 5) for v in x]
            row_extra["arm"] = arm
            if isinstance(agent, DistDQNAgent):
                row_extra["sigma"] = round(_horizon_sigma(feat, horizon), 2)
            if adj:
                row_extra["bias_adj"] = adj
        elif agent is not None:
            qs = agent.q.get(state)
            delta = (config.ACTION_DELTAS[max(range(len(qs)), key=lambda i: qs[i])]
                     if qs else 0)
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
            row_extra["lo"] = int(pred + k10 * sig)
            row_extra["hi"] = int(pred + k90 * sig)
            row_extra["band_src"] = "native"
        else:
            band = (bands or {}).get((variant, horizon))
            if band:
                row_extra["lo"] = pred + band[0]
                row_extra["hi"] = pred + band[1]
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
                snaps: list[dict]) -> dict:
    """Hourly retrain of every arm's OWN tables on the same replay window,
    each guarded by the hold-out no-regression gate. Bandits replay too —
    12 live updates/hour is a starvation diet next to control's replay."""
    now = datetime.now(tz=config.PACIFIC)
    bars = fetch_range(now - timedelta(hours=RETRAIN_WINDOW_H), now)
    cut_ts = int((now - timedelta(hours=VAL_HOLDOUT_H)).timestamp())
    train_eps = build_episodes({"replay": [b for b in bars if b["ts"] < cut_ts]},
                               {"replay": None})
    val_eps = build_episodes({"replay": [b for b in bars if b["ts"] >= cut_ts]},
                             {"replay": None})
    info: dict = {"at": now.isoformat(), "train_episodes": len(train_eps),
                  "val_episodes": len(val_eps), "epochs": RETRAIN_EPOCHS,
                  "arms": {}}
    for variant, agents in arms.items():
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
                        e.price_now + d, e.price_future, e.price_now, d))
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
                agent.learn(e.state, a, reward(pred, e.price_future, shaped=True))
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
    kb_made = {(r["ticker"], r["made_ts"]) for r in kb}
    kb_bets = _load_kb_bets()
    kb_bet_tickers = {b["ticker"] for b in kb_bets}
    last_retrain_slot = int(time.time()) // RETRAIN_EVERY
    retrain_info: dict = {}
    retrains = 0
    online_updates = 0
    warmed_this_session: set[str] = set()
    started = time.time()
    print(f"experiment runner up — arms: {', '.join(VARIANTS)}")

    while True:
        try:
            now = datetime.now(tz=config.PACIFIC)
            now_ts = int(now.timestamp())
            bars = fetch_range(now - timedelta(hours=BACKFILL_HOURS + 1), now)
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
                warm_eps = build_episodes({"warm": bars}, {"warm": None})
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
                                e.price_now, d))
                    _checkpoint(v, arms[v])
                    print(f"warmed up {v} on {len(warm_eps)} recent episodes")
                cold_bandits.clear()

            # 1. commit predictions per arm, at each arm's own cadence
            #    (uniformly covers live boundaries and first-run backfill)
            bias = _bias_map(ledger)
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
                    "actual": None, "abs_err": None, "hit": None,
                }
                cband = bands.get((ch, hz))
                if cband:
                    crow["lo"] = crow["pred"] + cband[0]
                    crow["hi"] = crow["pred"] + cband[1]
                ledger.append(crow)
                new_preds += 1

            # 1c. calibrated winner: shadow whichever arm currently leads on
            #     trailing +15m MAE and re-center its committed prediction
            #     with a full-strength correction fit on that arm's own
            #     trailing residuals. Reporting-layer meta-arm like the
            #     consensus — no model state, never learns.
            #     LEAKAGE GUARD: the residual window is built from the
            #     ledger as of NOW, so only unscored live commits are
            #     shadowed (backfill would calibrate on its own future).
            have_cal = {r["made_ts"] for r in ledger
                        if r["variant"] == CAL_VARIANT}
            win = _winner_variant(ledger, 15)
            if win:
                res = [r["actual"] - r["pred"] for r in ledger
                       if r["variant"] == win and r["horizon"] == 15
                       and r["actual"] is not None][-CAL_WINDOW:]
                adj = (_calibration_adj(res) or 0) if len(res) >= CAL_MIN else 0
                shadow = [r for r in ledger
                          if r["variant"] == win and r["horizon"] == 15
                          and r["actual"] is None
                          and r["made_ts"] >= int(started)
                          and r["made_ts"] not in have_cal]
                for src in shadow:
                    crow = {
                        "variant": CAL_VARIANT, "made_ts": src["made_ts"],
                        "target_ts": src["target_ts"], "horizon": 15,
                        "price_now": src["price_now"],
                        "pred": int(src["pred"] + adj),
                        "delta": int(src["pred"] + adj - src["price_now"]),
                        "src": win, "cal_adj": int(adj), "state": None,
                        "actual": None, "abs_err": None, "hit": None,
                    }
                    cband = bands.get((CAL_VARIANT, 15))
                    if cband:
                        crow["lo"] = crow["pred"] + cband[0]
                        crow["hi"] = crow["pred"] + cband[1]
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
            kb_changed = False
            slot1 = now_ts // 60 * 60
            if (pm_mkt and pm_mkt.get("strike") and k_close_ts
                    and (pm_mkt["ticker"], slot1) not in kb_made
                    and k_close_ts - now_ts >= 60):
                kfeat = compute_features(bars, fng)
                base = (brti["price"] if brti else None) or kfeat["price"]
                mins_left = (k_close_ts - now_ts) / 60
                p = _kb_p_up(arms, kfeat, snap,
                             pm_mkt["strike"], base, mins_left)
                if p is not None:
                    w = _kb_cal_weights(kb)[_kb_phase(mins_left)]
                    p_cal = round(0.5 + w * (p - 0.5), 4)
                    kb.append({
                        "variant": "kb", "ticker": pm_mkt["ticker"],
                        "made_ts": slot1, "close_ts": k_close_ts,
                        "strike": pm_mkt["strike"],
                        "base": round(base, 2),
                        "mins_left": round(mins_left, 1),
                        "p_up": p_cal, "p_raw": p, "cal_w": round(w, 3),
                        "call": int(p_cal >= 0.5),
                        "mkt_p_up": k_pup,
                        "actual": None, "hit": None,
                    })
                    kb_made.add((pm_mkt["ticker"], slot1))
                    kb_changed = True
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
                kb_changed = True
            if kb_changed:
                kb = kb[-KB_MAX_ROWS:]
                tmp = (RESULTS_DIR / KB_LOG_NAME).with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in kb))
                tmp.replace(RESULTS_DIR / KB_LOG_NAME)

            # 1e. the ONE paper bet per window the broker allows: entry only
            #     under 85c, settled at close. The first minute where the
            #     calibrated model sees >= KB_BET_EDGE_C cents of edge over
            #     the quoted price spends the window's single bet.
            bets_changed = False
            if (pm_mkt and pm_mkt.get("strike") and k_close_ts
                    and pm_mkt["ticker"] not in kb_bet_tickers
                    and k_close_ts - now_ts >= 60):
                yb, ya = pm_mkt.get("yes_bid"), pm_mkt.get("yes_ask")
                p_now = next((r["p_up"] for r in reversed(kb)
                              if r["ticker"] == pm_mkt["ticker"]), None)
                if p_now is not None and yb and ya:
                    # bet only the side the model actually calls — value-
                    # betting the other side degenerates into buying
                    # longshots (calibration shrinks p toward 0.5, so our
                    # p is systematically less extreme than a late market)
                    cands = []
                    if p_now >= 0.5 and ya < KB_BET_MAX_PRICE_C:
                        cands.append(("yes", ya, 100 * p_now - ya))
                    no_price = 100 - yb
                    if p_now < 0.5 and no_price < KB_BET_MAX_PRICE_C:
                        cands.append(("no", no_price,
                                      100 * (1 - p_now) - no_price))
                    best = max(cands, key=lambda c: c[2], default=None)
                    if best and best[2] >= KB_BET_EDGE_C:
                        kb_bets.append({
                            "ticker": pm_mkt["ticker"], "made_ts": now_ts,
                            "close_ts": k_close_ts,
                            "strike": pm_mkt["strike"],
                            "side": best[0], "price_c": round(best[1], 1),
                            "edge_c": round(best[2], 1),
                            "p_model": p_now,
                            "mins_left": round((k_close_ts - now_ts) / 60, 1),
                            "actual": None, "win": None, "pnl_c": None,
                        })
                        kb_bet_tickers.add(pm_mkt["ticker"])
                        bets_changed = True
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
            if bets_changed:
                tmp = (RESULTS_DIR / KB_BET_LOG_NAME).with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(b) + "\n" for b in kb_bets))
                tmp.replace(RESULTS_DIR / KB_BET_LOG_NAME)

            # 2. score matured predictions (all arms alike) and LEARN from
            #    each one: an immediate Q-update on the committed (s, a)
            hf_rows = _load_hf()  # human views for t11's blended reward
            scored = 0
            for row in ledger:
                if row["actual"] is not None:
                    continue
                bar = by_ts.get(row["target_ts"])
                if bar is None:
                    continue
                row["actual"] = bar["close"]
                row["err"] = round(row["pred"] - bar["close"], 2)  # + = predicted high
                row["abs_err"] = abs(row["err"])
                row["hit"] = int(row["pred"]) == int(bar["close"])
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
                                           row["price_now"], row["delta"])
                        if row["variant"].startswith("t11"):
                            r += _hf_bonus(hf_rows, row)
                        agent.update(row["x"], row["arm"], r)
                        online_updates += 1
                elif isinstance(agent, TabularQAgent) and row.get("state") \
                        and row["delta"] in config.ACTION_DELTAS:
                    r = reward(row["pred"], row["actual"], shaped=True)
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
                retrain_info = retrain_all(arms, snaps)
                retrains += 1
                last_retrain_slot = hour_slot

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
                "kb_bets": _kb_bets_summary(kb_bets),
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
        if once:
            for variant, agents in arms.items():
                _checkpoint(variant, agents)
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
