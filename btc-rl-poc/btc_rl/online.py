"""Always-on experiment runner: one prediction stream per horizon.

Each horizon predicts at its own natural cadence (cadence == horizon) with
its own Q-table, so streams never contaminate each other:

  All arms predict every 5 minutes (9:00, 9:05, 9:10…), each with its own
  model; the objective is pure price prediction (no betting layer).

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
from .agents import LinearQAgent, LinUCBAgent, TabularQAgent

BANDIT_TYPES = (LinUCBAgent, LinearQAgent)  # shared select/update API
from .env import build_episodes, reward
from .features import (BOOK_DIM, LIVE_DIM, LLM_DIM, OFI_DIM, TREND_DIM,
                       book_feature_vector, compute_features, discretize,
                       feature_vector, live_feature_vector,
                       llm_feature_vector, ofi_feature_vector,
                       trend_feature_vector)
from .llm_sentiment import sentiment_snapshot
from .sources import (fetch_book_stats, fetch_brti_composite,
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
# All arms predict every 5 minutes (9:00, 9:05, 9:10…). CONTROL arms (h5/h15/
# h30, tabular Q) are frozen — do not touch. TREATMENT 2 (t2-*) is a LinUCB
# contextual bandit over the same 21 integer-delta arms; every learner uses
# the prediction reward: +1 exact integer match, else -|error|/100.
VARIANTS: dict[str, dict] = {
    "h5": {"predict_every": 300, "horizons": [5], "agent": "tabular"},
    "h15": {"predict_every": 300, "horizons": [15], "agent": "tabular"},
    "h30": {"predict_every": 300, "horizons": [30], "agent": "tabular"},
    "t2-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb"},
    "t2-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb"},
    "t2-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb"},
    # TREATMENT 3: LinUCB + 5 live-streamed features (perp basis, funding,
    # cross-exchange dispersion, composite gap, mempool fee). t2 untouched.
    "t3-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb", "live": True},
    "t3-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb", "live": True},
    "t3-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb", "live": True},
    # TREATMENT 4 = t3 + trend features (4h momentum, alignment, range
    # position) and order-book features (imbalance, spread). t3/t2/ctl frozen.
    "t4-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb",
              "live": True, "trend": True, "book": True},
    "t4-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb",
               "live": True, "trend": True, "book": True},
    "t4-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb",
               "live": True, "trend": True, "book": True},
    # TREATMENT 5 = t4 + a frozen open fintech LLM trained on Bitcoin text
    # (ElKulako/cryptobert) as the perception module: live headline sentiment,
    # its momentum, and news intensity join the context. The RL policy stays
    # the decision maker — the FinRL-Contest "LLM signals + RL agent" pattern.
    "t5-h5": {"predict_every": 300, "horizons": [5], "agent": "linucb",
              "live": True, "trend": True, "book": True, "llm": True},
    "t5-h15": {"predict_every": 300, "horizons": [15], "agent": "linucb",
               "live": True, "trend": True, "book": True, "llm": True},
    "t5-h30": {"predict_every": 300, "horizons": [30], "agent": "linucb",
               "live": True, "trend": True, "book": True, "llm": True},
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
    return (FEATURE_DIM + (TREND_DIM if spec.get("trend") else 0)
            + (LIVE_DIM if spec.get("live") else 0)
            + (BOOK_DIM if spec.get("book") else 0)
            + (LLM_DIM if spec.get("llm") else 0)
            + (OFI_DIM if spec.get("ofi") else 0))


def _context(spec: dict, feat: dict, snap: dict | None) -> list[float]:
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


def _horizon_sigma(feat: dict, horizon: int) -> float:
    """Live dollar volatility of an h-minute move (per-min sigma x sqrt(h))."""
    return max(feat["vol_30m"] * math.sqrt(horizon) * feat["price"], SIGMA_FLOOR)


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
LEDGER_MAX_ROWS = 60_000       # ~2 weeks of both arms; keeps rewrites cheap


def _qtable_path(variant: str) -> Path:
    return RESULTS_DIR / f"q_table_online_{variant}.json"


def _bandit_path(variant: str) -> Path:
    return RESULTS_DIR / f"linucb_{variant}.json"


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
        agent = agents[horizon]
        row_extra: dict = {}
        adj = 0
        if isinstance(agent, BANDIT_TYPES):
            x = _context(spec, feat, snap)
            # commitments are GREEDY — exploration (UCB bonus / epsilon)
            # belongs in warm-up and hourly replay, never in the published
            # forecast (it was leaking ±3-sigma probes into predictions)
            arm = agent.select(x, greedy=True)
            delta = _vol_delta(config.K_FACTORS[arm], feat, horizon)
            # shrink (x0.5) and cap (±0.5 sigma_h) the bias intercept — the
            # uncapped version chased rallies and doubled the miss at turns
            raw_adj = (bias or {}).get((variant, horizon), 0)
            cap = 0.5 * _horizon_sigma(feat, horizon)
            adj = int(max(-cap, min(cap, 0.5 * raw_adj)))
            row_extra["x"] = [round(v, 5) for v in x]
            row_extra["arm"] = arm
            if adj:
                row_extra["bias_adj"] = adj
        else:
            qs = agent.q.get(state)
            delta = (config.ACTION_DELTAS[max(range(len(qs)), key=lambda i: qs[i])]
                     if qs else 0)
        pred = int(feat["price"] + delta + adj)
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
        d = _vol_delta(config.K_FACTORS[a], e.features, e.horizon_min)
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
        if any(isinstance(a, BANDIT_TYPES) for a in agents.values()):
            spec = VARIANTS[variant]
            gate = {}
            for h, agent in agents.items():
                eps = [e for e in train_eps if e.horizon_min == h]
                veps = [e for e in val_eps if e.horizon_min == h]
                before = ([a.copy() for a in agent.A],
                          [b.copy() for b in agent.b], list(agent.pulls))
                before_mae = _bandit_val_mae(agent, spec, veps, snaps)
                for e in eps:
                    x = _context(spec, e.features,
                                 _nearest_snap(snaps, e.minute_ts))
                    a = agent.select(x)
                    d = _vol_delta(config.K_FACTORS[a], e.features, h)
                    agent.update(x, a, _bandit_reward(
                        e.price_now + d, e.price_future, e.price_now, d))
                after_mae = _bandit_val_mae(agent, spec, veps, snaps)
                reverted = after_mae > before_mae
                if reverted:
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
                else _load_agents(v, spec["horizons"]))
            for v, spec in VARIANTS.items()}
    snaps = _load_snapshots()
    trades: dict[int, dict] = {}  # rolling taker-flow store (last ~20 min)
    cold_bandits = {v for v, spec in VARIANTS.items()
                    if spec.get("agent") in ("linucb", "linearq")
                    and all(a.total_pulls == 0 for a in arms[v].values())}
    ledger = _load_ledger()
    made = {(r.get("variant", "control"), r["made_ts"], r["horizon"])
            for r in ledger}
    last_retrain_slot = int(time.time()) // RETRAIN_EVERY
    retrain_info: dict = {}
    retrains = 0
    online_updates = 0
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
            snap = {
                "ts": now_ts,
                "funding": fetch_okx_funding_rate(),
                "basis_bp": ((mark - spot) / spot * 1e4
                             if mark and spot else None),
                "disp_bp": None, "gap_bp": None,
                "fee": fetch_mempool_fee(),
                "imb": book["imb"] if book else None,
                "spread_bp": book["spread_bp"] if book else None,
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
                warm_eps = build_episodes({"warm": bars}, {"warm": None})
                for v in sorted(cold_bandits):
                    vspec = VARIANTS[v]
                    for h, agent in arms[v].items():
                        for e in (e for e in warm_eps if e.horizon_min == h):
                            x = _context(vspec, e.features,
                                         _nearest_snap(snaps, e.minute_ts))
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
                for slot_ts in range(first, now_ts + 1, step):
                    if ((variant, slot_ts, spec["horizons"][0]) in made
                            or slot_ts not in by_ts):
                        continue
                    upto = [b for b in bars if b["ts"] <= slot_ts]
                    if len(upto) < config.LOOKBACK_MIN:
                        continue
                    rows = _predict_at(variant, arms[variant], upto, fng,
                                       slot_ts, spec["horizons"], spec=spec,
                                       snap=_nearest_snap(snaps, slot_ts),
                                       bias=bias, bands=bands)
                    ledger.extend(rows)
                    made.update((r["variant"], r["made_ts"], r["horizon"])
                                for r in rows)
                    new_preds += len(rows)

            # 1b. consensus: for every 5-min slot where all +5 predictors have
            #     committed, poll them (ctl-h5, t2-h5, persistence) and take
            #     the median as OUR final level for t+5. Scored like any
            #     predictor; never bets, never learns.
            have_consensus = {r["made_ts"] for r in ledger
                              if r["variant"] == "consensus"}
            by_slot: dict[int, dict] = {}
            for r in ledger:
                if r["horizon"] == 5 \
                        and r["variant"] in ("h5", "t2-h5", "t3-h5", "t4-h5",
                                             "t5-h5", "t6-h5") \
                        and r["made_ts"] not in have_consensus:
                    by_slot.setdefault(r["made_ts"], {})[r["variant"]] = r
            # skill-weighted poll: drop the voter with the worst trailing MAE
            trail: dict[str, float] = {}
            for v in ("h5", "t2-h5", "t3-h5", "t4-h5", "t5-h5", "t6-h5"):
                errs = [r["abs_err"] for r in ledger
                        if r["variant"] == v and r["actual"] is not None][-50:]
                if len(errs) >= 20:
                    trail[v] = sum(errs) / len(errs)
            worst = max(trail, key=trail.get) if len(trail) >= 3 else None
            for slot_ts, votes in sorted(by_slot.items()):
                if "h5" not in votes or "t2-h5" not in votes:
                    continue
                base = votes["h5"]
                polled = sorted([v["pred"] for name, v in votes.items()
                                 if name != worst]
                                + [int(base["price_now"])])
                mid = len(polled) // 2
                final = (polled[mid] if len(polled) % 2
                         else (polled[mid - 1] + polled[mid]) // 2)
                crow = {
                    "variant": "consensus", "made_ts": slot_ts,
                    "target_ts": slot_ts + 300, "horizon": 5,
                    "price_now": base["price_now"],
                    "pred": int(final),  # median of all voters
                    "delta": int(final) - int(base["price_now"]),
                    "votes": polled, "state": None,
                    "actual": None, "abs_err": None, "hit": None,
                }
                cband = bands.get(("consensus", 5))
                if cband:
                    crow["lo"] = crow["pred"] + cband[0]
                    crow["hi"] = crow["pred"] + cband[1]
                ledger.append(crow)
                new_preds += 1

            # 2. score matured predictions (all arms alike) and LEARN from
            #    each one: an immediate Q-update on the committed (s, a)
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
                if isinstance(agent, BANDIT_TYPES) and row.get("x") \
                        and row.get("arm") is not None:
                    if row["arm"] < agent.n_arms and len(row["x"]) == agent.dim:
                        agent.update(row["x"], row["arm"], _bandit_reward(
                            row["pred"], row["actual"],
                            row["price_now"], row["delta"]))
                        online_updates += 1
                elif isinstance(agent, TabularQAgent) and row.get("state") \
                        and row["delta"] in config.ACTION_DELTAS:
                    r = reward(row["pred"], row["actual"], shaped=True)
                    agent.learn(tuple(row["state"]),
                                config.ACTION_DELTAS.index(row["delta"]), r)
                    online_updates += 1
            if scored:  # persist what was just learned
                for variant, agents in arms.items():
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
