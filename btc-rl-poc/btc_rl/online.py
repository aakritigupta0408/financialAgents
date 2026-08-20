"""Always-on scheduler: predict every 15 minutes, retrain every hour.

BTC never closes, so the daemon runs three clocks:
  every 30s      score any prediction whose target minute has matured
  every 15 min   commit integer predictions for +15 and +30 minutes (:00/:15/:30/:45)
  every 1 hour   retrain — an experience-replay pass over the last 24h of bars,
                 warm-started from the current Q-table (never from scratch)

The dashboard reads results/prediction_log.jsonl (predicted vs actual at HH:MM),
results/recent_prices.json (the actual price line), and results/online_status.json.

Usage:  python -m btc_rl.online            # run forever
        python -m btc_rl.online --once     # backfill + one pass, then exit
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .agents import TabularQAgent
from .env import build_episodes, reward
from .features import compute_features, discretize
from .sources import fetch_fear_greed, fetch_range

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PRED_LOG = RESULTS_DIR / "prediction_log.jsonl"
STATUS = RESULTS_DIR / "online_status.json"
QTABLE = RESULTS_DIR / "q_table_online.json"
BATCH_QTABLE = RESULTS_DIR / "q_table.json"

PREDICT_EVERY = 900            # 15 minutes
RETRAIN_EVERY = 3600           # 1 hour
RETRAIN_WINDOW_H = 24          # replay the last day of bars
RETRAIN_EPOCHS = 3
POLL_SECONDS = 30
BACKFILL_HOURS = 6             # seed the chart with recent history on first run
ONLINE_EPSILON = 0.05


def _load_agents() -> dict[int, TabularQAgent]:
    agents: dict[int, TabularQAgent] = {}
    source = QTABLE if QTABLE.exists() else BATCH_QTABLE
    raw = json.loads(source.read_text()) if source.exists() else {}
    for horizon in config.HORIZONS_MIN:
        agent = TabularQAgent(seed=int(horizon))
        agent.epsilon = ONLINE_EPSILON
        key = (f"h{horizon}" if QTABLE.exists()
               else f"tabular-q-shaped_h{horizon}")
        for state_str, qs in raw.get(key, {}).items():
            agent.q[tuple(map(int, state_str.split("|")))] = list(qs)
        agents[horizon] = agent
    return agents


def _checkpoint(agents: dict[int, TabularQAgent]) -> None:
    payload = {f"h{h}": {"|".join(map(str, s)): qs for s, qs in a.q.items()}
               for h, a in agents.items()}
    tmp = QTABLE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(QTABLE)


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
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(PRED_LOG)


def _predict_at(agents, bars_upto: list[dict], fng, slot_ts: int) -> list[dict]:
    """Greedy integer predictions committed at slot_ts for +15 and +30 min."""
    feat = compute_features(bars_upto, fng)
    state = discretize(feat)
    rows = []
    for horizon in config.HORIZONS_MIN:
        agent = agents[horizon]
        qs = agent.q.get(state)
        delta = (config.ACTION_DELTAS[max(range(len(qs)), key=lambda i: qs[i])]
                 if qs else 0)
        rows.append({
            "made_ts": slot_ts, "target_ts": slot_ts + horizon * 60,
            "horizon": horizon, "price_now": feat["price"],
            "pred": int(feat["price"] + delta), "delta": delta,
            "state": list(state), "actual": None, "abs_err": None, "hit": None,
        })
    return rows


def _greedy_mae(agent: TabularQAgent, episodes: list) -> float:
    errs = [abs(e.price_now + agent.delta_for(
        agent.act(e.state, e.price_now, explore=False)) - e.price_future)
        for e in episodes]
    return sum(errs) / len(errs) if errs else 0.0


def retrain(agents: dict[int, TabularQAgent]) -> dict:
    """Hourly retrain with a no-regression gate against overfitting live noise.

    The newest VAL_HOLDOUT_H hours are never trained on; they validate the
    retrained table, and any horizon whose validation MAE worsens is reverted.
    """
    VAL_HOLDOUT_H = 3
    now = datetime.now(tz=config.PACIFIC)
    bars = fetch_range(now - timedelta(hours=RETRAIN_WINDOW_H), now)
    cut_ts = int((now - timedelta(hours=VAL_HOLDOUT_H)).timestamp())
    train_eps = build_episodes({"replay": [b for b in bars if b["ts"] < cut_ts]},
                               {"replay": None})
    val_eps = build_episodes({"replay": [b for b in bars if b["ts"] >= cut_ts]},
                             {"replay": None})
    before = {h: {s: list(qs) for s, qs in agents[h].q.items()}
              for h in config.HORIZONS_MIN}
    before_mae = {h: _greedy_mae(agents[h],
                                 [e for e in val_eps if e.horizon_min == h])
                  for h in config.HORIZONS_MIN}
    rng = random.Random(int(now.timestamp()) // RETRAIN_EVERY)
    updates = 0
    for _ in range(RETRAIN_EPOCHS):
        rng.shuffle(train_eps)
        for e in train_eps:
            agent = agents[e.horizon_min]
            a = agent.act(e.state, e.price_now, explore=True)
            pred = e.price_now + agent.delta_for(a)
            agent.learn(e.state, a, reward(pred, e.price_future, shaped=True))
            updates += 1
    gate = {}
    for h in config.HORIZONS_MIN:
        after_mae = _greedy_mae(agents[h],
                                [e for e in val_eps if e.horizon_min == h])
        reverted = after_mae > before_mae[h]
        if reverted:  # the new table chased noise — roll it back
            agents[h].q.clear()
            agents[h].q.update(before[h])
        gate[f"h{h}"] = {"val_mae_before": round(before_mae[h], 2),
                         "val_mae_after": round(after_mae, 2),
                         "reverted": reverted}
    _checkpoint(agents)
    return {"at": now.isoformat(), "train_episodes": len(train_eps),
            "val_episodes": len(val_eps), "epochs": RETRAIN_EPOCHS,
            "q_updates": updates, "gate": gate}


def run(once: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    agents = _load_agents()
    ledger = _load_ledger()
    made = {(r["made_ts"], r["horizon"]) for r in ledger}
    last_retrain_slot = int(time.time()) // RETRAIN_EVERY  # first retrain next hour
    retrain_info: dict = {}
    retrains = 0
    started = time.time()
    print("online scheduler up — predicting every 15m, retraining hourly")

    while True:
        try:
            now = datetime.now(tz=config.PACIFIC)
            now_ts = int(now.timestamp())
            bars = fetch_range(now - timedelta(hours=BACKFILL_HOURS + 1), now)
            by_ts = {b["ts"]: b for b in bars}
            fng = fetch_fear_greed().get(now.date().isoformat())

            # 1. commit predictions for every 15-min slot we haven't done yet
            #    (covers live boundaries AND first-run backfill uniformly)
            first_slot = ((now_ts - BACKFILL_HOURS * 3600) // PREDICT_EVERY + 1) \
                * PREDICT_EVERY
            new_preds = 0
            for slot_ts in range(first_slot, now_ts + 1, PREDICT_EVERY):
                if (slot_ts, config.HORIZONS_MIN[0]) in made or slot_ts not in by_ts:
                    continue
                upto = [b for b in bars if b["ts"] <= slot_ts]
                if len(upto) < config.LOOKBACK_MIN:
                    continue
                rows = _predict_at(agents, upto, fng, slot_ts)
                ledger.extend(rows)
                made.update((r["made_ts"], r["horizon"]) for r in rows)
                new_preds += len(rows)

            # 2. score matured predictions
            scored = 0
            for row in ledger:
                if row["actual"] is not None:
                    continue
                bar = by_ts.get(row["target_ts"])
                if bar is None:
                    continue
                row["actual"] = bar["close"]
                row["abs_err"] = round(abs(row["pred"] - bar["close"]), 2)
                row["hit"] = int(row["pred"]) == int(bar["close"])
                scored += 1
            if new_preds or scored:
                _save_ledger(ledger)

            # 3. hourly retrain
            hour_slot = now_ts // RETRAIN_EVERY
            if hour_slot > last_retrain_slot:
                print(f"{now:%H:%M:%S} hourly retrain...")
                retrain_info = retrain(agents)
                retrains += 1
                last_retrain_slot = hour_slot
                print(f"  replayed {retrain_info['episodes']} episodes, "
                      f"{retrain_info['q_updates']} Q-updates")

            # 4. status + actual-price series for the chart
            feat = compute_features(bars, fng)
            state = discretize(feat)
            upcoming = [r for r in ledger if r["actual"] is None]
            recent = [{"ts": b["ts"], "c": b["close"]}
                      for b in bars if b["ts"] >= now_ts - BACKFILL_HOURS * 3600]
            (RESULTS_DIR / "recent_prices.json").write_text(json.dumps(recent))
            STATUS.write_text(json.dumps({
                "alive_at": time.time(), "started_at": started,
                "price_now": feat["price"], "state": list(state),
                "predict_every_min": PREDICT_EVERY // 60,
                "retrain_every_min": RETRAIN_EVERY // 60,
                "retrains_this_session": retrains,
                "last_retrain": retrain_info or None,
                "states_known": {h: len(a.q) for h, a in agents.items()},
                "pending": upcoming[-4:],
                "predictions_total": len(ledger),
            }))
            if new_preds or scored:
                print(f"{now:%H:%M:%S} +{new_preds} predictions, {scored} scored "
                      f"({len(ledger)} total)")
        except Exception as exc:
            print(f"poll error (will retry): {exc}")
        if once:
            _checkpoint(agents)
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
