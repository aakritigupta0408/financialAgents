"""Always-on experiment runner: control vs treatment arms, trained separately.

The original system is frozen as the CONTROL arm (predict every 15 min at
+15/+30). Every addition from here on is a TREATMENT arm with its own
Q-tables and its own ledger rows, so arms never contaminate each other:

  control        predict every 15 min, horizons +15/+30   (the first model)
  t1-1min-multi  predict every  1 min, horizons +1/+5/+15/+30

Shared clocks: every 30s score matured predictions; every 1 hour retrain each
arm's own tables via experience replay with a hold-out no-regression gate.

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
from .sources import fetch_brti_composite, fetch_fear_greed, fetch_range

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PRED_LOG = RESULTS_DIR / "prediction_log.jsonl"
STATUS = RESULTS_DIR / "online_status.json"
BATCH_QTABLE = RESULTS_DIR / "q_table.json"

# Experiment arms. "control" is the original model — do not modify it;
# add new dicts here for each new treatment.
VARIANTS: dict[str, dict] = {
    "control": {"predict_every": 900, "horizons": [15, 30]},
    "t1-1min-multi": {"predict_every": 60, "horizons": [1, 5, 15, 30]},
}

RETRAIN_EVERY = 3600           # 1 hour
RETRAIN_WINDOW_H = 24          # replay the last day of bars
VAL_HOLDOUT_H = 3              # newest hours validate, never train
RETRAIN_EPOCHS = 3
POLL_SECONDS = 30
BACKFILL_HOURS = 6             # seed the charts with recent history on first run
ONLINE_EPSILON = 0.05          # exploration during learning only
LEDGER_MAX_ROWS = 60_000       # ~2 weeks of both arms; keeps rewrites cheap


def _qtable_path(variant: str) -> Path:
    return RESULTS_DIR / f"q_table_online_{variant}.json"


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


def _checkpoint(variant: str, agents: dict[int, TabularQAgent]) -> None:
    payload = {f"h{h}": {"|".join(map(str, s)): qs for s, qs in a.q.items()}
               for h, a in agents.items()}
    tmp = _qtable_path(variant).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(_qtable_path(variant))


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
                horizons: list[int]) -> list[dict]:
    """Greedy (no exploration) integer predictions committed at slot_ts."""
    feat = compute_features(bars_upto, fng)
    state = discretize(feat)
    rows = []
    for horizon in horizons:
        agent = agents[horizon]
        qs = agent.q.get(state)
        delta = (config.ACTION_DELTAS[max(range(len(qs)), key=lambda i: qs[i])]
                 if qs else 0)
        rows.append({
            "variant": variant,
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


def retrain_all(arms: dict[str, dict[int, TabularQAgent]]) -> dict:
    """Hourly retrain of every arm's OWN tables on the same replay window,
    each guarded by the hold-out no-regression gate."""
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
    arms = {v: _load_agents(v, spec["horizons"])
            for v, spec in VARIANTS.items()}
    ledger = _load_ledger()
    made = {(r.get("variant", "control"), r["made_ts"], r["horizon"])
            for r in ledger}
    last_retrain_slot = int(time.time()) // RETRAIN_EVERY
    retrain_info: dict = {}
    retrains = 0
    started = time.time()
    print(f"experiment runner up — arms: {', '.join(VARIANTS)}")

    while True:
        try:
            now = datetime.now(tz=config.PACIFIC)
            now_ts = int(now.timestamp())
            bars = fetch_range(now - timedelta(hours=BACKFILL_HOURS + 1), now)
            by_ts = {b["ts"]: b for b in bars}
            fng = fetch_fear_greed().get(now.date().isoformat())

            # 1. commit predictions per arm, at each arm's own cadence
            #    (uniformly covers live boundaries and first-run backfill)
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
                                       slot_ts, spec["horizons"])
                    ledger.extend(rows)
                    made.update((r["variant"], r["made_ts"], r["horizon"])
                                for r in rows)
                    new_preds += len(rows)

            # 2. score matured predictions (all arms alike)
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
                ledger = ledger[-LEDGER_MAX_ROWS:]
                _save_ledger(ledger)

            # 3. hourly retrain, each arm separately
            hour_slot = now_ts // RETRAIN_EVERY
            if hour_slot > last_retrain_slot:
                print(f"{now:%H:%M:%S} hourly retrain (all arms)...")
                retrain_info = retrain_all(arms)
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
                "brti": fetch_brti_composite(),
                "variants": {v: {"predict_every_min": s["predict_every"] // 60,
                                 "horizons": s["horizons"],
                                 "states_known": {h: len(a.q)
                                                  for h, a in arms[v].items()}}
                             for v, s in VARIANTS.items()},
                "retrain_every_min": RETRAIN_EVERY // 60,
                "retrains_this_session": retrains,
                "last_retrain": retrain_info or None,
                "predictions_total": len(ledger),
            }))
            if new_preds or scored:
                print(f"{now:%H:%M:%S} +{new_preds} predictions, {scored} scored "
                      f"({len(ledger)} total)")
        except Exception as exc:
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
