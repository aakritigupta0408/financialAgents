"""Always-on experiment runner: one prediction stream per horizon.

Each horizon predicts at its own natural cadence (cadence == horizon) with
its own Q-table, so streams never contaminate each other:

  h5 / h15 / h30 all predict every 5 minutes (9:00, 9:05, 9:10…), each with
  its own Q-table; h15/h30 warm-started from the original control model.

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
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .agents import LinUCBAgent, TabularQAgent
from .env import build_episodes, reward
from .features import (LIVE_DIM, compute_features, discretize, feature_vector,
                       live_feature_vector)
from .sources import (fetch_brti_composite, fetch_deribit_mark,
                      fetch_fear_greed, fetch_mempool_fee,
                      fetch_okx_funding_rate, fetch_range)

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
# contextual bandit over the same 21 integer-delta arms: reward = betting P&L
# (+$40 within ±$10, −$10 otherwise), context = the continuous feature vector.
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
}

# Betting rules: each committed prediction stakes $10 from a shared $1,000
# bankroll; |predicted - actual| <= $10 pays $50 (net +$40), a miss loses the
# stake. Bets start at first deployment of this feature — history never bets.
BET_STAKE = 10
BET_PAYOUT = 50
BET_TOLERANCE = 10
BANKROLL_START = 1000
BETTING_FILE_NAME = "betting.json"

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


def _bandit_path(variant: str) -> Path:
    return RESULTS_DIR / f"linucb_{variant}.json"


def _load_bandits(variant: str, horizons: list[int],
                  dim: int) -> dict[int, LinUCBAgent]:
    path = _bandit_path(variant)
    if path.exists():
        raw = json.loads(path.read_text())
        loaded = {h: LinUCBAgent.from_dict(raw[f"h{h}"]) for h in horizons
                  if f"h{h}" in raw and raw[f"h{h}"]["dim"] == dim}
        if loaded:
            return loaded
    return {h: LinUCBAgent(dim) for h in horizons}


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
    if any(isinstance(a, LinUCBAgent) for a in agents.values()):
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
                horizons: list[int], bet_start_ts: float,
                live_x: list[float] | None = None) -> list[dict]:
    """Greedy (no exploration) integer predictions committed at slot_ts."""
    feat = compute_features(bars_upto, fng)
    state = discretize(feat)
    rows = []
    for horizon in horizons:
        agent = agents[horizon]
        row_extra: dict = {}
        if isinstance(agent, LinUCBAgent):
            x = feature_vector(feat) + (live_x or [])
            delta = config.ACTION_DELTAS[agent.select(x)]
            row_extra["x"] = [round(v, 5) for v in x]
        else:
            qs = agent.q.get(state)
            delta = (config.ACTION_DELTAS[max(range(len(qs)), key=lambda i: qs[i])]
                     if qs else 0)
        rows.append({
            "variant": variant,
            "made_ts": slot_ts, "target_ts": slot_ts + horizon * 60,
            "horizon": horizon, "price_now": feat["price"],
            "pred": int(feat["price"] + delta), "delta": delta,
            "state": list(state), "actual": None, "abs_err": None, "hit": None,
            "bet": BET_STAKE if slot_ts >= bet_start_ts else 0,
            **row_extra,
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
        if any(isinstance(a, LinUCBAgent) for a in agents.values()):
            continue  # t2 bandits learn purely online — no batch retrain
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
    arms = {v: (_load_bandits(v, spec["horizons"],
                              FEATURE_DIM + (LIVE_DIM if spec.get("live") else 0))
                if spec.get("agent") == "linucb"
                else _load_agents(v, spec["horizons"]))
            for v, spec in VARIANTS.items()}
    snaps = _load_snapshots()
    cold_bandits = {v for v, spec in VARIANTS.items()
                    if spec.get("agent") == "linucb"
                    and all(a.total_pulls == 0 for a in arms[v].values())}
    ledger = _load_ledger()
    made = {(r.get("variant", "control"), r["made_ts"], r["horizon"])
            for r in ledger}
    last_retrain_slot = int(time.time()) // RETRAIN_EVERY
    retrain_info: dict = {}
    retrains = 0
    online_updates = 0
    started = time.time()

    # Betting starts at first-ever deployment and survives restarts.
    betting_file = RESULTS_DIR / BETTING_FILE_NAME
    if betting_file.exists():
        bet_start_ts = json.loads(betting_file.read_text())["start_ts"]
    else:
        bet_start_ts = time.time()
        betting_file.write_text(json.dumps(
            {"start_ts": bet_start_ts, "bankroll_start": BANKROLL_START,
             "stake": BET_STAKE, "payout": BET_PAYOUT,
             "tolerance": BET_TOLERANCE}))
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
            snap = {
                "ts": now_ts,
                "funding": fetch_okx_funding_rate(),
                "basis_bp": ((mark - spot) / spot * 1e4
                             if mark and spot else None),
                "disp_bp": None, "gap_bp": None,
                "fee": fetch_mempool_fee(),
            }
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
                    is_live = VARIANTS[v].get("live")
                    for h, agent in arms[v].items():
                        for e in (e for e in warm_eps if e.horizon_min == h):
                            x = feature_vector(e.features)
                            if is_live:
                                x = x + live_feature_vector(
                                    _nearest_snap(snaps, e.minute_ts))
                            a = agent.select(x)
                            pred = e.price_now + config.ACTION_DELTAS[a]
                            r = (BET_PAYOUT - BET_STAKE
                                 if abs(pred - e.price_future) <= BET_TOLERANCE
                                 else -BET_STAKE)
                            agent.update(x, a, r)
                    _checkpoint(v, arms[v])
                    print(f"warmed up {v} on {len(warm_eps)} recent episodes")
                cold_bandits.clear()

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
                    live_x = (live_feature_vector(_nearest_snap(snaps, slot_ts))
                              if spec.get("live") else None)
                    rows = _predict_at(variant, arms[variant], upto, fng,
                                       slot_ts, spec["horizons"], bet_start_ts,
                                       live_x=live_x)
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
                if r["horizon"] == 5 and r["variant"] in ("h5", "t2-h5", "t3-h5") \
                        and r["made_ts"] not in have_consensus:
                    by_slot.setdefault(r["made_ts"], {})[r["variant"]] = r
            for slot_ts, votes in sorted(by_slot.items()):
                if "h5" not in votes or "t2-h5" not in votes:
                    continue
                base = votes["h5"]
                polled = sorted([v["pred"] for v in votes.values()]
                                + [int(base["price_now"])])
                mid = len(polled) // 2
                final = (polled[mid] if len(polled) % 2
                         else (polled[mid - 1] + polled[mid]) // 2)
                ledger.append({
                    "variant": "consensus", "made_ts": slot_ts,
                    "target_ts": slot_ts + 300, "horizon": 5,
                    "price_now": base["price_now"],
                    "pred": int(final),  # median of all voters
                    "delta": int(final) - int(base["price_now"]),
                    "votes": polled, "state": None,
                    "actual": None, "abs_err": None, "hit": None, "bet": 0,
                })
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
                if row.get("bet"):  # settle the wager on evaluation
                    win = row["abs_err"] <= BET_TOLERANCE
                    row["payout"] = BET_PAYOUT if win else 0
                    row["pnl"] = row["payout"] - row["bet"]
                scored += 1
                agents = arms.get(row["variant"])
                agent = agents.get(row["horizon"]) if agents else None
                if agent is None or row["delta"] not in config.ACTION_DELTAS:
                    continue
                if isinstance(agent, LinUCBAgent) and row.get("x"):
                    r = (BET_PAYOUT - BET_STAKE
                         if row["abs_err"] <= BET_TOLERANCE else -BET_STAKE)
                    agent.update(row["x"],
                                 config.ACTION_DELTAS.index(row["delta"]), r)
                    online_updates += 1
                elif isinstance(agent, TabularQAgent) and row.get("state"):
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
                retrain_info = retrain_all(arms)
                retrains += 1
                last_retrain_slot = hour_slot

            # 4. betting book: settled P&L per stream + shared bankroll
            book: dict = {"per_stream": {}, "start_ts": bet_start_ts,
                          "stake": BET_STAKE, "payout": BET_PAYOUT,
                          "tolerance": BET_TOLERANCE,
                          "bankroll_start": BANKROLL_START}
            pending_staked = settled_pnl = 0
            for v in VARIANTS:
                vb = [r for r in ledger if r["variant"] == v and r.get("bet")]
                settled = [r for r in vb if r["actual"] is not None]
                wins = [r for r in settled if r["payout"]]
                pnl = sum(r["pnl"] for r in settled)
                book["per_stream"][v] = {
                    "bets": len(vb), "settled": len(settled),
                    "wins": len(wins), "staked": len(vb) * BET_STAKE,
                    "received": sum(r["payout"] for r in settled), "pnl": pnl}
                settled_pnl += pnl
                pending_staked += (len(vb) - len(settled)) * BET_STAKE
            book["settled_pnl"] = settled_pnl
            book["pending_staked"] = pending_staked
            book["bankroll"] = BANKROLL_START + settled_pnl - pending_staked

            # 5. status + actual-price series for the charts
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
                                     h: (a.total_pulls if isinstance(a, LinUCBAgent)
                                         else len(a.q))
                                     for h, a in arms[v].items()}}
                             for v, s in VARIANTS.items()},
                "betting": book,
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
