"""Fetch data, train agents, evaluate on a chronological holdout, dump metrics.

Usage:  python -m btc_rl.train [--days 120] [--epochs 30]

Outputs results/metrics.json (consumed by the dashboard site) and
results/q_table.json (consumed by live.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from . import config
from .agents import PersistenceAgent, TabularQAgent
from .env import Episode, build_episodes, reward
from .sources import fetch_fear_greed, fetch_history

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def split_by_day(episodes: list[Episode]) -> tuple[list[Episode], list[Episode]]:
    days = sorted({e.day for e in episodes})
    cut = int(len(days) * config.TRAIN_FRACTION)
    train_days = set(days[:cut])
    train = [e for e in episodes if e.day in train_days]
    test = [e for e in episodes if e.day not in train_days]
    return train, test


def train_q_agent(train_eps: list[Episode], horizon: int, shaped: bool,
                  epochs: int, seed: int = 7) -> TabularQAgent:
    agent = TabularQAgent(seed=seed)
    eps = [e for e in train_eps if e.horizon_min == horizon]
    rng = random.Random(seed)
    for epoch in range(epochs):
        agent.decay_epsilon(epoch, epochs)
        rng.shuffle(eps)
        for e in eps:
            a = agent.act(e.state, e.price_now, explore=True)
            pred = e.price_now + agent.delta_for(a)
            agent.learn(e.state, a, reward(pred, e.price_future, shaped))
    return agent


def evaluate(agent, test_eps: list[Episode], horizon: int) -> dict:
    eps = [e for e in test_eps if e.horizon_min == horizon]
    if not eps:
        return {}
    out = {"all": _score(agent, eps),
           "target_slots": _score(agent, [e for e in eps if e.is_target_slot])}
    return out


def _score(agent, eps: list[Episode]) -> dict:
    if not eps:
        return {"episodes": 0}
    errors, hits, sparse_rewards = [], 0, 0.0
    within = {1: 0, 10: 0, 50: 0}
    for e in eps:
        if isinstance(agent, TabularQAgent):
            a = agent.act(e.state, e.price_now, explore=False)
            pred = e.price_now + agent.delta_for(a)
        else:
            pred = e.price_now
        err = pred - e.price_future
        errors.append(err)
        if int(pred) == int(e.price_future):
            hits += 1
            sparse_rewards += config.REWARD_HIT
        else:
            sparse_rewards += config.REWARD_MISS
        for k in within:
            if abs(err) <= k:
                within[k] += 1
    n = len(eps)
    return {
        "episodes": n,
        "exact_int_hit_rate": hits / n,
        "exact_int_hits": hits,
        "within_$1": within[1] / n,
        "within_$10": within[10] / n,
        "within_$50": within[50] / n,
        "mae": sum(abs(x) for x in errors) / n,
        "rmse": math.sqrt(sum(x * x for x in errors) / n),
        "mean_sparse_reward": sparse_rewards / n,
        "cumulative_sparse_reward": sparse_rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=config.HISTORY_DAYS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    print(f"Fetching {args.days} days of 1m bars from Coinbase...")
    history = fetch_history(args.days)
    fng = fetch_fear_greed(limit=args.days + 10)
    print(f"  got {len(history)} days")

    episodes = build_episodes(history, fng)
    train_eps, test_eps = split_by_day(episodes)
    print(f"  episodes: {len(train_eps)} train / {len(test_eps)} test")

    metrics: dict = {
        "data": {
            "days": len(history),
            "first_day": min(history) if history else None,
            "last_day": max(history) if history else None,
            "train_episodes": len(train_eps),
            "test_episodes": len(test_eps),
            "action_deltas": config.ACTION_DELTAS,
        },
        "agents": {},
    }

    q_tables: dict = {}
    for horizon in config.HORIZONS_MIN:
        baseline = PersistenceAgent()
        metrics["agents"].setdefault(baseline.name, {})[f"h{horizon}"] = \
            evaluate(baseline, test_eps, horizon)
        for shaped in (False, True):
            name = "tabular-q-shaped" if shaped else "tabular-q-sparse"
            print(f"Training {name} (horizon {horizon}m)...")
            agent = train_q_agent(train_eps, horizon, shaped, args.epochs)
            metrics["agents"].setdefault(name, {})[f"h{horizon}"] = \
                evaluate(agent, test_eps, horizon)
            q_tables[f"{name}_h{horizon}"] = \
                {"|".join(map(str, k)): v for k, v in agent.q.items()}

    # Delta distribution on test data (for the dashboard's histogram).
    for horizon in config.HORIZONS_MIN:
        deltas = [e.price_future - e.price_now
                  for e in test_eps if e.horizon_min == horizon]
        metrics["data"][f"delta_stats_h{horizon}"] = _delta_stats(deltas)

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "q_table.json").write_text(json.dumps(q_tables))
    print(f"Wrote {RESULTS_DIR / 'metrics.json'}")
    _print_summary(metrics)


def _delta_stats(deltas: list[float]) -> dict:
    if not deltas:
        return {}
    n = len(deltas)
    mean = sum(deltas) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in deltas) / n)
    hist: dict[str, int] = {}
    for d in deltas:
        b = int(d // 25) * 25  # $25 bins
        hist[str(b)] = hist.get(str(b), 0) + 1
    return {"n": n, "mean": mean, "std": std,
            "min": min(deltas), "max": max(deltas), "hist_25": hist}


def _print_summary(metrics: dict) -> None:
    print("\n=== Test-set summary (all episodes) ===")
    for name, by_h in metrics["agents"].items():
        for h, ev in by_h.items():
            s = ev.get("all", {})
            if not s or not s.get("episodes"):
                continue
            print(f"{name:18s} {h:4s} hit={s['exact_int_hit_rate']:.4%} "
                  f"MAE=${s['mae']:.1f} within$10={s['within_$10']:.1%} "
                  f"reward/ep={s['mean_sparse_reward']:+.3f}")


if __name__ == "__main__":
    main()
