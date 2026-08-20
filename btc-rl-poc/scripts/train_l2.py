"""Batch-train the L2 linear-Q agents on >=50 days of history, up to today.

Usage: python scripts/train_l2.py [--days 60] [--epochs 20]
Writes results/linear_q.json (per-horizon weights) and prints test metrics
vs persistence. Context = 13 bar-derived dims (base 10 + trend 3) so batch
and live are identical; actions are volatility-scaled (k x sigma_h).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_rl import config                             # noqa: E402
from btc_rl.agents import LinearQAgent                # noqa: E402
from btc_rl.env import build_episodes, reward         # noqa: E402
from btc_rl.features import feature_vector, trend_feature_vector  # noqa: E402
from btc_rl.sources import fetch_fear_greed, fetch_history        # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DIM = 13


def ctx(e):
    return feature_vector(e.features) + trend_feature_vector(e.features)


def vol_delta(k, e):
    sigma = max(e.features["vol_30m"] * math.sqrt(e.horizon_min)
                * e.features["price"], 5.0)
    return int(round(k * sigma))


def bandit_reward(pred, actual, price_now, delta):
    r = reward(pred, actual, shaped=True)
    if delta and (delta > 0) == (actual > price_now) and actual != price_now:
        r += 0.1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    print(f"fetching {args.days} days of 1m bars (cached where possible)...")
    history = fetch_history(args.days)
    fng = fetch_fear_greed(limit=args.days + 10)
    episodes = build_episodes(history, fng)
    days = sorted({e.day for e in episodes})
    cut = set(days[:int(len(days) * 0.8)])
    train = [e for e in episodes if e.day in cut]
    test = [e for e in episodes if e.day not in cut]
    print(f"{len(history)} days -> {len(train)} train / {len(test)} test episodes")

    out, report = {}, {}
    for h in config.HORIZONS_MIN:
        tr = [e for e in train if e.horizon_min == h]
        te = [e for e in test if e.horizon_min == h]
        if not tr or not te:
            continue
        agent = LinearQAgent(DIM, lr=0.01, epsilon=0.3)
        rng = random.Random(7)
        for ep in range(args.epochs):
            agent.epsilon = 0.3 + (0.02 - 0.3) * ep / max(1, args.epochs - 1)
            rng.shuffle(tr)
            for e in tr:
                x = ctx(e)
                a = agent.select(x)
                d = vol_delta(config.K_FACTORS[a], e)
                agent.update(x, a, bandit_reward(
                    e.price_now + d, e.price_future, e.price_now, d))
        errs, pers = [], []
        for e in te:
            a = agent.select(ctx(e), greedy=True)
            d = vol_delta(config.K_FACTORS[a], e)
            errs.append(abs(e.price_now + d - e.price_future))
            pers.append(abs(e.price_future - e.price_now))
        out[f"h{h}"] = agent.to_dict()
        report[f"h{h}"] = {
            "test_mae": round(sum(errs) / len(errs), 2),
            "persistence_mae": round(sum(pers) / len(pers), 2),
            "test_episodes": len(te), "updates": agent.total_pulls}
        print(f"h{h}: L2 test MAE ${report[f'h{h}']['test_mae']:.1f} "
              f"vs persistence ${report[f'h{h}']['persistence_mae']:.1f} "
              f"({len(te)} episodes, {agent.total_pulls} updates)")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "linear_q.json").write_text(json.dumps(out))
    (RESULTS / "linear_q_report.json").write_text(json.dumps(report, indent=2))
    print("saved results/linear_q.json")


if __name__ == "__main__":
    main()
