"""Batch-train the L3 distributional DQN on historical data.

Usage: python scripts/train_l3.py [--days 60] [--epochs 6]
Predicts the vol-normalized delta distribution over K_FACTORS bins; acts on
the mode. Saves results/dqn_h{h}.pt and prints test MAE vs persistence.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_rl import config                          # noqa: E402
from btc_rl.agents import DistDQNAgent             # noqa: E402
from btc_rl.env import build_episodes              # noqa: E402
from btc_rl.features import feature_vector, trend_feature_vector  # noqa: E402
from btc_rl.sources import fetch_fear_greed, fetch_history        # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DIM = 13


def ctx(e):
    return feature_vector(e.features) + trend_feature_vector(e.features)


def sigma_of(e):
    return max(e.features["vol_30m"] * math.sqrt(e.horizon_min)
               * e.features["price"], 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    history = fetch_history(args.days)
    fng = fetch_fear_greed(limit=args.days + 10)
    episodes = [e for e in build_episodes(history, fng)
                if e.horizon_min in (5, 15, 30)]
    days = sorted({e.day for e in episodes})
    cut = set(days[:int(len(days) * 0.8)])
    for h in (5, 15, 30):
        tr = [e for e in episodes if e.day in cut and e.horizon_min == h]
        te = [e for e in episodes if e.day not in cut and e.horizon_min == h]
        agent = DistDQNAgent(DIM)
        X = [ctx(e) for e in tr]
        Z = [(e.price_future - e.price_now) / sigma_of(e) for e in tr]
        idx = list(range(len(tr)))
        rng = random.Random(7)
        for ep in range(args.epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                agent.learn_batch([X[j] for j in b], [Z[j] for j in b])
        errs, pers = [], []
        for e in te:
            a = agent.select(ctx(e), greedy=True)
            d = int(round(agent.bins[a] * sigma_of(e)))
            errs.append(abs(e.price_now + d - e.price_future))
            pers.append(abs(e.price_future - e.price_now))
        print(f"h{h}: L3 mode-action test MAE ${sum(errs)/len(errs):.1f} "
              f"vs persistence ${sum(pers)/len(pers):.1f} "
              f"({len(te)} episodes, {agent.steps} train steps)")
        agent.save(RESULTS / f"dqn_h{h}.pt")
    print("saved results/dqn_h*.pt")


if __name__ == "__main__":
    main()
