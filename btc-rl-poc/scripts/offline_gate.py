"""Offline evaluation gate — run BEFORE a treatment goes online.

Usage: python scripts/offline_gate.py [--days 45]

For every treatment configuration, trains its agent on historical episodes
(chronological 80/20) and reports test MAE / direction / mean deviation,
plus PAIRWISE SIMILARITY between treatments (fraction of identical action
choices on the same test episodes).

Gate rules
  PASS       test MAE <= 1.02 x persistence (an arm may trail the noise
             floor slightly while learning, but not more)
  DUPLICATE  >= 95% identical choices with a strictly simpler arm — the
             treatment adds no decision-relevant information; discard it.

Writes results/offline_gate.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_rl import config                                   # noqa: E402
from btc_rl.agents import LinearQAgent, LinUCBAgent         # noqa: E402
from btc_rl.env import build_episodes, reward               # noqa: E402
from btc_rl.features import (book_feature_vector, feature_vector,  # noqa: E402
                             live_feature_vector, llm_feature_vector,
                             ofi_feature_vector, trend_feature_vector)
from btc_rl.sources import fetch_fear_greed, fetch_history  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Offline context builders mirror online VARIANTS; snapshot-driven features
# have no history here (None -> neutral zeros) — that absence is part of the
# test: dims that are zero offline AND near-static online are dead weight.
SPECS = {
    "t2": {"kind": "linucb",
           "ctx": lambda e: feature_vector(e.features)},
    "t3": {"kind": "linucb",
           "ctx": lambda e: feature_vector(e.features) + live_feature_vector(None)},
    "t4": {"kind": "linucb",
           "ctx": lambda e: (feature_vector(e.features) + trend_feature_vector(e.features)
                             + live_feature_vector(None) + book_feature_vector(None))},
    "t5": {"kind": "linucb",
           "ctx": lambda e: (feature_vector(e.features) + trend_feature_vector(e.features)
                             + live_feature_vector(None) + book_feature_vector(None)
                             + llm_feature_vector(None))},
    "t6": {"kind": "linucb",
           "ctx": lambda e: (feature_vector(e.features) + trend_feature_vector(e.features)
                             + live_feature_vector(None) + book_feature_vector(None)
                             + llm_feature_vector(None) + ofi_feature_vector(None))},
    "t7": {"kind": "linearq",
           "ctx": lambda e: feature_vector(e.features) + trend_feature_vector(e.features)},
}
SIMPLER = {"t3": "t2", "t4": "t2", "t5": "t4", "t6": "t5"}


def vol_delta(k, e):
    sigma = max(e.features["vol_30m"] * math.sqrt(e.horizon_min)
                * e.features["price"], 5.0)
    return int(round(k * sigma))


def bandit_reward(pred, actual, price_now, delta):
    r = reward(pred, actual, shaped=True)
    if delta and (delta > 0) == (actual > price_now) and actual != price_now:
        r += 0.1
    return r


def run_arm(name, spec, train, test):
    dim = len(spec["ctx"](train[0]))
    per_h = {}
    choices = {}
    for h in (5, 15, 30):
        tr = [e for e in train if e.horizon_min == h]
        te = [e for e in test if e.horizon_min == h]
        if spec["kind"] == "linearq":
            agent = LinearQAgent(dim, lr=0.01, epsilon=0.3)
            rng = random.Random(7)
            for ep in range(5):
                agent.epsilon = 0.3 + (0.02 - 0.3) * ep / 4
                rng.shuffle(tr)
                for e in tr:
                    x = spec["ctx"](e)
                    a = agent.select(x)
                    d = vol_delta(config.K_FACTORS[a], e)
                    agent.update(x, a, bandit_reward(
                        e.price_now + d, e.price_future, e.price_now, d))
        else:
            agent = LinUCBAgent(dim, alpha=0.3, n_arms=len(config.K_FACTORS))
            for e in tr:
                x = spec["ctx"](e)
                a = agent.select(x)
                d = vol_delta(config.K_FACTORS[a], e)
                agent.update(x, a, bandit_reward(
                    e.price_now + d, e.price_future, e.price_now, d))
        errs, pers, dir_ok, moved = [], [], 0, 0
        ch = []
        for e in te:
            a = agent.select(spec["ctx"](e), greedy=True)
            d = vol_delta(config.K_FACTORS[a], e)
            ch.append(a)
            errs.append(abs(e.price_now + d - e.price_future))
            pers.append(abs(e.price_future - e.price_now))
            if d:
                moved += 1
                if (d > 0) == (e.price_future > e.price_now) \
                        and e.price_future != e.price_now:
                    dir_ok += 1
        per_h[f"h{h}"] = {
            "mae": round(sum(errs) / len(errs), 2),
            "persistence": round(sum(pers) / len(pers), 2),
            "direction": round(dir_ok / moved, 3) if moved else None,
            "moved_frac": round(moved / len(te), 3),
        }
        choices[h] = ch
    return per_h, choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45)
    args = ap.parse_args()
    history = fetch_history(args.days)
    fng = fetch_fear_greed(limit=args.days + 10)
    episodes = [e for e in build_episodes(history, fng)
                if e.horizon_min in (5, 15, 30)]
    days = sorted({e.day for e in episodes})
    cut = set(days[:int(len(days) * 0.8)])
    train = [e for e in episodes if e.day in cut]
    test = [e for e in episodes if e.day not in cut]
    print(f"{len(history)} days -> {len(train)} train / {len(test)} test\n")

    results, all_choices = {}, {}
    for name, spec in SPECS.items():
        per_h, choices = run_arm(name, spec, train, test)
        results[name] = per_h
        all_choices[name] = choices
        line = " ".join(
            f"h{h}: MAE ${per_h[f'h{h}']['mae']:.0f}"
            f" (pers ${per_h[f'h{h}']['persistence']:.0f})"
            f" dir {per_h[f'h{h}']['direction'] or 0:.0%}"
            for h in (5, 15, 30))
        print(f"{name}: {line}")

    print("\nPAIRWISE IDENTICAL-CHOICE RATE (test episodes):")
    names = list(SPECS)
    sims = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = tot = 0
            for h in (5, 15, 30):
                ca, cb = all_choices[a][h], all_choices[b][h]
                same += sum(1 for x, y in zip(ca, cb) if x == y)
                tot += len(ca)
            sims[f"{a}|{b}"] = round(same / tot, 4)
            print(f"  {a} vs {b}: {same / tot:6.1%}")

    print("\nGATE VERDICTS:")
    verdicts = {}
    for name in SPECS:
        fails = [h for h in ("h5", "h15", "h30")
                 if results[name][h]["mae"] > 1.02 * results[name][h]["persistence"]]
        dup = None
        simpler = SIMPLER.get(name)
        if simpler and sims.get(f"{simpler}|{name}", 0) >= 0.95:
            dup = simpler
        verdict = (f"DUPLICATE of {dup}" if dup
                   else ("PASS" if not fails else f"FAIL ({','.join(fails)})"))
        verdicts[name] = verdict
        print(f"  {name}: {verdict}")

    (RESULTS / "offline_gate.json").write_text(json.dumps(
        {"results": results, "similarity": sims, "verdicts": verdicts},
        indent=2))
    print("\nsaved results/offline_gate.json")


if __name__ == "__main__":
    main()
