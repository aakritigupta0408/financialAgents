"""Live mode: at ~6:45 PM Pacific, predict the 7:00 and 7:15 PM integer prices.

Usage:  python -m btc_rl.live            # make predictions now
        python -m btc_rl.live --score    # score any past unscored predictions

Predictions append to results/live_log.jsonl so accuracy accrues over days.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .features import compute_features, discretize
from .sources import (fetch_coinbase_candles, fetch_fear_greed,
                      fetch_okx_funding_rate)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOG = RESULTS_DIR / "live_log.jsonl"


def _load_q_tables() -> dict:
    path = RESULTS_DIR / "q_table.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {name: {tuple(map(int, k.split("|"))): v for k, v in tbl.items()}
            for name, tbl in raw.items()}


def predict_now() -> None:
    now = datetime.now(tz=config.PACIFIC)
    start = now - timedelta(minutes=config.LOOKBACK_MIN + 10)
    bars = fetch_coinbase_candles(start, now)
    if len(bars) < config.LOOKBACK_MIN:
        raise SystemExit(f"only {len(bars)} bars available; need "
                         f"{config.LOOKBACK_MIN}")
    fng = fetch_fear_greed().get(now.date().isoformat())
    feat = compute_features(bars, fng)
    state = discretize(feat)
    funding = fetch_okx_funding_rate()
    q_tables = _load_q_tables()

    records = []
    for (hh, mm), horizon in config.TARGET_SLOTS:
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target < now:
            target += timedelta(days=1)
        preds = {"persistence-baseline": int(feat["price"])}
        for mode in ("sparse", "shaped"):
            tbl = q_tables.get(f"tabular-q-{mode}_h{horizon}")
            if tbl and state in tbl:
                qs = tbl[state]
                best = max(range(len(qs)), key=lambda i: qs[i])
                delta = config.ACTION_DELTAS[best]
            else:
                delta = 0
            preds[f"tabular-q-{mode}"] = int(feat["price"] + delta)
        rec = {"made_at": now.isoformat(), "target": target.isoformat(),
               "horizon_min": horizon, "price_now": feat["price"],
               "state": list(state), "funding_rate": funding,
               "fng": feat["fng"], "predictions": preds, "actual": None}
        records.append(rec)
        print(f"target {target:%Y-%m-%d %H:%M %Z}: {preds}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def score_pending() -> None:
    if not LOG.exists():
        print("no live log yet")
        return
    rows = [json.loads(line) for line in LOG.read_text().splitlines()]
    now = datetime.now(tz=config.PACIFIC)
    changed = False
    for row in rows:
        if row["actual"] is not None:
            continue
        target = datetime.fromisoformat(row["target"])
        if target > now - timedelta(minutes=2):
            continue
        bars = fetch_coinbase_candles(target - timedelta(minutes=1),
                                      target + timedelta(minutes=1))
        match = [b for b in bars if b["ts"] == int(target.timestamp())]
        if not match:
            continue
        actual = match[0]["close"]
        row["actual"] = actual
        row["hits"] = {name: int(p) == int(actual)
                       for name, p in row["predictions"].items()}
        changed = True
        print(f"{row['target']}: actual={actual} hits={row['hits']}")
    if changed:
        LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    score_pending() if args.score else predict_now()
