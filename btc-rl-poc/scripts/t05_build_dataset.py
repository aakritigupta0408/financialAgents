"""T0.5 PRICE-DISCOVERY dataset builder (handoff §13/§80/Phase 5).

Question: do we know something BEFORE Kalshi reprices?

Streams the raw event shards (results/events/*.jsonl — the canonical dense
capture: kalshi quotes ~18/min, coinbase trades+L1 ~60/min) and, at sampled
decision points per active Kalshi ticker, computes PIT microstructure
features from Coinbase order flow (data strictly <= t) and labels = change
in Kalshi implied probability at t+5/+15/+30/+60s (data strictly > t).

No leakage: features use only receive_ts <= t; labels use the first kalshi
quote at-or-after t+N. Output: research/t05_repricing/dataset.jsonl.

Usage: python3 scripts/t05_build_dataset.py [n_recent_shards]
"""
import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "results" / "events"
OUT = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
SAMPLE_S = 15          # one decision point per ticker every ~15s
HORIZONS = [1, 2, 5, 15, 30, 60]
TRADE_WIN = 30         # order-flow lookback (s)


def kmid(q):
    yb, ya = q.get("yes_bid"), q.get("yes_ask")
    if yb is None or ya is None:
        return None
    return (yb + ya) / 2.0 / 100.0     # implied probability in [0,1]


def sgn_size(t):
    try:
        s = float(t.get("size") or 0)
    except (TypeError, ValueError):
        s = 0.0
    return s if t.get("side") == "buy" else -s


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    files = sorted(glob.glob(str(SHARDS / "*.jsonl")), key=os.path.getmtime)[-n:]
    # rolling coinbase state
    trades = collections.deque()      # (ts, signed_size, price)
    l1 = None                          # latest l1 dict
    mids = collections.deque()         # (ts, mid)
    # per-ticker kalshi quote series: ticker -> list[(ts, prob, spread)]
    kseries = collections.defaultdict(list)
    last_sample = {}                   # ticker -> last decision ts sampled
    pending = []                        # decision points awaiting labels
    rows = []

    def flush_ready(now):
        keep = []
        for d in pending:
            need = d["t"] + HORIZONS[-1]
            if now < need + 2:
                keep.append(d); continue
            ser = kseries[d["ticker"]]
            base = d["k_prob"]
            ok = True
            for H in HORIZONS:
                tgt = d["t"] + H
                fut = next((p for (ts, p, sp) in ser if ts >= tgt), None)
                if fut is None:
                    ok = False; break
                d["labels"]["dprob_%d" % H] = round(fut - base, 5)
            if ok:
                rows.append({"ts": d["t"], "ticker": d["ticker"],
                             **d["feat"], **d["labels"]})
        pending[:] = keep

    def feats(now, ticker, kprob, kspread):
        while trades and trades[0][0] < now - TRADE_WIN:
            trades.popleft()
        buy = sum(s for (_, s, _) in trades if s > 0)
        sell = -sum(s for (_, s, _) in trades if s < 0)
        tot = buy + sell
        ofi = (buy - sell) / tot if tot > 0 else 0.0
        # coinbase return + realized vol over window
        ms = [m for (ts, m) in mids if ts >= now - TRADE_WIN]
        ret = (ms[-1] / ms[0] - 1.0) if len(ms) >= 2 and ms[0] else 0.0
        rets = [ms[i] / ms[i - 1] - 1.0 for i in range(1, len(ms)) if ms[i - 1]]
        rvol = (sum(r * r for r in rets) / len(rets)) ** 0.5 if rets else 0.0
        l1imb, micro_dev = 0.0, 0.0
        if l1:
            try:
                bs, as_ = float(l1.get("bid_sz") or 0), float(l1.get("ask_sz") or 0)
                bid, ask = float(l1.get("bid") or 0), float(l1.get("ask") or 0)
                if bs + as_ > 0:
                    l1imb = (bs - as_) / (bs + as_)
                mid = (bid + ask) / 2.0
                if mid > 0 and (bs + as_) > 0:
                    microp = (bid * as_ + ask * bs) / (bs + as_)
                    micro_dev = (microp - mid) / mid
            except (TypeError, ValueError):
                pass
        cb_price = ms[-1] if ms else (l1 and (float(l1.get("bid") or 0) + float(l1.get("ask") or 0)) / 2.0) or None
        return {"cb_ofi_30s": round(ofi, 5), "cb_ret_30s": round(ret, 6),
                "cb_rvol_30s": round(rvol, 7), "cb_l1_imb": round(l1imb, 5),
                "cb_micro_dev": round(micro_dev, 8), "trade_n_30s": len(trades),
                "cb_price": round(cb_price, 2) if cb_price else None,
                "k_prob": round(kprob, 5), "k_spread": round(kspread, 4)}

    for fp in files:
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = e.get("receive_ts")
            if ts is None:
                continue
            src, kind = e.get("src"), e.get("kind")
            if src == "coinbase" and kind == "trade":
                try:
                    trades.append((ts, sgn_size(e), float(e.get("price") or 0)))
                except (TypeError, ValueError):
                    pass
            elif src == "coinbase" and kind == "l1":
                l1 = e
                try:
                    mids.append((ts, (float(e["bid"]) + float(e["ask"])) / 2.0))
                except (TypeError, ValueError, KeyError):
                    pass
                while mids and mids[0][0] < ts - 120:
                    mids.popleft()
            elif src == "kalshi" and kind == "quote":
                p = kmid(e)
                if p is None:
                    continue
                tk = e.get("ticker")
                sp = ((e.get("yes_ask") or 0) - (e.get("yes_bid") or 0)) / 100.0
                ser = kseries[tk]
                ser.append((ts, p, sp))
                if len(ser) > 4000:
                    del ser[:2000]
                # sample a decision point?
                if ts - last_sample.get(tk, 0) >= SAMPLE_S:
                    last_sample[tk] = ts
                    f = feats(ts, tk, p, sp)
                    pending.append({"t": ts, "ticker": tk, "k_prob": p,
                                    "feat": f, "labels": {}})
            flush_ready(ts)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    span = (rows[-1]["ts"] - rows[0]["ts"]) / 3600 if rows else 0
    print(f"t05 dataset: {len(rows)} labeled decision points from {len(files)} shards"
          f" (~{span:.1f}h span)")
    if rows:
        import statistics as st
        for H in HORIZONS:
            ds = [r["dprob_%d" % H] for r in rows]
            print(f"  |dprob_{H}s| median {st.median(abs(x) for x in ds):.4f}"
                  f"  moved>0.005: {sum(1 for x in ds if abs(x)>0.005)/len(ds):.2%}")


if __name__ == "__main__":
    main()
