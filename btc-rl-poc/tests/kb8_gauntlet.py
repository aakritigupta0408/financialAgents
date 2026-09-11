"""kb8/kb9 offline gauntlet — can any candidate beat the live kb7 recipe?

Replays one mid-window decision (6-9 mins left) per settled window, same
selection as tests/kb7_context_ablation.py, and scores every candidate on
the identical window set:

  kb7-replay   bolt-small @512, the live readout (baseline to beat)
  bolt-base    bigger Bolt checkpoint, same readout
  c2-uni-512   Chronos-2 univariate, same 512 context + readout
  c2-uni-2048  Chronos-2 univariate, long context (C2 supports 8k)
  c2-cov-512   Chronos-2 with past covariates: volume + high-low range

One decision per window => rows ARE windows: plain paired stats here are
already window-clustered. Pre-registered gate: a Chronos-2 candidate goes
live as kb9 only if it beats kb7-replay on paired Brier with |t| > 2.
Latency is measured per call (live budget ~2s/minute-loop).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.online import _chronos_p_up            # noqa: E402
from btc_rl.sources import fetch_range             # noqa: E402
from datetime import datetime, timedelta           # noqa: E402

QS = [i / 10 for i in range(1, 10)]


def quantile_p_up(vals, strike):
    """P(close >= strike) from decile forecasts — mirror of the live
    kb7 readout in btc_rl/online.py:_chronos_p_up (clamps included)."""
    if strike <= vals[0]:
        pr = 0.95
    elif strike >= vals[-1]:
        pr = 0.05
    else:
        pr = 0.5
        for i in range(len(vals) - 1):
            if vals[i] <= strike <= vals[i + 1]:
                frac = ((strike - vals[i]) / (vals[i + 1] - vals[i])
                        if vals[i + 1] > vals[i] else 0.5)
                pr = 1.0 - (QS[i] + frac * (QS[i + 1] - QS[i]))
                break
    return min(.95, max(.05, pr))


_BOLT_BASE = None
_C2 = None


def bolt_base_p(closes, strike, horizon):
    global _BOLT_BASE
    import torch
    if _BOLT_BASE is None:
        from chronos import BaseChronosPipeline
        _BOLT_BASE = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map="cpu",
            torch_dtype=torch.float32)
    ctx = torch.tensor(closes[-512:], dtype=torch.float32).unsqueeze(0)
    q, _ = _BOLT_BASE.predict_quantiles(
        ctx, prediction_length=max(1, horizon), quantile_levels=QS)
    return quantile_p_up([float(x) for x in q[0, -1]], strike)


def _c2():
    global _C2
    if _C2 is None:
        import torch
        from chronos import Chronos2Pipeline
        _C2 = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
    return _C2


def c2_p(closes, strike, horizon, n_ctx, covs=None):
    import numpy as np
    ctx = np.asarray(closes[-n_ctx:], dtype="float32")
    if covs is None:
        inp = [ctx]
    else:
        inp = [{"target": ctx,
                "past_covariates": {k: np.asarray(v[-n_ctx:], dtype="float32")
                                    for k, v in covs.items()}}]
    q, _ = _c2().predict_quantiles(
        inp, prediction_length=max(1, horizon), quantile_levels=QS)
    return quantile_p_up([float(x) for x in q[0][0, -1]], strike)


def main():
    kb = [json.loads(l) for l in
          (ROOT / "results" / "kalshi_binary_log.jsonl").open()]
    wins = {}
    for r in kb:
        if r.get("variant") == "kb2" and r.get("actual") is not None \
                and r.get("mkt_p_up") is not None \
                and 6 <= r["mins_left"] <= 9:
            wins.setdefault(r["ticker"], r)
    rows = sorted(wins.values(), key=lambda r: r["made_ts"])[-260:]
    print(f"windows: {len(rows)}")

    now = datetime.now().astimezone()
    bars = fetch_range(now - timedelta(hours=64), now)
    print(f"bars: {len(bars)}")
    keys = [b["ts"] for b in bars]
    closes = [b["close"] for b in bars]
    vols = [b["volume"] for b in bars]
    hlr = [b["high"] - b["low"] for b in bars]

    import bisect
    CANDS = ["kb7-replay", "bolt-base", "c2-uni-512", "c2-uni-2048",
             "c2-cov-512"]
    per = {c: [] for c in CANDS}          # (ticker, p, actual, mkt, secs)
    for r in rows:
        i = bisect.bisect_left(keys, r["made_ts"])
        if i < 520:
            continue
        upto, v, h = closes[:i], vols[:i], hlr[:i]
        horizon = int(max(1, round(r["mins_left"])))
        strike, y, mkt = r["strike"], r["actual"], r["mkt_p_up"]

        t0 = time.time()
        out = _chronos_p_up(upto, strike, horizon)
        if out:
            per["kb7-replay"].append(
                (r["ticker"], out[0], y, mkt, time.time() - t0))
        for name, fn in (
            ("bolt-base", lambda: bolt_base_p(upto, strike, horizon)),
            ("c2-uni-512", lambda: c2_p(upto, strike, horizon, 512)),
            ("c2-uni-2048", lambda: c2_p(upto, strike, horizon, 2048)),
            ("c2-cov-512", lambda: c2_p(upto, strike, horizon, 512,
                                        {"volume": v, "hl_range": h})),
        ):
            t0 = time.time()
            try:
                p = fn()
            except Exception as e:
                print(f"  {name} failed on {r['ticker']}: {e}")
                continue
            per[name].append((r["ticker"], p, y, mkt, time.time() - t0))

    base = {t: (p, y, mkt) for t, p, y, mkt, _ in per["kb7-replay"]}
    print(f"\n{'candidate':>12s} {'n':>4s} {'acc':>6s} {'brier':>7s} "
          f"{'t vs kb7':>9s} {'t vs mkt':>9s} {'med s':>6s}")
    for c in CANDS:
        rs = per[c]
        if not rs:
            continue
        n = len(rs)
        acc = sum((p >= .5) == bool(y) for _, p, y, _, _ in rs) / n
        br = sum((p - y) ** 2 for _, p, y, _, _ in rs) / n

        def paired_t(diffs):
            import math
            m = len(diffs)
            if m < 5:
                return float("nan")
            mu = sum(diffs) / m
            sd = (sum((d - mu) ** 2 for d in diffs) / (m - 1)) ** .5
            return mu / (sd / m ** .5) if sd else float("nan")

        d7 = [(p - y) ** 2 - (base[t][0] - y) ** 2
              for t, p, y, _, _ in rs if t in base]
        dm = [(p - y) ** 2 - (mkt - y) ** 2 for _, p, y, mkt, _ in rs]
        lat = sorted(s for *_, s in rs)[len(rs) // 2]
        print(f"{c:>12s} {n:4d} {acc:6.1%} {br:7.3f} "
              f"{paired_t(d7):9.2f} {paired_t(dm):9.2f} {lat:6.2f}")
    print("\n(negative t = candidate's Brier is LOWER = better; "
          "gate for kb9: t vs kb7 < -2)")


if __name__ == "__main__":
    main()
