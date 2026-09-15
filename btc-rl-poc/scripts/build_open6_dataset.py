"""Build the LARGE open+6min dataset from feature_snapshots (1366 windows), reconstructing the
first-6-minute price path per window and its features, decided at open+6min (~9 min left).
Strictly PIT: uses only closes within [open, open+360s]. Writes results/open6_dataset.jsonl.
"""
import json, math, statistics as st
from datetime import datetime, timezone
from pathlib import Path

RES = Path("results")
OUT = RES / "open6_dataset.jsonl"
SIX = 360.0


def _ep(t):
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def labels():
    m = {}
    for l in (RES / "contract_outcomes.jsonl").open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1):
            m[d["ticker"]] = (int(d["exact_yes"]), _ep(d.get("open_time")), _ep(d.get("close_time")),
                              d.get("floor_strike"))
    return m


def run():
    lab = labels()
    seen = set()
    n = 0
    with OUT.open("w") as out:
        for l in (RES / "feature_snapshots.jsonl").open():
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            wid = d.get("window_id")
            f = d.get("features") or {}
            if not (wid and isinstance(f, dict) and f.get("closes") and f.get("strike")):
                continue
            if wid in seen or wid not in lab:
                continue
            y, open_t, close_t, floor = lab[wid]
            dec = _ep(d.get("decision_ts") or d.get("event_ts") or d.get("persist_ts"))
            if not (open_t and close_t and dec) or dec <= open_t:
                continue
            closes = [c for c in f["closes"] if c]
            span = dec - open_t                       # seconds the closes cover
            if span < SIX or len(closes) < 20:
                continue
            idx6 = max(8, int(round(SIX / span * len(closes))))
            path = closes[:idx6]                      # first 6 minutes only (PIT)
            if len(path) < 8:
                continue
            tgt = f.get("strike"); cur = path[-1]
            tr = close_t - (open_t + SIX)             # time remaining at open+6min (~540s)
            rets = [math.log(path[i] / path[i - 1]) for i in range(1, len(path)) if path[i - 1] > 0]
            vol = st.pstdev(rets) if len(rets) > 1 else 1e-5
            dt = max(1.0, SIX / max(1, len(path) - 1))
            sig_s = vol / math.sqrt(dt)
            sigma_T = cur * sig_s * math.sqrt(max(1.0, tr))
            mom = cur - path[0]
            drift = 0.5 * mom * (tr / 900.0)
            z = (cur - tgt) / max(sigma_T, 1e-6)
            z_dr = (cur + drift - tgt) / max(sigma_T, 1e-6)
            up = sum(r for r in rets if r > 0); dn = -sum(r for r in rets if r < 0)
            rsi = up / (up + dn) if (up + dn) else 0.5
            k = min(5, len(path)); short_ma = sum(path[-k:]) / k
            row = {"ticker": wid, "y": y, "ts": dec, "tr": tr,
                   "z": z, "z_drift": z_dr, "theo": _phi(z), "theo_drift": _phi(z_dr),
                   "dist_bps": (cur - tgt) / cur * 1e4, "sigmaT_bps": sigma_T / cur * 1e4,
                   "mom6_bps": mom / cur * 1e4, "rsi": rsi,
                   "ma_dist_bps": (cur - short_ma) / cur * 1e4,
                   "range_bps": (max(path) - min(path)) / cur * 1e4, "npath": len(path)}
            out.write(json.dumps(row) + "\n"); seen.add(wid); n += 1
    print(f"open6_dataset: {n} windows -> {OUT}")


if __name__ == "__main__":
    run()
