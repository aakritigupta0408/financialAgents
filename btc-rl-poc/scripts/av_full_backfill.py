"""L2 — Alpha Vantage FULL historical context backfill + A_AV cohort + incremental
family test (directive P1.1/P7/P8/P15).

Backfills real equity/commodity intraday history (month-sliced), as-of joins each
contract T0 with EXPLICIT session semantics (market_open / observation_age /
previous_session_state) — never forward-filling a stale equity as "live". Produces
a per-instrument coverage table, AV context features, the A_AV cohort (coarse CORE
∩ contracts with fresh equity context at T0), and the shared-window incremental
test: BASE(coarse) vs BASE+AV. Absolute price is a primitive; only returns/RSI/vol
context enter features (§6).

Writes: av_full_coverage.json, av_context_features.jsonl, cohort_A_AV.json,
av_incremental_test.json. Emits events.
"""
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as MET          # noqa: E402
from btc_rl import research_events as EV    # noqa: E402

KEY = (Path.home() / ".alphavantage_key").read_text().strip()
BASE = "https://www.alphavantage.co/query"
T = ROOT / "research" / "true15m"
INV = T / "contract_inventory.jsonl"
COARSE = T / "coarse_features.jsonl"
UTC = ZoneInfo("UTC")
BAR_S = 300                      # 5-min bar; timestamped at START -> available at ts+BAR_S
MONTHS = ["2026-07", "2026-08", "2026-09"]
EQUITIES = ["SPY", "QQQ", "IWM", "COIN", "MSTR", "MARA", "RIOT"]
COMMOD = ["GLD", "USO"]          # gold / oil proxies
FRESH_S = 600                    # <=10 min since last bar => market effectively open


def _call(p):
    p["apikey"] = KEY
    with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(p), timeout=45) as r:
        return json.load(r)


def _fetch(sym):
    bars = []
    tz = "US/Eastern"
    for m in MONTHS:
        try:
            d = _call({"function": "TIME_SERIES_INTRADAY", "symbol": sym,
                       "interval": "5min", "outputsize": "full", "month": m,
                       "extended_hours": "false"})
        except Exception:
            continue
        meta = next((v for k, v in d.items() if "Meta Data" in k), {})
        tz = meta.get("6. Time Zone") or tz
        tskey = next((k for k in d if "Time Series" in k), None)
        if not tskey:
            continue
        zone = ZoneInfo(tz)
        for tstr, o in d[tskey].items():
            dt = datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone)
            bars.append((int(dt.astimezone(UTC).timestamp()), float(o["4. close"])))
    bars.sort()
    # dedup by ts
    out, seen = [], set()
    for t, p in bars:
        if t not in seen:
            seen.add(t); out.append((t, p))
    return out


def _feat(bars_t, bars_p, idx):
    P = bars_p[:idx + 1]
    if len(P) < 13:
        return None
    def ret(k):
        return round(math.log(P[-1] / P[-1 - k]), 6) if len(P) > k and P[-1 - k] > 0 else None
    rets = [math.log(P[i] / P[i - 1]) for i in range(max(1, len(P) - 12), len(P)) if P[i - 1] > 0]
    rvol = round((sum(x * x for x in rets) / len(rets)) ** 0.5, 6) if rets else None
    # RSI14 on 5min closes
    g = l = 0.0
    for i in range(-14, 0):
        dd = P[i] - P[i - 1]; g += max(dd, 0); l += max(-dd, 0)
    rsi = 50.0 if g + l == 0 else round(100 - 100 / (1 + (g / 14) / ((l / 14) or 1e-9)), 2)
    return {"ret_5m": ret(1), "ret_15m": ret(3), "ret_30m": ret(6),
            "ret_60m": ret(12), "rsi_14": rsi, "rvol": rvol}


def build():
    t0 = time.time()
    EV.emit("JOB_STARTED", "AV FULL historical context backfill", lane="L2",
            narrative=f"Backfilling {len(EQUITIES+COMMOD)} instruments x {len(MONTHS)} months "
                      "with session masks; resolving real T0 coverage over 6,337 contracts.")
    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    series = {}
    for sym in EQUITIES + COMMOD:
        EV.emit("DOWNLOAD_STARTED", f"AV {sym} intraday (3 months)", lane="L2")
        try:
            b = _fetch(sym)
        except Exception as e:
            b = []
            EV.emit("WARNING", f"AV {sym} fetch error", lane="L2", technical=str(e)[:120])
        series[sym] = b
        if b:
            EV.emit("DOWNLOAD_COMPLETE", f"AV {sym}: {len(b)} bars", lane="L2",
                    metrics={"bars": len(b)})

    # per-window context + coverage accounting
    cov = {sym: {"observed": 0, "market_open": 0, "stale_but_known": 0,
                 "ages": [], "pit_viol": 0} for sym in series}
    ctx_rows = []
    for w in inv:
        t = w["T0"]; row = {"market_window_id": w["market_window_id"], "T0": t,
                            "official_outcome": w["official_outcome"], "av": {}, "masks": {}}
        breadth = []
        for sym, bars in series.items():
            if not bars:
                continue
            ts = [x[0] for x in bars]; px = [x[1] for x in bars]
            # a 5-min bar at ts is only AVAILABLE at ts+BAR_S (bar close); require
            # the bar fully closed at or before T0 (no future-interval leakage)
            idx = bisect_right(ts, t - BAR_S) - 1
            if idx < 0:
                continue
            if ts[idx] + BAR_S > t:
                cov[sym]["pit_viol"] += 1; continue
            age = t - (ts[idx] + BAR_S)          # age since the bar's CLOSE
            cov[sym]["observed"] += 1; cov[sym]["ages"].append(age)
            fresh = age < FRESH_S
            cov[sym]["market_open" if fresh else "stale_but_known"] += 1
            f = _feat(ts, px, idx)
            if f:
                row["av"][sym] = f
                row["masks"][sym] = {"market_open": fresh, "observation_age_s": age}
                if sym in EQUITIES and fresh and f.get("ret_15m") is not None:
                    breadth.append(1 if f["ret_15m"] > 0 else 0)
        if breadth:
            row["av_agg"] = {"risk_on_breadth_15m": round(sum(breadth) / len(breadth), 4),
                             "n_fresh_equities": len(breadth)}
        ctx_rows.append(row)

    n = len(inv)
    coverage = {}
    for sym, c in cov.items():
        ages = sorted(c["ages"])
        coverage[sym] = {
            "eligible_N": n, "observed_N": c["observed"],
            "coverage_pct": round(100 * c["observed"] / n, 2),
            "market_open_N": c["market_open"], "market_closed_N": c["observed"] - c["market_open"],
            "fresh_at_T0_N": c["market_open"], "stale_but_known_N": c["stale_but_known"],
            "observation_age_s": {"min": ages[0] if ages else None,
                                  "median": round(statistics.median(ages), 0) if ages else None,
                                  "p95": ages[int(len(ages) * 0.95)] if ages else None,
                                  "max": ages[-1] if ages else None},
            "pit_violations": c["pit_viol"],
            "missing_reason": "no bar (fetch gap or pre-history)" if c["observed"] < n else "",
            "status": "BACKFILLED" if c["observed"] > 0 else "HISTORICALLY_UNAVAILABLE",
        }
    (T / "av_full_coverage.json").write_text(json.dumps(
        {"generated_at": time.time(), "months": MONTHS, "instruments": list(series.keys()),
         "fresh_threshold_s": FRESH_S, "coverage": coverage,
         "note": "session-masked; equities cover only US regular hours so BTC contracts "
                 "outside 09:30-16:00 ET are market_closed (stale_but_known), not forward-filled."},
        indent=1))
    body = "".join(json.dumps(r) + "\n" for r in ctx_rows)
    (T / "av_context_features.jsonl").write_text(body)

    # ---- A_AV cohort = coarse CORE windows with FRESH equity context (SPY market_open) ----
    coarse = {}
    for l in COARSE.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if all(v is not None for v in r["features"].values()):
            coarse[r["market_window_id"]] = r
    a_av = [r for r in ctx_rows if r["market_window_id"] in coarse
            and r["masks"].get("SPY", {}).get("market_open")]
    a_av_ids = [r["market_window_id"] for r in a_av]
    (T / "cohort_A_AV.json").write_text(json.dumps(
        {"cohort": "A_AV", "def": "COARSE_COMPLETE_20 windows with fresh (market-open) AV "
         "equity context at T0", "N": len(a_av_ids),
         "date_range": [a_av[0]["T0"], a_av[-1]["T0"]] if a_av else None,
         "membership_hash": hashlib.sha256("\n".join(sorted(a_av_ids)).encode()).hexdigest()[:16]},
        indent=1))
    EV.emit("DISCOVERY", f"A_AV cohort = {len(a_av_ids)} windows", lane="L2", severity="high",
            fact="Cross-asset context is only 'live' during US market hours; A_AV (coarse ∩ "
                 f"fresh equities) = {len(a_av_ids)} of {len(coarse)} coarse windows. "
                 f"Coverage: " + ", ".join(f"{s} {coverage[s]['coverage_pct']}%" for s in EQUITIES),
            interpretation="Cross-asset features can only apply to the market-hours subset; "
                           "a large fraction of 24/7 BTC contracts have no live equity context.",
            next_action="incremental test BASE vs BASE+AV on the shared A_AV windows.",
            files=["research/true15m/av_full_coverage.json", "research/true15m/cohort_A_AV.json"])

    # ---- incremental family test: BASE vs BASE+AV on shared A_AV windows (walk-forward) ----
    result = _incremental(a_av, coarse)
    (T / "av_incremental_test.json").write_text(json.dumps(result, indent=1))
    EV.emit("MODEL_TRAINING_COMPLETE", "AV incremental test done", lane="L15" if False else "L8",
            fact=f"shared N={result['shared_n']}: BASE logloss {result['base_val_logloss']} vs "
                 f"BASE+AV {result['base_plus_av_val_logloss']} (delta {result['delta_logloss']}).",
            interpretation=result["verdict"],
            files=["research/true15m/av_incremental_test.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"av_full_backfill: A_AV N={len(a_av_ids)} | BASE {result['base_val_logloss']} "
          f"BASE+AV {result['base_plus_av_val_logloss']} delta {result['delta_logloss']} -> {result['verdict']}")


def _incremental(a_av, coarse):
    """BASE(coarse 20) vs BASE+AV context, same windows, chronological 80/20 dev/val."""
    if len(a_av) < 200:
        return {"shared_n": len(a_av), "verdict": "INSUFFICIENT_SHARED_N",
                "base_val_logloss": None, "base_plus_av_val_logloss": None, "delta_logloss": None}
    a_av = sorted(a_av, key=lambda r: r["T0"])
    ck = list(next(iter(coarse.values()))["features"].keys())
    def av_vec(r):
        v = []
        for sym in EQUITIES:
            f = r["av"].get(sym) or {}
            v += [f.get("ret_15m") or 0.0, f.get("rsi_14") or 50.0, f.get("rvol") or 0.0]
        v.append((r.get("av_agg") or {}).get("risk_on_breadth_15m", 0.5))
        return v
    Xb, Xa, y = [], [], []
    for r in a_av:
        cf = coarse[r["market_window_id"]]["features"]
        base = [cf[k] for k in ck]
        Xb.append(base); Xa.append(base + av_vec(r)); y.append(r["official_outcome"])
    Xb, Xa, y = np.array(Xb, float), np.array(Xa, float), np.array(y, int)
    cut = int(len(y) * 0.8)
    def fit_ll(X):
        sc = StandardScaler().fit(X[:cut])
        clf = LogisticRegression(C=0.05, max_iter=3000, random_state=17).fit(sc.transform(X[:cut]), y[:cut])
        p = np.clip(clf.predict_proba(sc.transform(X[cut:]))[:, 1], 1e-6, 1 - 1e-6)
        return round(float(sk_ll(y[cut:], p, labels=[0, 1])), 5)
    base_ll = fit_ll(Xb); av_ll = fit_ll(Xa)
    delta = round(av_ll - base_ll, 5)
    verdict = ("AV_ADDS_SIGNAL" if delta < -0.002 else "AV_NO_OOS_VALUE")
    return {"shared_n": len(y), "val_n": len(y) - cut,
            "base_val_logloss": base_ll, "base_plus_av_val_logloss": av_ll,
            "delta_logloss": delta, "verdict": verdict,
            "note": "same-windows comparison (§15); AV adds value only if it lowers OOS log loss."}


if __name__ == "__main__":
    build()
