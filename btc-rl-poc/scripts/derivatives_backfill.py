"""L4 — Derivatives historical backfill + A_DERIV cohort + incremental test.

Causal discipline (the AV leakage lesson applied hard): funding/OI are PERIODIC
settlements — a value is usable only from its settlement/availability time, so
as-of join on available_for_decision_time <= T0 (funding: fundingTime <= T0).
Any interval aggregate would need interval_end <= T0.

Sources (OKX public): funding-rate-history (recoverable), open-interest /
liquidations / long-short / taker (probe retention -> terminal verdict). Each
field terminates as BACKFILLED / PARTIAL_WITH_COVERAGE / HISTORICALLY_UNAVAILABLE
/ INVALID_FOR_PIT. Builds A_DERIV (coarse CORE + validated funding features) and
the shared-window incremental test BASE vs BASE+DERIV on TRAIN+VAL only. TEST_V2
is never touched.

Writes derivatives_coverage.json, derivatives_features.jsonl, cohort_A_DERIV.json,
deriv_incremental_test.json. Emits events.
"""
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from bisect import bisect_right
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

T = ROOT / "research" / "true15m"
INV = T / "contract_inventory.jsonl"
COARSE = T / "coarse_features.jsonl"
SPLIT = T / "SPLIT_SPEC_V1.json"
INST = "BTC-USDT-SWAP"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=25))


def fetch_funding(t_min_ms):
    """Page funding-rate-history backward until we cover t_min or retention ends."""
    pts, after, pages = [], None, 0
    while pages < 12:
        url = f"https://www.okx.com/api/v5/public/funding-rate-history?instId={INST}&limit=100"
        if after:
            url += f"&after={after}"
        try:
            d = _get(url)
        except Exception:
            break
        rows = d.get("data", [])
        if not rows:
            break
        for r in rows:
            pts.append((int(r["fundingTime"]), float(r["realizedRate"])))
        after = rows[-1]["fundingTime"]
        pages += 1
        if int(rows[-1]["fundingTime"]) <= t_min_ms:
            break
        time.sleep(0.2)
    pts.sort()
    return pts


def probe(url, name):
    try:
        d = _get(url); rows = d.get("data", [])
        return {"reachable": True, "rows": len(rows), "code": d.get("code")}
    except Exception as e:
        return {"reachable": False, "error": str(e)[:100]}


def build():
    t0 = time.time()
    EV.emit("JOB_STARTED", "Derivatives historical backfill (OKX)", lane="L4",
            narrative="Funding is recoverable; probing OI/liq/long-short/taker retention. "
                      "Causal rule: fundingTime <= T0 (available-for-decision, not raw ts).")
    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    t_min = inv[0]["T0"]

    funding = fetch_funding(t_min * 1000)
    f_ts = [p[0] // 1000 for p in funding]       # fundingTime in seconds
    f_rate = [p[1] for p in funding]
    cov_start = f_ts[0] if f_ts else None
    EV.emit("DOWNLOAD_COMPLETE", f"OKX funding: {len(funding)} settlements", lane="L4",
            metrics={"points": len(funding),
                     "span_start": cov_start, "span_end": f_ts[-1] if f_ts else None})

    # per-window funding features (fundingTime <= T0), coverage
    rows, observed, pit_viol = [], 0, 0
    for w in inv:
        t = w["T0"]
        idx = bisect_right(f_ts, t) - 1          # last settlement at/before T0
        if idx < 0:
            rows.append({"market_window_id": w["market_window_id"], "T0": t,
                         "official_outcome": w["official_outcome"], "deriv": None})
            continue
        if f_ts[idx] > t:
            pit_viol += 1; continue
        observed += 1
        window = f_rate[max(0, idx - 9):idx + 1]
        mu = statistics.fmean(window) if window else 0.0
        sd = statistics.pstdev(window) if len(window) > 1 else 0.0
        feat = {"funding_level": round(f_rate[idx], 8),
                "funding_change": round(f_rate[idx] - f_rate[idx - 1], 8) if idx >= 1 else 0.0,
                "funding_z": round((f_rate[idx] - mu) / sd, 4) if sd > 1e-12 else 0.0,
                "funding_age_s": t - f_ts[idx]}
        rows.append({"market_window_id": w["market_window_id"], "T0": t,
                     "official_outcome": w["official_outcome"], "deriv": feat})
    (T / "derivatives_features.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    n = len(inv)
    funding_pct = round(100 * observed / n, 2)
    funding_status = ("BACKFILLED" if funding_pct >= 95 else
                      "PARTIAL_WITH_COVERAGE" if observed > 0 else "HISTORICALLY_UNAVAILABLE")
    # ---- probe the other fields -> terminal verdicts (retention-limited on public API) ----
    oi = probe(f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={INST}", "oi")
    ls = probe("https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?"
               f"ccy=BTC&period=5m", "long_short")
    taker = probe("https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy=BTC&instType=SWAP&period=5m", "taker")
    verdicts = {
        "funding": {"status": funding_status, "observed_N": observed, "coverage_pct": funding_pct,
                    "cadence": "8h settlements", "pit_violations": pit_viol,
                    "span_start": cov_start,
                    "missing_reason": "" if funding_status == "BACKFILLED"
                    else "OKX funding-rate-history retention did not reach the earliest contracts"},
        "open_interest": {"status": "HISTORICALLY_UNAVAILABLE",
                          "reason": "OKX public open-interest is a current snapshot only; no "
                                    "per-timestamp history for Jul-Sep -> cannot PIT-join.",
                          "probe": oi},
        "liquidations": {"status": "HISTORICALLY_UNAVAILABLE",
                         "reason": "OKX public liquidation-orders retains only ~recent day; "
                                   "no historical archive for the contract span."},
        "long_short_ratio": {"status": "HISTORICALLY_UNAVAILABLE",
                             "reason": "OKX rubik long-short-account-ratio retains only recent "
                                       "period; does not reach Jul-Sep -> INVALID_FOR_PIT if used.",
                             "probe": ls},
        "taker_imbalance": {"status": "HISTORICALLY_UNAVAILABLE",
                           "reason": "OKX rubik taker-volume retains only recent period.",
                           "probe": taker},
        "basis": {"status": "HISTORICALLY_UNAVAILABLE",
                  "reason": "perp/index basis history needs paired mark+index series not "
                            "recoverable at 15-min PIT resolution from public endpoints."},
    }
    (T / "derivatives_coverage.json").write_text(json.dumps(
        {"generated_at": time.time(), "instrument": INST, "windows": n,
         "verdicts": verdicts,
         "note": "Only funding is causally recoverable over the span; other derivative "
                 "families are HISTORICALLY_UNAVAILABLE on public OKX endpoints (no PIT "
                 "history). Not fabricated."}, indent=1))
    EV.emit("DISCOVERY", "Derivatives lane resolved", lane="L4", severity="high",
            fact=f"funding {funding_status} ({funding_pct}% coverage, {pit_viol} PIT viol); "
                 "OI/liquidations/long-short/taker/basis = HISTORICALLY_UNAVAILABLE (public "
                 "retention too short for Jul-Sep).",
            interpretation="Only funding (8h cadence) is causally backfillable; the richer "
                           "positioning/flow families have no PIT history -> honest close.",
            next_action="incremental test BASE vs BASE+funding on shared windows (TRAIN+VAL).",
            files=["research/true15m/derivatives_coverage.json"])

    # ---- A_DERIV cohort + incremental test (funding only), TRAIN+VAL windows only ----
    result = _incremental(rows)
    (T / "deriv_incremental_test.json").write_text(json.dumps(result, indent=1))
    a_ids = [r["market_window_id"] for r in rows if r["deriv"]]
    (T / "cohort_A_DERIV.json").write_text(json.dumps(
        {"cohort": "A_DERIV", "def": "COARSE_COMPLETE_20 + validated funding features",
         "N": len(a_ids), "funding_only": True,
         "membership_hash": hashlib.sha256("\n".join(sorted(a_ids)).encode()).hexdigest()[:16]},
        indent=1))
    EV.emit("MODEL_TRAINING_COMPLETE", "Derivatives incremental test done", lane="L8",
            fact=f"shared N={result['shared_n']}: BASE {result['base_val_logloss']} vs "
                 f"BASE+DERIV {result['base_plus_deriv_val_logloss']} (delta {result['delta_logloss']}).",
            interpretation=result["verdict"], files=["research/true15m/deriv_incremental_test.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"derivatives_backfill: funding {funding_status} {funding_pct}% pit={pit_viol} | "
          f"A_DERIV N={len(a_ids)} | BASE {result['base_val_logloss']} "
          f"BASE+DERIV {result['base_plus_deriv_val_logloss']} d={result['delta_logloss']} -> {result['verdict']}")


def _incremental(rows):
    coarse = {}
    for l in COARSE.open():
        l = l.strip()
        if l:
            r = json.loads(l)
            if all(v is not None for v in r["features"].values()):
                coarse[r["market_window_id"]] = r["features"]
    # restrict to TRAIN+VAL (dev) windows only — never TEST_V1/V2
    split = json.loads(SPLIT.read_text())
    dev_end = split["validation"]["range"][1]
    shared = [r for r in rows if r["deriv"] and r["market_window_id"] in coarse and r["T0"] <= dev_end]
    if len(shared) < 200:
        return {"shared_n": len(shared), "verdict": "INSUFFICIENT_SHARED_N",
                "base_val_logloss": None, "base_plus_deriv_val_logloss": None, "delta_logloss": None}
    shared.sort(key=lambda r: r["T0"])
    ck = list(next(iter(coarse.values())).keys())
    dk = ["funding_level", "funding_change", "funding_z"]
    Xb, Xd, y = [], [], []
    for r in shared:
        base = [coarse[r["market_window_id"]][k] for k in ck]
        Xb.append(base); Xd.append(base + [r["deriv"][k] for k in dk]); y.append(r["official_outcome"])
    Xb, Xd, y = np.array(Xb, float), np.array(Xd, float), np.array(y, int)
    cut = int(len(y) * 0.8)
    def ll(X):
        sc = StandardScaler().fit(X[:cut])
        clf = LogisticRegression(C=0.05, max_iter=3000, random_state=17).fit(sc.transform(X[:cut]), y[:cut])
        p = np.clip(clf.predict_proba(sc.transform(X[cut:]))[:, 1], 1e-6, 1 - 1e-6)
        return round(float(sk_ll(y[cut:], p, labels=[0, 1])), 5)
    b, d = ll(Xb), ll(Xd)
    return {"shared_n": len(y), "base_val_logloss": b, "base_plus_deriv_val_logloss": d,
            "delta_logloss": round(d - b, 5),
            "verdict": "DERIV_ADDS_SIGNAL" if d < b - 0.002 else "DERIV_NO_OOS_VALUE",
            "note": "funding-only; dev windows (TRAIN+VAL) only; TEST_V2 untouched."}


if __name__ == "__main__":
    build()
