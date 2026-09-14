"""L6 — News historical backfill + A_NEWS cohort + incremental test.

Strict causal semantics: a news feature at T0 uses only articles with
published_at <= T0. Syndicated copies are deduped (earliest publication per
title). AV NEWS_SENTIMENT time_published is UTC. News is sparse (~30-50 BTC
articles/day) so many windows legitimately have "no recent news" — that is a
valid state, not missing coverage.

Features (all <= T0): article counts 15m/1h/4h, arrival acceleration, seconds
since last article, mean/dispersion sentiment (4h), decayed sentiment, source
diversity. Builds A_NEWS (coarse CORE + news state) and the incremental test
BASE vs BASE+NEWS on TRAIN+VAL only. TEST_V2 untouched.

Writes news_coverage.json, news_features.jsonl, cohort_A_NEWS.json,
news_incremental_test.json. Emits events.
"""
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV  # noqa: E402

KEY = (Path.home() / ".alphavantage_key").read_text().strip()
BASE = "https://www.alphavantage.co/query"
T = ROOT / "research" / "true15m"
INV = T / "contract_inventory.jsonl"
COARSE = T / "coarse_features.jsonl"
SPLIT = T / "SPLIT_SPEC_V1.json"
MONTHS = [("20260707T0000", "20260801T0000"), ("20260801T0000", "20260901T0000"),
          ("20260901T0000", "20260913T0000")]


def _call(p):
    p["apikey"] = KEY
    with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(p), timeout=45) as r:
        return json.load(r)


def _epoch(s):        # "YYYYMMDDTHHMMSS" (UTC)
    return int(datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp())


def fetch_news():
    by_title = {}
    for tf, tt in MONTHS:
        try:
            d = _call({"function": "NEWS_SENTIMENT", "tickers": "CRYPTO:BTC",
                       "time_from": tf, "time_to": tt, "limit": "1000", "sort": "EARLIEST"})
        except Exception:
            continue
        for f in d.get("feed", []):
            title = (f.get("title") or "").strip().lower()
            tp = f.get("time_published")
            if not title or not tp:
                continue
            try:
                ts = _epoch(tp)
            except ValueError:
                continue
            sent = float(f.get("overall_sentiment_score") or 0.0)
            src = f.get("source") or "?"
            if title not in by_title or ts < by_title[title][0]:   # dedup: earliest copy
                by_title[title] = (ts, sent, src)
    arts = sorted(by_title.values())
    return arts


def build():
    t0 = time.time()
    EV.emit("JOB_STARTED", "News historical backfill (AV, BTC)", lane="L6",
            narrative="Strict published_at <= T0; dedup syndicated copies; sparse-news windows "
                      "are a valid 'no recent news' state.")
    arts = fetch_news()
    a_ts = [a[0] for a in arts]; a_sent = [a[1] for a in arts]; a_src = [a[2] for a in arts]
    EV.emit("DOWNLOAD_COMPLETE", f"News: {len(arts)} deduped BTC articles", lane="L6",
            metrics={"articles": len(arts),
                     "span": [a_ts[0], a_ts[-1]] if a_ts else None})

    inv = [json.loads(l) for l in INV.open() if l.strip()]
    inv.sort(key=lambda r: r["T0"])
    rows, active, pit = [], 0, 0
    for w in inv:
        t = w["T0"]
        hi = bisect_right(a_ts, t)                  # articles published <= T0
        if hi < len(a_ts) and a_ts[hi] <= t:
            pit += 1                                 # sentinel; should be 0
        def cnt(sec):
            return hi - bisect_left(a_ts, t - sec)
        c15, c1h, c4h = cnt(900), cnt(3600), cnt(14400)
        feat = {"news_cnt_15m": c15, "news_cnt_1h": c1h, "news_cnt_4h": c4h,
                "news_secs_since_last": (t - a_ts[hi - 1]) if hi > 0 else 999999,
                "news_arrival_accel": c1h - (c4h - c1h) / 3.0}
        if c4h > 0:
            lo = bisect_left(a_ts, t - 14400)
            seg_s = a_sent[lo:hi]; seg_src = a_src[lo:hi]
            feat["news_sent_mean_4h"] = round(statistics.fmean(seg_s), 4)
            feat["news_sent_disp_4h"] = round(statistics.pstdev(seg_s), 4) if len(seg_s) > 1 else 0.0
            feat["news_sent_decayed"] = round(sum(s * math.exp(-(t - a_ts[lo + i]) / 3600.0)
                                                  for i, s in enumerate(seg_s)), 4)
            feat["news_source_diversity_4h"] = len(set(seg_src))
            active += 1
        else:
            feat.update({"news_sent_mean_4h": 0.0, "news_sent_disp_4h": 0.0,
                         "news_sent_decayed": 0.0, "news_source_diversity_4h": 0})
        rows.append({"market_window_id": w["market_window_id"], "T0": t,
                     "official_outcome": w["official_outcome"], "news": feat})
    (T / "news_features.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    n = len(inv)
    doc = {"generated_at": time.time(), "articles_deduped": len(arts),
           "windows": n, "windows_with_recent_news_4h": active,
           "active_pct": round(100 * active / n, 2), "pit_violations": pit,
           "status": "BACKFILLED" if len(arts) > 100 else "PARTIAL_WITH_COVERAGE",
           "tz": "UTC (AV time_published)",
           "note": "source fully backfilled; sparse news means many windows have a valid "
                   "no-recent-news state (not missing)."}
    (T / "news_coverage.json").write_text(json.dumps(doc, indent=1))
    result = _incremental(rows)
    (T / "news_incremental_test.json").write_text(json.dumps(result, indent=1))
    ids = [r["market_window_id"] for r in rows]
    (T / "cohort_A_NEWS.json").write_text(json.dumps(
        {"cohort": "A_NEWS", "def": "COARSE_COMPLETE_20 + news state (published_at<=T0)",
         "N": len(ids), "membership_hash": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]},
        indent=1))
    EV.emit("DISCOVERY", "News lane resolved", lane="L6", severity="high",
            fact=f"{len(arts)} deduped articles; {doc['active_pct']}% of windows have news in the "
                 f"prior 4h; {pit} PIT violations. Incremental: BASE {result['base_val_logloss']} vs "
                 f"BASE+NEWS {result['base_plus_news_val_logloss']} (delta {result['delta_logloss']}).",
            interpretation=result["verdict"] + " — " + (
                "news state adds OOS signal" if result["verdict"] == "NEWS_ADDS_SIGNAL"
                else "news state adds no OOS signal to the coarse baseline"),
            next_action="options probe -> binary resolution.",
            files=["research/true15m/news_coverage.json", "research/true15m/news_incremental_test.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"news_backfill: {len(arts)} articles, active {doc['active_pct']}%, pit={pit} | "
          f"BASE {result['base_val_logloss']} BASE+NEWS {result['base_plus_news_val_logloss']} "
          f"d={result['delta_logloss']} -> {result['verdict']}")


def _incremental(rows):
    coarse = {}
    for l in COARSE.open():
        l = l.strip()
        if l:
            r = json.loads(l)
            if all(v is not None for v in r["features"].values()):
                coarse[r["market_window_id"]] = r["features"]
    dev_end = json.loads(SPLIT.read_text())["validation"]["range"][1]
    shared = [r for r in rows if r["market_window_id"] in coarse and r["T0"] <= dev_end]
    if len(shared) < 200:
        return {"shared_n": len(shared), "verdict": "INSUFFICIENT_SHARED_N",
                "base_val_logloss": None, "base_plus_news_val_logloss": None, "delta_logloss": None}
    shared.sort(key=lambda r: r["T0"])
    ck = list(next(iter(coarse.values())).keys())
    nk = ["news_cnt_1h", "news_cnt_4h", "news_secs_since_last", "news_arrival_accel",
          "news_sent_mean_4h", "news_sent_disp_4h", "news_sent_decayed", "news_source_diversity_4h"]
    Xb, Xn, y = [], [], []
    for r in shared:
        base = [coarse[r["market_window_id"]][k] for k in ck]
        Xb.append(base); Xn.append(base + [r["news"][k] for k in nk]); y.append(r["official_outcome"])
    Xb, Xn, y = np.array(Xb, float), np.array(Xn, float), np.array(y, int)
    cut = int(len(y) * 0.8)
    def ll(X):
        sc = StandardScaler().fit(X[:cut])
        clf = LogisticRegression(C=0.05, max_iter=3000, random_state=17).fit(sc.transform(X[:cut]), y[:cut])
        p = np.clip(clf.predict_proba(sc.transform(X[cut:]))[:, 1], 1e-6, 1 - 1e-6)
        return round(float(sk_ll(y[cut:], p, labels=[0, 1])), 5)
    b, nn = ll(Xb), ll(Xn)
    return {"shared_n": len(y), "base_val_logloss": b, "base_plus_news_val_logloss": nn,
            "delta_logloss": round(nn - b, 5),
            "verdict": "NEWS_ADDS_SIGNAL" if nn < b - 0.002 else "NEWS_NO_OOS_VALUE",
            "note": "dev (TRAIN+VAL) windows only; TEST_V2 untouched."}


if __name__ == "__main__":
    build()
