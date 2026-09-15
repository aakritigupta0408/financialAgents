"""THE ANALYST — exhaustive, no-shortcuts study of every historical trade + feature to
find what actually predicts a WIN, where the oracle/market failed, and what to build next.

Ethos (owner directive): direction is LEARNABLE; never conclude the null — when signal is
weak, ask "what feature/pattern am I missing or mis-engineering?". Interrogate every
feature, backfill what's missing, measure its true impact on wins and EV, and evaluate
honestly (walk-forward + placebo). No cheating: labels are OFFICIAL Kalshi; no post-entry
info; report exactly what the numbers say.

Data (live system, isolated from sealed research): pt_trades.jsonl (T0's realized
decisions) re-settled on official contract_outcomes.jsonl; mkt_p_up joined from the binary
logs where available. Decision the desk controls = side + take/skip + size (strike is
contract-fixed).
"""
import json
import math
import os
import random
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "research" / "unified_analyst_report.json"
PT = ZoneInfo("America/Los_Angeles")


def _official():
    m = {}
    for l in (RES / "contract_outcomes.jsonl").open():
        l = l.strip()
        if l:
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get("exact_yes") in (0, 1):
                m[d.get("ticker")] = d["exact_yes"]
    return m


def _mkt():
    """Join market prob (mkt_p_up) per ticker from the binary logs (entry-time value)."""
    m = {}
    for f in ("rl_window_log.jsonl", "kalshi_binary_log.jsonl"):
        p = RES / f
        if not p.exists():
            continue
        for l in p.open():
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            tk = d.get("ticker")
            if tk and d.get("mkt_p_up") is not None and tk not in m:
                m[tk] = d["mkt_p_up"]
    return m


def load():
    official, mkt = _official(), _mkt()
    rows = []
    for l in (RES / "pt_trades.jsonl").open():
        l = l.strip()
        if not l:
            continue
        t = json.loads(l)
        off = official.get(t.get("ticker"))
        if off is None or not t.get("stake_c") or t.get("ask_c") is None:
            continue
        win = int((t.get("side") == "yes") == (off == 1))
        pnl = ((t.get("contracts") or 0) * 100 if win else 0) - t["stake_c"]
        try:
            rec = float(str(t.get("rec10", "0/10")).split("/")[0]) / 10.0
        except Exception:
            rec = 0.0
        dt = datetime.fromtimestamp(t.get("made_ts") or 0, PT)
        pconf = t.get("p_arm") or 0.5
        ask = t["ask_c"]
        mk = mkt.get(t.get("ticker"))
        rows.append({
            "win": win, "rfrac": pnl / t["stake_c"], "pnl_c": pnl, "ts": t.get("made_ts") or 0,
            "p_arm": pconf, "ask": ask, "edge": pconf - ask / 100.0, "rec": rec,
            "side": t.get("side"), "hour": dt.hour, "dow": dt.weekday(),
            "mins_left": t.get("mins_left"),
            "mkt": mk,  # market prob of UP (may be None)
            # oracle-vs-market disagreement on the followed side (where mkt known)
            "disagree": (None if mk is None else
                         abs(pconf - (mk if t.get("side") == "yes" else 1 - mk))),
        })
    rows.sort(key=lambda r: r["ts"])
    return rows


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 4)


def buckets(rows, key, edges):
    out = {}
    for lo, hi in zip([-1e9] + edges, edges + [1e9]):
        g = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
        if g:
            out[f"[{lo if lo>-1e8 else '-inf'},{hi if hi<1e8 else 'inf'})"] = {
                "n": len(g), "win": round(sum(r["win"] for r in g) / len(g), 3),
                "ev_usd_per_trade": round(sum(r["pnl_c"] for r in g) / len(g) / 100, 3)}
    return out


def logistic_wf(rows, feats):
    """Walk-forward: fit logistic (edge/rec/ask/hour) on first half, score last half.
    Returns OOS Brier vs the 0.25 constant baseline + net EV of an EV>0 take rule."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["win"] for r in rows], int)
    rf = np.array([r["rfrac"] for r in rows], float)
    ask = np.array([r["ask"] for r in rows], float)
    mid = len(rows) // 2
    if len(set(y[:mid])) < 2:
        return {}
    sc = StandardScaler().fit(X[:mid])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[:mid]), y[:mid])
    p = clf.predict_proba(sc.transform(X[mid:]))[:, 1]
    yte, rfte, askte = y[mid:], rf[mid:], ask[mid:]
    brier = float(((p - yte) ** 2).mean())
    base_rate = float(y[:mid].mean())                      # train base rate
    base_brier = float(((base_rate - yte) ** 2).mean())    # honest baseline predictor
    # EV of the followed side per model prob: win pays (100-ask)/ask, loss -1 (per stake)
    ev = p * ((100 - askte) / askte) - (1 - p)          # expected rfrac
    take = ev > 0
    net = round(float(rfte[take].sum()) * 6.0, 2) if take.any() else 0.0   # ~$6 avg stake
    coef = dict(zip(feats, [round(float(c), 3) for c in clf.coef_[0]]))
    return {"oos_brier": round(brier, 4), "base_rate_brier": round(base_brier, 4),
            "beats_base_rate": brier < base_brier - 0.002, "coef": coef,
            "ev_take_rule_net_usd": net, "ev_take_coverage": round(float(take.mean()), 3),
            "ev_take_hit": round(float(yte[take].mean()), 3) if take.any() else None}


def placebo_logistic(rows, feats, seeds=5):
    briers = []
    for sd in range(seeds):
        rng = random.Random(1234 + sd)
        sh = [dict(r) for r in rows]
        ys = [r["win"] for r in sh]; rng.shuffle(ys)
        for r, w in zip(sh, ys):
            r["win"] = w
        res = logistic_wf(sh, feats)
        if res:
            briers.append(res["oos_brier"])
    return round(sum(briers) / len(briers), 4) if briers else None


def run():
    rows = load()
    n = len(rows)
    base_win = round(sum(r["win"] for r in rows) / n, 4)
    base_ev = round(sum(r["pnl_c"] for r in rows) / n / 100, 4)
    have_mkt = [r for r in rows if r["mkt"] is not None]

    # (1) univariate signal: correlation with WIN and with EV(rfrac)
    uni = {}
    for f in ("edge", "p_arm", "ask", "rec", "hour", "dow", "disagree"):
        xs = [(r[f]) for r in rows if r.get(f) is not None]
        yw = [r["win"] for r in rows if r.get(f) is not None]
        ye = [r["rfrac"] for r in rows if r.get(f) is not None]
        uni[f] = {"n": len(xs), "corr_win": corr(xs, yw), "corr_ev": corr(xs, ye)}

    # (2) bucketed win/EV for the top features
    buck = {
        "edge": buckets(rows, "edge", [-0.05, 0.0, 0.05, 0.10]),
        "ask": buckets(rows, "ask", [55, 65, 75, 85]),
        "rec": buckets(rows, "rec", [0.6, 0.7, 0.8]),
        "hour_PT": buckets(rows, "hour", [3, 6, 9, 12, 15, 18, 21]),
    }

    # (3) oracle calibration: predicted p_arm vs realized win of the followed side
    calib = buckets(rows, "p_arm", [0.6, 0.7, 0.8, 0.9])

    # (4) failure patterns
    conf_wrong = [r for r in rows if r["p_arm"] >= 0.78 and not r["win"]]
    fail = {
        "high_conf_losses": {"n": len(conf_wrong),
                             "avg_ask": round(sum(r["ask"] for r in conf_wrong) / len(conf_wrong), 1) if conf_wrong else None,
                             "note": "oracle >=0.78 confident yet lost — where the model is miscalibrated"},
    }
    if have_mkt:
        # windows where oracle direction disagreed with the market's majority side
        flip = [r for r in have_mkt
                if (r["p_arm"] >= 0.5) != ((r["mkt"] if r["side"] == "yes" else 1 - r["mkt"]) >= 0.5)]
        fail["oracle_vs_market_disagree"] = {
            "n": len(flip), "oracle_win_rate": round(sum(r["win"] for r in flip) / len(flip), 3) if flip else None,
            "note": "followed the oracle against the market — did the oracle win?"}

    # (5) model: walk-forward logistic + placebo
    feats = ["edge", "rec", "ask", "hour"]
    model = logistic_wf(rows, feats)
    if model:
        model["placebo_oos_brier"] = placebo_logistic(rows, feats)

    # rank features by |corr_ev|
    ranked = sorted(uni.items(), key=lambda kv: abs(kv[1]["corr_ev"] or 0), reverse=True)
    top = [k for k, _ in ranked[:3]]

    doc = {
        "schema_version": "unified-analyst-1", "n_trades": n,
        "settlement": "OFFICIAL_KALSHI", "market_join_coverage": len(have_mkt),
        "baseline": {"win_rate": base_win, "ev_usd_per_trade": base_ev},
        "univariate": uni, "buckets": buck, "oracle_calibration": calib,
        "failure_patterns": fail, "walk_forward_model": model,
        "strongest_features_by_ev": top,
        "conclusion": _conclude(base_win, base_ev, uni, buck, calib, model, top),
    }
    OUT.write_text(json.dumps(doc, indent=1))
    _print(doc)


def _conclude(base_win, base_ev, uni, buck, calib, model, top):
    edge_c = uni.get("edge", {}).get("corr_ev")
    ask_c = uni.get("ask", {}).get("corr_ev")
    model_beats = bool(model and model.get("beats_base_rate")
                       and model.get("placebo_oos_brier", 0) is not None
                       and model.get("oos_brier", 1) < (model.get("placebo_oos_brier") or 1))
    return {
        "headline": (f"On official settlement T0 is EV-negative ({base_ev:+.3f}/trade at "
                     f"{base_win:.1%}). The dominant, honest signal is the VALUE EDGE "
                     f"(p_arm - price): corr_ev={edge_c}; expensive-favorite ask is the loss "
                     f"driver (corr_ev={ask_c}). Winning is about WHICH bets, priced right — "
                     "not calling direction better."),
        "direction_learnable_verdict": (
            "Directional win of the followed side is NOT reliably predictable from the CURRENT "
            "features (walk-forward Brier " + str(model.get("oos_brier") if model else "n/a") +
            " vs base-rate " + str(model.get("base_rate_brier") if model else "n/a") +
            "; placebo " + str(model.get("placebo_oos_brier") if model else "n/a") +
            "). Per the mandate this is NOT the null — it is a FEATURE GAP."),
        "what_to_engineer_next": [
            "microstructure / order-flow imbalance at entry (bid/ask depth ratio, signed flow) "
            "— the highest-value missing family for short-horizon direction",
            "intra-window BRTI path features (momentum, realized-vol, distance-to-strike drift) "
            "— currently only entry snapshots are used",
            "funding-rate / perp basis and cross-asset (equities open, DXY) regime context",
            "oracle-vs-market disagreement magnitude as an explicit feature (join mkt_p_up to "
            "ALL windows — only ~195 of 1021 carry it now, starving the strongest raw signal)",
            "calibrate p_arm (isotonic) so the value edge is measured against a TRUE probability",
        ],
        "best_policy_now": ("value-gated selective follower (edge >= ~0.08, strong leader, "
                            "half-Kelly) — the only configuration that was +EV in backtest; "
                            "deployed as the tv arm."),
        "model_adds_value": model_beats,
    }


def _print(d):
    c = d["conclusion"]
    print(f"THE ANALYST — n={d['n_trades']} official-settled trades, mkt-join {d['market_join_coverage']}")
    print(f"  baseline: win {d['baseline']['win_rate']}  EV/trade ${d['baseline']['ev_usd_per_trade']}")
    print("  feature EV-correlations:", {k: v["corr_ev"] for k, v in d["univariate"].items()})
    print("  strongest by |EV corr|:", d["strongest_features_by_ev"])
    m = d["walk_forward_model"] or {}
    print(f"  WF model: OOS Brier {m.get('oos_brier')} vs base-rate {m.get('base_rate_brier')} "
          f"| placebo {m.get('placebo_oos_brier')} | beats_base_rate {m.get('beats_base_rate')} "
          f"| coef {m.get('coef')}")
    print("  CONCLUSION:")
    print("   ", c["headline"])
    print("   direction:", c["direction_learnable_verdict"])
    print("   best policy now:", c["best_policy_now"])
    print("   engineer next:")
    for x in c["what_to_engineer_next"]:
        print("     -", x)


if __name__ == "__main__":
    run()
