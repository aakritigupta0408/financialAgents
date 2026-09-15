"""Find the single best trader that beats T0 (past data, honest walk-forward).

Phase 1 showed always-skip ($0) already beats T0 (-$209): T0 loses by OVER-TRADING a
near-efficient market. This searches a small family of INTERPRETABLE, low-capacity
policies (1-2 params, thresholds fit on train only) for the one with the best OOS net
PnL, and reports honestly whether it merely beats T0 or is actually good (beats
always-skip = net positive) with a clean placebo.

Reuses the Phase 1 episode/payoff machinery. Isolated: reads only the live durable log;
T0 untouched; no research/TEST_V2 contact.
"""
import copy
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rl_treatment_phase1 as P1  # noqa: E402

OUTF = ROOT / "results" / "rl_ideal_trader.json"
N_FOLDS = P1.N_FOLDS
EMB = P1.EMBARGO
BANK = P1.BANKROLL0


def _bucket_ev(train, phi=0.05):
    """Train EV/trade (hold-to-close) per entry-price bucket."""
    ev = defaultdict(list)
    for ep in train:
        b = P1._bucket(P1.PRICE_BUCKETS, ep["price"])
        ev[b].append(P1._settle_return(ep, phi, 1.0, 1.0))
    return {b: sum(v) / len(v) for b, v in ev.items()}


# ---- policy families. each: (train_eps) -> decision(ep)->(phi,tp,sl) ----
def pol_always_skip(train):
    return lambda ep: (0.0, 1.0, 1.0)


def pol_t0_mimic(train):
    return lambda ep: (0.10, 1.0, 1.0)


def make_ev_gated(margin, phi=0.05, exit_=False):
    def build(train):
        ev = _bucket_ev(train, phi)
        tp, sl = (0.05, 0.10) if exit_ else (1.0, 1.0)
        def dec(ep):
            b = P1._bucket(P1.PRICE_BUCKETS, ep["price"])
            return (phi, tp, sl) if ev.get(b, -1) > margin else (0.0, 1.0, 1.0)
        return dec
    return build


def make_conf_gated(cmin, phi=0.05):
    def build(train):
        return lambda ep: ((phi, 1.0, 1.0) if ep["conf"] >= cmin else (0.0, 1.0, 1.0))
    return build


def make_ev_kelly(margin, kf=0.5, cap=0.15):
    def build(train):
        ev = _bucket_ev(train, 0.05)
        def dec(ep):
            b = P1._bucket(P1.PRICE_BUCKETS, ep["price"])
            e = ev.get(b, -1)
            if e <= margin:
                return (0.0, 1.0, 1.0)
            phi = min(cap, kf * max(0.0, e))    # fractional-Kelly-ish on train edge
            return (phi, 1.0, 1.0)
        return dec
    return build


POLICIES = {
    "always_skip": pol_always_skip,
    "T0_mimic": pol_t0_mimic,
    "ev_gated_m0.00": make_ev_gated(0.00),
    "ev_gated_m0.05": make_ev_gated(0.05),
    "ev_gated_m0.10": make_ev_gated(0.10),
    "ev_gated_m0.00_exit": make_ev_gated(0.00, exit_=True),
    "conf_gated_0.20": make_conf_gated(0.20),
    "conf_gated_0.40": make_conf_gated(0.40),
    "conf_gated_0.60": make_conf_gated(0.60),
    "ev_kelly_m0.00": make_ev_kelly(0.00),
    "ev_kelly_m0.05": make_ev_kelly(0.05),
}


def walk_forward(eps, build):
    n = len(eps); fold = n // (N_FOLDS + 1)
    rets, taken, hits = [], 0, 0
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        train = eps[:tr_end]; te = eps[tr_end + EMB: fold * (k + 1)]
        if not train or not te:
            continue
        dec = build(train)
        for ep in te:
            phi, tp, sl = dec(ep)
            rets.append(P1._settle_return(ep, phi, tp, sl))
            if phi > 0:
                taken += 1
                hits += 1 if ep["win"] else 0
    net = round(sum(r * BANK for r in rets), 2)
    m = sum(rets) / len(rets) if rets else 0
    s = st.pstdev(rets) if len(rets) > 1 else 0
    sharpe = round((m / (s + 1e-9)) * (len(rets) ** 0.5), 3) if s > 0 else 0.0
    return {"net_pnl": net, "n": len(rets), "taken": taken,
            "coverage": round(taken / max(1, len(rets)), 3),
            "hit_rate": round(hits / taken, 3) if taken else None, "sharpe": sharpe}


def placebo(eps, build, seeds=5):
    nets = []
    for sd in range(seeds):
        rng = random.Random(500 + sd)
        sh = copy.deepcopy(eps); fl = [e["win"] for e in sh]; rng.shuffle(fl)
        for e, w in zip(sh, fl):
            e["win"] = w
        nets.append(walk_forward(sh, build)["net_pnl"])
    return round(sum(nets) / len(nets), 2)


def run():
    eps = P1.build_episodes()
    res = {}
    for name, build in POLICIES.items():
        r = walk_forward(eps, build)
        r["placebo_net"] = placebo(eps, build)
        res[name] = r
    t0 = res["T0_mimic"]["net_pnl"]
    skip = res["always_skip"]["net_pnl"]
    # candidates that beat T0, ranked by OOS net; "ideal" also wants clean placebo + positivity
    beats_t0 = {k: v for k, v in res.items() if k not in ("T0_mimic",) and v["net_pnl"] > t0}
    best = max(res, key=lambda k: res[k]["net_pnl"])
    # "genuinely good" = beats always-skip (net>0) AND placebo doesn't rival its edge
    good = {k: v for k, v in res.items()
            if v["net_pnl"] > 0 and v["placebo_net"] < 0.5 * v["net_pnl"]}
    doc = {
        "schema_version": "rl-ideal-trader-1", "n_windows": len(eps),
        "T0_net": t0, "always_skip_net": skip,
        "policies": res,
        "best_by_oos_net": best,
        "count_beating_T0": len(beats_t0),
        "genuinely_positive_and_placebo_clean": list(good.keys()),
        "verdict": ("IDEAL_IS_SELECTIVITY" if best in ("always_skip",) or not good
                    else f"CANDIDATE:{best}"),
        "interpretation": (
            "T0 loses by over-trading a near-efficient market, so many policies beat it "
            "simply by trading less. The honest 'ideal trader that beats T0' is the most "
            "SELECTIVE one; whether any is truly PROFITABLE (beats always-skip with a clean "
            "placebo) is the real test. On this sample: " +
            ("no policy is both net-positive and placebo-clean -> the ideal trader is one "
             "that mostly STANDS DOWN (skip), i.e. recognizing there is no edge to trade."
             if not good else f"candidates survive: {list(good.keys())} (still n-limited).")),
        "data_caveat": f"n={len(eps)} windows — provisional; accrue before trusting magnitudes.",
    }
    OUTF.write_text(json.dumps(doc, indent=1))
    print(f"ideal_trader: n={len(eps)}  T0 net {t0}  always-skip {skip}")
    for k in sorted(res, key=lambda x: -res[x]["net_pnl"]):
        v = res[k]
        flag = "  <-BEST" if k == best else ""
        print(f"  {k:22} net ${v['net_pnl']:>8}  cov {v['coverage']:.2f}  hit {v['hit_rate']}"
              f"  sharpe {v['sharpe']:>6}  placebo ${v['placebo_net']:>8}{flag}")
    print(f"  verdict: {doc['verdict']}  | beat T0: {len(beats_t0)}  | good: {doc['genuinely_positive_and_placebo_clean']}")


if __name__ == "__main__":
    run()
