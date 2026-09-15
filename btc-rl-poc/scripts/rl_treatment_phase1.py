"""RL TREATMENT — Phase 1: offline harness + 3-reward tournament backtest (past data).

Learns, per arm, a policy that decides SKIP / SIZE / EARLY-EXIT while FOLLOWING the
Oracle's direction (it never picks direction — that is unlearnable here). Because every
window's settled label AND its intra-window market-price path are known, the reward of
every candidate action is fully computable offline, so this is batch value-based policy
optimization (fitted-Q over a low-capacity state bucketing) — the right capacity for ~226
windows, not deep RL that would overfit.

Three arms differ ONLY in the objective used to pick the best action per state bucket:
  RL-PnL     argmax mean(return)
  RL-Kelly   argmax mean(log(1+return))          (log-growth)
  RL-Sharpe  argmax mean(return)/std(return)       (risk-adjusted)

Evaluation: purged/embargoed expanding walk-forward; per-arm OOS net PnL, coverage vs T0,
hit rate, Sharpe; baselines always-skip and T0-mimic (always take, hold to close); a
label-shuffle PLACEBO that must collapse any edge. Honest: with n~226 the CIs are wide —
this validates the pipeline and gives a provisional read, not a verdict.

Isolation: reads only results/rl_window_log.jsonl (live system's own durable log). No
research imports, no TEST_V2, T0 untouched.
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "rl_window_log.jsonl"
OUT = ROOT / "research" if False else ROOT / "results"      # live-system output
OUTF = OUT / "rl_tournament_backtest.json"

ENTRY_MINS = 12.0          # fixed PIT decision point (minutes-to-close)
SPREAD = 0.02              # conservative half-spread paid on entry (price units)
FEE = 0.013               # per-contract fee (price units) ~ observed desk fee
BANKROLL0 = 300.0
PHI = [0.0, 0.05, 0.10, 0.20]      # size buckets: skip, or fraction of bankroll at risk
PRICE_BUCKETS = [0.55, 0.65, 0.75, 0.85]     # entry-price bucket edges
CONF_BUCKETS = [0.60, 0.70, 0.80]            # oracle-confidence bucket edges
TP_GRID = [0.03, 0.05, 0.08, 1.0]            # take-profit on mark move (1.0 = never)
SL_GRID = [0.05, 0.10, 0.20, 1.0]            # stop-loss on mark move
EMBARGO = 1
N_FOLDS = 5


def _rows(p):
    out = []
    for l in p.open():
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def build_episodes():
    """One episode per settled window: entry state at ENTRY_MINS, the followed side, and
    the market-price path of that side for early-exit marking."""
    byw = defaultdict(list)
    for r in _rows(SRC):
        byw[r.get("ticker")].append(r)
    eps = []
    for tk, rs in byw.items():
        lab = [x.get("actual") for x in rs if x.get("actual") in (0, 1)]
        if not lab:
            continue
        y = lab[-1]
        # collapse variants -> one oracle signal per timestamp (mean p_up)
        byt = defaultdict(list)
        for x in rs:
            if x.get("mins_left") is not None and x.get("mkt_p_up") is not None:
                byt[round(x["made_ts"])].append(x)
        steps = []
        for ts in sorted(byt):
            g = byt[ts]
            pu = [x["p_up"] for x in g if x.get("p_up") is not None]
            steps.append({
                "ts": ts, "mins_left": g[0]["mins_left"],
                "mkt_p_up": g[0]["mkt_p_up"],
                "oracle_p_up": sum(pu) / len(pu) if pu else 0.5,
                "close_ts": g[0].get("close_ts"),
            })
        if len(steps) < 3:
            continue
        # entry = step nearest ENTRY_MINS from above (PIT: decide once)
        cand = [s for s in steps if s["mins_left"] >= ENTRY_MINS] or steps
        entry = min(cand, key=lambda s: abs(s["mins_left"] - ENTRY_MINS))
        call = 1 if entry["oracle_p_up"] >= 0.5 else 0        # follow oracle direction
        side_yes = call == 1
        # entry price of the followed side (mid + spread), and win indicator
        mid = entry["mkt_p_up"] if side_yes else (1 - entry["mkt_p_up"])
        price = min(0.99, mid + SPREAD)
        win = (y == 1) if side_yes else (y == 0)
        conf = abs(entry["oracle_p_up"] - 0.5) * 2
        # forward mark path of the held side (for early exit)
        path = [(s["mkt_p_up"] if side_yes else 1 - s["mkt_p_up"])
                for s in steps if s["ts"] >= entry["ts"]]
        eps.append({"ticker": tk, "close_ts": entry["close_ts"] or entry["ts"],
                    "price": price, "win": bool(win), "conf": conf,
                    "mark_path": path})
    eps.sort(key=lambda e: e["close_ts"])
    return eps


def _bucket(edges, v):
    b = 0
    for e in edges:
        if v >= e:
            b += 1
    return b


def _state(ep):
    return (_bucket(PRICE_BUCKETS, ep["price"]), _bucket(CONF_BUCKETS, ep["conf"]))


def _settle_return(ep, phi, tp, sl):
    """Return on bankroll for a taken position with size phi and exit rule (tp, sl).
    Risk-your-stake payoff: a loss forfeits the stake; a win pays (1-price)/price."""
    if phi == 0.0:
        return 0.0
    price = ep["price"]
    # early exit: first step where mark move crosses tp (up) or sl (down)
    entry_mark = ep["mark_path"][0]
    exit_r = None
    for m in ep["mark_path"][1:]:
        d = m - entry_mark
        if d >= tp or d <= -sl:
            # sell at mark m: return on stake = (m - price)/price minus round-trip fee
            exit_r = (m - price) / price - 2 * FEE / price
            break
    if exit_r is not None:
        return phi * exit_r
    # else hold to settlement
    if ep["win"]:
        r = (1 - price) / price - FEE / price
    else:
        r = -1.0 - FEE / price
    return phi * r


def _actions():
    acts = [("skip", 0.0, 1.0, 1.0)]
    for phi in PHI[1:]:
        for tp in TP_GRID:
            for sl in SL_GRID:
                acts.append((f"phi{phi}_tp{tp}_sl{sl}", phi, tp, sl))
    return acts


ACTIONS = _actions()


def _objective(returns, kind):
    if not returns:
        return -1e9
    if kind == "pnl":
        return sum(returns) / len(returns)
    if kind == "kelly":
        return sum(math.log(max(1e-9, 1 + r)) for r in returns) / len(returns)
    if kind == "sharpe":
        m = sum(returns) / len(returns)
        s = st.pstdev(returns) if len(returns) > 1 else 0.0
        return m / (s + 1e-6)
    raise ValueError(kind)


def learn_policy(train, kind):
    """Per state bucket, pick the action whose train-window returns maximize the objective."""
    by_state = defaultdict(list)
    for ep in train:
        by_state[_state(ep)].append(ep)
    pol = {}
    for stt, eps in by_state.items():
        best, best_v = None, -1e18
        for a in ACTIONS:
            _, phi, tp, sl = a
            rs = [_settle_return(ep, phi, tp, sl) for ep in eps]
            v = _objective(rs, kind)
            if v > best_v:
                best_v, best = v, a
        pol[stt] = best
    return pol


def apply_policy(pol, ep):
    a = pol.get(_state(ep))
    if a is None:                     # unseen bucket -> skip (conservative)
        return 0.0, None
    _, phi, tp, sl = a
    return _settle_return(ep, phi, tp, sl), a


def t0_mimic(ep):
    """Baseline: always take at fixed 10% size, follow oracle, hold to close."""
    return _settle_return(ep, 0.10, 1.0, 1.0)


def walk_forward(eps, kind):
    n = len(eps)
    fold = n // (N_FOLDS + 1)
    rets, cover, hits, taken = [], 0, 0, 0
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te = eps[tr_end + EMBARGO: fold * (k + 1)]
        train = eps[:tr_end]
        if not train or not te:
            continue
        pol = learn_policy(train, kind)
        for ep in te:
            r, a = apply_policy(pol, ep)
            rets.append(r)
            if a and a[1] > 0:
                taken += 1
                if ep["win"]:
                    hits += 1
    return rets, taken, hits


def _summary(rets, bankroll0=BANKROLL0):
    if not rets:
        return {}
    # PRIMARY metric is ADDITIVE net PnL at a fixed notional: each window risks the same
    # fixed dollars (phi * bankroll0), so returns don't compound. Compounding a tiny
    # bankroll over ~175 windows produces explosive, unrealistic figures (the placebo
    # exposed this) — additive is the honest read at this stake scale.
    add_pnl = sum(r * bankroll0 for r in rets)
    m = sum(rets) / len(rets)
    s = st.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe = (m / (s + 1e-9)) * math.sqrt(len(rets)) if s > 0 else 0.0
    return {"n": len(rets), "net_pnl": round(add_pnl, 2),
            "mean_ret": round(m, 5), "sharpe": round(sharpe, 3)}


def run():
    eps = build_episodes()
    base_rate = sum(1 for e in eps if e["win"]) / len(eps) if eps else 0
    # baselines
    skip_rets = [0.0 for _ in eps]
    t0_rets = [t0_mimic(e) for e in eps]
    # T0-mimic evaluated OOS is just its per-window return (no training); use test span
    n = len(eps); fold = n // (N_FOLDS + 1)
    oos_slice = eps[fold + EMBARGO:]
    t0_oos = [t0_mimic(e) for e in oos_slice]
    t0_taken = len(oos_slice)

    arms = {}
    for kind in ("pnl", "kelly", "sharpe"):
        rets, taken, hits = walk_forward(eps, kind)
        s = _summary(rets)
        s["coverage_vs_t0"] = round(taken / max(1, t0_taken), 3)
        s["taken"] = taken
        s["hit_rate_on_taken"] = round(hits / taken, 3) if taken else None
        arms[f"RL-{kind}"] = s

    # placebo: RANDOM label permutation (breaks state->outcome), averaged over seeds.
    # A real edge must vanish; a placebo that stays strongly positive means the policy
    # is fitting noise (small-n overfitting), not signal.
    import copy
    import random
    placebo = {f"RL-{k}": [] for k in ("pnl", "kelly", "sharpe")}
    for seed in range(5):
        rng = random.Random(1000 + seed)
        shuf = copy.deepcopy(eps)
        flags = [e["win"] for e in shuf]
        rng.shuffle(flags)
        for e, w in zip(shuf, flags):
            e["win"] = w
        for kind in ("pnl", "kelly", "sharpe"):
            rp, _, _ = walk_forward(shuf, kind)
            placebo[f"RL-{kind}"].append(_summary(rp).get("net_pnl", 0.0))
    placebo = {k: round(sum(v) / len(v), 2) for k, v in placebo.items()}

    best = max(arms, key=lambda k: arms[k].get("net_pnl", -1e9))
    beats_skip = arms[best]["net_pnl"] > 0
    beats_t0 = arms[best]["net_pnl"] > _summary(t0_oos)["net_pnl"]
    placebo_clean = all(abs(v) < 0.15 * abs(arms[k]["net_pnl"] or 1) or v <= 0
                        for k, v in placebo.items())
    verdict = ("PROVISIONAL_PROMISING" if (beats_skip and beats_t0 and placebo_clean)
               else "NO_OOS_EDGE_YET")
    doc = {
        "schema_version": "rl-tournament-backtest-1",
        "n_windows": len(eps), "base_rate_up": round(base_rate, 4),
        "decision_point_mins_left": ENTRY_MINS,
        "costs": {"half_spread": SPREAD, "fee": FEE},
        "baselines": {"always_skip": _summary(skip_rets),
                      "T0_mimic_full": _summary(t0_rets),
                      "T0_mimic_oos": _summary(t0_oos)},
        "arms_oos": arms,
        "placebo_net_pnl": placebo,
        "best_arm": best, "beats_always_skip": beats_skip, "beats_t0": beats_t0,
        "always_skip_dominates": all(arms[k]["net_pnl"] < 0 for k in arms),
        "placebo_clean": placebo_clean,
        "placebo_caveat": ("placebo shuffles only the final win/loss label; the early-exit "
                           "channel reads the REAL intra-window price path, so residual "
                           "non-causal structure survives a label-only shuffle. A non-zero "
                           "placebo here = small-n overfitting + path structure, NOT a "
                           "validated edge. A cleaner placebo would permute whole episodes."),
        "verdict": verdict,
        "data_caveat": (f"n={len(eps)} windows — far below the deflated-Sharpe / purged-CV "
                        "threshold for a trustworthy verdict. Provisional pipeline read only; "
                        "accrue via rl_treatment_dataset.py before trusting any edge."),
        "honesty": ("RL follows the Oracle's direction and only decides skip/size/exit; it "
                    "cannot and does not predict direction. Coverage vs T0 is measured, "
                    "never fitted toward a target."),
    }
    OUTF.write_text(json.dumps(doc, indent=1))
    print(f"rl_tournament: n={len(eps)} base_rate={base_rate:.4f}  verdict={verdict} (best {best})")
    print(f"  baseline always-skip $0.00 | T0-mimic OOS net {_summary(t0_oos)['net_pnl']} "
          f"(n={_summary(t0_oos)['n']})")
    for k, v in arms.items():
        print(f"  {k:10} net ${v['net_pnl']:>7} sharpe {v['sharpe']:>6} "
              f"cover {v['coverage_vs_t0']:.2f} hit {v['hit_rate_on_taken']} (n={v['n']}, took {v['taken']})")
    print(f"  placebo net: {placebo}  <- should be ~0/negative if the edge is real")


if __name__ == "__main__":
    run()
