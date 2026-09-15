"""T0 improvement study — PLAN/IDEATE/BACKTEST ONLY. Changes nothing live.

Uses T0's OWN realized trades (results/pt_trades.jsonl) as ground truth of what the
Follower did, re-settled on the OFFICIAL Kalshi outcome (the live ledger used the buggy
Coinbase-candle proxy), then evaluates counterfactual improvement levers on those exact
trades. Because we know each trade's features (p_arm, ask, rec10, side, time) and its
official outcome, every lever's P&L is computed exactly — no new alpha assumed.

Levers (none invent direction; all are gate/size/risk on the Follower):
  G  tighter confidence gate      keep p_arm >= tau
  P  entry-price discipline       skip expensive favorites (ask > cap) / skip mid band
  L  leader-strength gate         keep rec10 >= k/10
  S  sizing                       flat 5/10/20% + fractional-Kelly (risk, not net)

Evaluation: baseline vs lever, IN-SAMPLE and OUT-OF-SAMPLE (chronological walk-forward:
fit the threshold on the first half, apply to the last half). Placebo: shuffle outcomes;
a real filter's edge should collapse. Honest: single desk, ~1000 windows, multiple levers
tried -> treat as directional evidence, not proof.

Writes research/t0_improvement_backtest.json.
"""
import json
import math
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / "results" / "pt_trades.jsonl"
CO = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "research" / "t0_improvement_backtest.json"


def _load():
    official = {}
    for l in CO.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("exact_yes") in (0, 1):
            official[d["ticker"]] = d["exact_yes"]
    trades = []
    for l in PT.open():
        l = l.strip()
        if not l:
            continue
        t = json.loads(l)
        off = official.get(t.get("ticker"))
        if off is None or t.get("stake_c") in (None, 0) or t.get("ask_c") is None:
            continue
        win = int((t.get("side") == "yes") == (off == 1))
        payout = (t.get("contracts") or 0) * 100 if win else 0
        pnl = payout - t["stake_c"]
        try:
            rec = float(str(t.get("rec10", "0/10")).split("/")[0]) / 10.0
        except Exception:
            rec = 0.0
        trades.append({
            "ts": t.get("close_ts") or t.get("made_ts") or 0,
            "p_arm": t.get("p_arm"), "ask": t.get("ask_c"), "rec": rec,
            "side": t.get("side"), "stake_c": t["stake_c"],
            "win": win, "pnl_c": pnl, "rfrac": pnl / t["stake_c"],
        })
    trades.sort(key=lambda x: x["ts"])
    return trades


def _summ(kept):
    if not kept:
        return {"n": 0, "net_c": 0, "hit": None, "sharpe": 0.0}
    net = sum(t["pnl_c"] for t in kept)
    wins = sum(1 for t in kept if t["win"])
    rf = [t["rfrac"] for t in kept]
    m = sum(rf) / len(rf); s = st.pstdev(rf) if len(rf) > 1 else 0
    return {"n": len(kept), "net_c": net, "net_usd": round(net / 100, 2),
            "hit": round(wins / len(kept), 3),
            "sharpe": round((m / (s + 1e-9)) * math.sqrt(len(rf)), 2) if s > 0 else 0.0}


# ---- levers: each returns a predicate keep(t) given a param ----
LEVERS = {
    "G_conf>=": ("keep only leader confidence >= tau", [0.62, 0.66, 0.70, 0.74, 0.78],
                 lambda tau: lambda t: (t["p_arm"] or 0) >= tau),
    "P_ask<=": ("skip expensive favorites: keep ask <= cap", [90, 80, 75, 70, 65],
                lambda cap: lambda t: (t["ask"] or 99) <= cap),
    "P_skipmid": ("skip mid-price band [lo,70)", [55, 60, 65],
                  lambda lo: lambda t: not (lo <= (t["ask"] or 0) < 70)),
    "L_rec>=": ("keep only strong leader: rec10 >= k", [0.6, 0.7, 0.8, 0.9],
                lambda k: lambda t: t["rec"] >= k),
    "E_edge>=": ("value gate: keep only p_arm - ask/100 >= margin (model beats price)",
                 [0.0, 0.03, 0.05, 0.08, 0.12],
                 lambda mrg: lambda t: ((t["p_arm"] or 0) - (t["ask"] or 100) / 100.0) >= mrg),
}


def _wf(trades, param_grid, mk):
    """Chronological walk-forward: pick the param with best net on the first half,
    apply to the second half; report OOS net vs baseline OOS net."""
    mid = len(trades) // 2
    tr, te = trades[:mid], trades[mid:]
    best_p, best_net = None, -1e18
    for p in param_grid:
        keep = mk(p)
        net = sum(t["pnl_c"] for t in tr if keep(t))
        if net > best_net:
            best_net, best_p = net, p
    keep = mk(best_p)
    te_kept = [t for t in te if keep(t)]
    return {"chosen_param": best_p, "oos": _summ(te_kept),
            "oos_baseline": _summ(te), "oos_delta_usd": round(
                (_summ(te_kept)["net_c"] - _summ(te)["net_c"]) / 100, 2)}


def _placebo(trades, param_grid, mk, seeds=5):
    outs = []
    for sd in range(seeds):
        rng = random.Random(900 + sd)
        sh = [dict(t) for t in trades]
        wins = [t["win"] for t in sh]; rng.shuffle(wins)
        for t, w in zip(sh, wins):
            t["win"] = w
            # rebuild rfrac/pnl under shuffled outcome at same stake/ask
            r = (100 - t["ask"]) / t["ask"] if w else -1.0
            t["rfrac"] = r; t["pnl_c"] = int(r * t["stake_c"])
        outs.append(_wf(sh, param_grid, mk)["oos_delta_usd"])
    return round(sum(outs) / len(outs), 2)


def _sizing(trades):
    """Compounded bankroll + max drawdown at flat 5/10/20% and fractional-Kelly, on the
    official return sequence. Sizing changes RISK, not hit rate."""
    def run(phi_fn):
        bank = 300.0; peak = 300.0; dd = 0.0
        for t in trades:
            phi = phi_fn(t)
            bank *= (1 + phi * t["rfrac"])
            peak = max(peak, bank); dd = max(dd, (peak - bank) / peak)
        return {"end_bankroll": round(bank, 2), "max_dd_pct": round(100 * dd, 1)}
    out = {f"flat_{int(p*100)}%": run(lambda t, p=p: p) for p in (0.05, 0.10, 0.20)}
    # fractional-Kelly: edge = p_win_implied? use realized conf as p, ask/100 as price
    def kelly(t):
        p = t["p_arm"] or 0.5; price = (t["ask"] or 50) / 100.0
        b = (1 - price) / price
        f = (p * b - (1 - p)) / b if b > 0 else 0
        return max(0.0, min(0.10, 0.5 * f))     # half-Kelly, capped 10%
    out["half_kelly_cap10%"] = run(kelly)
    return out


def run():
    trades = _load()
    base = _summ(trades)
    levers = {}
    for name, (desc, grid, mk) in LEVERS.items():
        insample = {str(p): _summ([t for t in trades if mk(p)(t)]) for p in grid}
        wf = _wf(trades, grid, mk)
        wf["placebo_oos_delta_usd"] = _placebo(trades, grid, mk)
        levers[name] = {"desc": desc, "in_sample": insample, "walk_forward": wf}
    sizing = _sizing(trades)
    # rank levers by OOS delta that also survives placebo (placebo ~ 0/negative)
    ranked = sorted(levers.items(),
                    key=lambda kv: kv[1]["walk_forward"]["oos_delta_usd"], reverse=True)
    doc = {
        "schema_version": "t0-improvement-backtest-1",
        "note": "PLAN/BACKTEST ONLY — nothing changed live. T0's own trades re-settled on "
                "OFFICIAL Kalshi outcomes.",
        "n_trades": base["n"],
        "baseline_T0_official": base,
        "levers": levers,
        "sizing_on_baseline": sizing,
        "ranking_by_oos_delta": [
            {"lever": k, "chosen_param": v["walk_forward"]["chosen_param"],
             "oos_delta_usd": v["walk_forward"]["oos_delta_usd"],
             "placebo_oos_delta_usd": v["walk_forward"]["placebo_oos_delta_usd"],
             "survives_placebo": v["walk_forward"]["oos_delta_usd"]
                                  > 2 * abs(v["walk_forward"]["placebo_oos_delta_usd"])}
            for k, v in ranked],
        "caveats": "single paper desk, ~1000 windows, 4 levers x grids tried (multiple "
                   "testing); OOS = one chronological split. Directional evidence, not proof. "
                   "T0 has no directional alpha (base rate ~0.5) so gains come only from "
                   "declining -EV windows and controlling risk/size.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"BASELINE T0 (official, n={base['n']}): net ${base['net_usd']} hit {base['hit']} sharpe {base['sharpe']}")
    print("LEVERS (walk-forward OOS delta vs baseline, + placebo):")
    for k, v in ranked:
        wf = v["walk_forward"]
        print(f"  {k:12} param {str(wf['chosen_param']):5} OOS Δ ${wf['oos_delta_usd']:>7} "
              f"| placebo Δ ${wf['placebo_oos_delta_usd']:>7} | {v['desc']}")
    print("SIZING on baseline (end bankroll from $300 / max DD):")
    for k, v in sizing.items():
        print(f"  {k:18} end ${v['end_bankroll']:>8}  maxDD {v['max_dd_pct']}%")


if __name__ == "__main__":
    run()
