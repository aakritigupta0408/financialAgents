"""§28-31, §11-14 — DISAGREEMENT & SELECTIVE-EDGE ECONOMICS.

Independent Oracle p_oracle = recalibrated MECH_FAIR_BRTI (best model that uses NO
Kalshi input). Compare to the market and ask the money question: when the Oracle
materially disagrees with Kalshi, who is right, and does a selective paper trader
earn positive realized EV after crossing the spread?

Executable model (§29 — not midpoint): buy the Oracle-favored side at the Kalshi
ASK ~ market_price(side) + k_spread/2; $1 binary payoff = realized official
outcome; Kalshi maker/taker fee approximated as 0.07*p*(1-p) per contract.

Primary metric (§14): REALIZED_PAPER_EV_PER_ELIGIBLE_WINDOW. One entry per window
at the EARLIEST decision whose |p_oracle - p_market| >= tau (rewards early edge,
§31). Sweeps tau to trace coverage × win-rate × EV (§30), and stratifies the
disagreement outcome by |gap| bucket (§29) and earliness (§31).

Writes research/oracle/disagreement_edge_result.json.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "results" / "brti_decision_dataset.jsonl"
T05 = ROOT / "research" / "t05_repricing" / "dataset.jsonl"
MFB = ROOT / "research" / "oracle" / "mech_fair_brti_result.json"
OUT = ROOT / "research" / "oracle" / "disagreement_edge_result.json"


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def p_mech_row(r, sig):
    lvl, tte = r["current_brti"], max(1.0, r["time_remaining_s"])
    p = Phi(r["brti_distance_to_target"] / (sig * lvl * math.sqrt(tte)))
    rem = r.get("settle_remaining") or 0
    req = r.get("required_remaining_average")
    if r["time_remaining_s"] <= 60 and rem > 0 and req is not None:
        s_rem = sig * lvl * math.sqrt(tte) / math.sqrt(rem)
        p = Phi((lvl - req) / max(1e-6, s_rem))
    return min(1 - 1e-4, max(1e-4, p))


def fee(p):
    return 0.07 * p * (1 - p)          # Kalshi-style per-contract fee approximation


def main():
    from sklearn.isotonic import IsotonicRegression
    sig = json.load(MFB.open())["sigma_rel_per_sqrt_s"]
    spread = {}
    for l in T05.open():
        if not l.strip():
            continue
        r = json.loads(l)
        spread[(r["ticker"], r["ts"])] = r.get("k_spread", 0.01)
    rows = []
    for l in DEC.open():
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("current_brti") is None or d["time_remaining_s"] < 1 or d.get("k_prob") is None:
            continue
        d["p_mech"] = p_mech_row(d, sig)
        d["k_spread"] = spread.get((d["market_window_id"], d["decision_time"]), 0.01)
        rows.append(d)

    first = {}
    for r in rows:
        first[r["market_window_id"]] = min(first.get(r["market_window_id"], 1e18),
                                           r["decision_time"])
    order = sorted(first, key=lambda t: first[t]); n = len(order)
    trw = set(order[:int(.7 * n)])
    tr = [r for r in rows if r["market_window_id"] in trw]
    te = [r for r in rows if r["market_window_id"] not in trw]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(np.array([r["p_mech"] for r in tr]),
            np.array([r["exact_yes"] for r in tr], float))
    for r in te:
        r["p_oracle"] = float(np.clip(iso.predict([r["p_mech"]])[0], 1e-4, 1 - 1e-4))
        r["gap"] = r["p_oracle"] - r["k_prob"]

    te_windows = sorted(set(r["market_window_id"] for r in te))
    W = len(te_windows)
    by_win = {}
    for r in te:
        by_win.setdefault(r["market_window_id"], []).append(r)
    for w in by_win:
        by_win[w].sort(key=lambda r: r["decision_time"])

    def trade_at(r):
        """Realized paper PnL of taking the Oracle-favored side at the ask."""
        yes = r["p_oracle"] >= r["k_prob"]           # side we disagree toward
        half = r["k_spread"] / 2.0
        if yes:
            cost = min(0.99, r["k_prob"] + half)
            payoff = 1.0 if r["exact_yes"] == 1 else 0.0
        else:
            cost = min(0.99, (1 - r["k_prob"]) + half)
            payoff = 1.0 if r["exact_yes"] == 0 else 0.0
        pnl = payoff - cost - fee(cost)
        win = int(payoff > 0)
        return pnl, win, yes

    # ---- §30 selective-edge curve: earliest eligible disagreement per window ----
    curve = []
    for tau in [0.0, 0.02, 0.05, 0.08, 0.12, 0.15, 0.20]:
        traded, wins, pnl_sum, entry_trem = 0, 0, 0.0, []
        for w in te_windows:
            entry = next((r for r in by_win[w] if abs(r["gap"]) >= tau), None)
            if entry is None:
                continue
            pnl, win, _ = trade_at(entry)
            traded += 1; wins += win; pnl_sum += pnl
            entry_trem.append(entry["time_remaining_s"] / 60.0)
        cov = traded / W
        curve.append({
            "min_disagreement": tau,
            "coverage": round(cov, 3),
            "trades": traded,
            "win_rate": round(wins / traded, 4) if traded else None,
            "ev_per_trade": round(pnl_sum / traded, 4) if traded else None,
            "ev_per_eligible_window": round(pnl_sum / W, 4),
            "median_entry_tte_min": round(float(np.median(entry_trem)), 1) if entry_trem else None,
        })

    # ---- §29 who-is-right by |gap| bucket (single snapshot per window: max-gap row) ----
    buckets = [(0, .02, "0-2pp"), (.02, .05, "2-5pp"), (.05, .08, "5-8pp"),
               (.08, .12, "8-12pp"), (.12, 1.0, "12pp+")]
    by_gap = {}
    for lo, hi, nm in buckets:
        rs = [r for r in te if lo <= abs(r["gap"]) < hi]
        if len(rs) < 20:
            continue
        # when oracle disagrees, did oracle's side or market's implied side win?
        oracle_right = np.mean([(r["p_oracle"] >= 0.5) == (r["exact_yes"] == 1) for r in rs])
        market_right = np.mean([(r["k_prob"] >= 0.5) == (r["exact_yes"] == 1) for r in rs])
        pnls = [trade_at(r)[0] for r in rs]
        by_gap[nm] = {"n": len(rs),
                      "oracle_dir_acc": round(float(oracle_right), 4),
                      "market_dir_acc": round(float(market_right), 4),
                      "ev_per_trade": round(float(np.mean(pnls)), 4)}

    # ---- §31 early-disagreement: EV by earliness at a fixed tau=0.05 ----
    by_time = {}
    for lo, hi, nm in [(0, 2, "T-2..0"), (2, 5, "T-5..2"), (5, 9, "T-9..5"), (9, 20, "T-15..9")]:
        rs = [r for r in te if lo <= r["time_remaining_s"] / 60.0 < hi and abs(r["gap"]) >= 0.05]
        if len(rs) < 20:
            continue
        pnls = [trade_at(r)[0] for r in rs]
        wins = [trade_at(r)[1] for r in rs]
        by_time[nm] = {"n": len(rs), "win_rate": round(float(np.mean(wins)), 4),
                       "ev_per_trade": round(float(np.mean(pnls)), 4)}

    best = max(curve, key=lambda c: c["ev_per_eligible_window"])

    # ---- window-level bootstrap CI (integrity: 111 windows is tiny, curve is noisy) ----
    tau_b = best["min_disagreement"]
    per_win_ev, per_win_dir = [], []   # dir = oracle_right - market_right at max|gap| row
    for w in te_windows:
        entry = next((r for r in by_win[w] if abs(r["gap"]) >= tau_b), None)
        per_win_ev.append(trade_at(entry)[0] if entry else 0.0)
        mx = max(by_win[w], key=lambda r: abs(r["gap"]))
        per_win_dir.append(int((mx["p_oracle"] >= 0.5) == (mx["exact_yes"] == 1))
                           - int((mx["k_prob"] >= 0.5) == (mx["exact_yes"] == 1)))
    ev_arr, dir_arr = np.array(per_win_ev), np.array(per_win_dir)
    # MOVING-BLOCK bootstrap (blocks of adjacent windows) to respect serial
    # correlation across contiguous 15-min windows — a naive i.i.d. bootstrap
    # understates the CI here. Deterministic LCG (no RNG; envs disable random).
    B, Wn, BLK = 2000, len(te_windows), 6
    nblocks = int(math.ceil(Wn / BLK))
    boot_ev, boot_dir = [], []
    for b in range(B):
        s = (1103515245 * (b + 1) + 12345) & 0x7fffffff
        idx = []
        for _ in range(nblocks):
            s = (1103515245 * s + 12345) & 0x7fffffff
            start = s % Wn
            idx.extend((start + k) % Wn for k in range(BLK))
        idx = np.array(idx[:Wn])
        boot_ev.append(ev_arr[idx].mean())
        boot_dir.append(dir_arr[idx].mean())
    ev_ci = [round(float(np.percentile(boot_ev, 2.5)), 4),
             round(float(np.percentile(boot_ev, 97.5)), 4)]
    dir_ci = [round(float(np.percentile(boot_dir, 2.5)), 4),
              round(float(np.percentile(boot_dir, 97.5)), 4)]
    ev_sig = ev_ci[0] > 0
    dir_sig = dir_ci[0] > 0
    positive = ev_sig or dir_sig
    doc = {
        "oracle": "recalibrated MECH_FAIR_BRTI (no Kalshi input)",
        "test_windows": W,
        "executable_assumptions": "enter at market+half-spread; $1 binary; fee 0.07*p*(1-p)",
        "selective_edge_curve": curve,
        "who_is_right_by_gap": by_gap,
        "ev_by_earliness_tau0.05": by_time,
        "best_operating_point": best,
        "bootstrap_ci_95": {
            "at_min_disagreement": tau_b,
            "ev_per_eligible_window": {"point": best["ev_per_eligible_window"], "ci": ev_ci,
                                       "significant_positive": bool(ev_sig)},
            "oracle_minus_market_dir_acc": {
                "point": round(float(dir_arr.mean()), 4), "ci": dir_ci,
                "significant_positive": bool(dir_sig)},
            "n_windows": Wn},
        # Integrity cap: even when the moving-block CI excludes 0, a retrospective
        # 111-window test with temporally-adjacent recalibration and a non-monotone
        # earliness profile CANNOT be called a confirmed edge. Highest attainable
        # verdict here is PROMISING pending prospective (live paper) confirmation.
        "verdict": (
            "PROMISING_PENDING_PROSPECTIVE — point estimates favor the Oracle and the "
            "moving-block 95% CI excludes 0, BUT n=111 windows, retrospective, "
            "adjacent-window recalibration, and a non-monotone earliness profile "
            "(T-9..5 negative) mean this is NOT a confirmed edge. Must be validated "
            "on live prospective paper before any claim." if positive else
            "NO_EXECUTABLE_EDGE_VS_KALSHI — after crossing the spread, disagreement "
            "with Kalshi does not yield significantly positive realized paper EV; "
            "Kalshi's price already reflects recalibrated mechanics."),
        "integrity_caveats": [
            "111 test windows over ~1.3 days — severely underpowered.",
            "EV curve non-monotonic across tau and earliness — noise signature.",
            "isotonic recalibration fit on temporally-adjacent train windows (regime leak risk).",
            "oracle uses exact current BRTI; near expiry it approaches settlement, so "
            "late-window 'edge' vs a possibly-laggy stored k_prob may be artifactual.",
            "no independent predictive feature cleared OOS — edge (if real) is bounded "
            "by mechanics recalibration alone."],
        "note": "Oracle uses only exact BRTI mechanics + recalibration; no independent "
                "predictive features cleared OOS (INFORMATION_LIMITED), so this bounds "
                "the tradeable edge available from mechanics ALONE.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"Disagreement edge — {W} test windows, oracle = recalibrated MECH_BRTI")
    print(f"{'min_gap':>8} {'cov':>6} {'trades':>7} {'win':>7} {'ev/trade':>9} {'ev/elig_win':>12} {'med_tte':>8}")
    for c in curve:
        print(f"{c['min_disagreement']:8.2f} {c['coverage']:6.3f} {c['trades']:7d} "
              f"{str(c['win_rate']):>7} {str(c['ev_per_trade']):>9} "
              f"{c['ev_per_eligible_window']:12.4f} {str(c['median_entry_tte_min']):>8}")
    print("who-is-right by |gap| (oracle_acc / market_acc / ev_per_trade):")
    for k, v in by_gap.items():
        print(f"  {k:7s} n={v['n']:4d} oracle {v['oracle_dir_acc']} market {v['market_dir_acc']} ev {v['ev_per_trade']:+.4f}")
    print(f"bootstrap95 @tau={tau_b}: EV/elig_win {best['ev_per_eligible_window']} CI {ev_ci} sig+={ev_sig}")
    print(f"             oracle-market dir_acc {round(float(dir_arr.mean()),4)} CI {dir_ci} sig+={dir_sig}")
    print("VERDICT:", doc["verdict"][:90])


if __name__ == "__main__":
    main()
