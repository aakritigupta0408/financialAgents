"""Directive check 1 — FORMAL T1-vs-T0 PAIRED HOLDOUT EFFECT.

On the untouched 93-window holdout, per market_window_id:
    delta_i = T1_realized_pnl_i - T0_realized_pnl_i
Every ELIGIBLE (in-envelope) window stays in the estimand; when T1 abstains its
P&L is 0 (abstention IS the treatment). Determines whether the ~+7.2c/eligible T1
advantage is diffuse or concentrated, with an abstention mechanism decomposition
and identity residual. NO retuning; reuses the frozen replay + policies.

Writes research/replay/t1_effect_result.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import replay_backtest as RB  # noqa: E402
from btc_rl import economics as E  # noqa: E402

OUT = ROOT / "research" / "replay" / "t1_effect_result.json"


def main():
    rows, leak = RB.load()
    dev, val, hold = RB.split(rows)
    devval = dev | val
    sigma, iso = RB.fit_oracle([r for r in rows if r["market_window_id"] in devval])
    RB.attach_oracle(rows, sigma, iso)
    wr_hold = RB.by_window(rows, hold)

    t0 = {x["wid"]: x for x in RB.run_trader(wr_hold, "T0")}
    t1 = {x["wid"]: x for x in RB.run_trader(wr_hold, "T1", RB.T.T1_EDGE_TAU)}

    elig = [w for w in t0 if t0[w]["eligible"]]        # in-envelope windows = estimand base
    def pnl(x):
        return x["traded"]["pnl_c"] if x["traded"] else 0.0
    t0_p = [pnl(t0[w]) for w in elig]
    t1_p = [pnl(t1[w]) for w in elig]
    deltas = [t1_p[i] - t0_p[i] for i in range(len(elig))]

    n = len(elig)
    total_delta = sum(deltas)
    paired = E.paired_delta(t0_p, t1_p)               # moving-block bootstrap CI
    frac_pos = sum(1 for d in deltas if d > 0) / n if n else None
    frac_zero = sum(1 for d in deltas if abs(d) < 1e-9) / n if n else None

    # concentration of the POSITIVE advantage: how much of total_delta comes from the
    # few biggest contributors
    order = sorted(range(n), key=lambda i: deltas[i], reverse=True)
    def topk(k):
        return round(sum(deltas[order[j]] for j in range(min(k, n))) / total_delta, 3) \
            if abs(total_delta) > 1e-9 else None
    top5pct = max(1, n // 20)
    outlier_conc = round(sum(deltas[order[j]] for j in range(top5pct)) / total_delta, 3) \
        if abs(total_delta) > 1e-9 else None

    # abstention decomposition (T1 abstains where T0 trades)
    loss_avoided_n = loss_avoided_val = 0
    profit_missed_n = profit_missed_val = 0.0
    both_trade_delta = 0.0
    for i, w in enumerate(elig):
        t0_traded = t0[w]["traded"] is not None
        t1_traded = t1[w]["traded"] is not None
        if t0_traded and not t1_traded:
            p0 = pnl(t0[w])
            if p0 < 0:
                loss_avoided_n += 1; loss_avoided_val += -p0     # delta = -p0 > 0
            elif p0 > 0:
                profit_missed_n += 1; profit_missed_val += p0    # delta = -p0 < 0
        elif t0_traded and t1_traded:
            both_trade_delta += deltas[i]
    abstention_net = loss_avoided_val - profit_missed_val
    identity_residual = round(total_delta - (both_trade_delta + abstention_net), 4)

    doc = {
        "holdout_windows": len(hold), "eligible_windows": n,
        "paired_completeness": 1.0,
        "t0_ev_per_eligible_c": round(sum(t0_p) / n, 4) if n else None,
        "t1_ev_per_eligible_c": round(sum(t1_p) / n, 4) if n else None,
        "paired_delta_ev_per_eligible_c": round(total_delta / n, 4) if n else None,
        "median_paired_delta_c": round(sorted(deltas)[n // 2], 4) if n else None,
        "moving_block_bootstrap_95ci_c": paired["ci95"] if paired else None,
        "significant": paired["significant"] if paired else None,
        "fraction_delta_gt_0": round(frac_pos, 4) if frac_pos is not None else None,
        "fraction_delta_zero": round(frac_zero, 4) if frac_zero is not None else None,
        "concentration": {"top1_contribution": topk(1), "top3_contribution": topk(3),
                          "top5_contribution": topk(5),
                          "top5pct_outlier_contribution": outlier_conc,
                          "reading": "share of the TOTAL paired advantage from the k "
                                     "largest per-window deltas"},
        "abstention_decomposition": {
            "t0_losing_trades_avoided_by_t1": {"n": loss_avoided_n,
                                               "value_c": round(loss_avoided_val, 2)},
            "profitable_t0_trades_missed_by_t1": {"n": profit_missed_n,
                                                  "value_c": round(profit_missed_val, 2)},
            "net_abstention_value_c": round(abstention_net, 2),
            "both_trade_entry_delta_c": round(both_trade_delta, 2)},
        "identity": {"total_delta_c": round(total_delta, 2),
                     "abstention_net_c": round(abstention_net, 2),
                     "both_trade_delta_c": round(both_trade_delta, 2),
                     "residual_c": identity_residual},
        "diffuse_or_concentrated": None,
        "verdict": None,
        "mechanism_finding": (
            "The positive paired advantage does NOT come from the registered mechanism "
            "(selective abstention), which is net-NEGATIVE here (-297c: missed more "
            "profitable T0 trades than losses avoided). It comes from an INCIDENTAL "
            "second effect: T1 enters the first DISAGREEMENT-eligible row, which differs "
            "from T0's first-eligible row, shifting entry timing/price (+1007c, "
            "concentrated in ~5 windows). So frozen T1 differs from T0 in TWO ways "
            "(abstention AND entry-row selection) — an experiment-integrity note for "
            "PROSPECTIVE-T0-vs-T1 (§10 treatment purity). NOT retuned per directive; "
            "flagged for the owner."),
        "note": "estimand base = in-envelope eligible windows; T1 abstention -> P&L 0, "
                "window retained. Frozen policies; no retuning. OFFLINE holdout — "
                "prospective evidence still required to establish the effect.",
    }
    tc = doc["concentration"]["top3_contribution"]
    doc["diffuse_or_concentrated"] = ("CONCENTRATED" if (tc is not None and tc >= 0.5)
                                      else "DIFFUSE" if tc is not None else "UNKNOWN")
    doc["verdict"] = (
        "T1_ADVANTAGE_ROBUST_OFFLINE" if (paired and paired["ci95"][0] > 0
                                          and doc["diffuse_or_concentrated"] == "DIFFUSE")
        else "T1_ADVANTAGE_CONCENTRATED_NOT_ESTABLISHED" if doc["diffuse_or_concentrated"] == "CONCENTRATED"
        else "T1_ADVANTAGE_NOT_STATISTICALLY_ESTABLISHED")
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"T1-vs-T0 holdout: {n} eligible windows")
    print(f"  T0 EV/elig {doc['t0_ev_per_eligible_c']}c  T1 EV/elig {doc['t1_ev_per_eligible_c']}c  "
          f"paired Δ {doc['paired_delta_ev_per_eligible_c']}c (median {doc['median_paired_delta_c']}c)")
    print(f"  moving-block 95% CI {doc['moving_block_bootstrap_95ci_c']}  sig={doc['significant']}")
    print(f"  frac Δ>0 {doc['fraction_delta_gt_0']}  frac Δ=0 {doc['fraction_delta_zero']}")
    print(f"  concentration top1/3/5 {tc and doc['concentration']['top1_contribution']}/"
          f"{tc}/{doc['concentration']['top5_contribution']} -> {doc['diffuse_or_concentrated']}")
    ad = doc["abstention_decomposition"]
    print(f"  abstention: avoided {ad['t0_losing_trades_avoided_by_t1']['n']} losses "
          f"(+{ad['t0_losing_trades_avoided_by_t1']['value_c']}c), missed "
          f"{ad['profitable_t0_trades_missed_by_t1']['n']} profits "
          f"(-{ad['profitable_t0_trades_missed_by_t1']['value_c']}c), net "
          f"{ad['net_abstention_value_c']}c; identity residual {identity_residual}c")
    print(f"  VERDICT: {doc['verdict']}")


if __name__ == "__main__":
    main()
