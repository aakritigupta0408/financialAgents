"""Evaluate the Confidence-Gated Follower across stake sizes (5% -> 33%).

Position size at the risk-your-stake payoff is a VARIANCE multiplier: additive PnL scales
linearly with stake but ruin/drawdown scale far worse. This reports the metrics that
actually matter at large stake — COMPOUNDED bankroll path from $300, max drawdown, minimum
bankroll, and a ruin flag — so a 33% stake can be judged on risk, not just headline PnL.

Isolated: reads the live durable log via the Phase-1 machinery; T0 untouched.
"""
import copy
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rl_treatment_phase1 as P1  # noqa: E402

OUTF = ROOT / "results" / "rl_stake_eval.json"
CONF_MIN = 0.20
STAKES = [0.05, 0.10, 0.20, 0.33]
BANK0 = P1.BANKROLL0
RUIN_FLOOR = 0.20 * BANK0        # bankroll below 20% of start = practical ruin
N_FOLDS = P1.N_FOLDS
EMB = P1.EMBARGO


def conf_gate(phi):
    return lambda ep: ((phi, 1.0, 1.0) if ep["conf"] >= CONF_MIN else (0.0, 1.0, 1.0))


def wf_returns(eps, dec):
    """Ordered OOS per-window returns across the expanding walk-forward test folds."""
    n = len(eps); fold = n // (N_FOLDS + 1)
    seq = []
    for k in range(1, N_FOLDS + 1):
        te = eps[fold * k + EMB: fold * (k + 1)]
        for ep in te:
            phi, tp, sl = dec(ep)
            seq.append((P1._settle_return(ep, phi, tp, sl), phi > 0, ep["win"]))
    return seq


def risk_metrics(seq):
    rets = [r for r, _, _ in seq]
    taken = [(r, w) for r, t, w in seq if t]
    # compounded path
    bank = BANK0; peak = BANK0; maxdd = 0.0; minb = BANK0; ruined = False
    for r, _, _ in seq:
        bank *= (1 + r)
        peak = max(peak, bank)
        maxdd = max(maxdd, (peak - bank) / peak)
        minb = min(minb, bank)
        if bank <= RUIN_FLOOR:
            ruined = True
    add = sum(r * BANK0 for r in rets)
    m = sum(rets) / len(rets) if rets else 0
    s = st.pstdev(rets) if len(rets) > 1 else 0
    sharpe = round((m / (s + 1e-9)) * (len(rets) ** 0.5), 3) if s > 0 else 0.0
    worst = min((r for r, _, _ in seq), default=0.0)
    return {
        "additive_net": round(add, 2),
        "compounded_end_bankroll": round(bank, 2),
        "min_bankroll": round(minb, 2),
        "max_drawdown_pct": round(100 * maxdd, 1),
        "worst_window_ret_pct": round(100 * worst, 1),
        "ruined_below_20pct": ruined,
        "n": len(rets), "taken": len(taken),
        "hit_rate": round(sum(1 for _, w in taken if w) / len(taken), 3) if taken else None,
        "sharpe": sharpe,
    }


def placebo(eps, dec, seeds=5):
    out = []
    for sd in range(seeds):
        rng = random.Random(700 + sd); sh = copy.deepcopy(eps)
        fl = [e["win"] for e in sh]; rng.shuffle(fl)
        for e, w in zip(sh, fl):
            e["win"] = w
        out.append(round(sum(r * BANK0 for r, _, _ in wf_returns(sh, dec)), 2))
    return round(sum(out) / len(out), 2)


def run():
    eps = P1.build_episodes()
    by_stake = {}
    for phi in STAKES:
        dec = conf_gate(phi)
        m = risk_metrics(wf_returns(eps, dec))
        m["placebo_additive"] = placebo(eps, dec)
        by_stake[f"{int(phi*100)}%"] = m
    doc = {"schema_version": "rl-stake-eval-1", "policy": f"conf_gate>={CONF_MIN}",
           "n_windows": len(eps), "bankroll0": BANK0, "ruin_floor": RUIN_FLOOR,
           "by_stake": by_stake,
           "note": ("Additive net scales ~linearly with stake; the honest cost of stake is in "
                    "compounded_end_bankroll, max_drawdown, min_bankroll and ruin. n is small."),
           "data_caveat": f"n={len(eps)} windows — provisional; magnitudes not yet trustworthy."}
    OUTF.write_text(json.dumps(doc, indent=1))
    print(f"stake_eval: conf_gate>={CONF_MIN}  n={len(eps)}  bankroll0=${BANK0:.0f}")
    hdr = f"  {'stake':6} {'add_net':>9} {'end_bank':>9} {'min_bank':>9} {'maxDD%':>7} {'worst%':>7} {'ruin':>5} {'sharpe':>7} {'placebo':>9}"
    print(hdr)
    for s, m in by_stake.items():
        print(f"  {s:6} {m['additive_net']:>9} {m['compounded_end_bankroll']:>9} "
              f"{m['min_bankroll']:>9} {m['max_drawdown_pct']:>7} {m['worst_window_ret_pct']:>7} "
              f"{str(m['ruined_below_20pct']):>5} {m['sharpe']:>7} {m['placebo_additive']:>9}")


if __name__ == "__main__":
    run()
