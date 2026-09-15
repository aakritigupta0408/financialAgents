"""STRIKE-LADDER reproduction — is the professor's 89%@90% the distance-to-strike task?

Our capture keeps only the at-the-money KXBTC15M contract per window (strike = open,
~50/50). The Kalshi ladder has MANY strikes per window; most are far in/out-of-the-money, so
"settlement >= strike K" is near-certain from moneyness alone — directionally empty but
high-accuracy. We reconstruct the ladder from what we DO have: each window's open (= the
at-the-money floor_strike) and its settlement (expiration_value), giving the empirical 15-min
settlement-return distribution over 6,340 windows. For a realistic ladder of strikes around
the open, the at-open predictor is sign(open - K): YES if strike below open. We report
hit-rate at coverage (confident = strike far from open in vol units) to show the ladder task
trivially reaches ~89%@90% with ZERO forecasting — reconciling it with our hard ~0.50 label.

Also reports the HARD at-the-money label (strike=open) for contrast.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "contract_outcomes.jsonl"
OUT = ROOT / "research" / "strike_ladder_test.json"


def returns():
    r = []
    for l in SRC.open():
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        o = d.get("floor_strike"); s = d.get("expiration_value")
        if o and s:
            r.append((s - o) / o)          # 15-min settlement return vs open
    return np.array(r, float)


def run():
    ret = returns()
    sigma = ret.std()
    rep = {"n_windows": len(ret), "sigma_15m_bps": round(float(sigma) * 1e4, 1),
           "at_the_money": {}, "ladder": {}}

    # HARD label: at-the-money strike = open. predictor sign(open-K)=sign(0) -> coin flip.
    # accuracy of "predict UP" = P(ret>=0); moneyness gives nothing at K=open.
    rep["at_the_money"]["base_up"] = round(float((ret >= 0).mean()), 4)
    rep["at_the_money"]["moneyness_predictor_acc"] = 0.5  # sign(0) is undefined -> 50/50

    # LADDER: strikes at open*(1+delta) for a grid of delta (in sigma units), like Kalshi's
    # ladder. Per (window, strike): outcome = ret >= delta. At-open predictor: YES iff delta<=0
    # (strike at/below open). Confidence = |delta|/sigma (far strikes = confident).
    grid_sigmas = np.concatenate([np.linspace(-4, -0.05, 40), np.linspace(0.05, 4, 40)])
    deltas = grid_sigmas * sigma
    preds, correct, conf = [], [], []
    for dz, d in zip(grid_sigmas, deltas):
        outcome = (ret >= d).astype(int)          # settles YES for this strike
        pred = int(d <= 0)                         # moneyness: YES if strike <= open
        acc_col = (pred == outcome).astype(float)
        correct.append(acc_col.mean())
        conf.append(abs(dz))
        preds.append((dz, round(float(acc_col.mean()), 4)))
    # build the full (window x strike) hit/confidence matrix for a coverage sweep
    allcorrect = []
    allconf = []
    for dz, d in zip(grid_sigmas, deltas):
        outcome = (ret >= d).astype(int)
        pred = int(d <= 0)
        allcorrect.append((pred == outcome).astype(float))
        allconf.append(np.full(len(ret), abs(dz)))
    C = np.concatenate(allcorrect); Z = np.concatenate(allconf)
    rep["ladder"]["overall_hit_all_strikes"] = round(float(C.mean()), 4)
    sweep = []
    for cov_target in (1.0, 0.95, 0.90, 0.80, 0.70, 0.50):
        # keep the most-confident fraction cov_target
        thr = np.quantile(Z, 1 - cov_target)
        m = Z >= thr
        sweep.append({"coverage": round(float(m.mean()), 3),
                      "conf_thr_sigma": round(float(thr), 3),
                      "hit": round(float(C[m].mean()), 4)})
    rep["ladder"]["hit_at_coverage"] = sweep
    # the headline: hit at ~90% coverage
    at90 = min(sweep, key=lambda s: abs(s["coverage"] - 0.90))
    rep["ladder"]["headline_at_~90pct_coverage"] = at90
    rep["ladder"]["per_strike_acc_sample"] = preds[::8]
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"n={rep['n_windows']} 15-min sigma={rep['sigma_15m_bps']}bps  base-up(ATM)={rep['at_the_money']['base_up']}")
    print("LADDER hit @ coverage (distance-to-strike predictor, ZERO forecasting):")
    for s in sweep:
        print(f"  coverage {s['coverage']:.2f}  (|z|>={s['conf_thr_sigma']})  ->  hit {s['hit']}")
    print(f"\n  HEADLINE ~90% coverage: hit {at90['hit']} @ coverage {at90['coverage']}")
    print("  => the ladder / distance-to-strike task reaches ~this with NO directional skill.")
    print("  ATM label (strike=open, our hard problem): moneyness gives 0.50 — needs real forecasting.")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
