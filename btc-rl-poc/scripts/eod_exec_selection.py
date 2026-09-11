"""EOD ALPHA ASSAULT — Attack #1: execution-selection trader.

Hypothesis: of the champion's already-selected bets, can a simple
model using ONLY pre-entry state predict which to actually take, so
the surviving subset has POSITIVE realized net EV — recovering the
~6.6c/$1 execution leak by abstaining from bad fills?

This is an EXECUTION / SELECTION edge (Attack #1), NOT information
alpha (§4 taxonomy). Per PM 09-10 the honest verdicts are:
  realized EV > 0  ->  PROFITABLE SELECTION/EXECUTION CANDIDATE
  realized EV <= 0 ->  NO CANDIDATE (evidence preserved)

INTEGRITY (code-enforced, so honesty is structural):
  * chronological split frozen by timestamp BEFORE any fitting;
  * features are PRE-ENTRY only (known at decision time) — outcomes
    (pnl/win/actual) are labels, never features (0 look-ahead);
  * the selection gate is chosen on VALIDATION, then the HOLDOUT is
    scored EXACTLY ONCE — no retuning after it opens;
  * fragility guards: coverage floor, top-3 concentration, drawdown.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEED = 20260910

# ---- FROZEN EOD GATE (pre-registered before fitting) --------------
COVERAGE_MIN = 0.20            # must trade >=20% of holdout windows
TOP3_MAX_SHARE = 0.50         # top-3 wins < 50% of total gain
GATE_GRID = [round(x, 2) for x in np.arange(0.50, 0.86, 0.02)]


def load():
    rows = [json.loads(l) for l in
            (RES / "kb_bets.jsonl").open() if l.strip()]
    rows = [r for r in rows if r.get("actual") is not None
            and r.get("price_c") and r.get("pnl_c") is not None]
    rows.sort(key=lambda r: r.get("made_ts", 0))
    return rows


def features(r):
    """PRE-ENTRY state only — everything here is knowable at made_ts."""
    p = float(r.get("p_model") or 0.5)
    price = float(r["price_c"])
    side_yes = 1.0 if r.get("side") == "yes" else 0.0
    return [
        price,                              # what we pay (expensive=bad)
        p,                                  # model probability
        abs(p - 0.5),                       # confidence
        float(r.get("mins_left") or 0),     # time to expiry
        float(r.get("edge_c") or 0),        # our estimated edge
        side_yes,
        price / 100.0 - p,                  # price vs belief (overpay?)
    ]


def ev_per_dollar(r):
    """Realized EV per $1 of cost = pnl_c / price_c."""
    return float(r["pnl_c"]) / float(r["price_c"])


def econ(sub):
    """Economics of a set of taken bets."""
    if not sub:
        return {"n": 0, "mean_ev": 0.0, "total_pnl": 0.0}
    evs = [ev_per_dollar(r) for r in sub]
    pnl = [float(r["pnl_c"]) for r in sub]
    # equity curve -> max drawdown (on cumulative realized pnl)
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) if len(cum) else 0.0
    gains = sorted((x for x in pnl if x > 0), reverse=True)
    top3 = sum(gains[:3]); tot = sum(gains) or 1.0
    return {"n": len(sub),
            "mean_ev_per_$1_c": round(100 * float(np.mean(evs)), 2),
            "total_pnl_c": round(float(sum(pnl)), 0),
            "win_rate": round(float(np.mean([r["pnl_c"] > 0
                                             for r in sub])), 3),
            "max_drawdown_c": round(dd, 0),
            "top3_share_of_gains": round(top3 / tot, 3)}


def main():
    rows = load()
    n = len(rows)
    i_tr, i_va = int(n * 0.60), int(n * 0.80)
    train, val, hold = rows[:i_tr], rows[i_tr:i_va], rows[i_va:]
    split_ts = {"train": [train[0]["made_ts"], train[-1]["made_ts"]],
                "val": [val[0]["made_ts"], val[-1]["made_ts"]],
                "holdout": [hold[0]["made_ts"], hold[-1]["made_ts"]]}

    Xtr = np.array([features(r) for r in train])
    ytr = np.array([1 if r["pnl_c"] > 0 else 0 for r in train])
    Xva = np.array([features(r) for r in val])
    Xho = np.array([features(r) for r in hold])

    # two candidates: regularized logistic + LightGBM
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    lr = LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)
    lr.fit((Xtr - mu) / sd, ytr)
    gb = lgb.LGBMClassifier(n_estimators=120, num_leaves=15,
                            learning_rate=0.05, min_child_samples=40,
                            reg_lambda=1.0, random_state=SEED,
                            verbose=-1)
    gb.fit(Xtr, ytr)

    def probs(model, X, std=False):
        Z = (X - mu) / sd if std else X
        return model.predict_proba(Z)[:, 1]

    champion_val = econ(val)      # trade-all baseline on val
    # pick (model, gate) that maximizes val mean-EV subject to coverage
    best = None
    for name, m, std in (("logistic", lr, True), ("lgbm", gb, False)):
        pv = probs(m, Xva, std)
        for g in GATE_GRID:
            take = [r for r, p in zip(val, pv) if p >= g]
            if len(take) < COVERAGE_MIN * len(val):
                continue
            e = econ(take)
            key = e["mean_ev_per_$1_c"]
            if best is None or key > best["val_mean_ev"]:
                best = {"model": name, "m": m, "std": std, "gate": g,
                        "val_mean_ev": key, "val_cov":
                        round(len(take) / len(val), 3)}
    if best is None:
        result = {"verdict": "NO_CANDIDATE",
                  "reason": "no (model,gate) met coverage on validation"}
    else:
        # ---- ONE-SHOT HOLDOUT (never touched until now) ----------
        ph = probs(best["m"], Xho, best["std"])
        take = [r for r, p in zip(hold, ph) if p >= best["gate"]]
        sel = econ(take)
        champ = econ(hold)               # champion trades-all holdout
        cov = round(len(take) / len(hold), 3)
        passes = (sel["n"] > 0
                  and sel["mean_ev_per_$1_c"] > 0
                  and sel["mean_ev_per_$1_c"] > champ["mean_ev_per_$1_c"]
                  and cov >= COVERAGE_MIN
                  and sel["top3_share_of_gains"] < TOP3_MAX_SHARE
                  and sel["max_drawdown_c"] <= champ["max_drawdown_c"])
        verdict = ("PROFITABLE_SELECTION_EXECUTION_CANDIDATE"
                   if passes else "NO_CANDIDATE")
        result = {"model": best["model"], "gate": best["gate"],
                  "holdout_coverage": cov,
                  "selective": sel, "champion_trade_all": champ,
                  "gate_checks": {
                      "realized_ev_positive": sel["mean_ev_per_$1_c"] > 0,
                      "beats_champion": sel["mean_ev_per_$1_c"]
                      > champ["mean_ev_per_$1_c"],
                      "coverage_ge_20pct": cov >= COVERAGE_MIN,
                      "top3_below_50pct":
                      sel["top3_share_of_gains"] < TOP3_MAX_SHARE,
                      "drawdown_le_champion":
                      sel["max_drawdown_c"] <= champ["max_drawdown_c"]},
                  "verdict": verdict}

    doc = {"generated_ts": int(time.time()), "attack": "1-execution-selection",
           "alpha_class": "EXECUTION_UTILIZATION (not information alpha)",
           "n_bets": n, "split_ts": split_ts,
           "seed": SEED, "one_shot_holdout": True,
           "frozen_gate": {"coverage_min": COVERAGE_MIN,
                           "top3_max_share": TOP3_MAX_SHARE},
           **result}
    (RES / "eod_exec_selection.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("verdict", "model", "gate",
          "holdout_coverage", "selective", "champion_trade_all")
          if k in doc}, indent=1))


if __name__ == "__main__":
    main()
