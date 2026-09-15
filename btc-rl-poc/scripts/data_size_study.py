"""How much training data do we need, and how often should we retrain?

Learning curve (OOS accuracy vs training-set size) + retraining-cadence study
(expanding-window vs rolling-last-K vs a frozen model, over time — to expose concept
drift) on the engineered PIT features. Uses the ~1.3k open-regime windows (the largest
clean set). Fast: features only, no foundation-model recompute. Walk-forward throughout.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from insample_reconstruction import load_windows          # noqa: E402
from combined_student import rich_features                # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
import json

OUT = ROOT / "research" / "data_size_retrain_study.json"


def _acc(clf, X, y):
    return float((clf.predict(X) == y).mean())


def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                          l2_regularization=2.0, min_samples_leaf=20, random_state=17)


def run():
    W = load_windows()
    W.sort(key=lambda w: w["ts"])
    y = np.array([w["y"] for w in W], int)
    F = np.nan_to_num(np.array([[rich_features(w)[k] for k in rich_features(W[0])] for w in W], float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    n = len(W)
    rep = {"n": n, "base_up": round(float(y.mean()), 4)}

    # ---- (1) LEARNING CURVE: train on the first `sz`, test on a FIXED held-out tail ----
    test_lo = int(n * 0.75)
    Xte, yte = F[test_lo:], y[test_lo:]
    curve = []
    for sz in [100, 150, 200, 300, 400, 600, 800, test_lo]:
        if sz >= test_lo:
            sz = test_lo
        clf = gbm().fit(F[:sz], y[:sz])
        curve.append({"train_n": sz, "oos_acc": round(_acc(clf, Xte, yte), 4)})
    rep["learning_curve"] = curve

    # ---- (2) RETRAIN CADENCE: walk forward in blocks; compare expanding vs rolling-K vs frozen ----
    block = max(30, n // 20)          # ~5% blocks
    start = int(n * 0.4)              # begin after a warmup
    roll_K = 400
    frozen = gbm().fit(F[:start], y[:start])   # trained once at `start`, never updated
    exp_acc, roll_acc, frz_acc, marks = [], [], [], []
    i = start
    while i + block <= n:
        Xb, yb = F[i:i + block], y[i:i + block]
        # expanding: all data up to i
        ce = gbm().fit(F[:i], y[:i])
        # rolling: last roll_K windows
        lo = max(0, i - roll_K)
        cr = gbm().fit(F[lo:i], y[lo:i])
        exp_acc.append(_acc(ce, Xb, yb)); roll_acc.append(_acc(cr, Xb, yb)); frz_acc.append(_acc(frozen, Xb, yb))
        marks.append(i)
        i += block
    rep["retrain"] = {
        "block": block, "roll_K": roll_K, "n_blocks": len(marks),
        "expanding_mean_oos": round(float(np.mean(exp_acc)), 4),
        "rolling_mean_oos": round(float(np.mean(roll_acc)), 4),
        "frozen_mean_oos": round(float(np.mean(frz_acc)), 4),
        "frozen_drift": round(float(frz_acc[0] - frz_acc[-1]), 4) if len(frz_acc) > 1 else None,
        "per_block": [{"at": int(m), "expanding": round(e, 3), "rolling": round(r, 3), "frozen": round(f, 3)}
                      for m, e, r, f in zip(marks, exp_acc, roll_acc, frz_acc)],
    }
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"n={n} base-up={rep['base_up']}")
    print("LEARNING CURVE (train_n -> OOS acc on fixed tail):")
    for c in curve:
        print(f"  {c['train_n']:5} -> {c['oos_acc']}")
    r = rep["retrain"]
    print(f"RETRAIN ({r['n_blocks']} blocks of {r['block']}): expanding {r['expanding_mean_oos']} | "
          f"rolling-{r['roll_K']} {r['rolling_mean_oos']} | frozen {r['frozen_mean_oos']} "
          f"(frozen drift {r['frozen_drift']})")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
