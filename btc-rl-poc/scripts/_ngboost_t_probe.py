"""Isolated NGBoost Student-t (TGT-E2) probe — run as a SUBPROCESS.

NGBoost's Student-t score function overflows (df -> inf) on this near-symmetric,
zero-mean D target and SEGFAULTS the interpreter — a native crash a Python
try/except cannot trap. Running it here in a child process lets the parent sweep
observe the crash empirically (non-zero / negative exit) and record TGT-E2 as
UNSTABLE with evidence, instead of dying itself.

On success: prints one JSON line with P(D>=0) metrics on VAL to stdout.
On instability: crashes / exits non-zero — the parent interprets that as UNSTABLE.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss as sk_ll, roc_auc_score
from ngboost import NGBRegressor
from ngboost.distns import T as TDist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import metrics as MET  # noqa: E402

TT = ROOT / "research" / "true15m"
DS = TT / "TRUE15M_DATASET_V1.jsonl"
SPLIT = TT / "SPLIT_SPEC_V1.json"
SEED = 17


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    rows = [json.loads(l) for l in DS.open() if l.strip()]
    rows.sort(key=lambda r: r["T0"])
    fids = list(rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in fids] for r in rows], float)
    y = np.array([r["Y"] for r in rows], int)
    D = np.array([r["D"] for r in rows], float)
    sp = json.loads(SPLIT.read_text())
    tr_end = sp["train"]["n"]; va_end = tr_end + sp["validation"]["n"]
    sc = StandardScaler().fit(X[:tr_end])
    Xtr, Xva = sc.transform(X[:tr_end]), sc.transform(X[tr_end:va_end])
    yva = y[tr_end:va_end]
    Dtr = D[:tr_end]
    Dtr_n = Dtr / (Dtr.std() or 1.0)

    ng = NGBRegressor(Dist=TDist, n_estimators=250, learning_rate=0.02,
                      verbose=False, random_state=SEED).fit(Xtr, Dtr_n)
    dd = ng.pred_dist(Xva); loc = dd.loc; scale = getattr(dd, "scale", None)
    sc2 = scale if scale is not None else np.full(len(loc), 1.0)
    p = np.clip(np.array([_phi(loc[i] / (sc2[i] or 1.0)) for i in range(len(loc))]), 1e-6, 1 - 1e-6)
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite predictive probabilities")
    auc = round(float(roc_auc_score(yva, p)), 4) if len(set(yva)) > 1 else None
    out = {"log_loss": round(float(sk_ll(yva, p, labels=[0, 1])), 5),
           "brier": round(float(MET.brier(p.tolist(), yva.tolist())), 5),
           "auc": auc,
           "accuracy": round(float(((p >= 0.5).astype(int) == yva).mean()), 4)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
