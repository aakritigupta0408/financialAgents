"""LEARNABILITY / BAYES-ERROR analysis — is 15-min BTC direction a function of our features?

The overfit experiment shows 100% in-sample, ~0.55-0.70 OOS. That gap is either (a)
irreducible label noise (windows with the SAME features finish up ~half the time -> not a
function of these features), or (b) modelling/data headroom we haven't captured. This
decides which, with data, not opinion:

  * kNN neighbourhood determinism: for each TEST window, look at its k nearest TRAIN windows
    in feature space. p_hat = mean neighbour label. If neighbours agree ~100%, the label is
    locally DETERMINED by features (learnable); if ~50%, it's a coin flip given features.
  * Bayes-ceiling proxy = mean over test of max(p_hat, 1-p_hat): the accuracy an oracle that
    knew the local label rate would reach. Compare to what our best model actually gets:
      - model_acc  ~= Bayes_ceiling  -> we are AT the feature limit; need NEW features.
      - model_acc  << Bayes_ceiling  -> real modelling headroom; keep improving the model.
  * Conditional entropy H(y | neighbourhood) — the residual noise in bits.
  * Does ENRICHING features (barrier -> rich) tighten neighbourhoods (lower Bayes error)?
    If yes, more/better features can raise the ceiling (supports the forward program).

Run at both entry regimes (mid-window ~11min, open ~15min). Feature families compared.
PAPER / SIMULATION research; read-only.
"""
import sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from insample_reconstruction import load_windows          # open regime
from combined_student import rich_features
from fm_benchmark import windows as bench_windows          # mid-window regime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
import json

OUT = ROOT / "research" / "learnability_report.json"


def feat_matrix(W, keys=None):
    feats = [rich_features(w) for w in W]
    if keys is None:
        keys = list(feats[0].keys())
    X = np.nan_to_num(np.array([[f[k] for k in keys] for f in feats], float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return X, keys


def bayes_probe(X, y, ks=(5, 15, 30, 50)):
    n = len(y); mid = int(n * 0.6)
    sc = StandardScaler().fit(X[:mid])
    Xtr, Xte = sc.transform(X[:mid]), sc.transform(X[mid:])
    ytr, yte = y[:mid], y[mid:]
    out = {}
    for k in ks:
        nn = NearestNeighbors(n_neighbors=min(k, mid)).fit(Xtr)
        _, idx = nn.kneighbors(Xte)
        phat = ytr[idx].mean(axis=1)                         # local label rate
        ceil = float(np.mean(np.maximum(phat, 1 - phat)))    # Bayes-accuracy proxy
        knn_acc = float(((phat >= 0.5).astype(int) == yte).mean())
        det = float(np.mean((phat >= 0.8) | (phat <= 0.2)))  # fraction locally determined
        flip = float(np.mean((phat > 0.4) & (phat < 0.6)))   # fraction coin-flip
        # conditional entropy of y given the neighbourhood rate (binned)
        pe = np.clip(phat, 1e-6, 1 - 1e-6)
        H = float(np.mean(-(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe))))
        out[f"k{k}"] = {"bayes_ceiling_acc": round(ceil, 4), "knn_acc": round(knn_acc, 4),
                        "frac_determined_>.8": round(det, 4), "frac_coinflip_.4-.6": round(flip, 4),
                        "cond_entropy_bits": round(H, 4)}
    return out


def near_dup(X, y, eps_pct=5):
    """Label agreement among the closest feature pairs (bottom eps_pct% of pairwise
    distances by a sample). ~0.5 => identical features, opposite outcomes => irreducible."""
    sc = StandardScaler().fit(X); Z = sc.transform(X)
    rng = np.random.default_rng(0)
    n = len(Z); m = min(n, 400)
    sub = rng.choice(n, m, replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    d, idx = nn.kneighbors(Z[sub])                  # nearest OTHER point
    dd = d[:, 1]; jj = idx[:, 1]
    thr = np.percentile(dd, eps_pct)
    close = dd <= thr
    if close.sum() == 0:
        return None
    agree = (y[sub][close] == y[jj][close]).mean()
    return {"n_pairs": int(close.sum()), "label_agreement": round(float(agree), 4),
            "median_dist_all": round(float(np.median(dd)), 3), "dist_thr": round(float(thr), 3)}


def model_acc(X, y):
    n = len(y); mid = int(n * 0.6)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                        l2_regularization=2.0, min_samples_leaf=20, random_state=17)
    gb.fit(X[:mid], y[:mid])
    return round(float((gb.predict(X[mid:]) == y[mid:]).mean()), 4)


def regime(name, W):
    y = np.array([w["y"] for w in W], int)
    Xr, keys = feat_matrix(W)
    # barrier-only subset (the physics baseline)
    bkeys = [k for k in keys if k in ("z", "theo", "z_drift", "theo_drift", "dist_bps", "sigT_bps")]
    Xb = Xr[:, [keys.index(k) for k in bkeys]]
    return {
        "n": len(W), "base_up": round(float(y.mean()), 4),
        "bayes_rich": bayes_probe(Xr, y),
        "bayes_barrier": bayes_probe(Xb, y, ks=(15, 30)),
        "near_dup_rich": near_dup(Xr, y),
        "model_acc_rich": model_acc(Xr, y),
        "model_acc_barrier": model_acc(Xb, y),
    }


def run():
    rep = {}
    print("=== MID-WINDOW (~11 min left) ===")
    Wm = bench_windows()
    for w in Wm:
        w.setdefault("id", "")
    rep["mid_window"] = regime("mid", Wm)
    print(json.dumps(rep["mid_window"], indent=1))
    print("=== OPEN (~15 min left) ===")
    Wo = load_windows()
    rep["open"] = regime("open", Wo)
    print(json.dumps(rep["open"], indent=1))
    OUT.write_text(json.dumps(rep, indent=1))
    # headroom verdict
    for reg in ("mid_window", "open"):
        r = rep[reg]
        ceil = r["bayes_rich"]["k30"]["bayes_ceiling_acc"]; ma = r["model_acc_rich"]
        print(f"\n[{reg}] Bayes-ceiling(k30) {ceil} | model {ma} | headroom {ceil-ma:+.3f} | "
              f"near-dup label agreement {r['near_dup_rich']['label_agreement'] if r['near_dup_rich'] else 'na'}")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    run()
