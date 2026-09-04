"""T1.1 ABLATION RUNNER — executes T1_1_SPEC.yaml exactly.

Same linear quantile probe both arms (information, not
architecture); train folds < k, evaluate fold k (k=2..N); paired
pinball endpoint with seeded block bootstrap; the FROZEN decision
rule from the spec. REGISTERED mode refuses to run without a FROZEN
manifest; --dry-run exercises machinery on the dry-run dataset and
its artifact is stamped DRY_RUN (never evidence, never a verdict).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SEED = 20260904
QS = (0.1, 0.5, 0.9)
HORIZONS = ("h5", "h15", "h30")
BLOCK = 24                    # ~6h of 15-min obs per bootstrap block
BOOT_N = 2000
QUALIFY_REL = 0.03            # >=3% paired pinball improvement
WORST_FOLD_FLOOR = -0.01      # no fold pays for another
COVERAGE_BAND = (0.72, 0.88)


def fit_quantile(X, y, q, iters=600):
    """Deterministic full-batch subgradient descent on pinball."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    w[0] = np.quantile(y, q)
    scale = np.abs(y - w[0]).mean() or 1.0
    for t in range(iters):
        r = y - Xb @ w
        g = -Xb.T @ np.where(r > 0, q, q - 1.0) / len(y)
        w -= (0.5 * scale / np.sqrt(t + 1)) * g
    return w


def pinball(y, pred, q):
    r = y - pred
    return np.where(r > 0, q * r, (q - 1.0) * r)


def main():
    dry = "--dry-run" in sys.argv
    tag = "_dryrun" if dry else ""
    man_p = RES / f"t1_dataset_manifest{tag}.json"
    if not man_p.exists():
        print(f"REFUSED: no manifest {man_p.name}")
        return 1
    manifest = json.loads(man_p.read_text())
    if not dry and manifest.get("mode") != "FROZEN":
        print("REFUSED: registered run requires a FROZEN manifest")
        return 1
    rows = [json.loads(l) for l in
            (RES / f"t1_dataset{tag}.jsonl").open()]
    n_folds = manifest["folds"]["n"]
    rng = np.random.default_rng(SEED)

    per_h = {}
    for h in HORIZONS:
        y_all = np.array([r["y"][h] for r in rows])
        folds = np.array([r["fold"] for r in rows])
        Xc = np.array([r["x_ctl"] for r in rows])
        Xt = np.hstack([Xc, np.array([r["x_xv"] for r in rows])])
        oof = {arm: {q: np.full(len(rows), np.nan) for q in QS}
               for arm in ("ctl", "trt")}
        fold_rel = []
        for k in range(1, n_folds):
            tr, ev = folds < k, folds == k
            if tr.sum() < 40 or ev.sum() < 10:
                continue
            for arm, X in (("ctl", Xc), ("trt", Xt)):
                mu = X[tr].mean(0)
                sg = X[tr].std(0)
                sg[sg == 0] = 1.0
                Z = (X - mu) / sg
                for q in QS:
                    w = fit_quantile(Z[tr], y_all[tr], q)
                    oof[arm][q][ev] = np.hstack(
                        [np.ones((ev.sum(), 1)), Z[ev]]) @ w
            pc = np.mean([pinball(y_all[ev], oof["ctl"][q][ev], q)
                          for q in QS], axis=0)
            pt = np.mean([pinball(y_all[ev], oof["trt"][q][ev], q)
                          for q in QS], axis=0)
            fold_rel.append(float((pc - pt).mean() / pc.mean()))
        m = ~np.isnan(oof["ctl"][0.5])
        pc = np.mean([pinball(y_all[m], oof["ctl"][q][m], q)
                      for q in QS], axis=0)
        pt = np.mean([pinball(y_all[m], oof["trt"][q][m], q)
                      for q in QS], axis=0)
        diff = pc - pt
        rel = float(diff.mean() / pc.mean())
        nb = len(diff) // BLOCK
        boots = []
        for _ in range(BOOT_N):
            idx = np.hstack([np.arange(b * BLOCK, (b + 1) * BLOCK)
                             for b in rng.integers(0, nb, nb)])
            boots.append(diff[idx].mean() / pc[idx].mean())
        ci = [float(np.quantile(boots, 0.025)),
              float(np.quantile(boots, 0.975))]
        cov = {arm: float(np.mean(
            (y_all[m] >= oof[arm][0.1][m])
            & (y_all[m] <= oof[arm][0.9][m])))
            for arm in ("ctl", "trt")}
        per_h[h] = {
            "n_oof": int(m.sum()),
            "pinball_ctl": round(float(pc.mean()), 3),
            "pinball_trt": round(float(pt.mean()), 3),
            "rel_improvement": round(rel, 4),
            "ci95_block_bootstrap": [round(c, 4) for c in ci],
            "fold_rel": [round(f, 4) for f in fold_rel],
            "worst_fold_rel": round(min(fold_rel), 4),
            "coverage80": {a: round(c, 3) for a, c in cov.items()},
            "mae_ctl": round(float(np.abs(
                y_all[m] - oof["ctl"][0.5][m]).mean()), 2),
            "mae_trt": round(float(np.abs(
                y_all[m] - oof["trt"][0.5][m]).mean()), 2),
            "qualifies": bool(rel >= QUALIFY_REL and ci[0] > 0
                              and min(fold_rel) > WORST_FOLD_FLOOR
                              and all(COVERAGE_BAND[0] <= c
                                      <= COVERAGE_BAND[1]
                                      for c in cov.values()))}
    n_q = sum(1 for h in HORIZONS if per_h[h]["qualifies"])
    verdict = ("DRY_RUN_NO_VERDICT" if dry else
               "QUALIFY" if n_q >= 2 else "REJECT")
    doc = {"generated_ts": int(time.time()),
           "experiment": "T1.1", "spec_sha": manifest["spec_sha"],
           "dataset_sha": manifest["dataset_sha"],
           "mode": "DRY_RUN" if dry else "REGISTERED",
           "seed": SEED, "horizons": per_h,
           "decision_rule": "QUALIFY iff >=2/3 horizons pass "
           f"(rel>={QUALIFY_REL}, CI>0, worst-fold>"
           f"{WORST_FOLD_FLOOR}, coverage in {COVERAGE_BAND})",
           "horizons_qualifying": n_q,
           "verdict": verdict}
    (RES / f"t1_1_result{tag}.json").write_text(
        json.dumps(doc, indent=1))
    print(f"t1_ablation[{doc['mode']}]: verdict {verdict} | " +
          " | ".join(f"{h}: rel {per_h[h]['rel_improvement']:+.2%} "
                     f"CI {per_h[h]['ci95_block_bootstrap']}"
                     for h in HORIZONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
