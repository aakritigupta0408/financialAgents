"""Pre-register TEST_V2 target N from a power analysis (directive).

Is 672 ("one week = 96x7") scientifically enough, or arbitrary? This estimates the
per-window variance of the evaluation metric on DEV data ONLY (TRAIN/VAL/TEST_V1 —
TEST_V2 outcomes remain unseen) and derives the sample size needed to detect a
pre-registered minimum effect at a target power. Safe: nothing here touches V2.

Model of the test: the registered reference is the p=0.5 baseline, whose per-window
log-loss is the constant -ln(0.5)=0.6931. A finalist's per-window log-loss LL_i has
mean mu and std sigma_d (estimated OOS on DEV). Detecting an edge delta = 0.6931 - mu
is a one-sample test of mean(LL_i) < 0.6931, so:

    N = (z_{1-alpha/2} + z_{1-beta})^2 * sigma_d^2 / delta^2      (two-sided alpha=0.05)

We report N across a delta grid and the minimum detectable effect (MDE) at N=672,
then pre-register the target N for the chosen meaningful delta*.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl import research_events as EV        # noqa: E402
from btc_rl.test_v2_firewall import guard_rows  # noqa: E402

T = ROOT / "research" / "true15m"
DS = T / "TRUE15M_DATASET_V1.jsonl"
SPLIT = T / "SPLIT_SPEC_V1.json"
OUT = T / "TEST_V2_POWER_ANALYSIS.json"
SEED = 17
Z_A = 1.959963985           # z_{0.975}, two-sided alpha=0.05
Z_POW = {0.80: 0.8416212336, 0.90: 1.2815515655}
BASE_LL = -math.log(0.5)    # 0.693147 — constant per-window log-loss of the 0.5 baseline
DELTA_GRID = [0.002, 0.003, 0.005, 0.01, 0.02]
DELTA_STAR = 0.01           # PRE-REGISTERED minimum meaningful edge (nats of log-loss)
FALLBACK_N = 672


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _req_n(sigma, delta, power):
    return int(math.ceil(((Z_A + Z_POW[power]) ** 2) * (sigma ** 2) / (delta ** 2)))


def _mde(sigma, n, power):
    return round((Z_A + Z_POW[power]) * sigma / math.sqrt(n), 5)


def _quantile_p(Xtr, Dtr, Xte):
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    qp = {a: CatBoostRegressor(loss_function=f"Quantile:alpha={a}", depth=3, iterations=150,
                               learning_rate=0.03, random_seed=SEED, verbose=False)
          .fit(Xtr, Dtr).predict(Xte) for a in alphas}
    out = []
    for i in range(len(Xte)):
        qs = [qp[a][i] for a in alphas]; p_lt = 0.5
        for j in range(len(alphas) - 1):
            if qs[j] <= 0 <= qs[j + 1] or qs[j] >= 0 >= qs[j + 1]:
                den = (qs[j + 1] - qs[j]) or 1e-9
                p_lt = alphas[j] + (0 - qs[j]) / den * (alphas[j + 1] - alphas[j]); break
        else:
            p_lt = 0.02 if qs[-1] < 0 else (0.98 if qs[0] > 0 else 0.5)
        out.append(1 - p_lt)
    return np.array(out)


def _per_window_ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run():
    t0 = time.time()
    rows = guard_rows([json.loads(l) for l in DS.open() if l.strip()], "test_v2_power_analysis")
    rows.sort(key=lambda r: r["T0"])
    fids = list(rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in fids] for r in rows], float)
    y = np.array([r["Y"] for r in rows], int)
    D = np.array([r["D"] for r in rows], float)
    sp = json.loads(SPLIT.read_text())
    tr_end = sp["train"]["n"]; va_end = tr_end + sp["validation"]["n"]
    # DEV-eval pool = VAL + TEST_V1 (V1 is spent; its outcomes are allowed for variance est.)
    sc = StandardScaler().fit(X[:tr_end])
    Xtr = sc.transform(X[:tr_end]); Xev = sc.transform(X[tr_end:]); yev = y[tr_end:]

    # two representative OOS scorers; use the higher-variance one (conservative)
    ll = {}
    pq = _quantile_p(Xtr, D[:tr_end], Xev)
    ll["quantile"] = _per_window_ll(pq, yev)
    lr = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED).fit(Xtr, y[:tr_end])
    ll["logistic"] = _per_window_ll(lr.predict_proba(Xev)[:, 1], yev)

    scorer_stats = {}
    for name, v in ll.items():
        scorer_stats[name] = {"mean_ll": round(float(v.mean()), 5),
                              "sigma_per_window": round(float(v.std(ddof=1)), 5),
                              "n_eval": int(len(v))}
    # conservative sigma: the LARGER per-window std across scorers
    sigma = max(s["sigma_per_window"] for s in scorer_stats.values())

    req = {f"delta={d}": {"power_0.80": _req_n(sigma, d, 0.80),
                          "power_0.90": _req_n(sigma, d, 0.90)} for d in DELTA_GRID}
    mde_672 = {"power_0.80": _mde(sigma, FALLBACK_N, 0.80),
               "power_0.90": _mde(sigma, FALLBACK_N, 0.90)}
    n_star = _req_n(sigma, DELTA_STAR, 0.80)
    # pre-register: round up to a whole number of days (96/day), never below fallback
    target = max(FALLBACK_N, int(math.ceil(n_star / 96.0) * 96))
    is_672_enough = n_star <= FALLBACK_N

    doc = {
        "schema_version": "test-v2-power-analysis-1", "generated_at": time.time(),
        "basis": "DEV only (TRAIN fit; VAL+TEST_V1 eval). TEST_V2 outcomes UNSEEN.",
        "reference_baseline": "p=0.5 constant (per-window log-loss = 0.693147)",
        "metric": "per-window log-loss; sigma = std across DEV-eval windows",
        "sigma_per_window_used": sigma,
        "scorer_stats": scorer_stats,
        "test": "one-sample, two-sided alpha=0.05; N=(z_a+z_b)^2 * sigma^2 / delta^2",
        "required_N_by_delta": req,
        "mde_at_672_windows": mde_672,
        "pre_registered_delta_star": DELTA_STAR,
        "pre_registered_delta_star_rationale": (
            "0.01 nats log-loss (~1.4% of the 0.693 baseline) is the smallest edge we would "
            "call practically meaningful for a 15-min ~50%-base-rate binary; smaller dev blips "
            "(~0.004, unstable) are treated as noise."),
        "required_N_for_delta_star_power80": n_star,
        "fallback_N": FALLBACK_N,
        "pre_registered_target_N": target,
        "is_672_enough_for_delta_star": is_672_enough,
        "interpretation": (
            f"sigma≈{sigma}. At N=672 the test resolves only an edge of "
            f"~{mde_672['power_0.80']} nats at 80% power; detecting the pre-registered "
            f"delta*={DELTA_STAR} needs N≈{n_star}. Pre-registered target set to {target} "
            f"({'672 is sufficient' if is_672_enough else 'raised above 672'}). "
            "Edges below the resolvable MDE are TEST-POWER-LIMITED, not disprovable by V2 — "
            "a distinct, honest limit from information-limited."),
        "seal_note": "This analysis fixes N BEFORE opening V2; it never inspects V2 outcomes.",
    }
    OUT.write_text(json.dumps(doc, indent=1))
    EV.emit("DISCOVERY", f"TEST_V2 target pre-registered: N={target} (delta*={DELTA_STAR})",
            lane="L9", severity="high",
            fact=f"sigma_per_window≈{sigma}; MDE@672≈{mde_672['power_0.80']} (80% power); "
                 f"N for delta*={DELTA_STAR} is ≈{n_star}; 672_enough={is_672_enough}.",
            interpretation=doc["interpretation"],
            next_action="freeze finalists on DEV+WF+VAL; open TEST_V2 once it reaches target N.",
            files=["research/true15m/TEST_V2_POWER_ANALYSIS.json"],
            duration_ms=int((time.time() - t0) * 1000))
    print(f"power_analysis: sigma≈{sigma}  MDE@672(80%)≈{mde_672['power_0.80']}  "
          f"N*(delta*={DELTA_STAR})≈{n_star}  -> pre_registered_target_N={target} "
          f"(672_enough={is_672_enough})")
    for d, r in req.items():
        print(f"  {d:12} N@80%={r['power_0.80']:>7}  N@90%={r['power_0.90']:>7}")


if __name__ == "__main__":
    run()
