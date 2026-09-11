"""F-AV1 redundancy screen — the elimination gate. For each AV feature,
partial correlation with the settlement residual controlling for market
price (mkt_p_up). If the CI includes 0, the feature adds nothing
conditional on price -> ELIMINATE. Screen-only; never QUALIFY.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FEATS = ["spy_ret_15m", "spy_ret_60m", "spy_rvol_30m", "spy_staleness_min",
         "spy_session_open", "uup_ret_60m", "yield_10y_level",
         "yield_10y_chg_1d", "fed_funds_level"]


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def residualize(y, ctrl):
    """Linear residual of y on ctrl (single control var)."""
    r = pearson(ctrl, y)
    n = len(y)
    mc, my = sum(ctrl) / n, sum(y) / n
    sc = math.sqrt(sum((c - mc) ** 2 for c in ctrl) / n)
    sy = math.sqrt(sum((v - my) ** 2 for v in y) / n)
    if sc == 0:
        return [v - my for v in y]
    beta = r * sy / sc
    return [v - (my + beta * (c - mc)) for v, c in zip(y, ctrl)]


def fisher_ci(r, n):
    if abs(r) >= 1 or n < 4:
        return [None, None]
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return [round(math.tanh(lo), 3), round(math.tanh(hi), 3)]


def main():
    rows = [json.loads(l) for l in (RES / "av_features.jsonl").open()
            if l.strip()]
    resid = [r["residual"] for r in rows]
    price = [r["mkt_p_up"] for r in rows]
    out = []
    for f in FEATS:
        xy = [(r["features"].get(f), rr, p)
              for r, rr, p in zip(rows, resid, price)
              if r["features"].get(f) is not None]
        n = len(xy)
        if n < 30:
            out.append({"feature": f, "n": n, "verdict": "INSUFFICIENT_N"})
            continue
        xf = [a for a, _, _ in xy]
        yf = [b for _, b, _ in xy]
        pf = [c for _, _, c in xy]
        # partial corr = corr of residualized feature & residualized target
        pc = pearson(residualize(xf, pf), residualize(yf, pf))
        ci = fisher_ci(pc, n)
        raw = pearson(xf, yf)
        elim = (ci[0] is None) or (ci[0] <= 0 <= ci[1])
        out.append({"feature": f, "n": n,
                    "raw_corr_with_residual": round(raw, 3),
                    "partial_corr_given_price": round(pc, 3),
                    "ci95": ci,
                    "verdict": "ELIMINATE" if elim
                    else "SURVIVES_REDUNDANCY_GATE"})
    survivors = [o for o in out if o.get("verdict")
                 == "SURVIVES_REDUNDANCY_GATE"]
    doc = {"generated_ts": rows and json.loads(
        (RES / "av_features_manifest.json").read_text())["generated_ts"],
        "spec": "F_AV1_SPEC.yaml", "gate": "redundancy (partial corr | price)",
        "n_windows": len(rows), "features": out,
        "n_survivors": len(survivors),
        "survivors": [s["feature"] for s in survivors],
        "verdict": ("ALL_ELIMINATED — no AV feature adds information about "
                    "the residual conditional on market price (expected on "
                    "an efficient short horizon)" if not survivors else
                    "SOME_SURVIVE_TO_RESOLUTION_GATE")}
    (RES / "f_av1_screen.json").write_text(json.dumps(doc, indent=1))
    print(f"F-AV1 redundancy screen: n={len(rows)}  "
          f"survivors={len(survivors)}  verdict={doc['verdict'].split(' ')[0]}")
    for o in out:
        if "partial_corr_given_price" in o:
            print(f"  {o['feature']:20s} raw {o['raw_corr_with_residual']:+.3f}"
                  f"  partial|price {o['partial_corr_given_price']:+.3f}"
                  f"  ci{o['ci95']}  {o['verdict']}")


if __name__ == "__main__":
    main()
