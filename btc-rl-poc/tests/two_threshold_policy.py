"""Two-threshold class policy for the 80/80 target, leakage-free.

Blend p with the market mid (weight fit on train), then call UP when
p >= tau_up, DOWN when p <= 1 - tau_dn, else abstain. Tune (tau_up,
tau_dn) on the chronological train split for per-class precision >= 0.8
(with margin) maximizing min(recall_up, recall_dn); evaluate once on the
held-out tail. Recall denominators include abstained instances (the
strict definition)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

rows = [json.loads(l) for l in
        (ROOT / "results" / "kalshi_binary_log.jsonl").read_text().splitlines()]
done = sorted((r for r in rows if r.get("variant", "kb") == "kb"
               and r["actual"] is not None), key=lambda r: r["made_ts"])
cut = int(len(done) * 0.6)
train, test = done[:cut], done[cut:]


def blend_weight(rows_):
    q = [r for r in rows_ if r.get("mkt_p_up") is not None]
    num = den = 0.0
    for r in q:
        d = r["p_up"] - r["mkt_p_up"]
        num += d * (r["actual"] - r["mkt_p_up"])
        den += d * d
    return max(0.0, min(1.0, num / den)) if den > 1e-9 else 0.5


W = blend_weight(train)


def pb(r):
    mk = r.get("mkt_p_up")
    return W * r["p_up"] + (1 - W) * mk if mk is not None else r["p_up"]


def evaluate(rows_, tu, td):
    stats = {"UP": [0, 0, 0], "DOWN": [0, 0, 0]}  # called, correct, actual
    for r in rows_:
        p = pb(r)
        stats["UP" if r["actual"] else "DOWN"][2] += 1
        if p >= tu:
            stats["UP"][0] += 1
            stats["UP"][1] += r["actual"]
        elif p <= 1 - td:
            stats["DOWN"][0] += 1
            stats["DOWN"][1] += 1 - r["actual"]
    out = {}
    for cls, (c, k, a) in stats.items():
        out[cls] = {"prec": k / c if c else 0, "rec": k / a if a else 0,
                    "called": c, "actual": a}
    out["coverage"] = (stats["UP"][0] + stats["DOWN"][0]) / len(rows_)
    return out


best = None
for tu100 in range(50, 86):
    for td100 in range(50, 86):
        tu, td = tu100 / 100, td100 / 100
        e = evaluate(train, tu, td)
        if (e["UP"]["prec"] >= 0.82 and e["DOWN"]["prec"] >= 0.82
                and e["UP"]["called"] >= 25 and e["DOWN"]["called"] >= 25):
            score = min(e["UP"]["rec"], e["DOWN"]["rec"])
            if best is None or score > best[0]:
                best = (score, tu, td, e)

print(f"blend weight (train-fit): {W:.2f}")
if best is None:
    print("no threshold pair reaches 82/82 precision on train")
else:
    _, tu, td, e = best
    print(f"TRAIN pick: tau_up={tu:.2f} tau_dn={td:.2f}")
    for cls in ("UP", "DOWN"):
        print(f"  {cls:>5} train: prec {e[cls]['prec']:.1%} rec {e[cls]['rec']:.1%}")
    h = evaluate(test, tu, td)
    print(f"HELD-OUT (n={len(test)}, coverage {h['coverage']:.1%}):")
    import math
    for cls in ("UP", "DOWN"):
        s = h[cls]
        # Wilson 95% lower bound on precision
        n2, p2 = s["called"], s["prec"]
        z = 1.96
        lo = ((p2 + z*z/(2*n2) - z*math.sqrt((p2*(1-p2) + z*z/(4*n2))/n2))
              / (1 + z*z/n2)) if n2 else 0
        print(f"  {cls:>5}: precision {s['prec']:.1%} (95% lower {lo:.1%}, "
              f"n={s['called']}) · recall {s['rec']:.1%} of {s['actual']}")
