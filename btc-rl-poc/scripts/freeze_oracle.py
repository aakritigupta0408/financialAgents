"""Freeze the Independent Oracle for live prospective use.

Fits the MECH_FAIR_BRTI sigma + the isotonic recalibration on ALL backfilled
windows and writes research/oracle/oracle_frozen.json — a self-contained spec the
live capture step applies deterministically (no training at runtime). Includes a
content hash so drift is detectable.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "results" / "brti_decision_dataset.jsonl"
MFB = ROOT / "research" / "oracle" / "mech_fair_brti_result.json"
OUT = ROOT / "research" / "oracle" / "oracle_frozen.json"


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def p_mech_row(r, sig):
    lvl, tte = r["current_brti"], max(1.0, r["time_remaining_s"])
    p = Phi(r["brti_distance_to_target"] / (sig * lvl * math.sqrt(tte)))
    rem = r.get("settle_remaining") or 0
    req = r.get("required_remaining_average")
    if r["time_remaining_s"] <= 60 and rem > 0 and req is not None:
        s_rem = sig * lvl * math.sqrt(tte) / math.sqrt(rem)
        p = Phi((lvl - req) / max(1e-6, s_rem))
    return min(1 - 1e-4, max(1e-4, p))


def main():
    from sklearn.isotonic import IsotonicRegression
    sig = json.load(MFB.open())["sigma_rel_per_sqrt_s"]
    rows = [json.loads(l) for l in DEC.open() if l.strip()]
    rows = [r for r in rows if r.get("current_brti") is not None and r["time_remaining_s"] >= 1]
    perwin = {}
    for r in rows:
        perwin[r["market_window_id"]] = perwin.get(r["market_window_id"], 0) + 1
    pm = np.array([p_mech_row(r, sig) for r in rows])
    y = np.array([r["exact_yes"] for r in rows], float)
    w = np.array([1.0 / perwin[r["market_window_id"]] for r in rows])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(pm, y, sample_weight=w)
    # export breakpoints (piecewise-linear); apply via np.interp at runtime
    xs = np.linspace(0, 1, 101)
    ys = np.clip(iso.predict(xs), 1e-4, 1 - 1e-4)
    spec = {
        "model": "independent_oracle_v1",
        "mech": "MECH_FAIR_BRTI (Gaussian diffusion + settlement-aware final-60s)",
        "sigma_rel_per_sqrt_s": sig,
        "recalibration": "isotonic (monotone), applied via linear interp on p_mech",
        "recal_x": [round(float(v), 4) for v in xs],
        "recal_y": [round(float(v), 5) for v in ys],
        "fit_windows": len(perwin), "fit_rows": len(rows),
        "provenance": "scripts/freeze_oracle.py on results/brti_decision_dataset.jsonl",
        "usage": "p_mech = mech(state); p_oracle = interp(p_mech, recal_x, recal_y)",
    }
    spec["spec_hash"] = hashlib.sha256(
        json.dumps({k: spec[k] for k in ("sigma_rel_per_sqrt_s", "recal_x", "recal_y")},
                   sort_keys=True).encode()).hexdigest()[:12]
    OUT.write_text(json.dumps(spec, indent=1))
    print(f"froze oracle -> {OUT.name}  sigma={sig}  hash={spec['spec_hash']}  "
          f"({len(rows)} rows / {len(perwin)} windows)")


if __name__ == "__main__":
    main()
