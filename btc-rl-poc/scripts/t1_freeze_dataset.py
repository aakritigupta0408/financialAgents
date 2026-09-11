"""T1.1 PIT DATASET FREEZER (prepare-only harness, task PM 09-03).

Builds the frozen dataset the moment F1 v2.1 becomes governing:
  15-min grid (v2.1 stitched usability) -> control features (bar-
  derived, PIT-safe) + F-XVENUE block (via xvenue_sync ONLY) +
  forward labels -> results/t1_dataset.jsonl + manifest with hashes.

LAWS:
- FREEZE mode refuses to run unless BOTH the F1 gate artifact shows
  the stitched rule governing/PASS-equivalent AND the missingness
  audit verdict is PASS. --dry-run exercises the machinery on
  current data and writes *_dryrun files (never evidence).
- fng is pinned to 50 for BOTH arms (identical constant cannot
  affect an ablation; live fng history is not PIT-reconstructable).
- One writer: this script owns t1_dataset*.jsonl + manifest.
"""
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_f1_sufficiency import machine_outages, scan  # noqa: E402
from btc_rl.features import (compute_features, feature_vector,  # noqa
                             tech_feature_vector,
                             trend_feature_vector)
from btc_rl.sources import fetch_range  # noqa: E402
import xvenue_sync  # noqa: E402

GRID_STEP_MIN = 15
HORIZONS = (5, 15, 30)
PATH_LO, PATH_HI = 60, 30
N_FOLDS = 5
BARS_NEEDED = 300


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def gate_allows_freeze():
    try:
        g = json.loads((RES / "f1_capture_qualification.json")
                       .read_text())
        m = json.loads((RES / "f1_missingness_audit.json")
                       .read_text())
        stitched_ok = g.get("proposed_v2_1_stitched", {}).get(
            "would_pass") is True
        governing = "GOVERNING" in str(
            g.get("proposed_v2_1_stitched", {}).get("status", ""))
        return (m.get("verdict") == "PASS"
                and stitched_ok and governing), \
            {"missingness": m.get("verdict"),
             "stitched_would_pass": stitched_ok,
             "stitched_governing": governing}
    except Exception as e:
        return False, {"error": str(e)}


def main():
    dry = "--dry-run" in sys.argv
    ok, why = gate_allows_freeze()
    if not ok and not dry:
        print(f"REFUSED: freeze gate not satisfied {why} "
              "(use --dry-run to test machinery)")
        return 1
    tag = "_dryrun" if dry else ""

    per_min, _sec, _integ = scan()
    outages = machine_outages(per_min)
    out_min = set()
    for a, b in outages:
        out_min.update(range(a + 1, b))
    all_minutes = set()
    for v in per_min:
        all_minutes |= set(per_min[v])
    m_lo, m_hi = min(all_minutes), max(all_minutes)

    fetch_end = min((m_hi + HORIZONS[-1] + 5) * 60,
                    int(time.time()) - 120)
    bars = fetch_range(
        datetime.fromtimestamp((m_lo - BARS_NEEDED - 5) * 60,
                               tz=timezone.utc),
        datetime.fromtimestamp(fetch_end, tz=timezone.utc))
    by_min = {int(b["ts"] // 60): b for b in bars}

    grid = [m for m in range(m_lo + PATH_LO, m_hi - PATH_HI,
                             GRID_STEP_MIN)
            if not any(x in out_min
                       for x in range(m - PATH_LO, m + PATH_HI + 1))]
    xv = xvenue_sync.features_at_batch([m * 60 for m in grid])

    rows, dropped = [], {"bars": 0, "xv": 0, "label": 0}
    for m in grid:
        window = [by_min[x] for x in range(m - BARS_NEEDED + 1, m + 1)
                  if x in by_min]
        if len(window) < BARS_NEEDED - 5 or m not in by_min:
            dropped["bars"] += 1
            continue
        x_xv = xv.get(m * 60)
        if x_xv is None:
            dropped["xv"] += 1
            continue
        y = {}
        base = by_min[m]["close"]
        for h in HORIZONS:
            if m + h in by_min:
                y[f"h{h}"] = round(
                    1e4 * math.log(by_min[m + h]["close"] / base), 2)
        if len(y) != len(HORIZONS):
            dropped["label"] += 1
            continue
        feat = compute_features(window, 50)
        x_ctl = (feature_vector(feat) + trend_feature_vector(feat)
                 + tech_feature_vector(feat))
        rows.append({"ts": m * 60,
                     "x_ctl": [round(v, 6) for v in x_ctl],
                     "x_xv": [round(v, 6) for v in x_xv],
                     "y": y})

    fold_sz = len(rows) // N_FOLDS
    for i, r in enumerate(rows):
        r["fold"] = min(i // max(1, fold_sz), N_FOLDS - 1)

    data_path = RES / f"t1_dataset{tag}.jsonl"
    with data_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    shard_manifest = xvenue_sync.shard_manifest()
    manifest = {
        "frozen_ts": int(time.time()),
        "mode": "DRY_RUN" if dry else "FROZEN",
        "gate_state_at_freeze": why,
        "spec": "T1_1_SPEC.yaml", "spec_sha": sha(ROOT /
                                                  "T1_1_SPEC.yaml"),
        "extractor_shas": {
            "xvenue_sync.py": sha(ROOT / "scripts/xvenue_sync.py"),
            "t1_freeze_dataset.py": sha(__file__),
            "btc_rl/features.py": sha(ROOT / "btc_rl/features.py")},
        "xv_feature_order": list(xvenue_sync.XV_FEATURES),
        "grid": {"step_min": GRID_STEP_MIN, "path_rule":
                 f"[m-{PATH_LO}, m+{PATH_HI}] outage-free",
                 "span_utc": [time.strftime(
                     "%m-%d %H:%M", time.gmtime(m_lo * 60)),
                     time.strftime("%m-%d %H:%M",
                                   time.gmtime(m_hi * 60))]},
        "machine_outages": [[a, b] for a, b in outages],
        "n_rows": len(rows), "dropped": dropped,
        "folds": {"n": N_FOLDS, "sizes": [
            sum(1 for r in rows if r["fold"] == k)
            for k in range(N_FOLDS)]},
        "fng_pinned": 50,
        "label": "coinbase 1m candle close, forward log-return bps",
        "dataset_sha": sha(data_path),
        "shard_shas": shard_manifest}
    (RES / f"t1_dataset_manifest{tag}.json").write_text(
        json.dumps(manifest, indent=1))
    print(f"t1_freeze_dataset[{manifest['mode']}]: {len(rows)} rows "
          f"(dropped {dropped}), folds {manifest['folds']['sizes']}, "
          f"dataset_sha {manifest['dataset_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
