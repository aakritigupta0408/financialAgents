"""Warm-start the M1 per-arm calibrators on history, settle-ordered.

No leakage: rows are replayed in close_ts order and each calibrator is
scored on a window BEFORE training on it (prequential). Prints the
shadow verdict — decayed mean log-loss, calibrated vs raw. Calibrated
below raw means the layer earns its place.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btc_rl.agents import PlattCalibrator          # noqa: E402
from btc_rl.online import KB_CALIB_ARMS, KB_CALIB_NAME  # noqa: E402

kb = [json.loads(l) for l in
      (ROOT / "results" / "kalshi_binary_log.jsonl").open() if l.strip()]

# one decision per (arm, window) — the same envelope the desk trades in
dec = defaultdict(dict)
for r in kb:
    v = r.get("variant") or "kb"
    if (v not in KB_CALIB_ARMS or r.get("actual") is None
            or r.get("mins_left") is None or r["mins_left"] > 12):
        continue
    d = dec[v]
    tk = r["ticker"]
    if tk not in d or r["mins_left"] > d[tk]["mins_left"]:
        d[tk] = r

cals = {}
print(f"{'arm':5s} {'n':>4s} {'a':>7s} {'b':>6s} "
      f"{'LL cal':>8s} {'LL raw':>8s} {'gain':>8s}  verdict")
wins = 0
for v in KB_CALIB_ARMS:
    rows = sorted(dec[v].values(), key=lambda r: r["close_ts"])
    if not rows:
        continue
    c = PlattCalibrator()
    for r in rows:
        c.update(r["p_up"], r["actual"])
    cals[v] = c
    m = c.mean_ll()
    if m is None:
        continue
    gain = m[1] - m[0]                # positive = calibration helps
    wins += gain > 0
    print(f"{v:5s} {c.updates:4d} {c.a:+7.3f} {c.b:6.3f} "
          f"{m[0]:8.4f} {m[1]:8.4f} {gain:+8.4f}  "
          f"{'HELPS' if gain > 0 else 'no gain'}")

out = ROOT / "results" / KB_CALIB_NAME
out.write_text(json.dumps({v: c.to_dict() for v, c in cals.items()}))
print(f"\nwrote {out.name} — {len(cals)} calibrators, "
      f"{wins}/{len(cals)} arms improved by calibration")
print("shadow mode: p_cal is stamped on every row; nothing trades on it")
print("until the shadow week clears M1's gate.")
