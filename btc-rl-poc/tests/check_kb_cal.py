"""Unit checks for kb per-phase calibration + leakage audit."""
from btc_rl.online import _kb_cal_weights, _kb_phase

assert _kb_phase(12) == "early" and _kb_phase(7) == "mid" and _kb_phase(2) == "late"

def rows(ph_min, pairs):
    return [{"actual": y, "p_raw": p, "p_up": p, "mins_left": ph_min}
            for p, y in pairs]

# perfectly informative early calls -> full strength (clamped 1.2)
w = _kb_cal_weights(rows(12, [(0.9, 1), (0.1, 0)] * 15))
assert w["early"] == 1.2, w
# uninformative early calls (confident but coin-flip outcomes) -> shrink to ~0
w = _kb_cal_weights(rows(12, [(0.9, 1), (0.9, 0)] * 15))
assert w["early"] == 0.0, w
# under 20 settled rows -> identity (no correction from tiny samples)
w = _kb_cal_weights(rows(12, [(0.9, 1)] * 10))
assert w["early"] == 1.0
# unsettled rows are excluded from the fit (leakage guard)
mixed = rows(12, [(0.9, 1), (0.9, 0)] * 15) + [
    {"actual": None, "p_raw": 0.99, "p_up": 0.99, "mins_left": 12}] * 50
assert _kb_cal_weights(mixed)["early"] == 0.0
# phases are independent: late stays 1.0 with no late data
assert _kb_cal_weights(rows(12, [(0.9, 1), (0.1, 0)] * 15))["late"] == 1.0
print("kb calibration unit checks ok")
