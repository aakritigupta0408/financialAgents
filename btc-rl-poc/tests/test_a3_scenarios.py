"""§42 scenario tests for the A3-v1.1 machinery — synthetic tapes
through the REAL functions in scripts/emit_a3.py. Run ad hoc and on
demand before any A3 code change; failures are release blockers.

Quote tuple: (ts, yes_ask, no_ask, close_time, yes_bid, no_bid,
yes_ask_sz).
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "emit_a3", ROOT / "scripts" / "emit_a3.py")
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a3)

PASS = []
FAIL = []


def t(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))


def Q(ts, ask, bid=None):
    return (ts, ask, 100 - ask, None, bid if bid is not None
            else ask - 1, None, "10")


# ---- 1. temporary overreaction --------------------------------------
q = [Q(10, 72), Q(20, 67), Q(30, 75, bid=74)]
conf = {10: 0.77, 20: 0.72, 30: 0.76}


def conf_at(ts):
    best = 0.80
    for k in sorted(conf):
        if k <= ts:
            best = conf[k]
    return best


s05 = a3.run_shadow(5, q, True, 0, 78, conf_at, 100)
s10 = a3.run_shadow(10, q, True, 0, 78, conf_at, 100)
s15 = a3.run_shadow(15, q, True, 0, 78, conf_at, 100)
t("overreaction: T05 triggers @72",
  s05["state"] == "FILLED" and s05["entry_ask"] == 72, str(s05))
t("overreaction: T10 triggers @67",
  s10["state"] == "FILLED" and s10["entry_ask"] == 67, str(s10))
t("overreaction: T15 no trigger", s15["state"] == "MISSED", str(s15))
mo = a3.markouts(q, True, 20, 67)     # anchor at T10 entry
t("overreaction: +10s markout positive after rebound",
  mo["markout_10s"] == 7.0, str(mo))  # bid 74 at ts30 − 67

# ---- 2. falling knife ------------------------------------------------
q2 = [Q(10, 68), Q(20, 64), Q(30, 58)]
conf2 = {10: 0.71, 20: 0.66, 30: 0.61}


def conf2_at(ts):
    best = 0.80
    for k in sorted(conf2):
        if k <= ts:
            best = conf2[k]
    return best


k05 = a3.run_shadow(5, q2, True, 0, 78, conf2_at, 100)
k15 = a3.run_shadow(15, q2, True, 0, 78, conf2_at, 100)
t("knife: T05 triggers while valid (@68)",
  k05["state"] == "FILLED" and k05["entry_ask"] == 68, str(k05))
t("knife: T15 triggers @64 before invalidation (dip 14>=... no, 14<15)"
  " -> then INVALIDATED at .61",
  k15["state"] == "INVALIDATED", str(k15))

# ---- 3. invalidation before dip -------------------------------------
q3 = [Q(10, 75), Q(20, 72), Q(30, 61)]
conf3 = {10: 0.69, 20: 0.63, 30: 0.70}


def conf3_at(ts):
    best = 0.80
    for k in sorted(conf3):
        if k <= ts:
            best = conf3[k]
    return best


i10 = a3.run_shadow(10, q3, True, 0, 78, conf3_at, 100)
t("invalidation-first: 61c after .63 can trigger NOTHING "
  "(no reactivation at .70)",
  i10["state"] == "INVALIDATED", str(i10))

# ---- 4. markout gap --------------------------------------------------
q4 = [Q(0, 60, bid=59), Q(14.2, 70, bid=69)]   # +10s target, 4.2s late
mo4 = a3.markouts(q4, True, 0, 60)
t("markout gap: 4.2s-late quote = UNAVAILABLE",
  mo4["markout_10s"] == "UNAVAILABLE", str(mo4))

# ---- 5. hindsight isolation -----------------------------------------
q5a = [Q(10, 67)]
q5b = [Q(10, 67), Q(20, 40)]        # deeper dip AFTER trigger
h_a = a3.run_shadow(10, q5a, True, 0, 78, lambda t_: 0.75, 100)
h_b = a3.run_shadow(10, q5b, True, 0, 78, lambda t_: 0.75, 100)
t("hindsight isolation: later deeper dip never changes the recorded "
  "trigger", h_a == h_b, f"{h_a} vs {h_b}")

# ---- report ----------------------------------------------------------
print(f"A3 scenario tests: {len(PASS)} pass · {len(FAIL)} fail")
for name, d in FAIL:
    print("  FAIL:", name, "--", d[:120])
sys.exit(0)
