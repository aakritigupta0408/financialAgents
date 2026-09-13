"""PROSPECTIVE CAPTURE — the merge-ready live step that turns the retrospective
findings into confirmable ones (§40-41, §46).

At each decision, snapshot POINT-IN-TIME:
  - exact BRTI contract state (official_target, current_brti, distance, settlement)
  - the Independent Oracle probability p_oracle (frozen spec; NO Kalshi input)
  - the market p_market and the oracle-market disagreement
  - independent predictor families B-E (multi-venue / derivatives / options / news)
    when their adapters are available — for the OOS test they could not get before
Then, after the window settles, the desk appends the official outcome so the
promising disagreement edge can be validated PROSPECTIVELY.

SAFE TO MERGE DISABLED: every entry point is a no-op unless
PROSPECTIVE_CAPTURE_ENABLED=1. Adapters are imported defensively. A leak-guard
refuses to capture any settlement/outcome field at decision time.

Contract truth is BRTI. If live BRTI is unhealthy the record is written with
contract_state_quality=DEGRADED/PROXY (§40) rather than silently substituting.
"""
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "research" / "oracle" / "oracle_frozen.json"
OUT = ROOT / "results" / "prospective_capture.jsonl"

_LEAK_FIELDS = {"exact_yes", "result", "expiration_value", "official_expiration_value",
                "close_avg", "settlement", "outcome", "actual"}


def enabled():
    return os.environ.get("PROSPECTIVE_CAPTURE_ENABLED", "").strip() in ("1", "true", "yes")


_spec = None


def _oracle_spec():
    global _spec
    if _spec is None and FROZEN.exists():
        _spec = json.loads(FROZEN.read_text())
    return _spec


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def p_mech(current_brti, official_target, time_remaining_s,
           settle_remaining=0, required_remaining_average=None, sigma=None):
    """MECH_FAIR_BRTI for a live contract state (mirror of the frozen research model)."""
    sp = _oracle_spec()
    sig = sigma if sigma is not None else (sp["sigma_rel_per_sqrt_s"] if sp else 6e-5)
    tte = max(1.0, time_remaining_s)
    p = _Phi((current_brti - official_target) / (sig * current_brti * math.sqrt(tte)))
    if time_remaining_s <= 60 and settle_remaining and settle_remaining > 0 \
            and required_remaining_average is not None:
        s_rem = sig * current_brti * math.sqrt(tte) / math.sqrt(settle_remaining)
        p = _Phi((current_brti - required_remaining_average) / max(1e-6, s_rem))
    return min(1 - 1e-4, max(1e-4, p))


def p_oracle(current_brti, official_target, time_remaining_s, **kw):
    """Independent Oracle probability: recalibrated MECH_FAIR_BRTI. No Kalshi input."""
    pm = p_mech(current_brti, official_target, time_remaining_s, **kw)
    sp = _oracle_spec()
    if not sp:
        return pm
    # piecewise-linear isotonic recalibration
    xs, ys = sp["recal_x"], sp["recal_y"]
    lo, hi = 0, len(xs) - 1
    if pm <= xs[0]:
        return ys[0]
    if pm >= xs[-1]:
        return ys[-1]
    # binary search the interval
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= pm:
            lo = mid
        else:
            hi = mid
    frac = (pm - xs[lo]) / (xs[hi] - xs[lo]) if xs[hi] > xs[lo] else 0.0
    return float(ys[lo] + frac * (ys[hi] - ys[lo]))


def _independent_snapshot():
    """PIT snapshot of independent predictor families, if adapters are available."""
    snap, health = {}, {}
    try:
        from data.adapters import derivatives as dv
        snap["derivatives"] = dv.snapshot() if hasattr(dv, "snapshot") else None
        health["derivatives"] = "OK" if snap.get("derivatives") else "EMPTY"
    except Exception as e:
        health["derivatives"] = f"NA:{str(e)[:40]}"
    for name, mod in (("options", "av_options"), ("news", "av_news")):
        try:
            m = __import__(f"data.adapters.{mod}", fromlist=[mod])
            snap[name] = m.snapshot() if hasattr(m, "snapshot") else None
            health[name] = "OK" if snap.get(name) else "EMPTY"
        except Exception as e:
            health[name] = f"NA:{str(e)[:40]}"
    return snap, health


def capture(window_ctx):
    """Append one PIT prospective-capture record. No-op unless enabled.

    window_ctx must carry live decision-time state ONLY:
      market_window_id, decision_time, current_brti, official_target,
      time_remaining_s, k_prob (market), optional settle_remaining/
      required_remaining_average, brti_quality ('OK'|'DEGRADED'|'PROXY').
    Returns the record (or None if disabled/invalid)."""
    if not enabled():
        return None
    leak = _LEAK_FIELDS & set(window_ctx)
    assert not leak, f"leak-guard: settlement/outcome present at capture: {leak}"
    req = ("market_window_id", "decision_time", "current_brti",
           "official_target", "time_remaining_s", "k_prob")
    if any(window_ctx.get(k) is None for k in req):
        return None
    quality = window_ctx.get("brti_quality", "OK")
    po = p_oracle(window_ctx["current_brti"], window_ctx["official_target"],
                  window_ctx["time_remaining_s"],
                  settle_remaining=window_ctx.get("settle_remaining", 0),
                  required_remaining_average=window_ctx.get("required_remaining_average"))
    pm = p_mech(window_ctx["current_brti"], window_ctx["official_target"],
                window_ctx["time_remaining_s"],
                settle_remaining=window_ctx.get("settle_remaining", 0),
                required_remaining_average=window_ctx.get("required_remaining_average"))
    ind, ind_health = _independent_snapshot()
    sp = _oracle_spec()
    rec = {
        "captured_ts": time.time(),
        "market_window_id": window_ctx["market_window_id"],
        "decision_time": window_ctx["decision_time"],
        "current_brti": window_ctx["current_brti"],
        "official_target": window_ctx["official_target"],
        "brti_distance_to_target": round(window_ctx["current_brti"]
                                         - window_ctx["official_target"], 4),
        "time_remaining_s": window_ctx["time_remaining_s"],
        "contract_state_quality": quality,
        "p_mech": round(pm, 5),
        "p_oracle": round(po, 5),
        "p_market": window_ctx["k_prob"],
        "oracle_market_gap": round(po - window_ctx["k_prob"], 5),
        "independent": ind,
        "independent_health": ind_health,
        "oracle_spec_hash": sp["spec_hash"] if sp else None,
        # outcome is appended later by the settlement hook, never here
        "exact_yes": None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    # smoke: compute oracle probs for a synthetic mid-window state (no capture write)
    st = dict(current_brti=77250.0, official_target=77240.0, time_remaining_s=420)
    print("frozen oracle:", (_oracle_spec() or {}).get("spec_hash"))
    print("p_mech  =", round(p_mech(**st), 4))
    print("p_oracle=", round(p_oracle(**st), 4))
    print("enabled =", enabled())
